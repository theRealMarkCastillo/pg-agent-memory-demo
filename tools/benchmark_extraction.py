"""Benchmark our companion fact extraction against production ground truth.

Loads LangSmith traces that contain BOTH a conversation and the production
system's FactExtractionOutput (predicate/object_value/importance). Replays each
conversation through our extraction prompt, then scores our facts against the
production ground-truth facts using normalized entity + predicate matching.

Metrics reported:
  - Precision: fraction of our facts that match a production fact
  - Recall:    fraction of production facts we recovered
  - F1

Usage:
    python tools/benchmark_extraction.py --traces-dir /tmp/conv --sample 50
    python tools/benchmark_extraction.py --traces-dir /tmp/conv --all
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from parse_traces import parse_trace_file
from replay_traces import PREDICATE_VOCAB, _strip_code_fence, get_openai_client

# production predicates we care about (skip the generic HAS_FACT noise)
_SKIP_PREDICATES = {"HAS_FACT"}


def load_env(env_path: Path) -> dict:
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _norm(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def _predicate_key(pred: str) -> str:
    """Normalize a predicate to a canonical key for matching."""
    return _norm(pred).replace(" ", "_")


# Fallback map for legacy free-form relations -> production predicates. The
# extractor now emits UPPER_SNAKE directly, so this only catches stragglers.
_OUR_RELATION_TO_PRED = {
    "LIVES_IN": "LIVES_IN",
    "FROM": "FROM",
    "MOVED_TO": "MOVED_TO",
    "WORKS_AS": "HAS_JOB",
    "WORKS_AT": "WORKS_AT",
    "WORKED_AT": "WORKED_AT",
    "MARRIED_TO": "MARRIED_TO",
    "DATING": "DATING",
    "PARENT_OF": "PARENT_OF",
    "SIBLING_OF": "SIBLING_OF",
    "FRIEND_OF": "FRIEND_OF",
    "HAS_PET": "HAS_PET",
    "LIKES": "LIKES",
    "LOVES": "LOVES",
    "DISLIKES": "DISLIKES",
    "HATES": "HATES",
    "ENJOYS": "ENJOYS",
    "INTERESTED_IN": "INTERESTED_IN",
    "PREFERS": "PREFERS",
    "ALLERGIC_TO": "ALLERGIC_TO",
    "HAS_CONDITION": "HAS_CONDITION",
    "IDENTIFIES_AS": "IDENTIFIES_AS",
    "HAS_NAME": "HAS_NAME",
    "WANTS_TO": "WANTS_TO",
    "PLANS_TO": "PLANS_TO",
    "AVOIDS": "AVOIDS",
    "SKILLED_AT": "SKILLED_AT",
    "HAS_HOBBY": "ENJOYS",
    "VALUES": "VALUES",
    "USES": "USES",
    "ROMANTIC_PARTNER_OF": "DATING",
    "SPENDS_MONEY_ON": "HAS_FACT",
    "HAS_CHILD": "HAS_FACT",
    "HAS_CHILDREN": "HAS_FACT",
    "HAS_COWORKER": "HAS_FACT",
    "COWORKER_OF": "HAS_FACT",
    "PREGNANT_WITH": "HAS_FACT",
    "CO_PARENT_WITH": "HAS_FACT",
    "IS": "HAS_FACT",
    "HAS": "HAS_FACT",
    "PREFERS_TO_BE_CALLED": "HAS_NAME",
}


def our_fact_predicate(fact: dict) -> str:
    rel = _norm(fact.get("relationship_type") or "")
    if not rel:
        return "HAS_FACT"
    # Extractor emits UPPER_SNAKE predicates aligned with production; pass through.
    upper = rel.replace(" ", "_").upper()
    return _OUR_RELATION_TO_PRED.get(upper, upper)


# Production scopes: 'user' facts have implicit subject "user"; the entity is object_value.
# Our facts: name is the subject, relationship_to is the object. Normalize both so the
# "real entity" (the thing that isn't the user/companion) is what we compare.
def _real_entity_ours(fact: dict) -> str:
    name = _norm(fact.get("name") or "")
    rel_to = _norm(fact.get("relationship_to") or "")
    return rel_to or name


def _real_entity_prod(fact: dict) -> str:
    return _norm(fact.get("object_value") or "")


def _entity_overlap(a: str, b: str) -> bool:
    """Match normalized entity identity exactly; aliases must be explicit in labels."""
    return bool(a and b and a == b)


def match_fact(our_fact: dict, prod_fact: dict) -> bool:
    """Return True if our fact aligns with a production fact (entity+predicate)."""
    if our_fact.get("subject", "user") != prod_fact.get(
        "scope", prod_fact.get("subject", "user")
    ):
        return False
    our_entity = _real_entity_ours(our_fact)
    prod_entity = _real_entity_prod(prod_fact)
    if not _entity_overlap(our_entity, prod_entity):
        return False
    our_pred = our_fact_predicate(our_fact).lower()
    prod_pred = _predicate_key(prod_fact.get("predicate") or "").lower()
    # predicate must be the same family
    return our_pred == prod_pred


def _dedup_facts(facts: list[dict]) -> list[dict]:
    """Drop near-duplicate facts (same entity+predicate+target) after per-turn merging."""
    seen: set = set()
    out = []
    for f in facts:
        key = (
            f.get("subject", "user"),
            _norm(f.get("name") or ""),
            our_fact_predicate(f),
            _norm(f.get("relationship_to") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


async def _extract_single(client, model: str, user_msg: str, agent_msg: str) -> dict:
    """Run the extraction prompt on ONE user/companion turn pair."""
    prompt = (
        "Extract durable, identity-relevant facts from this exchange. Capture facts "
        "about the USER, facts the COMPANION reveals about ITSELF, and SHARED "
        "relationship facts.\n"
        "User facts: preferences, people, locations, goals, pets, jobs, hobbies, "
        "health, relationships. Self facts: what the companion says about itself. "
        "Shared facts: 'we/our' relationship details.\n\n"
        f"User: {user_msg}\nCompanion: {agent_msg}\n\n"
        "Return ONLY valid JSON:\n"
        '{"user_facts": [{"name": "<entity>", "entity_type": "<type>", '
        '"relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", '
        '"valence": <number -1..1>, "intensity": <number 0..1>}],\n'
        '"self_facts": [{"name": "<companion entity>", "entity_type": "<type>", '
        '"relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", '
        '"valence": <number -1..1>, "intensity": <number 0..1>}],\n'
        '"shared_facts": [{"name": "<entity>", "entity_type": "<type>", '
        '"relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", '
        '"valence": <number -1..1>, "intensity": <number 0..1>}],\n'
        '"terminated_edges": [], "episode_content": ""}\n'
        "Rules: name=subject entity, relationship_to=object. One canonical direction only. "
        "Use entity_type: person, location, goal, preference, hobby, job, pet, event, self. "
        "valence: emotional tone (-1 sad/loss, +1 joyful); intensity: how strongly felt (0..1). "
        "relationship_type MUST be one of:\n"
        f"{PREDICATE_VOCAB}\n"
        'Examples: "I live in Seattle"->LIVES_IN, "I love sushi"->LOVES, '
        '"I play guitar"->PLAYS, "my wife Sarah"->MARRIED_TO, '
        '"I have a cat Luna"->HAS_PET, "I want to learn Japanese"->WANTS_TO. '
        "If nothing meaningful in a category, return an empty list for it."
    )
    response = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(_strip_code_fence(content))
    except Exception:
        return {}
    if isinstance(data, list):
        data = {"user_facts": data}
    if not isinstance(data, dict):
        return {}
    for key in ("user_facts", "self_facts", "shared_facts", "terminated_edges"):
        if not isinstance(data.get(key), list):
            data[key] = []
        data[key] = [x for x in data[key] if isinstance(x, dict)]
    return {
        "user_facts": data["user_facts"],
        "self_facts": data["self_facts"],
        "shared_facts": data["shared_facts"],
        "terminated_edges": data["terminated_edges"],
        "episode_content": "",
    }


async def extract_turn(
    client, model: str, turns, max_chars: int = 1200, per_turn_concurrency: int = 4
) -> dict:
    """Run extraction per turn over the whole conversation and merge the facts.

    Production's extractor processes short conversation snippets, not one huge
    transcript. We extract from every user/companion turn pair (concurrently),
    merge the per-turn facts, and dedupe. This recovers facts from the full
    conversation instead of truncating to a window.
    """
    clean = [
        t
        for t in turns
        if not t.user.startswith("[CONTEXT]") and "YOUR MUSING" not in t.user
    ]
    if not clean:
        clean = turns[-1:]

    sem = asyncio.Semaphore(per_turn_concurrency)

    async def one(t):
        async with sem:
            try:
                return await asyncio.wait_for(
                    _extract_single(client, model, t.user, t.assistant), timeout=45
                )
            except (asyncio.TimeoutError, Exception):
                return {}

    results = await asyncio.gather(*(one(t) for t in clean))
    merged = {"user_facts": [], "self_facts": [], "shared_facts": []}
    for data in results:
        for k in merged:
            merged[k].extend(data.get(k, []) or [])
    return {
        "user_facts": _dedup_facts(merged["user_facts"]),
        "self_facts": _dedup_facts(merged["self_facts"]),
        "shared_facts": _dedup_facts(merged["shared_facts"]),
        "terminated_edges": [],
        "episode_content": "",
    }


def score_facts(predictions, labels):
    """Maximum one-to-one matching; duplicate predictions count as false positives."""
    matched = {}

    def assign(i, seen):
        for j, label in enumerate(labels):
            if j in seen or not match_fact(predictions[i], label):
                continue
            seen.add(j)
            if j not in matched or assign(matched[j], seen):
                matched[j] = i
                return True
        return False

    tp = sum(assign(i, set()) for i in range(len(predictions)))
    return tp, len(predictions) - tp, len(labels) - tp


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", default="/tmp/conv")
    parser.add_argument("--sample", type=int, default=30, help="max traces to evaluate")
    parser.add_argument(
        "--all", action="store_true", help="evaluate every paired trace"
    )
    args = parser.parse_args()

    env = load_env(Path(".env"))
    model = env.get("LLM_MODEL_NAME") or "gpt-4o-mini"

    paired = []
    for p in Path(args.traces_dir).glob("*.json"):
        turns, meta = parse_trace_file(str(p))
        prod = [
            f
            for f in (meta.get("production_facts") or [])
            if f.get("predicate") not in _SKIP_PREDICATES
        ]
        if turns and prod:
            paired.append((str(p), turns, prod))

    if args.all:
        sample = paired
    else:
        import random

        random.seed(42)
        sample = random.sample(paired, min(args.sample, len(paired)))

    print(
        f"Evaluating {len(sample)} conversations with ground-truth facts "
        f"(of {len(paired)} paired traces)"
    )

    client = get_openai_client(env)
    tp = fp = fn = 0
    per_trace = []
    sem = asyncio.Semaphore(2)

    async def evaluate(path, turns, prod):
        nonlocal tp, fp, fn
        async with sem:
            try:
                data = await asyncio.wait_for(
                    extract_turn(client, model, turns), timeout=100
                )
            except (asyncio.TimeoutError, Exception):
                data = {}
        our_facts = [
            f
            for f in [
                dict(f, subject=subject)
                for subject, key in (
                    ("user", "user_facts"),
                    ("self", "self_facts"),
                    ("shared", "shared_facts"),
                )
                for f in data.get(key, [])
            ]
            if f.get("name")
        ]
        local_tp, local_fp, local_fn = score_facts(our_facts, prod)
        tp += local_tp
        fp += local_fp
        fn += local_fn
        precision = local_tp / (local_tp + local_fp) if local_tp + local_fp else 0.0
        recall = local_tp / (local_tp + local_fn) if local_tp + local_fn else 0.0
        per_trace.append(
            (
                path.split("/")[-1][:12],
                len(prod),
                len(our_facts),
                round(precision, 2),
                round(recall, 2),
            )
        )
        print(
            f"  {path.split('/')[-1][:12]} prod={len(prod)} ours={len(our_facts)} "
            f"P={precision:.2f} R={recall:.2f}",
            flush=True,
        )

    await asyncio.gather(*(evaluate(*args) for args in sample))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\n=== SUMMARY ({len(sample)} conversations) ===")
    print(f"true positives: {tp}, false positives: {fp}, false negatives: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
