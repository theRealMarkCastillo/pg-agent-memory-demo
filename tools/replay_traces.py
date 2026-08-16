"""Replay LangSmith trace conversations through the companion memory engine.

For each (user, assistant) turn parsed from the traces, this script:
  1. Runs the same extraction prompt used by the companion agent's
     ``extract_memory`` node (demo_agents/agents/companion_agent.py).
  2. POSTs extracted facts / terminated edges / episodes to the memory engine.
  3. Optionally evaluates recall by querying /companion/context.

Usage:
    python tools/replay_traces.py                     # replay into memory engine
    python tools/replay_traces.py --traces-dir traces --user-id replay-demo
    python tools/replay_traces.py --dry-run           # just show what would be stored
    python tools/replay_traces.py --max-turns 5       # limit turns per trace

Reads LLM_* and EMBEDDING_* from .env like the rest of the stack.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from parse_traces import parse_trace_dir

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://localhost:8001")

# Canonical relation predicates, aligned with the production extractor's vocabulary.
# Our extractor emits these as relationship_type so facts match ground truth.
PREDICATE_VOCAB = """LIVES_IN, FROM, MOVED_TO, GREW_UP_IN, HAS_JOB, WORKS_AT, WORKED_AT, WORKS_ON,
MARRIED_TO, DATING, PARENT_OF, SIBLING_OF, FRIEND_OF, FAMILY_MEMBER, HAS_PET,
LIKES, LOVES, DISLIKES, HATES, ENJOYS, INTERESTED_IN, PREFERS, ALLERGIC_TO,
HAS_CONDITION, DIETARY_RESTRICTION, WATCHES, READS, PLAYS, LISTENS_TO, WRITES,
HAS_AGE, BIRTHDAY_ON, SPEAKS, IDENTIFIES_AS, PRACTICES, HAS_NAME, WANTS_TO,
PLANS_TO, AVOIDS, SKILLED_AT, CERTIFIED_IN, VALUES, USES, SUPPORTS, TRUSTS,
APPRECIATES, BUILT, MISSES, SHARED_EXPERIENCE, SHARED_INTEREST, SHARED_RITUAL,
INSIDE_JOKE, AGREED_ON, SHARED_MEMORY, RELATIONSHIP_MILESTONE, DISAGREE_ON, HAS_FACT"""

EXTRACTION_PROMPT = """Analyze this conversation between a user and an AI companion. Extract durable,
identity-relevant memories. Capture facts about the USER, facts the COMPANION reveals about
ITSELF (its own preferences, history, values, how it's changing), and SHARED relationship
facts (inside jokes, shared rituals, milestones, what "we/us/our" — the growing bond).

User said: {user_msg}
Companion said: {agent_msg}

Return ONLY valid JSON with this exact shape:
{{
  "user_facts": [
    {{"name": "<entity>", "entity_type": "<type>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", "valence": <number -1..1>, "intensity": <number 0..1>}}
  ],
  "self_facts": [
    {{"name": "<companion entity>", "entity_type": "<type>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", "valence": <number -1..1>, "intensity": <number 0..1>}}
  ],
  "shared_facts": [
    {{"name": "<entity>", "entity_type": "<type>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", "valence": <number -1..1>, "intensity": <number 0..1>}}
  ],
  "terminated_edges": [
    {{"name": "<entity>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>"}}
  ],
  "episode_content": "<1-2 sentence summary of the exchange>"
}}

Rules:
- user_facts: durable facts the USER revealed (preferences, people, locations, goals, pets, jobs, hobbies, health, relationships). These are about the user.
- self_facts: durable facts the COMPANION revealed about ITSELF — its own stated preferences, history, values, how it feels or is changing. Only extract these if the companion says something concrete about itself (NOT the user). Prefer the companion's name as name.
- shared_facts: relationship facts that belong to BOTH — "we always...", "our inside joke", shared rituals, relationship milestones, agreements.
- Use entity_type values like: person, location, goal, preference, hobby, job, pet, event, self
- DIRECTION: name is the SUBJECT, relationship_to is the OBJECT. Express each fact in ONE canonical direction and never emit the reverse. E.g. "user likes Gemini" -> name="user", relationship_to="Gemini", relationship_type="LIKES".
- CONSISTENCY: reuse stable entity names (the user's name or "user"; the companion's name for self_facts). Never duplicate a fact already stated.
- PREDICATES: relationship_type MUST be one of these UPPER_SNAKE values:
{PREDICATE_VOCAB}
  - Match the verb: "I enjoy hiking" -> ENJOYS. "I love sushi" -> LOVES. "I play guitar" -> PLAYS. "I live in Seattle" -> LIVES_IN. "I'm a software developer" -> HAS_JOB. "My wife Sarah is a nurse" -> MARRIED_TO (relationship_to="Sarah"). "I have a cat named Luna" -> HAS_PET (relationship_to="Luna"). "I want to learn Japanese" -> WANTS_TO (relationship_to="learn Japanese"). "I used to work at Google" -> WORKED_AT. "I hate spiders" -> HATES. "I'm allergic to peanuts" -> ALLERGIC_TO. "I'm vegetarian" -> DIETARY_RESTRICTION.
  - TEMPORAL: extract CURRENT state. Past jobs use WORKED_AT, current use WORKS_AT.
- VALENCE & INTENSITY: valence reflects emotional tone (-1 very sad/loss, +1 joyful); intensity how strongly felt (0..1). Pet dying -> valence -0.9 intensity 0.9. New job -> +0.6/0.5. Neutral -> 0/0.3.
- terminated_edges: relationships that are NO LONGER true (user or companion moved on, dropped a goal, ended a fact)
- If nothing meaningful in a category, return an empty list for it.
"""


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


def get_openai_client(env: dict):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=env.get("LLM_BASE_URL"),
        api_key=env.get("LLM_API_KEY"),
        timeout=110.0,
        max_retries=1,
    )


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    return content.strip()


async def extract_turn(client, model: str, user_msg: str, agent_msg: str, timeout: float = 90.0) -> dict:
    """Run the companion extraction prompt for a single turn."""
    prompt = EXTRACTION_PROMPT.format(user_msg=user_msg, agent_msg=agent_msg)

    async def _call():
        return await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

    try:
        response = await asyncio.wait_for(_call(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        return {}
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(_strip_code_fence(content))
    except Exception:
        return {}
    if isinstance(data, list):
        # Legacy bare-list response: treat as user facts.
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
        "episode_content": data.get("episode_content") or "",
    }


async def store_extraction(engine, user_id: str, data: dict):
    """POST extracted facts (user/self/shared), terminations, and episode."""
    results = {"facts": 0, "terminated": 0, "episodes": 0}
    for subject, key in (("user", "user_facts"), ("self", "self_facts"),
                         ("shared", "shared_facts")):
        for f in data.get(key, []) or []:
            if not f.get("name"):
                continue
            r = await engine.post(
                "/companion/facts",
                json={
                    "user_id": user_id,
                    "name": f["name"],
                    "entity_type": f.get("entity_type") or "entity",
                    "relationship_to": f.get("relationship_to"),
                    "relationship_type": f.get("relationship_type"),
                    "subject": subject,
                    "valence": f.get("valence", 0.0),
                    "intensity": f.get("intensity", 0.5),
                },
            )
            if r.status_code == 200:
                results["facts"] += 1
    for t in data.get("terminated_edges", []) or []:
        if not t.get("name"):
            continue
        r = await engine.post(
            "/companion/facts/terminate",
            json={
                "user_id": user_id,
                "name": t["name"],
                "relationship_to": t.get("relationship_to"),
                "relationship_type": t.get("relationship_type"),
            },
        )
        if r.status_code == 200:
            results["terminated"] += 1
    episode_content = data.get("episode_content")
    if episode_content:
        r = await engine.post(
            "/companion/episodes",
            json={"user_id": user_id, "content": episode_content},
        )
        if r.status_code == 200:
            results["episodes"] += 1
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", default="traces", help="directory of trace exports")
    parser.add_argument("--user-id", default="trace-replay", help="memory engine user_id")
    parser.add_argument("--dry-run", action="store_true", help="print instead of store")
    parser.add_argument("--max-turns", type=int, default=0, help="limit turns per trace (0=all)")
    args = parser.parse_args()

    env = load_env(Path(".env"))
    model = env.get("LLM_MODEL_NAME") or "gpt-4o-mini"
    base_url = env.get("LLM_BASE_URL") or "http://localhost:8001"
    llm = get_openai_client(env) if not args.dry_run else None
    engine_url = os.getenv("MEMORY_ENGINE_URL", MEMORY_ENGINE_URL)

    parsed = parse_trace_dir(args.traces_dir)
    if not parsed:
        print(f"No conversations found in {args.traces_dir}/")
        return 1
    print(f"Parsed {len(parsed)} traces with conversations.")

    engine = httpx.AsyncClient(base_url=engine_url, timeout=30.0)
    try:
        total = {"facts": 0, "terminated": 0, "episodes": 0}
        for turns, meta in parsed:
            print(f"=== {meta['file']} [{meta.get('shape')}] turns={len(turns)} ===")
            if args.max_turns:
                turns = turns[: args.max_turns]
            if args.dry_run:
                for i, turn in enumerate(turns):
                    print(f"  turn {i}: user={turn.user[:60]!r}")
                continue
            results = await _process_turns(llm, engine, model, args.user_id, turns)
            for k in total:
                total[k] += results[k]
            print(f"  -> {results}")
        print(f"\nTotal stored: {total}")
    finally:
        await engine.aclose()
    return 0


async def _process_turns(llm, engine, model, user_id, turns, concurrency: int = 4):
    """Extract + store turns concurrently. Returns aggregated counters."""
    sem = asyncio.Semaphore(concurrency)
    results = {"facts": 0, "terminated": 0, "episodes": 0}

    async def process(i, turn):
        async with sem:
            data = await extract_turn(llm, model, turn.user, turn.assistant)
            res = await store_extraction(engine, user_id, data)
            facts = [f["name"] for f in data.get("new_facts", [])]
            print(f"  turn {i}: {len(facts)} facts {facts[:5]} / {res}", flush=True)
            return res

    for res in await asyncio.gather(*(process(i, t) for i, t in enumerate(turns))):
        for k in results:
            results[k] += res[k]
    return results


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
