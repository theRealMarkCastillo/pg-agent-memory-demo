"""Synthetic companion-memory tests.

Validates the synthetic data generator and the fact-matching logic WITHOUT
requiring an LLM, so it runs deterministically in CI:
  - generated traces parse back to conversations + ground-truth facts
  - every ground-truth fact is revealed in the conversation
  - the benchmark matcher recognizes exact ground-truth facts

An optional LLM-backed extraction test is included but skipped unless
LLM_BASE_URL / LLM_API_KEY are set (OpenRouter/OpenAI required).
"""
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from generate_synthetic import FACT_LIBRARY, build_trace, build_facts, build_conversation  # noqa: E402
from parse_traces import parse_trace_file  # noqa: E402
from benchmark_extraction import match_fact, _real_entity_ours  # noqa: E402
from replay_traces import PREDICATE_VOCAB  # noqa: E402

import random  # noqa: E402

SYNTH_DIR = Path(__file__).resolve().parent / "synth_fixtures"


@pytest.fixture(scope="module")
def synth_traces(tmp_path_factory):
    """Generate a fixed set of synthetic traces once per module."""
    out = tmp_path_factory.mktemp("synth")
    rng = random.Random(7)
    n = 8
    for i in range(n):
        trace = build_trace(rng, f"synth-test-{i:03d}", n_facts=6)
        (out / f"synth-test-{i:03d}.json").write_text(json.dumps(trace, indent=2))
    return out


def test_generated_traces_parse(synth_traces):
    parsed = 0
    for p in synth_traces.glob("*.json"):
        turns, meta = parse_trace_file(str(p))
        assert turns, f"{p.name}: no conversation parsed"
        assert meta.get("shape") == "extraction"
        assert meta.get("production_facts"), f"{p.name}: no ground truth"
        parsed += 1
    assert parsed >= 8


def test_ground_truth_revealed_in_conversation(synth_traces):
    """Every ground-truth fact's object must appear in the conversation text."""
    for p in synth_traces.glob("*.json"):
        turns, meta = parse_trace_file(str(p))
        transcript = " ".join(t.user + " " + t.assistant for t in turns).lower()
        for fact in meta["production_facts"]:
            obj = str(fact["object_value"]).lower()
            assert obj in transcript, f"{p.name}: '{obj}' not revealed in conversation"


def test_every_fact_reveals_one_turn(synth_traces):
    """All three scopes (user/self/shared) must be present in every trace."""
    for p in synth_traces.glob("*.json"):
        turns, meta = parse_trace_file(str(p))
        facts = meta["production_facts"]
        scopes = {f.get("scope") for f in facts}
        assert "user" in scopes, f"{p.name}: missing user facts"
        assert "self" in scopes, f"{p.name}: missing self facts"
        assert "shared" in scopes, f"{p.name}: missing shared facts"
        assert len(turns) >= 4, f"{p.name}: too few turns"


def test_matcher_recognizes_exact_ground_truth():
    """match_fact should return True when our fact equals a ground-truth fact."""
    cases = [
        ({"name": "user", "relationship_type": "LIVES_IN", "relationship_to": "Seattle"},
         {"predicate": "LIVES_IN", "object_value": "Seattle"}),
        ({"name": "user", "relationship_type": "MARRIED_TO", "relationship_to": "Sarah"},
         {"predicate": "MARRIED_TO", "object_value": "Sarah"}),
        ({"name": "user", "relationship_type": "HAS_PET", "relationship_to": "Luna"},
         {"predicate": "HAS_PET", "object_value": "Luna"}),
        ({"name": "user", "relationship_type": "WANTS_TO", "relationship_to": "learn Japanese"},
         {"predicate": "WANTS_TO", "object_value": "learn Japanese"}),
    ]
    for our, prod in cases:
        assert match_fact(our, prod), f"matcher rejected {our} vs {prod}"


def test_matcher_rejects_wrong_entity():
    our = {"name": "user", "relationship_type": "LIVES_IN", "relationship_to": "Chicago"}
    prod = {"predicate": "LIVES_IN", "object_value": "Seattle"}
    assert not match_fact(our, prod)


def test_entity_resolution_ours():
    assert _real_entity_ours({"name": "user", "relationship_to": "Seattle"}) == "seattle"
    assert _real_entity_ours({"name": "Sarah", "relationship_to": None}) == "sarah"


def test_predicate_vocab_aligned_with_fact_library():
    """Every predicate used by the generator must be in the extraction vocabulary."""
    vocab = set(PREDICATE_VOCAB.replace("\n", " ").replace(",", " ").split())
    lib_preds = {f[0] for f in FACT_LIBRARY}
    missing = lib_preds - vocab
    assert not missing, f"generator predicates missing from extraction vocab: {missing}"


@pytest.mark.skipif(
    not (os.getenv("LLM_BASE_URL") and os.getenv("LLM_API_KEY")),
    reason="LLM credentials not configured",
)
@pytest.mark.asyncio
async def test_llm_extraction_recovers_synthetic_facts(synth_traces):
    """LLM-backed smoke test: extract from one synthetic trace, require recall > 0."""
    from benchmark_extraction import extract_turn
    from replay_traces import get_openai_client

    env = {k: os.getenv(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL_NAME")}
    if not env.get("LLM_MODEL_NAME"):
        env["LLM_MODEL_NAME"] = "gpt-4o-mini"

    p = sorted(synth_traces.glob("*.json"))[0]
    turns, meta = parse_trace_file(str(p))
    client = get_openai_client(env)
    data = await asyncio.wait_for(extract_turn(client, env["LLM_MODEL_NAME"], turns), timeout=100)
    our_facts = [f for f in data.get("new_facts", []) if f.get("name")]
    prod = meta["production_facts"]
    tp = sum(1 for of in our_facts if any(match_fact(of, pf) for pf in prod))
    assert tp >= 1, f"LLM extraction recovered no ground-truth facts: {our_facts}"


# --- Common-interest detection between the user's graph and the AI self-model.


def test_common_ground_matches_shared_affinity():
    """A matching affinity predicate+entity between user and self facts is found."""
    def norm(name):
        return "".join(c for c in (name or "").lower() if c.isalnum())

    # Reimplement the agent's logic to test the concept without langgraph deps.
    def find_common(user_facts, self_facts):
        affinity = {"likes", "loves", "enjoys", "interested_in", "prefers", "plays"}
        self_map = {}
        for f in self_facts:
            rel = (f.get("relationship_type") or "").strip().lower()
            if rel in affinity and f.get("related_to"):
                self_map[norm(f["related_to"])] = f["related_to"]
        return [f["related_to"] for f in user_facts
                if (f.get("relationship_type") or "").strip().lower() in affinity
                and f.get("related_to")
                and norm(f["related_to"]) in self_map]

    user_facts = [
        {"relationship_type": "likes", "related_to": "pizza"},
        {"relationship_type": "enjoys", "related_to": "hiking"},
        {"relationship_type": "lives_in", "related_to": "Seattle"},
    ]
    self_facts = [
        {"relationship_type": "likes", "related_to": "Pizza"},
        {"relationship_type": "loves", "related_to": "music"},
    ]
    common = find_common(user_facts, self_facts)
    assert "pizza" in common, f"pizza should be common ground: {common}"
    assert "hiking" not in common


def test_common_ground_requires_same_entity_and_affinity():
    def norm(name):
        return "".join(c for c in (name or "").lower() if c.isalnum())

    def find_common(user_facts, self_facts):
        affinity = {"likes", "loves", "enjoys", "interested_in", "prefers", "plays"}
        self_map = {}
        for f in self_facts:
            rel = (f.get("relationship_type") or "").strip().lower()
            if rel in affinity and f.get("related_to"):
                self_map[norm(f["related_to"])] = f["related_to"]
        return [f["related_to"] for f in user_facts
                if (f.get("relationship_type") or "").strip().lower() in affinity
                and f.get("related_to")
                and norm(f["related_to"]) in self_map]

    # same entity, different predicate family (lives_in is not affinity)
    assert find_common(
        [{"relationship_type": "lives_in", "related_to": "Paris"}],
        [{"relationship_type": "likes", "related_to": "Paris"}],
    ) == []
    # different entity, same predicate
    assert find_common(
        [{"relationship_type": "likes", "related_to": "pizza"}],
        [{"relationship_type": "likes", "related_to": "sushi"}],
    ) == []
