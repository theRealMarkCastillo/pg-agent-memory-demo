import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_extraction import match_fact, score_facts
from parse_traces import _turns_from_conversation_block, Turn
import replay_traces


def test_matcher_rejects_contradiction_scope_and_substrings():
    prediction = {
        "name": "user",
        "relationship_type": "LIKES",
        "relationship_to": "coffee",
        "subject": "user",
    }
    assert not match_fact(
        prediction, {"predicate": "DISLIKES", "object_value": "coffee"}
    )
    assert not match_fact(
        prediction, {"predicate": "LIKES", "object_value": "coffee", "scope": "self"}
    )
    assert not match_fact(
        dict(prediction, relationship_to="York"),
        {"predicate": "LIKES", "object_value": "New York"},
    )
    assert score_facts(
        [prediction] * 5,
        [
            {"predicate": "LIKES", "object_value": "coffee"},
            {"predicate": "LIKES", "object_value": "tea"},
        ],
    ) == (1, 4, 1)


def test_parser_preserves_trailing_user():
    turns = _turns_from_conversation_block(
        "User: Seattle\nCompanion: Nice city\nUser: Moved to Austin"
    )
    assert turns == [Turn("Seattle", "Nice city"), Turn("Moved to Austin", "")]


async def test_replay_commits_source_order_with_provenance(monkeypatch):
    stored = []

    async def extract(llm, model, user, assistant):
        if user == "old":
            await asyncio.sleep(0.03)
        return {
            "user_facts": [
                {
                    "name": "user",
                    "relationship_to": user,
                    "relationship_type": "lives_in",
                }
            ]
        }

    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

        def raise_for_status(self):
            pass

    class Engine:
        async def post(self, path, json):
            stored.append((path, json))
            return Response({"episode_id": "ep-" + str(len(stored))})

    monkeypatch.setattr(replay_traces, "extract_turn", extract)
    await replay_traces._process_turns(
        None, Engine(), "fake", "alice", [Turn("old", "a"), Turn("new", "b")]
    )
    facts = [data for path, data in stored if path == "/companion/facts"]
    assert [f["relationship_to"] for f in facts] == ["old", "new"]
    assert [f["source_episode_id"] for f in facts] == ["ep-1", "ep-3"]
