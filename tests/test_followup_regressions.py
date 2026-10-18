"""Regressions for the follow-up review: scoped terminations, single-valued facts,
repeatable demo seeding, swarm pull mode with code-held leases, and lexical recall."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage
from conftest import post, get

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_agents"))


@pytest.fixture
def agents(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-no-external-calls")
    monkeypatch.setenv("LLM_MODEL_NAME", "test")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1")
    return SimpleNamespace(
        companion=importlib.import_module("agents.companion_agent"),
        swarm=importlib.import_module("agents.swarm_agent"),
        tools=importlib.import_module("agents.tools"),
    )


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    async def ainvoke(self, messages):
        return next(self.outputs)


def edges(facts, relationship_type):
    return {f["related_to"] for f in facts if f["relationship_type"] == relationship_type}


async def seed_user(client, user):
    for target, rel in (("Brooklyn", "lives_in"), ("hiking", "enjoys"), ("TechCorp", "works_at")):
        await post(
            client,
            "/companion/facts",
            user_id=user,
            name="user",
            entity_type="person",
            relationship_to=target,
            relationship_type=rel,
        )


async def test_terminate_tool_with_target_only_ends_just_that_fact(client, agents):
    user = "scoped_terminate_user"
    await seed_user(client, user)
    await agents.tools.terminate_companion_relationship.ainvoke(
        {"user_id": user, "name": "user", "relationship_to": "Brooklyn"}
    )
    facts = (await get(client, "/companion/context", user_id=user))["graph_facts"]
    assert edges(facts, "lives_in") == set()
    assert edges(facts, "enjoys") == {"hiking"}
    assert edges(facts, "works_at") == {"TechCorp"}


async def test_unfiltered_terminate_is_rejected(client, agents):
    user = "unfiltered_terminate_user"
    await seed_user(client, user)
    with pytest.raises(Exception):
        await agents.tools.terminate_companion_relationship.ainvoke(
            {"user_id": user, "name": "user"}
        )
    resp = await client.post(
        "/companion/facts/terminate", json={"user_id": user, "name": "user"}
    )
    assert resp.status_code == 422
    facts = (await get(client, "/companion/context", user_id=user))["graph_facts"]
    assert len(facts) == 3


async def test_single_valued_predicate_supersedes_previous_value(client):
    user = "moved_user"
    await seed_user(client, user)
    stored = await post(
        client,
        "/companion/facts",
        user_id=user,
        name="user",
        entity_type="person",
        relationship_to="Manhattan",
        relationship_type="LIVES_IN",
    )
    assert stored["superseded"] == 1
    # Multi-valued predicates accumulate instead.
    await post(
        client,
        "/companion/facts",
        user_id=user,
        name="user",
        entity_type="person",
        relationship_to="poetry",
        relationship_type="enjoys",
    )
    facts = (await get(client, "/companion/context", user_id=user))["graph_facts"]
    assert edges(facts, "lives_in") == {"Manhattan"}
    assert edges(facts, "enjoys") == {"hiking", "poetry"}
    history = await get(
        client,
        "/companion/facts/provenance",
        user_id=user,
        name="user",
        relationship_to="Brooklyn",
        relationship_type="lives_in",
    )
    assert [s["status"] for s in history["sources"]] == ["INACTIVE"]
    assert history["sources"][0]["valid_until"]


async def test_single_valued_supersession_is_scoped_by_subject(client):
    user = "subject_scope_user"
    for subject, place in (("user", "Brooklyn"), ("self", "a quiet coastal town")):
        await post(
            client,
            "/companion/facts",
            user_id=user,
            name="Iris",
            entity_type="person",
            relationship_to=place,
            relationship_type="lives_in",
            subject=subject,
        )
    ctx = await get(client, "/companion/context", user_id=user)
    assert edges(ctx["graph_facts"], "lives_in") == {"Brooklyn"}
    assert edges(ctx["self_facts"], "lives_in") == {"a quiet coastal town"}


async def test_extraction_terminates_self_fact_in_self_scope(client, agents, monkeypatch):
    from agents.companion_extraction import EXTRACTION_PROMPT

    # The extractor must be asked which scope an ended fact lived in.
    terminated_shape = EXTRACTION_PROMPT.split('"terminated_edges"')[1].split("]")[0]
    assert '"subject"' in terminated_shape
    user = "self_terminate_user"
    await post(
        client,
        "/companion/facts",
        user_id=user,
        name="Iris",
        entity_type="self",
        relationship_to="poetry",
        relationship_type="writes",
        subject="self",
    )
    monkeypatch.setattr(
        agents.companion, "llm_with_tools", ScriptedModel([AIMessage(content="I stopped writing poetry.")])
    )
    monkeypatch.setattr(
        agents.companion,
        "extraction_llm",
        ScriptedModel(
            [
                AIMessage(
                    content='{"terminated_edges": [{"subject": "self", "name": "Iris", '
                    '"relationship_to": "poetry", "relationship_type": "WRITES"}]}'
                )
            ]
        ),
    )
    await agents.companion.build_companion_graph().ainvoke(
        {"user_id": user, "user_message": "Do you still write poetry?"}
    )
    ctx = await get(client, "/companion/context", user_id=user)
    assert edges(ctx["self_facts"], "writes") == set()


async def test_demo_seed_is_repeatable_after_forget(client, agents, monkeypatch):
    import main as demo_main

    base = str(client.base_url).rstrip("/")
    monkeypatch.setattr(demo_main, "MEMORY_ENGINE_URL", base)
    async with httpx.AsyncClient(
        base_url=base,
        timeout=30.0,
        headers=client.headers,
        event_hooks={"response": [demo_main.raise_http_error]},
    ) as strict:
        await demo_main.seed_data(strict)
        # Demo 6d: forgetting disables memory for the user.
        await strict.delete("/companion/memory/usr_anthony")
        await demo_main.seed_data(strict)

    state = await get(client, "/companion/memory/usr_anthony/state")
    assert state["enabled"] is True
    ctx = await get(client, "/companion/context", user_id="usr_anthony")
    assert edges(ctx["graph_facts"], "lives_in") == {"Brooklyn"}
    assert edges(ctx["self_facts"], "values") == {"long conversations"}
    trajectories = await post(
        client,
        "/task/trajectories/search",
        goal_description="Write a Python script to count words",
        agent_id="task-bot-2",
        min_similarity=-1,
    )
    goals = [t["goal_description"] for t in trajectories]
    assert len(goals) == len(set(goals)), "re-seeding duplicated trajectories"


async def test_swarm_pull_mode_workers_split_queue_and_hold_leases(
    client, agents, monkeypatch
):
    workflow = "pull-graph-wf"
    for name in ("analyze_sentiment", "extract_entities", "summarize_text"):
        await post(client, "/swarm/tasks", workflow_id=workflow, task_name=name, payload={})

    class PullWorker:
        async def ainvoke(self, messages):
            results = [m for m in messages if m.type == "tool"]
            if not results:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "claim",
                            "name": "claim_next_task",
                            "args": {"agent_name": "x", "workflow_id": "x"},
                        }
                    ],
                )
            if len(results) == 1:
                # The model never sees or copies the lease; bogus values are overridden.
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "complete",
                            "name": "complete_swarm_task",
                            "args": {
                                "task_id": "00000000-0000-0000-0000-000000000000",
                                "agent_name": "x",
                                "lease_token": "not-a-real-lease",
                                "result_summary": "done",
                            },
                        }
                    ],
                )
            return AIMessage(content="done")

    monkeypatch.setattr(agents.swarm, "llm_with_tools", PullWorker())
    result = await agents.swarm.build_swarm_graph().ainvoke(
        {"workflow_id": workflow, "claim_mode": "pull"}
    )
    assert len(result["reports"]) == 3
    tasks = await get(client, f"/swarm/tasks/{workflow}")
    assert all(t["status"] == "COMPLETED" for t in tasks)
    reported = sorted(r.split("] ")[1].split(":")[0] for r in result["reports"])
    assert reported == sorted(t["task_name"] for t in tasks)


async def test_identifier_lexical_arm_fires_for_natural_language_query(client):
    await post(
        client,
        "/developer/symbols",
        project_id="lexical-proj",
        git_branch="main",
        file_path="src/store.py",
        symbol_name="embed_memory",
        symbol_type="function",
        signature="def embed_memory(text: str) -> list[float]",
        code_content="zzz qqq",
    )
    rows = await post(
        client,
        "/developer/symbols/search",
        project_id="lexical-proj",
        git_branch="main",
        # Shares no fake-embedding token with the symbol (vector arm empty) and has
        # whole-string trigram similarity ~0.1, so only word_similarity can match.
        query="How can I embed things before searching?",
    )
    assert rows and rows[0]["symbol_name"] == "embed_memory"


async def test_trajectory_store_is_idempotent(client):
    body = dict(
        agent_id="dedupe-bot",
        goal_description="Count words in a file",
        action_sequence=[{"action": "write_file"}],
        execution_result="ok",
        success_score=0.9,
    )
    assert (await post(client, "/task/trajectories", **body))["status"] == "stored"
    assert (await post(client, "/task/trajectories", **body))["status"] == "duplicate"
