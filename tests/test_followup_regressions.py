"""Boundary regressions from the second full code and design review."""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from conftest import post, get
from test_review_regressions import connect
import test_agent_regressions as agent_helpers

agents = agent_helpers.agents
ScriptedModel = agent_helpers.ScriptedModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


async def test_scope_conflicts_cannot_read_or_claim(client):
    await post(
        client, "/companion/episodes", user_id="bob", content="Bob private appointment"
    )
    task = await post(
        client, "/swarm/tasks", workflow_id="forbidden", task_name="private"
    )
    headers = {"Authorization": "Bearer " + os.environ["MEMORY_EMPLOYEE_TEST_TOKEN"]}
    response = await client.post(
        "/companion/context/search",
        params={"user_id": "bob", "query": "appointment"},
        json={"user_id": "alice"},
        headers=headers,
    )
    assert response.status_code in (403, 422)
    response = await client.post(
        "/swarm/tasks/claim-next",
        params={"workflow_id": "forbidden", "agent_name": "impostor"},
        json={"workflow_id": "allowed", "agent_name": "worker"},
        headers=headers,
    )
    assert response.status_code in (403, 422)
    board = await get(client, "/swarm/tasks/forbidden")
    assert board[0]["task_id"] == task["task_id"] and board[0]["status"] == "PENDING"


async def test_duplicate_query_scopes_rejected(client):
    response = await client.post(
        "/companion/context/search",
        params=[("user_id", "bob"), ("user_id", "alice"), ("query", "appointment")],
        headers={"Authorization": "Bearer " + os.environ["MEMORY_EMPLOYEE_TEST_TOKEN"]},
    )
    assert response.status_code in (403, 422)


@pytest.mark.parametrize(
    "partial", [{"relationship_type": "LIVES_IN"}, {"relationship_to": "Tokyo"}]
)
async def test_partial_termination_preserves_unrelated_facts(client, agents, partial):
    for relation, target in [("LIVES_IN", "Tokyo"), ("LIKES", "coffee")]:
        await post(
            client,
            "/companion/facts",
            user_id="alice",
            name="user",
            entity_type="person",
            relationship_to=target,
            relationship_type=relation,
        )
    await agents.tools.terminate_companion_relationship.ainvoke(
        {"user_id": "alice", "name": "user", **partial}
    )
    facts = (await get(client, "/companion/context", user_id="alice"))["graph_facts"]
    assert any(f.get("relationship_type") == "likes" for f in facts)
    assert not any(f.get("relationship_type") == "lives_in" for f in facts)


@pytest.mark.parametrize(
    "message",
    [
        ToolMessage(
            content="Error: permission denied",
            name="search_policy_documents",
            tool_call_id="failed",
            status="error",
        ),
        ToolMessage(
            content="Title: fabricated policy",
            name="search_policy_documents",
            tool_call_id="empty",
        ),
    ],
)
async def test_enterprise_requires_successful_policy_evidence(
    agents, monkeypatch, message
):
    model = ScriptedModel([AIMessage(content="Invented policy: all expenses approved")])
    monkeypatch.setattr(agents.enterprise, "llm_with_tools", model)
    result = await agents.enterprise.agent_node(
        {
            "retrieved_docs": "",
            "messages": [HumanMessage(content="What is policy?"), message],
        }
    )
    assert "No current authorized policy" in result["messages"][-1].content


async def test_swarm_failed_claim_never_invokes_model_or_execution(
    client, agents, monkeypatch
):
    from agents import swarm_agent

    task = await post(client, "/swarm/tasks", workflow_id="wf", task_name="work")
    await post(
        client, "/swarm/tasks/claim", task_id=task["task_id"], agent_name="winner"
    )
    model = ScriptedModel([AIMessage(content="I will execute without claiming")])
    monkeypatch.setattr(swarm_agent, "llm_with_tools", model)
    result = await swarm_agent.worker_node(
        {"workflow_id": "wf", "worker_task": task, "agent_name": "loser"}
    )
    assert not model.inputs
    assert "not claimed" in result["reports"][0].lower()


async def test_swarm_claims_before_model_and_binds_completion_lease(
    client, agents, monkeypatch
):
    from agents import swarm_agent

    task = await post(client, "/swarm/tasks", workflow_id="wf", task_name="work")

    class Worker:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            board = await get(client, "/swarm/tasks/wf")
            if self.calls == 1:
                assert board[0]["status"] == "IN_PROGRESS"
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "complete",
                            "name": "complete_swarm_task",
                            "args": {
                                "task_id": "wrong",
                                "agent_name": "wrong",
                                "lease_token": "wrong",
                                "result_summary": "done",
                            },
                        }
                    ],
                )
            return AIMessage(content="done")

    monkeypatch.setattr(swarm_agent, "llm_with_tools", Worker())
    await swarm_agent.worker_node(
        {"workflow_id": "wf", "worker_task": task, "agent_name": "worker"}
    )
    assert (await get(client, "/swarm/tasks/wf"))[0]["status"] == "COMPLETED"


async def test_swarm_expired_lease_blocks_execution(client, agents, monkeypatch):
    from agents import swarm_agent

    task = await post(client, "/swarm/tasks", workflow_id="wf", task_name="work")
    dispatched = []

    class Worker:
        async def ainvoke(self, messages):
            conn = await connect()
            try:
                await conn.execute(
                    "UPDATE swarm_blackboard SET lease_until=now()-interval '1 second'"
                )
            finally:
                await conn.close()
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "execute",
                        "name": "fetch_url",
                        "args": {"url": "https://example.com"},
                    }
                ],
            )

    class Execution:
        async def ainvoke(self, args):
            dispatched.append(args)
            return "executed"

    monkeypatch.setattr(swarm_agent, "llm_with_tools", Worker())
    monkeypatch.setitem(swarm_agent.TOOLS_BY_NAME, "fetch_url", Execution())
    await swarm_agent.worker_node(
        {"workflow_id": "wf", "worker_task": task, "agent_name": "worker"}
    )
    assert not dispatched


@pytest.mark.parametrize("subject", ["self", "shared"])
async def test_explicit_extracted_termination_scope(
    client, agents, monkeypatch, subject
):
    from agents.companion_extraction import build_extraction_prompt

    prompt = build_extraction_prompt("question", "answer")
    assert (
        '"subject"'
        in prompt.split('"terminated_edges"', 1)[1].split('"episode_content"', 1)[0]
    )
    await post(
        client,
        "/companion/facts",
        user_id="alice",
        name="Iris",
        entity_type="self",
        relationship_to="coffee",
        relationship_type="LIKES",
        subject=subject,
    )
    output = {
        "terminated_edges": [
            {
                "name": "Iris",
                "relationship_to": "coffee",
                "relationship_type": "LIKES",
                "subject": subject,
            }
        ]
    }
    monkeypatch.setattr(
        agents.companion,
        "extraction_llm",
        ScriptedModel([AIMessage(content=json.dumps(output))]),
    )
    await agents.companion.extract_memory(
        {
            "user_id": "alice",
            "user_message": "Do you still like coffee?",
            "messages": [AIMessage(content="No.")],
        }
    )
    facts = (await get(client, "/companion/context", user_id="alice"))[
        subject + "_facts"
    ]
    assert not any(f.get("relationship_type") == "likes" for f in facts)


@pytest.mark.parametrize("subject", [None, "invalid"])
def test_extraction_rejects_ambiguous_termination(subject):
    from companion_extraction import parse_extraction_response

    termination = {"name": "Iris", "relationship_type": "LIKES"}
    if subject is not None:
        termination["subject"] = subject
    with pytest.raises(ValueError):
        parse_extraction_response(json.dumps({"terminated_edges": [termination]}))


async def test_reindex_updates_chunks_and_records_model(client, monkeypatch):
    await post(
        client, "/companion/episodes", user_id="alice", content="old model evidence"
    )
    await post(
        client,
        "/companion/facts",
        user_id="alice",
        name="user",
        entity_type="person",
        relationship_to="Tokyo",
        relationship_type="LIVES_IN",
    )
    sys.path.insert(0, str(ROOT / "memory_engine"))
    from routers import companion

    conn = await connect()
    replacement = json.dumps([0.0, 1.0] + [0.0] * 1534)

    async def embed(text):
        return replacement

    monkeypatch.setattr(companion, "_embed", embed)
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "replacement-model")
    request = SimpleNamespace(state=SimpleNamespace(connection=conn))
    try:
        await companion.reindex_memory("alice", request)
        for table in (
            "companion_chunks",
            "companion_graph_nodes",
            "companion_graph_edges",
        ):
            for row in await conn.fetch(
                f"SELECT embedding::text AS vector FROM {table}"
            ):
                assert json.loads(row["vector"]) == json.loads(replacement)
        from embedding import embedding_space

        assert (
            await conn.fetchval(
                "SELECT embedding_space FROM companion_memory_state WHERE user_id='alice'"
            )
            == embedding_space()
        )
    finally:
        await conn.close()


@pytest.mark.parametrize("recorded_space", [None, "different-model"])
async def test_model_mismatch_requires_reindex(client, recorded_space):
    await post(
        client, "/companion/episodes", user_id="alice", content="existing evidence"
    )
    conn = await connect()
    try:
        await conn.execute(
            "UPDATE companion_memory_state SET embedding_space=$1 WHERE user_id='alice'",
            recorded_space,
        )
    finally:
        await conn.close()
    assert (
        await client.post(
            "/companion/context/search",
            params={"user_id": "alice", "query": "evidence"},
        )
    ).status_code == 409
    assert (
        await client.get(
            "/companion/context", params={"user_id": "alice", "query": "evidence"}
        )
    ).status_code == 409
    assert (
        await client.post(
            "/companion/episodes", json={"user_id": "alice", "content": "new evidence"}
        )
    ).status_code == 409
    await post(client, "/companion/memory/alice/reindex")
    assert (
        await client.post(
            "/companion/context/search",
            params={"user_id": "alice", "query": "evidence"},
        )
    ).status_code == 200


async def test_companion_actual_candidate_plan_uses_hnsw(client, monkeypatch):
    sys.path.insert(0, str(ROOT / "memory_engine"))
    from routers import companion
    from fake_embeddings import FakeEmbeddings, vector

    monkeypatch.setattr(
        companion,
        "get_embedding_client",
        lambda: SimpleNamespace(embeddings=FakeEmbeddings()),
    )
    await post(
        client, "/companion/facts", user_id="alice", name="user", entity_type="person"
    )
    conn = await connect()
    plans = []

    class Proxy:
        def __getattr__(self, key):
            return getattr(conn, key)

        async def fetch(self, sql, *args):
            if "edge_candidates" in sql:
                plans.append(await conn.fetchval("EXPLAIN (FORMAT JSON) " + sql, *args))
            return await conn.fetch(sql, *args)

    try:
        await conn.execute(
            "INSERT INTO companion_graph_nodes(user_id,name,normalize_name_key,entity_type,subject,is_subject,embedding) SELECT 'alice',i::text,i::text,'entity','user',true,$1::halfvec FROM generate_series(1,1000) i",
            json.dumps(vector("probe")),
        )
        await conn.execute(
            "INSERT INTO companion_graph_edges(user_id,source_node_id,target_node_id,relationship_type,subject,embedding) SELECT user_id,node_id,node_id,'likes','user',$1::halfvec FROM companion_graph_nodes",
            json.dumps(vector("probe")),
        )
        await conn.execute(
            "ANALYZE companion_graph_edges; ANALYZE companion_graph_nodes; SET enable_seqscan=off; SET enable_sort=off"
        )
        request = SimpleNamespace(state=SimpleNamespace(connection=Proxy()))
        await companion.get_companion_context("alice", request, "probe", 5)
        assert "companion_edge_embedding" in plans[0]
    finally:
        await conn.close()


def test_scoring_requires_source_identity_and_explicit_aliases():
    from benchmark_extraction import score_facts

    prediction = {
        "name": "Bob",
        "subject": "user",
        "relationship_type": "LIVES_IN",
        "relationship_to": "London",
    }
    label = {"scope": "user", "predicate": "LIVES_IN", "object_value": "London"}
    assert score_facts([prediction], [label]) == (0, 1, 1)
    assert score_facts([dict(prediction, name="user")], [label]) == (1, 0, 0)
    assert score_facts([prediction], [dict(label, source_entity="Bob")]) == (1, 0, 0)
    assert score_facts(
        [dict(prediction, name="Alice")], [dict(label, source_aliases=["Alice"])]
    ) == (1, 0, 0)


async def test_cleanup_fails_when_any_deletion_fails(monkeypatch):
    import cleanup_replay

    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def delete(self, path):
            requests.append(path)
            return SimpleNamespace(
                status_code=403 if len(requests) == 1 else 200,
                text="denied",
                json=lambda: {"status": "forgotten"},
            )

    monkeypatch.setattr(cleanup_replay.httpx, "AsyncClient", Client)
    monkeypatch.setenv("MEMORY_API_TOKEN", "test-token")
    monkeypatch.setattr(sys, "argv", ["cleanup", "--user-id", "one", "two"])
    assert await cleanup_replay.main() == 1
    assert len(requests) == 2


def test_trace_download_defaults_to_current_time(monkeypatch, tmp_path):
    import pull_conversations

    queries = []
    monkeypatch.setattr(
        pull_conversations, "load_env", lambda _: {"LANGSMITH_API_KEY": "test"}
    )
    monkeypatch.setattr(
        pull_conversations,
        "api_post",
        lambda path, key, body: queries.append(body) or {"runs": []},
    )
    monkeypatch.setattr(
        sys, "argv", ["pull", "--session", "test", "--out", str(tmp_path)]
    )
    before = datetime.now(timezone.utc)
    assert pull_conversations.main() == 0
    after = datetime.now(timezone.utc)
    bound = datetime.fromisoformat(queries[0]["end_time"].replace("Z", "+00:00"))
    assert before <= bound <= after


async def test_reindex_failure_preserves_vectors_and_identity(client, monkeypatch):
    await post(
        client, "/companion/episodes", user_id="alice", content="original episode"
    )
    await post(
        client,
        "/companion/facts",
        user_id="alice",
        name="user",
        entity_type="person",
        relationship_to="Tokyo",
        relationship_type="LIVES_IN",
    )
    sys.path.insert(0, str(ROOT / "memory_engine"))
    from routers import companion

    conn = await connect()

    async def snapshot():
        rows = []
        for table in (
            "companion_graph_nodes",
            "companion_graph_edges",
            "companion_chunks",
        ):
            rows.append(
                sorted(
                    await conn.fetchval(
                        f"SELECT array_agg(embedding::text) FROM {table}"
                    )
                    or []
                )
            )
        return rows, await conn.fetchval(
            "SELECT embedding_space FROM companion_memory_state WHERE user_id='alice'"
        )

    count = 0

    async def embed(text):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("provider unavailable")
        return json.dumps([1.0] + [0.0] * 1535)

    monkeypatch.setattr(companion, "_embed", embed)
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "replacement")
    try:
        before = await snapshot()
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await companion.reindex_memory(
                "alice", SimpleNamespace(state=SimpleNamespace(connection=conn))
            )
        assert await snapshot() == before
    finally:
        await conn.close()


async def test_replay_rejects_missing_scope_before_any_writes():
    import replay_traces

    class Engine:
        async def post(self, *args, **kwargs):
            pytest.fail("Ambiguous terminations must be rejected before publication")

    with pytest.raises(ValueError):
        await replay_traces.store_extraction(
            Engine(),
            "alice",
            {"episode_content": "evidence", "terminated_edges": [{"name": "Iris"}]},
        )


async def test_policy_search_artifact_survives_actual_tool_node(
    client, agents, monkeypatch
):
    await post(
        client,
        "/enterprise/documents",
        doc_title="Expenses",
        allowed_role="employee",
        content="Meals are reimbursable up to fifty dollars.",
    )
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search",
                        "name": "search_policy_documents",
                        "args": {
                            "query": "reimbursable meals",
                            "user_role": "employee",
                        },
                    }
                ],
            ),
            AIMessage(content="Meals have a fifty-dollar limit."),
        ]
    )
    monkeypatch.setattr(agents.enterprise, "llm_with_tools", model)
    result = await agents.enterprise.build_enterprise_graph().ainvoke(
        {"query": "zzzzunrelated", "user_role": "employee"}
    )
    assert not result["retrieved_docs"]
    assert result["messages"][-1].content == "Meals have a fifty-dollar limit."
    evidence = next(m for m in result["messages"] if m.type == "tool")
    assert evidence.status == "success" and evidence.artifact["documents"]


async def test_swarm_competing_workers_execute_only_the_claimant(
    client, agents, monkeypatch
):
    from agents import swarm_agent

    task = await post(client, "/swarm/tasks", workflow_id="wf", task_name="work")
    executed = []

    class Execution:
        async def ainvoke(self, args):
            executed.append(args)
            return "done"

    class Worker:
        async def ainvoke(self, messages):
            if not any(m.type == "tool" for m in messages):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "exec",
                            "name": "fetch_url",
                            "args": {"url": "https://example.com"},
                        }
                    ],
                )
            return AIMessage(content="done")

    monkeypatch.setattr(swarm_agent, "llm_with_tools", Worker())
    monkeypatch.setitem(swarm_agent.TOOLS_BY_NAME, "fetch_url", Execution())
    results = await asyncio.gather(
        *(
            swarm_agent.worker_node(
                {"workflow_id": "wf", "worker_task": task, "agent_name": name}
            )
            for name in ("a", "b")
        )
    )
    assert len(executed) == 1
    assert sum("Not claimed" in r["reports"][0] for r in results) == 1


def test_trace_download_respects_explicit_window(monkeypatch, tmp_path):
    import pull_conversations

    queries = []

    def request(path, key, body):
        queries.append(body)
        if len(queries) == 1:
            return {
                "runs": [{"trace_id": "too-old", "start_time": "2024-01-01T00:00:00Z"}]
            }
        pytest.fail("Out-of-window runs must not be downloaded")

    monkeypatch.setattr(
        pull_conversations, "load_env", lambda _: {"LANGSMITH_API_KEY": "test"}
    )
    monkeypatch.setattr(pull_conversations, "api_post", request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pull",
            "--session",
            "test",
            "--out",
            str(tmp_path),
            "--end-time",
            "2025-01-02T00:00:00Z",
            "--min-time",
            "2025-01-01T00:00:00Z",
        ],
    )
    assert pull_conversations.main() == 0
    assert datetime.fromisoformat(queries[0]["end_time"]) == datetime(
        2025, 1, 2, tzinfo=timezone.utc
    )
    assert not list(tmp_path.iterdir())


async def test_embedding_space_migration_preserves_existing_privacy_state(client):
    conn = await connect()
    try:
        # A temporary table shadows the live test table on this connection only.
        await conn.execute(
            "CREATE TEMP TABLE companion_memory_state(user_id text PRIMARY KEY, enabled boolean NOT NULL, generation integer NOT NULL)"
        )
        await conn.execute(
            "INSERT INTO companion_memory_state VALUES('forgotten-user',false,7)"
        )
        migration = (
            ROOT / "memory_engine/migrations/002_embedding_space.sql"
        ).read_text()
        await conn.execute(migration)
        await conn.execute(migration)
        row = await conn.fetchrow("SELECT * FROM companion_memory_state")
        assert dict(row) == {
            "user_id": "forgotten-user",
            "enabled": False,
            "generation": 7,
            "embedding_space": None,
        }
    finally:
        await conn.close()
