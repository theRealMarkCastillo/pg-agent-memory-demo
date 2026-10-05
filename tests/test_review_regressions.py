"""Regression coverage for the reproduced review failures; no external model calls."""

import os
from uuid import uuid4
import asyncpg
import httpx
import pytest
from conftest import post, get, delete


async def connect():
    return await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DB"],
    )


def fact(user="alice", **kw):
    return dict(
        user_id=user,
        name="Alice",
        entity_type="person",
        relationship_to="Tokyo",
        relationship_type="lives_in",
        **kw,
    )


async def test_missing_and_forged_credentials(client):
    response = await client.get(
        "/companion/context",
        params={"user_id": "alice"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401
    token = os.environ["MEMORY_EMPLOYEE_TEST_TOKEN"]
    async with httpx.AsyncClient(
        base_url=str(client.base_url), headers={"Authorization": "Bearer " + token}
    ) as employee:
        assert (
            await employee.get("/companion/context", params={"user_id": "bob"})
        ).status_code == 403
        assert (
            await employee.post(
                "/enterprise/documents/search",
                json={"query": "policy", "user_role": "admin"},
            )
        ).status_code == 403
        assert (
            await employee.post(
                "/developer/symbols/search",
                json={"query": "x", "project_id": "other", "git_branch": "main"},
            )
        ).status_code == 403
        assert (
            await employee.post(
                "/task/trajectories/search", json={"goal_description": "x"}
            )
        ).status_code == 403


async def test_forget_provenance_and_disable_future_writes(client):
    episode = await post(
        client, "/companion/episodes", user_id="alice", content="private evidence"
    )
    await post(
        client, "/companion/facts", **fact(source_episode_id=episode["episode_id"])
    )
    await delete(client, "/companion/memory/alice")
    assert (await get(client, "/companion/context", user_id="alice"))[
        "graph_facts"
    ] == []
    response = await client.post(
        "/companion/episodes",
        json={"user_id": "alice", "content": "late sensitive write"},
    )
    assert response.status_code == 409
    await post(client, "/companion/memory/alice/resume")
    stale = await client.post(
        "/companion/episodes",
        json={"user_id": "alice", "content": "old generation"},
        headers={"X-Memory-Generation": "0"},
    )
    assert stale.status_code == 409
    await post(
        client, "/companion/episodes", user_id="alice", content="explicitly resumed"
    )


async def test_provenance_owner_validation_is_atomic(client):
    episode = await post(
        client, "/companion/episodes", user_id="bob", content="bob private evidence"
    )
    response = await client.post(
        "/companion/facts", json=fact(source_episode_id=episode["episode_id"])
    )
    assert response.status_code == 403
    assert not (await get(client, "/companion/context", user_id="alice"))["graph_facts"]


async def test_episode_failure_and_retry_idempotency(client):
    response = await client.post(
        "/companion/episodes",
        json={"user_id": "alice", "content": "one " * 501 + "[EMBEDDING_FAILURE]"},
    )
    assert response.status_code == 502
    conn = await connect()
    try:
        assert await conn.fetchval("SELECT count(*) FROM companion_episodes") == 0
    finally:
        await conn.close()
    body = {
        "user_id": "alice",
        "content": "retryable episode",
        "ingestion_id": "request-1",
    }
    a = await post(client, "/companion/episodes", **body)
    b = await post(client, "/companion/episodes", **body)
    assert a == b
    assert (
        await client.post("/companion/episodes", json=dict(body, content="different"))
    ).status_code == 409


async def test_unicode_entities_remain_distinct(client):
    for name in ("東京", "大阪"):
        await post(
            client,
            "/companion/facts",
            user_id="alice",
            name=name,
            entity_type="location",
        )
    names = {
        f["name"]
        for f in (await get(client, "/companion/context", user_id="alice"))[
            "graph_facts"
        ]
    }
    assert names == {"東京", "大阪"}
    assert (
        await client.post(
            "/companion/facts",
            json={"user_id": "alice", "name": "!!!", "entity_type": "x"},
        )
    ).status_code == 422


async def test_complete_assertion_ranking_and_history(client):
    for relation, target in [("likes", "coffee"), ("lives_in", "Tokyo")]:
        await post(
            client,
            "/companion/facts",
            user_id="alice",
            name="Alice",
            entity_type="person",
            relationship_to=target,
            relationship_type=relation,
        )
    result = await get(
        client,
        "/companion/context",
        user_id="alice",
        query="Alice lives in Tokyo",
        limit=1,
    )
    assert result["graph_facts"][0]["relationship_type"] == "lives_in"
    await post(client, "/companion/facts/terminate", **fact())
    await post(client, "/companion/facts", **fact())
    conn = await connect()
    try:
        rows = await conn.fetch(
            "SELECT status,valid_until FROM companion_graph_edges WHERE relationship_type='lives_in'"
        )
        assert len(rows) == 2 and sum(r["status"] == "ACTIVE" for r in rows) == 1
        assert next(r for r in rows if r["status"] == "INACTIVE")["valid_until"]
    finally:
        await conn.close()


async def test_expiry_and_policy_replacement(client):
    base = {
        "doc_title": "Policy",
        "allowed_role": "employee",
        "content": "remote work policy",
    }
    await post(
        client, "/enterprise/documents", **base, valid_until="2030-01-01T00:00:00Z"
    )
    await post(
        client,
        "/enterprise/documents",
        **dict(base, content="updated remote work policy"),
    )
    rows = await post(
        client,
        "/enterprise/documents/search",
        query="remote work policy",
        user_role="employee",
    )
    assert len(rows) == 1 and rows[0]["content"] == "updated remote work policy"
    await post(
        client, "/enterprise/documents", **base, valid_until="2000-01-01T00:00:00Z"
    )
    assert not await post(
        client,
        "/enterprise/documents/search",
        query="remote work policy",
        user_role="employee",
    )


async def test_symbol_update_and_delete(client):
    body = dict(
        project_id="x",
        git_branch="main",
        file_path="x.py",
        symbol_name="reconcile_memory",
        symbol_type="function",
        signature="def reconcile_memory()",
        code_content="old implementation",
    )
    await post(client, "/developer/symbols", **body)
    for _ in range(12):
        await post(
            client,
            "/developer/symbols",
            **dict(body, code_content="new implementation"),
        )
    rows = await post(
        client,
        "/developer/symbols/search",
        project_id="x",
        git_branch="main",
        query="reconcile_memory",
    )
    assert len(rows) == 1 and rows[0]["code_content"] == "new implementation"
    response = await client.delete(
        "/developer/symbols",
        params={"project_id": "x", "git_branch": "main", "file_path": "x.py"},
    )
    assert response.status_code == 200
    assert not await post(
        client,
        "/developer/symbols/search",
        project_id="x",
        git_branch="main",
        query="reconcile_memory",
    )


async def test_swarm_completion_requires_live_owned_lease(client):
    task = await post(client, "/swarm/tasks", workflow_id="wf", task_name="work")
    body = dict(
        task_id=task["task_id"],
        agent_name="worker",
        lease_token=str(uuid4()),
        payload={"result": "done"},
    )
    assert (await client.post("/swarm/tasks/complete", json=body)).status_code == 409
    claim = await post(
        client, "/swarm/tasks/claim", task_id=task["task_id"], agent_name="worker"
    )
    body["lease_token"] = claim["task"]["lease_token"]
    assert (
        await client.post(
            "/swarm/tasks/complete", json=dict(body, agent_name="impostor")
        )
    ).status_code == 409
    conn = await connect()
    try:
        await conn.execute(
            "UPDATE swarm_blackboard SET lease_until=now()-interval '1 second'"
        )
    finally:
        await conn.close()
    replacement = await post(
        client, "/swarm/tasks/claim", task_id=task["task_id"], agent_name="replacement"
    )
    assert replacement["status"] == "claimed"
    assert (await client.post("/swarm/tasks/complete", json=body)).status_code == 409
    body.update(
        agent_name="replacement", lease_token=replacement["task"]["lease_token"]
    )
    await post(client, "/swarm/tasks/complete", **body)
    assert (await client.post("/swarm/tasks/complete", json=body)).status_code == 409


async def test_tutor_transitive_prerequisites_and_score_bounds(client):
    for name, parent in [("one", None), ("two", "one"), ("three", "two")]:
        await post(client, "/tutor/skills", skill_name=name, parent_skill_name=parent)
    await post(
        client,
        "/tutor/progress",
        user_id="alice",
        skill_name="two",
        proficiency_score=1,
    )
    results = await get(client, "/tutor/gaps/alice")
    assert not next(x for x in results if x["skill_name"] == "three")["ready"]
    assert (
        await client.post(
            "/tutor/progress",
            json={"user_id": "alice", "skill_name": "one", "proficiency_score": 5},
        )
    ).status_code == 422


async def test_pool_checkout_restores_iterative_scan(client):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory_engine"))
    from db import get_db_pool

    pool = await get_db_pool()
    try:
        for _ in range(2):
            async with pool.acquire() as conn:
                assert await conn.fetchval("SHOW hnsw.iterative_scan") == "strict_order"
                await conn.execute("SET hnsw.iterative_scan=off")
    finally:
        await pool.close()


async def test_actual_hybrid_query_retains_hnsw_plan(client, monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from contextlib import asynccontextmanager

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory_engine"))
    from routers import developer
    from fake_embeddings import FakeEmbeddings

    monkeypatch.setattr(
        developer,
        "get_embedding_client",
        lambda: SimpleNamespace(embeddings=FakeEmbeddings()),
    )
    conn = await connect()
    plans = []

    class Proxy:
        async def fetch(self, sql, *args):
            plan = await conn.fetchval("EXPLAIN (FORMAT JSON) " + sql, *args)
            plans.append(plan)
            return await conn.fetch(sql, *args)

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield Proxy()

    try:
        import json
        from fake_embeddings import vector

        await conn.execute(
            """INSERT INTO dev_code_symbols(project_id,git_branch,file_path,symbol_name,symbol_type,signature,code_content,embedding)
            SELECT 'empty','main',i::text,'symbol_'||i,'function','f()','content',$1::halfvec FROM generate_series(1,1000) i""",
            json.dumps(vector("probe")),
        )
        await conn.execute("ANALYZE dev_code_symbols")
        await conn.execute("SET enable_seqscan=off")
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(pool=Pool()))
        )
        await developer.search_symbols(
            developer.SymbolSearch(
                project_id="empty", git_branch="main", query="probe"
            ),
            request,
        )
        assert "idx_dev_symbols_hnsw" in plans[0]
    finally:
        await conn.close()


async def test_file_reconcile_removes_renames_atomically(client):
    base = {"project_id": "x", "git_branch": "main", "file_path": "x.py"}
    old = {
        "symbol_name": "old",
        "symbol_type": "function",
        "signature": "old()",
        "code_content": "old implementation",
    }
    new = dict(old, symbol_name="new", code_content="new implementation")
    await post(client, "/developer/symbols/reconcile", **base, symbols=[old])
    failure = await client.post(
        "/developer/symbols/reconcile",
        json=dict(base, symbols=[dict(new, code_content="[EMBEDDING_FAILURE]")]),
    )
    assert failure.status_code == 502
    conn = await connect()
    try:
        assert (
            await conn.fetchval(
                "SELECT symbol_name FROM dev_code_symbols WHERE project_id='x'"
            )
            == "old"
        )
    finally:
        await conn.close()
    await post(client, "/developer/symbols/reconcile", **base, symbols=[new])
    conn = await connect()
    try:
        assert [
            r["symbol_name"]
            for r in await conn.fetch(
                "SELECT symbol_name FROM dev_code_symbols WHERE project_id='x'"
            )
        ] == ["new"]
    finally:
        await conn.close()


async def test_rrf_uses_ranks_instead_of_raw_scores(client):
    await post(
        client,
        "/enterprise/documents",
        doc_title="Keyword",
        allowed_role="employee",
        content="rarekeyword",
    )
    rows = await post(
        client,
        "/enterprise/documents/search",
        query="rarekeyword",
        user_role="employee",
    )
    assert rows and float(rows[0]["rrf_score"]) == pytest.approx(2 / 61)


async def test_legacy_checkpoints_require_explicit_admin_cleanup(client):
    conn = await connect()
    try:
        # This fixture models an unowned checkpoint from the previous schema.
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints(thread_id text,checkpoint_ns text,checkpoint_id text,checkpoint jsonb)"
        )
        await conn.execute(
            "INSERT INTO checkpoints(thread_id,checkpoint_ns,checkpoint_id,checkpoint) VALUES('old-companion-thread','','legacy','{}')"
        )
        response = await client.delete("/companion/memory/alice")
        assert response.status_code == 409
        response = await client.post(
            "/companion/legacy-checkpoints/purge",
            json={"confirmation": "delete unowned checkpoint history"},
            headers={
                "Authorization": "Bearer " + os.environ["MEMORY_EMPLOYEE_TEST_TOKEN"]
            },
        )
        assert response.status_code == 403
        await post(
            client,
            "/companion/legacy-checkpoints/purge",
            confirmation="delete unowned checkpoint history",
        )
        await delete(client, "/companion/memory/alice")
    finally:
        await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id='old-companion-thread'"
        )
        await conn.close()


async def test_swarm_scope_checks_canonical_uuid(client):
    task = await post(
        client, "/swarm/tasks", workflow_id="forbidden", task_name="private"
    )
    for identifier in (task["task_id"].upper(), task["task_id"].replace("-", "")):
        response = await client.post(
            "/swarm/tasks/claim",
            json={"task_id": identifier, "agent_name": "worker"},
            headers={
                "Authorization": "Bearer " + os.environ["MEMORY_EMPLOYEE_TEST_TOKEN"]
            },
        )
        assert response.status_code == 403
