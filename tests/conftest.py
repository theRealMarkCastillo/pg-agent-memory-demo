import os
import httpx
import pytest_asyncio
import asyncpg

BASE_URL = os.getenv("MEMORY_ENGINE_TEST_URL", "http://localhost:8001")
HEADERS = {"Content-Type": "application/json"}


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=30.0,
        headers={"Authorization": "Bearer " + os.environ.get("MEMORY_API_TOKEN", "")},
    ) as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def isolated_database(request):
    if os.getenv("MEMORY_TEST_RESET") == "1" and "client" in request.fixturenames:
        # Opt-in safeguard: the runner creates this uniquely named disposable DB.
        database = os.environ["POSTGRES_DB"]
        if not database.startswith("memory_test_"):
            raise RuntimeError("Refusing to reset a non-test database")
        conn = await asyncpg.connect(
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ["POSTGRES_PORT"]),
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            database=database,
        )
        try:
            await conn.execute(
                "TRUNCATE dev_code_symbols,task_trajectories,enterprise_documents,tutor_skills,tutor_user_progress,swarm_blackboard,companion_graph_edges,companion_graph_nodes,companion_episodes,companion_ephemerals,companion_memory_state CASCADE"
            )
        finally:
            await conn.close()


async def post(client, path, **json_body):
    resp = await client.post(path, json=json_body, headers=HEADERS)
    assert resp.status_code == 200, (
        f"POST {path} failed ({resp.status_code}): {resp.text}"
    )
    return resp.json()


async def get(client, path, **params):
    if params:
        resp = await client.get(path, params=params, headers=HEADERS)
    else:
        resp = await client.get(path, headers=HEADERS)
    assert resp.status_code == 200, (
        f"GET {path} failed ({resp.status_code}): {resp.text}"
    )
    return resp.json()


async def post_params(client, path, **params):
    """POST with query params (for endpoints like /claim-next)."""
    resp = await client.post(path, params=params, headers=HEADERS)
    assert resp.status_code == 200, (
        f"POST {path} failed ({resp.status_code}): {resp.text}"
    )
    return resp.json()


async def delete(client, path):
    resp = await client.delete(path, headers=HEADERS)
    assert resp.status_code == 200, (
        f"DELETE {path} failed ({resp.status_code}): {resp.text}"
    )
    return resp.json()
