import os
import httpx
import pytest_asyncio

BASE_URL = os.getenv("MEMORY_ENGINE_TEST_URL", "http://localhost:8001")
HEADERS = {"Content-Type": "application/json"}


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def verify_engine_up(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def post(client, path, **json_body):
    resp = await client.post(path, json=json_body, headers=HEADERS)
    assert resp.status_code == 200, f"POST {path} failed ({resp.status_code}): {resp.text}"
    return resp.json()


async def get(client, path, **params):
    if params:
        resp = await client.get(path, params=params, headers=HEADERS)
    else:
        resp = await client.get(path, headers=HEADERS)
    assert resp.status_code == 200, f"GET {path} failed ({resp.status_code}): {resp.text}"
    return resp.json()


async def post_params(client, path, **params):
    """POST with query params (for endpoints like /claim-next)."""
    resp = await client.post(path, params=params, headers=HEADERS)
    assert resp.status_code == 200, f"POST {path} failed ({resp.status_code}): {resp.text}"
    return resp.json()
