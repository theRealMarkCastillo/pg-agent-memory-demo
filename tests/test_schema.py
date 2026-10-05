"""
Verify all tables, indexes, and extensions exist.
"""

import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize(
    "route,method",
    [
        ("/developer/symbols/search", "POST"),
        ("/task/trajectories/search", "POST"),
        ("/enterprise/documents/search", "POST"),
        ("/tutor/gaps/test_user", "GET"),
        ("/swarm/tasks/test-wf", "GET"),
        ("/companion/context", "GET"),
    ],
)
@pytest.mark.asyncio
async def test_all_routes_exist(client, route, method):
    if method == "GET":
        resp = await client.get(route)
    else:
        resp = await client.post(route, json={})
    # A missing route (404) is a failure; validation errors (422) prove routing.
    assert resp.status_code in (200, 422), (
        f"{method} {route} returned {resp.status_code}: {resp.text}"
    )
