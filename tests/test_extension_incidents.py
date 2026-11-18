import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import httpx
import pytest
import pytest_asyncio
from extension_host import ExtensionHost, OTHER, headers


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    host = ExtensionHost(tmp_path_factory.mktemp("incident-host"))
    try:
        yield host.start()
    finally:
        host.stop()


@pytest_asyncio.fixture
async def ext(host, client):
    async with httpx.AsyncClient(
        base_url=host.url, headers=headers(), timeout=30
    ) as conn:
        yield conn


def event(incident, **overrides):
    return {
        "event_id": str(uuid4()),
        "incident_id": incident,
        "user_id": "alice",
        "source_id": str(uuid4()),
        "source_revision": 1,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "content": "Synthetic service recovered",
        **overrides,
    }


@pytest.mark.asyncio
async def test_late_correction_revocation_and_stale_summary(ext):
    incident = str(uuid4())
    path = f"/extensions/incidents/{incident}"
    recovered = event(incident)
    assert (
        await ext.post("/extensions/incidents/events", json=recovered)
    ).status_code == 200
    assert (await ext.post(path + "/summary", json={"user_id": "alice"})).json()[
        "stale"
    ] is False
    outage = event(
        incident,
        content="Outage observed",
        occurred_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    )
    assert (
        await ext.post("/extensions/incidents/events", json=outage)
    ).status_code == 200
    result = (await ext.get(path + "/timeline", params={"user_id": "alice"})).json()
    assert [e["event_id"] for e in result["timeline"]] == [
        outage["event_id"],
        recovered["event_id"],
    ]
    assert [e["event_id"] for e in result["history"]] == [
        recovered["event_id"],
        outage["event_id"],
    ]
    assert (await ext.get(path + "/summary", params={"user_id": "alice"})).json()[
        "stale"
    ]
    correction = event(
        incident,
        source_id=outage["source_id"],
        source_revision=2,
        supersedes=outage["event_id"],
        content="Partial outage only",
    )
    assert (
        await ext.post("/extensions/incidents/events", json=correction)
    ).status_code == 200
    revoked = event(
        incident,
        source_id=recovered["source_id"],
        source_revision=2,
        revokes=recovered["event_id"],
        content="Recovery source withdrawn",
    )
    assert (
        await ext.post("/extensions/incidents/events", json=revoked)
    ).status_code == 200
    result = (await ext.get(path + "/timeline", params={"user_id": "alice"})).json()
    assert [e["event_id"] for e in result["timeline"]] == [correction["event_id"]]
    assert len(result["history"]) == 4
    assert result["summary"]["hypotheses"] == []
    assert result["summary"]["causal_claim"] is None


@pytest.mark.asyncio
async def test_duplicate_concurrency_clocks_and_isolation(ext):
    incident = str(uuid4())
    body = event(
        incident,
        occurred_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )
    responses = await asyncio.gather(
        *(ext.post("/extensions/incidents/events", json=body) for _ in range(3))
    )
    assert all(r.status_code == 200 for r in responses)
    assert len({r.json()["arrival_order"] for r in responses}) == 1
    assert (
        await ext.post(
            "/extensions/incidents/events", json=dict(body, content="different")
        )
    ).status_code == 409
    path = f"/extensions/incidents/{incident}/timeline"
    result = (await ext.get(path, params={"user_id": "alice"})).json()
    assert len(result["history"]) == 1 and result["timeline"][0]["clock_conflict"]
    assert (
        await ext.get(path, params={"user_id": "alice"}, headers=headers(OTHER))
    ).status_code == 403
    assert (
        await ext.get(path, params={"user_id": "bob"}, headers=headers(OTHER))
    ).json()["timeline"] == []
    bad = event(
        str(uuid4()),
        source_id=body["source_id"],
        source_revision=2,
        supersedes=body["event_id"],
    )
    assert (await ext.post("/extensions/incidents/events", json=bad)).status_code == 403
