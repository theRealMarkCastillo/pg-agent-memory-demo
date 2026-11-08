from uuid import uuid4
import httpx
import pytest
import pytest_asyncio
from extension_host import ExtensionHost, REVIEWER, OTHER, headers


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    host = ExtensionHost(tmp_path_factory.mktemp("procedure-host"))
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


def attempt(**overrides):
    return {
        "attempt_id": str(uuid4()),
        "user_id": "alice",
        "source_csv": "name,quantity\npear,2\napple,1\n",
        "artifact": '[{"name":"apple","quantity":1},{"name":"pear","quantity":2}]',
        "procedure": "Parse CSV; validate integers; sort by name; emit JSON.",
        **overrides,
    }


@pytest.mark.asyncio
async def test_verified_promotion_retirement_scope_and_duplicate(ext):
    body = attempt()
    first = await ext.post("/extensions/procedures/attempts", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["verified"]
    assert (
        await ext.post("/extensions/procedures/attempts", json=body)
    ).json() == first.json()
    assert (
        await ext.get("/extensions/procedures", params={"user_id": "alice"})
    ).json()["procedures"] == []
    path = f"/extensions/procedures/{body['attempt_id']}"
    assert (await ext.post(path + "/promote", json={})).status_code == 403
    assert (
        await ext.post(path + "/promote", headers=headers(OTHER), json={})
    ).status_code == 403
    assert (
        await ext.post(path + "/promote", headers=headers(REVIEWER), json={})
    ).status_code == 200
    assert (
        await ext.post(path + "/promote", headers=headers(REVIEWER), json={})
    ).status_code == 200
    assert (
        len(
            (
                await ext.get("/extensions/procedures", params={"user_id": "alice"})
            ).json()["procedures"]
        )
        == 1
    )
    assert (
        await ext.get(
            "/extensions/procedures",
            params={"user_id": "alice", "environment": "other"},
        )
    ).json()["procedures"] == []
    assert (
        await ext.get(
            "/extensions/procedures",
            params={"user_id": "alice", "checker_version": "old"},
        )
    ).json()["procedures"] == []
    assert (
        await ext.get(
            "/extensions/procedures", params={"user_id": "bob"}, headers=headers(OTHER)
        )
    ).json()["procedures"] == []
    assert (
        await ext.post(path + "/retire", headers=headers(REVIEWER), json={})
    ).status_code == 200
    assert (
        await ext.get("/extensions/procedures", params={"user_id": "alice"})
    ).json()["procedures"] == []
    assert (
        await ext.post(path + "/promote", headers=headers(REVIEWER), json={})
    ).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"artifact": '[{"name":"apple","quantity":1},{"name":"pear","quantity":200}]'},
        {"artifact": None},
        {"artifact": "not JSON"},
        {"artifact": '[{"name":"apple","quantity":true}]'},
        {"spec_version": "changed/2"},
        {"checker_version": "changed/2"},
    ],
)
async def test_failed_attempts_cannot_be_promoted(ext, changes):
    body = attempt(**changes)
    result = await ext.post("/extensions/procedures/attempts", json=body)
    assert result.status_code == 200, result.text
    assert not result.json()["verified"]
    assert (
        await ext.post(
            f"/extensions/procedures/{body['attempt_id']}/promote",
            headers=headers(REVIEWER),
            json={},
        )
    ).status_code == 409


@pytest.mark.asyncio
async def test_forged_score_and_mutated_evidence_rejected(ext):
    assert (
        await ext.post("/extensions/procedures/attempts", json=attempt(score=1.0))
    ).status_code == 422
    assert (
        await ext.post(
            "/extensions/procedures/attempts", json=attempt(environment="trusted")
        )
    ).status_code == 422
    body = attempt()
    assert (
        await ext.post("/extensions/procedures/attempts", json=body)
    ).status_code == 200
    assert (
        await ext.post(
            "/extensions/procedures/attempts", json=dict(body, artifact="[]")
        )
    ).status_code == 409
