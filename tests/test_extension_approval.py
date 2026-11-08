"""Approval tests cross HTTP, graph, PostgreSQL, process and filesystem boundaries."""

import asyncio
from hashlib import sha256
from uuid import uuid4
import httpx
import pytest
import pytest_asyncio
from extension_host import ExtensionHost, REVIEWER, OTHER, headers


def digest(text):
    return sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    host = ExtensionHost(tmp_path_factory.mktemp("approval-host"))
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


async def propose(ext, **overrides):
    body = {
        "proposal_id": str(uuid4()),
        "user_id": "alice",
        "target": f"{uuid4()}.txt",
        "base_digest": digest(""),
        "content": "reviewed revision\n",
        **overrides,
    }
    result = await ext.post("/extensions/proposals", json=body)
    assert result.status_code == 200, result.text
    return result.json(), body


async def review(ext, row, decision="approved", **overrides):
    return await ext.post(
        f"/extensions/proposals/{row['proposal_id']}/review",
        headers=headers(REVIEWER),
        json={
            "revision": row["revision"],
            "digest": row["digest"],
            "decision": decision,
            **overrides,
        },
    )


async def run(ext, row):
    return await ext.post(f"/extensions/proposals/{row['proposal_id']}/run", json={})


def destination(host, row):
    return host.workspace / digest(row["user_id"]) / row["target"]


@pytest.mark.asyncio
async def test_review_boundaries_and_idempotency(ext, host):
    row, body = await propose(ext)
    assert row["thread_id"].startswith("approval:" + digest("alice") + ":")
    assert (await run(ext, row)).json()["status"] == "pending"
    assert not destination(host, row).exists()
    assert (await ext.post("/extensions/proposals", json=body)).json()[
        "effect_key"
    ] == row["effect_key"]
    path = f"/extensions/proposals/{row['proposal_id']}"
    assert (await ext.get(path, headers=headers(OTHER))).status_code == 403
    assert (
        await ext.post(path + "/run", headers=headers(OTHER), json={})
    ).status_code == 403
    assert (
        await ext.post(path + "/run", json={"thread_id": row["thread_id"]})
    ).status_code == 422
    data = {"revision": 1, "digest": row["digest"], "decision": "approved"}
    assert (await ext.post(path + "/review", json=data)).status_code == 403
    assert (await review(ext, row, reviewer="forged-admin")).status_code == 422
    assert (await review(ext, row, revision=2)).status_code == 409
    results = await asyncio.gather(review(ext, row), review(ext, row))
    assert [r.status_code for r in results] == [200, 200]
    results = await asyncio.gather(run(ext, row), run(ext, row))
    assert all(r.json()["status"] == "applied" for r in results)
    file = destination(host, row)
    stat = file.stat().st_mtime_ns
    assert file.read_text() == row["content"]
    assert (await run(ext, row)).json()["status"] == "applied"
    assert stat == file.stat().st_mtime_ns
    assert (await review(ext, row, "rejected")).status_code == 409


@pytest.mark.asyncio
async def test_reject_edit_reapproval_and_cancel(ext, host):
    row, _ = await propose(ext)
    assert (await run(ext, row)).json()["status"] == "pending"
    assert (await review(ext, row, "rejected")).status_code == 200
    assert (await run(ext, row)).json()["status"] == "rejected"
    assert not destination(host, row).exists()
    path = f"/extensions/proposals/{row['proposal_id']}"
    edited = await ext.post(
        path + "/edit", json={"revision": 1, "content": "corrected\n"}
    )
    assert edited.status_code == 200
    new = edited.json()
    assert new["thread_id"] != row["thread_id"]
    assert (await review(ext, row)).status_code == 409
    assert (await run(ext, new)).json()["status"] == "pending"
    assert (await review(ext, new)).status_code == 200
    assert (await run(ext, new)).json()["status"] == "applied"
    assert destination(host, row).read_text() == "corrected\n"
    cancelled, _ = await propose(ext)
    assert (await run(ext, cancelled)).json()["status"] == "pending"
    assert (
        await ext.post(
            f"/extensions/proposals/{cancelled['proposal_id']}/cancel", json={}
        )
    ).status_code == 200
    assert (await review(ext, cancelled)).status_code == 409
    assert (await run(ext, cancelled)).json()["status"] == "cancelled"
    assert not destination(host, cancelled).exists()


@pytest.mark.asyncio
async def test_changed_base_and_scope_spoof(ext, host):
    row, body = await propose(ext)
    assert (
        await ext.post("/extensions/proposals", json=dict(body, user_id="bob"))
    ).status_code == 403
    assert (
        await ext.post("/extensions/proposals", json=dict(body, owner="admin"))
    ).status_code == 422
    assert (
        await ext.post("/extensions/proposals", json=dict(body, target="../escape"))
    ).status_code == 422
    await run(ext, row)
    await review(ext, row)
    destination(host, row).write_text("concurrent edit")
    assert (await run(ext, row)).status_code == 409
    assert destination(host, row).read_text() == "concurrent edit"


@pytest.mark.asyncio
async def test_actual_host_death_pending_and_after_effect(tmp_path, client):
    host = ExtensionHost(tmp_path / "restart")
    try:
        host.start()
        async with httpx.AsyncClient(
            base_url=host.url, headers=headers(), timeout=30
        ) as ext:
            row, _ = await propose(ext)
            assert (await run(ext, row)).json()["status"] == "pending"
            pid = host.process.pid
            host.stop(kill=True)
            host.start()
            assert host.process.pid != pid
            assert (await run(ext, row)).json()["status"] == "pending"
            assert (await review(ext, row)).status_code == 200
            host.stop()
            host.env["EXTENSION_CRASH_AFTER_REPLACE_MARKER"] = str(
                tmp_path / "crashed-once"
            )
            host.start()
            with pytest.raises(httpx.RemoteProtocolError):
                await run(ext, row)
            host.process.wait(timeout=10)
            assert host.process.returncode == 86
            file = destination(host, row)
            assert file.read_text() == row["content"]
            stat = file.stat().st_mtime_ns
            host.stop()
            host.start()
            # File effect committed but transaction/checkpoint did not; reconcile.
            assert (await run(ext, row)).json()["status"] == "applied"
            assert file.stat().st_mtime_ns == stat
    finally:
        host.stop()
