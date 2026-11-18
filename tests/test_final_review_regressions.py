"""Adversarial regressions found in the final code/design review."""

import asyncio
from datetime import datetime, timedelta, timezone
import os
from uuid import UUID, uuid4
import asyncpg
import httpx
import pytest
import pytest_asyncio
from extension_host import ExtensionHost, REVIEWER, headers
from test_extension_approval import propose, review, run, destination
from test_extension_support import policy, assertion, draft
from demo_agents.extensions.research import build_research_graph, extract_fixture


async def database():
    return await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


async def wait_for_lock_waiters(conn, expected):
    for _ in range(150):
        count = await conn.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event='advisory'"
        )
        if count >= expected:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"Expected {expected} lock waiters, saw {count}")


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    host = ExtensionHost(tmp_path_factory.mktemp("final-review"))
    try:
        yield host.start()
    finally:
        host.stop()


@pytest_asyncio.fixture
async def ext(host, client):
    async with httpx.AsyncClient(
        base_url=host.url, headers=headers(), timeout=15
    ) as conn:
        yield conn


@pytest.mark.asyncio
async def test_approval_burst_does_not_deadlock_node_connections(tmp_path, client):
    host = ExtensionHost(tmp_path / "contended-host")
    conn = await database()
    tasks = []
    try:
        host.start()
        async with httpx.AsyncClient(
            base_url=host.url, headers=headers(), timeout=10
        ) as ext:
            row, _ = await propose(ext)
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                    f"graph:{row['proposal_id']}",
                )
                tasks = [asyncio.create_task(run(ext, row)) for _ in range(12)]
                # Fill all ten API connections with waiting requests before
                # releasing the first graph, making the old starvation repeatable.
                await wait_for_lock_waiters(conn, 10)
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=6)
            assert all(
                r.status_code == 200 and r.json()["status"] == "pending"
                for r in results
            )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        host.stop(kill=True)
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intervening_action", ["edit", "cancel", "policy_update", "overwritten"]
)
async def test_crashed_file_effect_cannot_be_erased_by_later_actions(
    tmp_path, client, intervening_action
):
    host = ExtensionHost(tmp_path / "effect-recovery")
    host.env["EXTENSION_CRASH_AFTER_REPLACE_MARKER"] = str(tmp_path / "crash-once")
    try:
        host.start()
        async with httpx.AsyncClient(
            base_url=host.url, headers=headers(), timeout=15
        ) as ext:
            if intervening_action == "policy_update":
                p = await policy(ext)
                case = str(uuid4())
                await assertion(ext, case, 3)
                row = (await draft(ext, p, case))["proposal"]
            else:
                row, _ = await propose(ext)
            assert (await run(ext, row)).json()["status"] == "pending"
            assert (await review(ext, row)).status_code == 200
            with pytest.raises(httpx.RemoteProtocolError):
                await run(ext, row)
            host.process.wait(timeout=10)
            assert host.process.returncode == 86
            file = destination(host, row)
            before = file.stat().st_mtime_ns
            assert file.read_text() == row["content"]
            host.stop()
            host.start()
            path = f"/extensions/proposals/{row['proposal_id']}"
            persisted = (await ext.get(path)).json()
            assert persisted["status"] == "applying"
            assert persisted["effect_started_at"] is not None
            if intervening_action == "policy_update":
                await policy(
                    ext,
                    **dict(
                        p,
                        version=2,
                        content="Refunds are available within 1 days.",
                        refund_days=1,
                    ),
                )
            elif intervening_action == "overwritten":
                later, _ = await propose(
                    ext,
                    target=row["target"],
                    base_digest=row["digest"],
                    content="Later reviewed file revision",
                )
                await run(ext, later)
                await review(ext, later)
                assert (await run(ext, later)).json()["status"] == "applied"
                assert (await ext.post(path + "/cancel", json={})).status_code == 409
                assert (await run(ext, row)).status_code == 409
                assert (await ext.get(path)).json()["status"] == "applying"
                assert file.read_text() == later["content"]
                return
            else:
                body = (
                    {"revision": 1, "content": "revised"}
                    if intervening_action == "edit"
                    else {}
                )
                response = await ext.post(path + "/" + intervening_action, json=body)
                assert response.status_code == 409, response.text
            assert (await run(ext, row)).json()["status"] == "applied"
            assert file.stat().st_mtime_ns == before
    finally:
        host.stop(kill=True)


@pytest.mark.asyncio
async def test_policy_expiry_is_checked_after_waiting_for_evidence_lock(ext):
    p = await policy(
        ext, valid_until=(datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
    )
    case = str(uuid4())
    await assertion(ext, case, 1)
    row = (await draft(ext, p, case))["proposal"]
    conn = await database()
    task = None
    try:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,0))", "support:alice"
            )
            task = asyncio.create_task(review(ext, row))
            await wait_for_lock_waiters(conn, 1)
            await asyncio.sleep(2.1)
        response = await task
        assert response.status_code == 409, response.text
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_model", [False, True])
async def test_research_cannot_certify_support_after_omitting_contradiction(
    partial_model,
):
    source = {
        "document_id": str(uuid4()),
        "version": 1,
        "digest": "fixture",
        "content": "\n".join(
            ["SUPPORT cache_reduces_latency: Lower latency."] * 16
            + ["CONTRADICT cache_reduces_latency: Higher latency."]
        ),
    }

    async def partial(document, claims):
        return [
            e for e in extract_fixture(document, claims) if e["relation"] == "supports"
        ][:1]

    result = await build_research_graph(
        [source], ["cache_reduces_latency"], generate=partial if partial_model else None
    ).ainvoke({"items": []})
    assert result["report"]["claims"]["cache_reduces_latency"] in (
        "disputed",
        "insufficient",
    )


@pytest.mark.asyncio
async def test_deep_malformed_artifact_is_retained_as_failed_attempt(ext):
    response = await ext.post(
        "/extensions/procedures/attempts",
        json={
            "attempt_id": str(uuid4()),
            "user_id": "alice",
            "source_csv": "name,quantity\na,1\n",
            "artifact": "[" * 12000 + "0" + "]" * 12000,
            "procedure": "Invalid candidate",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["verified"] is False
    assert response.json()["errors"]


@pytest.mark.asyncio
async def test_older_incomplete_extraction_protocol_is_not_reused(ext):
    notebook, document, report_id = (str(uuid4()) for _ in range(3))
    response = await ext.post(
        "/extensions/research/documents",
        json={
            "document_id": document,
            "notebook_id": notebook,
            "user_id": "alice",
            "version": 1,
            "content": "SUPPORT cache_reduces_latency: Lower latency.",
        },
    )
    assert response.status_code == 200
    report = (
        await ext.post(
            "/extensions/research/reports",
            json={
                "report_id": report_id,
                "notebook_id": notebook,
                "user_id": "alice",
                "claims": ["cache_reduces_latency"],
            },
        )
    ).json()
    path = f"/extensions/research/reports/{report_id}/review"
    decision = {"digest": report["digest"], "accept": True}
    assert (
        await ext.post(path, json=decision, headers=headers(REVIEWER))
    ).status_code == 200
    conn = await database()
    try:
        # Represent an accepted report produced before completeness was recorded.
        await conn.execute(
            "UPDATE ext_research_reports SET report=report-'extraction_protocol'-'coverage' WHERE report_id=$1",
            UUID(report_id),
        )
    finally:
        await conn.close()
    result = await ext.get(
        f"/extensions/research/{notebook}/assertions", params={"user_id": "alice"}
    )
    assert result.json()["assertions"] == []
    assert (
        await ext.post(path, json=decision, headers=headers(REVIEWER))
    ).status_code == 409
