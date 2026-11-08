import json
import os
import subprocess
import sys
from uuid import uuid4
import asyncpg
import httpx
import pytest
import pytest_asyncio
from extension_host import ExtensionHost, headers


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    root = tmp_path_factory.mktemp("analytics-host")
    config = root / "logins.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "demo_agents.extensions.provision",
            "--user",
            "alice",
            "--user",
            "bob",
            "--output",
            str(config),
            "--seed-fixtures",
        ],
        check=True,
    )
    host = ExtensionHost(root)
    host.env["EXTENSION_ANALYTICS_CONFIG"] = str(config)
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


def query(**plan):
    return {"run_id": str(uuid4()), "user_id": "alice", "plan": plan}


async def connect(login=None):
    return await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        database=os.environ["POSTGRES_DB"],
        user=login["role"] if login else os.environ["POSTGRES_USER"],
        password=login["password"] if login else os.environ["POSTGRES_PASSWORD"],
    )


@pytest.mark.asyncio
async def test_actual_restricted_aggregation_join_and_empty(ext):
    body = query(operation="sum_amount", status="paid")
    response = await ext.post("/extensions/analytics/query", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["summary"]["observed_rows"] == [{"total": 40}]
    assert (
        await ext.post("/extensions/analytics/query", json=body)
    ).json() == response.json()
    joined = await ext.post(
        "/extensions/analytics/query",
        json=query(view="orders_by_region", columns=["order_id", "region"]),
    )
    assert joined.json()["summary"]["observed_rows"] == [
        {"order_id": 1, "region": "west"},
        {"order_id": 2, "region": "east"},
        {"order_id": 3, "region": "west"},
    ]
    empty = await ext.post("/extensions/analytics/query", json=query(max_amount=0))
    assert empty.json()["summary"] == {
        "observed_rows": [],
        "row_count": 0,
        "scope": "authenticated tenant only",
        "empty": True,
    }
    assert (
        await ext.post("/extensions/analytics/query", json=dict(body, user_id="bob"))
    ).status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan",
    [
        {"sql": "DELETE FROM ext_orders"},
        {"operation": "delete"},
        {"view": "ext_analytics_identities"},
        {"columns": ["internal_note"]},
        {"join": "cross"},
        {"limit": 100000},
        {"status": "paid'; DROP TABLE ext_orders;--"},
        {"columns": ["region"]},
    ],
)
async def test_plan_denials(ext, plan):
    assert (
        await ext.post("/extensions/analytics/query", json=query(**plan))
    ).status_code == 422


@pytest.mark.asyncio
async def test_database_enforces_scope_without_application_validator(host, client):
    logins = json.loads(open(host.env["EXTENSION_ANALYTICS_CONFIG"]).read())
    conn = await connect(logins["alice"])
    try:
        assert await conn.fetchval("SELECT count(*) FROM ext_order_metrics") == 3
        assert (
            await conn.fetchval(
                "SELECT count(user_id) FROM ext_orders WHERE user_id='bob'"
            )
            == 0
        )
        await conn.execute("SET app.user_id='bob'")
        assert await conn.fetchval("SELECT ext_analytics_tenant()") == "alice"
        for sql in (
            "SELECT internal_note FROM ext_orders",
            "SELECT * FROM ext_analytics_identities",
            "SELECT * FROM ext_proposals",
            f'SET ROLE "{logins["bob"]["role"]}"',
            f'SET SESSION AUTHORIZATION "{logins["bob"]["role"]}"',
        ):
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await conn.execute(sql)
        await conn.execute("SET default_transaction_read_only=off")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("UPDATE ext_orders SET amount=0")
        with pytest.raises(asyncpg.QueryCanceledError):
            await conn.execute("SELECT pg_sleep(1)")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_statement_budget_in_actual_graph(ext):
    admin = await connect()
    try:
        await admin.execute(
            "CREATE OR REPLACE VIEW ext_order_metrics WITH(security_invoker=true,security_barrier=true) AS SELECT order_id,customer_id,amount,status FROM ext_orders WHERE pg_sleep(0.5) IS NULL"
        )
        response = await ext.post("/extensions/analytics/query", json=query())
        assert response.status_code == 408, response.text
    finally:
        await admin.execute(
            "CREATE OR REPLACE VIEW ext_order_metrics WITH(security_invoker=true,security_barrier=true) AS SELECT order_id,customer_id,amount,status FROM ext_orders"
        )
        await admin.close()
