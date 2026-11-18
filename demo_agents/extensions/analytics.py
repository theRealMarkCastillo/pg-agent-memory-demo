"""Strict plans compile to application-owned SQL over restricted views."""

import json
import os
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID
import asyncpg
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, model_validator
from langgraph.graph import START, END, StateGraph
from .common import StrictModel, canonical, locked, owned, permission

from .responses import AnalyticsResult

router = APIRouter()


class QueryPlan(StrictModel):
    view: Literal["orders", "orders_by_region"] = "orders"
    operation: Literal["rows", "count", "sum_amount"] = "rows"
    columns: list[Literal["order_id", "customer_id", "amount", "status", "region"]] = (
        Field(
            default_factory=lambda: ["order_id", "amount", "status"],
            min_length=1,
            max_length=4,
        )
    )
    status: Literal["open", "paid"] | None = None
    max_amount: int | None = Field(default=None, ge=0, le=1000000)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def view_columns(self):
        permitted = {
            "order_id",
            "amount",
            "status",
            "region" if self.view == "orders_by_region" else "customer_id",
        }
        if not set(self.columns) <= permitted or len(set(self.columns)) != len(
            self.columns
        ):
            raise ValueError("Columns must be unique and belong to the chosen view")
        return self


class Query(StrictModel):
    run_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    plan: QueryPlan


def compile_plan(plan):
    table = {
        "orders": "public.ext_order_metrics",
        "orders_by_region": "public.ext_order_regions",
    }[plan.view]
    # Every interpolated identifier originates from the validated enum/mapping.
    select = {
        "count": "count(*) AS count",
        "sum_amount": "COALESCE(sum(amount),0) AS total",
    }.get(plan.operation, ",".join(plan.columns))
    params, where = [], []
    for column, op, value in (
        ("status", "=", plan.status),
        ("amount", "<=", plan.max_amount),
    ):
        if value is not None:
            params.append(value)
            where.append(f"{column}{op}${len(params)}")
    sql = f"SELECT {select} FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if plan.operation == "rows":
        sql += " ORDER BY order_id"
    params.append(plan.limit)
    sql += f" LIMIT ${len(params)}"
    return sql, params


async def open_pools():
    path = os.getenv("EXTENSION_ANALYTICS_CONFIG")
    if not path:
        return {}
    config = json.loads(Path(path).read_text())
    pools = {}
    try:
        for user, login in config.items():
            pool = await asyncpg.create_pool(
                host=os.environ["POSTGRES_HOST"],
                port=int(os.environ["POSTGRES_PORT"]),
                database=os.environ["POSTGRES_DB"],
                user=login["role"],
                password=login["password"],
                min_size=1,
                max_size=2,
                command_timeout=2,
                server_settings={
                    "statement_timeout": "250ms",
                    "default_transaction_read_only": "on",
                },
            )
            pools[user] = pool
            async with pool.acquire() as conn:
                identity = await conn.fetchval("SELECT public.ext_analytics_tenant()")
                elevated = await conn.fetchval(
                    "SELECT rolsuper OR rolbypassrls OR rolcreaterole FROM pg_roles WHERE rolname=session_user"
                )
                if identity != user or elevated:
                    raise RuntimeError(
                        "Analytics login does not match restricted tenant identity"
                    )
        return pools
    except BaseException:
        for pool in pools.values():
            await pool.close()
        raise


class QueryState(TypedDict, total=False):
    sql: str
    params: list
    rows: list[dict]
    summary: dict


def build_analytics_graph(plan, pool):
    def compile_node(state):
        sql, params = compile_plan(plan)
        return {"sql": sql, "params": params}

    async def execute(state):
        async with pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                await conn.execute("SET LOCAL statement_timeout='250ms'")
                rows = await conn.fetch(state["sql"], *state["params"])
        return {"rows": [dict(row) for row in rows]}

    def summarize(state):
        return {
            "summary": {
                "observed_rows": state["rows"],
                "row_count": len(state["rows"]),
                "scope": "authenticated tenant only",
                "empty": not state["rows"],
            }
        }

    graph = StateGraph(QueryState)
    graph.add_node("compile", compile_node)
    graph.add_node("execute", execute)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "compile")
    graph.add_edge("compile", "execute")
    graph.add_edge("execute", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


@router.post("/analytics/query", response_model=AnalyticsResult)
async def query(body: Query, request: Request):
    permission(request, "run")
    pool = request.app.state.analytics_pools.get(body.user_id)
    if pool is None:
        raise HTTPException(
            503, "No restricted analytics login configured for this tenant"
        )
    async with locked(request.app.state.pool, f"analytics:{body.run_id}") as conn:
        old = await conn.fetchrow(
            "SELECT * FROM ext_analytics_runs WHERE run_id=$1", body.run_id
        )
        if old:
            owned(request, old)
            if (
                old["user_id"] != body.user_id
                or json.loads(old["plan"]) != body.plan.model_dump()
            ):
                raise HTTPException(409, "Run ID already contains a different plan")
            return json.loads(old["result"])
        try:
            result = await build_analytics_graph(body.plan, pool).ainvoke(
                {}, {"recursion_limit": 8}
            )
        except (asyncpg.QueryCanceledError, TimeoutError):
            raise HTTPException(
                408, "Analytics statement exceeded its execution budget"
            )
        observed = {
            "run_id": str(body.run_id),
            "plan": body.plan.model_dump(),
            "summary": result["summary"],
        }
        await conn.execute(
            "INSERT INTO ext_analytics_runs(run_id,user_id,plan,result) VALUES($1,$2,$3::jsonb,$4::jsonb)",
            body.run_id,
            body.user_id,
            canonical(body.plan.model_dump()),
            canonical(observed),
        )
    return observed
