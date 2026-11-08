"""Artifact checking and promotion; generated procedures never grade themselves."""

import csv
import io
import json
from typing import TypedDict
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field
from langgraph.graph import START, END, StateGraph
from .common import StrictModel, actor, canonical, digest, locked, owned, permission
from .approval import Empty

from .responses import AttemptRecord, PromotionResult, ProcedureRecall

router = APIRouter()
SPEC = "csv-name-quantity/1"
CHECKER = "independent-json-checker/1"
ENVIRONMENT = "python-stdlib-csv-json/1"


class Attempt(StrictModel):
    attempt_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    source_csv: str = Field(max_length=20000)
    artifact: str | None = Field(default=None, max_length=30000)
    procedure: str = Field(max_length=20000)
    spec_version: str = Field(default=SPEC, max_length=100)
    checker_version: str = Field(default=CHECKER, max_length=100)


def check_artifact(source_csv, artifact, spec=SPEC, checker=CHECKER):
    """Check structural/semantic invariants directly against the source rows.

    This is intentionally separate from the model/fixture producer. There is no
    model-supplied score, evaluation tool, executable recipe, or dynamic import.
    """
    if spec != SPEC or checker != CHECKER:
        return ["Unsupported specification or checker version"]
    if artifact is None:
        return ["Missing output artifact"]
    try:
        reader = csv.DictReader(io.StringIO(source_csv), strict=True)
        if reader.fieldnames != ["name", "quantity"]:
            return ["CSV header must be name,quantity"]
        source = list(reader)
        if len(source) > 100 or any(
            set(row) != {"name", "quantity"}
            or not row["name"]
            or row["quantity"] is None
            for row in source
        ):
            return ["Invalid source rows or row budget exceeded"]
        quantities = [int(row["quantity"]) for row in source]
        if any(q < 0 or q > 1000000 for q in quantities):
            return ["Quantity out of range"]
        if len({row["name"] for row in source}) != len(source):
            return ["Names must be unique"]
        output = json.loads(artifact)
        if not isinstance(output, list) or len(output) != len(source):
            return ["Output must be a list with one object per source row"]
        if any(
            not isinstance(row, dict)
            or set(row) != {"name", "quantity"}
            or type(row["name"]) is not str
            or type(row["quantity"]) is not int
            for row in output
        ):
            return ["Output fields/types do not match the specification"]
        if [row["name"] for row in output] != sorted(row["name"] for row in source):
            return ["Names must match source and be sorted ascending"]
        source_quantities = dict(zip((row["name"] for row in source), quantities))
        if any(row["quantity"] != source_quantities[row["name"]] for row in output):
            return ["Quantity mismatch"]
        return []
    except (ValueError, TypeError, RecursionError, csv.Error):
        return ["Artifact or CSV could not be parsed"]


class AttemptState(TypedDict, total=False):
    artifact: str | None
    procedure: str
    errors: list[str]
    verified: bool


def build_procedure_graph(body, generate=None):
    async def produce(state):
        if generate:
            result = await generate(body.source_csv)
            # Model adapter output is still just an untrusted artifact.
            return {"artifact": result["artifact"], "procedure": result["procedure"]}
        return {"artifact": body.artifact, "procedure": body.procedure}

    def verify(state):
        errors = check_artifact(
            body.source_csv, state["artifact"], body.spec_version, body.checker_version
        )
        return {"errors": errors, "verified": not errors}

    graph = StateGraph(AttemptState)
    graph.add_node("produce", produce)
    graph.add_node("independent_check", verify)
    graph.add_edge(START, "produce")
    graph.add_edge("produce", "independent_check")
    graph.add_edge("independent_check", END)
    return graph.compile()


@router.post("/procedures/attempts", response_model=AttemptRecord)
async def attempt(body: Attempt, request: Request):
    permission(request, "submit")
    result = await build_procedure_graph(body).ainvoke({}, {"recursion_limit": 6})
    async with locked(request.app.state.pool, f"attempt:{body.attempt_id}") as conn:
        old = await conn.fetchrow(
            "SELECT * FROM ext_attempts WHERE attempt_id=$1", body.attempt_id
        )
        if old:
            owned(request, old)
            if any(old[key] != value for key, value in body.model_dump().items()):
                raise HTTPException(
                    409, "Attempt ID already contains different evidence"
                )
            return dict(old)
        row = await conn.fetchrow(
            "INSERT INTO ext_attempts(attempt_id,user_id,source_csv,artifact,artifact_digest,procedure,spec_version,checker_version,environment,verified,errors) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb) RETURNING *",
            body.attempt_id,
            body.user_id,
            body.source_csv,
            body.artifact,
            digest(body.artifact) if body.artifact is not None else None,
            body.procedure,
            body.spec_version,
            body.checker_version,
            ENVIRONMENT,
            result["verified"],
            canonical(result["errors"]),
        )
    return dict(row)


@router.post(
    "/procedures/{attempt_id}/promote",
    response_model=PromotionResult,
    response_model_exclude_none=True,
)
async def promote(attempt_id: UUID, body: Empty, request: Request):
    permission(request, "promote")
    promoter = actor(request)
    async with locked(request.app.state.pool, f"attempt:{attempt_id}") as conn:
        row = owned(
            request,
            await conn.fetchrow(
                "SELECT * FROM ext_attempts WHERE attempt_id=$1 FOR UPDATE", attempt_id
            ),
        )
        errors = check_artifact(
            row["source_csv"],
            row["artifact"],
            row["spec_version"],
            row["checker_version"],
        )
        if (
            not row["verified"]
            or errors
            or row["environment"] != ENVIRONMENT
            or row["status"] == "retired"
        ):
            raise HTTPException(
                409, "Only a currently verified, compatible attempt may be promoted"
            )
        await conn.execute(
            "UPDATE ext_attempts SET status='promoted',promoter=COALESCE(promoter,$2) WHERE attempt_id=$1",
            attempt_id,
            promoter,
        )
    return {"status": "promoted", "attempt_id": attempt_id}


@router.post(
    "/procedures/{attempt_id}/retire",
    response_model=PromotionResult,
    response_model_exclude_none=True,
)
async def retire(attempt_id: UUID, body: Empty, request: Request):
    permission(request, "promote")
    async with locked(request.app.state.pool, f"attempt:{attempt_id}") as conn:
        owned(
            request,
            await conn.fetchrow(
                "SELECT * FROM ext_attempts WHERE attempt_id=$1 FOR UPDATE", attempt_id
            ),
        )
        await conn.execute(
            "UPDATE ext_attempts SET status='retired' WHERE attempt_id=$1", attempt_id
        )
    return {"status": "retired"}


@router.get("/procedures", response_model=ProcedureRecall)
async def recall(
    request: Request,
    user_id: str,
    spec_version: str = SPEC,
    checker_version: str = CHECKER,
    environment: str = ENVIRONMENT,
):
    if (spec_version, checker_version, environment) != (SPEC, CHECKER, ENVIRONMENT):
        return {"procedures": []}
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM ext_attempts WHERE user_id=$1 AND status='promoted' AND verified AND spec_version=$2 AND checker_version=$3 AND environment=$4 ORDER BY attempt_id LIMIT 10",
            user_id,
            SPEC,
            CHECKER,
            ENVIRONMENT,
        )
    return {"procedures": [dict(row) for row in rows]}
