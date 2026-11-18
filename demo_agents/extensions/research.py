"""Versioned local evidence with bounded fan-out and a sequential baseline."""

import json
import operator
from typing import Annotated, Literal, TypedDict
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from .common import StrictModel, actor, canonical, digest, locked, owned, permission

from .responses import (
    DocumentRecord,
    ResearchReportRecord,
    AcceptedResult,
    ResearchRecall,
)

router = APIRouter()
Claim = Literal["cache_reduces_latency", "replicas_improve_availability"]
EXTRACTION_PROTOCOL = "complete-labelled-evidence/2"


class Document(StrictModel):
    document_id: UUID
    notebook_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    content: str = Field(max_length=8000)


class ReportRequest(StrictModel):
    report_id: UUID
    notebook_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    claims: list[Claim] = Field(min_length=1, max_length=2)
    parallel: bool = True


class Evidence(StrictModel):
    claim: Claim
    relation: Literal["supports", "contradicts"]
    quote: str = Field(max_length=8000)
    document_id: UUID
    version: int = Field(ge=1)


class ReportReview(StrictModel):
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    accept: bool


@router.post("/research/documents", response_model=DocumentRecord)
async def document(body: Document, request: Request):
    permission(request, "submit")
    async with locked(
        request.app.state.pool, f"research:{body.user_id}:{body.notebook_id}"
    ) as conn:
        previous = await conn.fetchrow(
            "SELECT * FROM ext_research_documents WHERE document_id=$1 ORDER BY version DESC LIMIT 1",
            body.document_id,
        )
        if previous:
            owned(request, previous)
            if (previous["user_id"], previous["notebook_id"]) != (
                body.user_id,
                body.notebook_id,
            ):
                raise HTTPException(409, "Source cannot change notebooks or owners")
        exact = await conn.fetchrow(
            "SELECT * FROM ext_research_documents WHERE document_id=$1 AND version=$2",
            body.document_id,
            body.version,
        )
        if exact:
            if exact["digest"] != digest(body.content):
                raise HTTPException(409, "Version already has different content")
            return dict(exact)
        if body.version != (previous["version"] + 1 if previous else 1):
            raise HTTPException(409, "Source revisions must be consecutive")
        count = await conn.fetchval(
            "SELECT count(DISTINCT document_id) FROM ext_research_documents WHERE user_id=$1 AND notebook_id=$2",
            body.user_id,
            body.notebook_id,
        )
        if not previous and count >= 8:
            raise HTTPException(422, "Notebook source budget is eight documents")
        row = await conn.fetchrow(
            "INSERT INTO ext_research_documents VALUES($1,$2,$3,$4,$5,$6) RETURNING *",
            body.document_id,
            body.version,
            body.notebook_id,
            body.user_id,
            body.content,
            digest(body.content),
        )
    return dict(row)


async def sources(conn, user, notebook):
    rows = await conn.fetch(
        "SELECT DISTINCT ON(document_id) * FROM ext_research_documents WHERE user_id=$1 AND notebook_id=$2 ORDER BY document_id,version DESC",
        user,
        notebook,
    )
    return [
        {
            **dict(row),
            "document_id": str(row["document_id"]),
            "notebook_id": str(row["notebook_id"]),
        }
        for row in rows
    ]


def snapshot(documents):
    return digest(
        canonical([(d["document_id"], d["version"], d["digest"]) for d in documents])
    )


def extract_fixture(document, claims):
    items = []
    for line in document["content"].splitlines():
        for claim in claims:
            for label, relation in (
                ("SUPPORT", "supports"),
                ("CONTRADICT", "contradicts"),
            ):
                if (
                    line.startswith(f"{label} {claim}: ")
                    and line[len(f"{label} {claim}: ") :].strip()
                ):
                    items.append(
                        {
                            "claim": claim,
                            "relation": relation,
                            "quote": line,
                            "document_id": document["document_id"],
                            "version": document["version"],
                        }
                    )
    return items


class ResearchState(TypedDict, total=False):
    documents: list[dict]
    items: Annotated[list[dict], operator.add]
    coverage: Annotated[list[dict], operator.add]
    report: dict


def build_research_graph(documents, claims, parallel=True, generate=None):
    async def worker(state):
        document = state["document"]
        expected = {canonical(item): item for item in extract_fixture(document, claims)}
        candidates = (
            await generate(document, claims) if generate else list(expected.values())
        )
        # Exact quote membership alone does not establish broad natural-language
        # entailment. This teaching fixture recognizes only labelled statements.
        accepted = {}
        for candidate in candidates if isinstance(candidates, list) else []:
            try:
                item = Evidence.model_validate(candidate).model_dump(mode="json")
            except ValueError:
                continue
            key = canonical(item)
            if key in expected:
                accepted[key] = item
        # Source size is bounded at ingestion. Do not silently slice evidence:
        # a missing contrary line must prevent promotion of a one-sided answer.
        coverage = [
            {
                "document_id": document["document_id"],
                "version": document["version"],
                "claim": claim,
                "complete": all(
                    key in accepted
                    for key, item in expected.items()
                    if item["claim"] == claim
                ),
            }
            for claim in sorted(set(claims))
        ]
        return {"items": list(accepted.values()), "coverage": coverage}

    async def sequential(state):
        items, coverage = [], []
        for document in state["documents"]:
            result = await worker({"document": document})
            items.extend(result["items"])
            coverage.extend(result["coverage"])
        return {"items": items, "coverage": coverage}

    def route(state):
        if not parallel or not state["documents"]:
            return "sequential"
        return [
            Send("extract", {"document": document}) for document in state["documents"]
        ]

    def aggregate(state):
        items = [
            json.loads(item)
            for item in sorted({canonical(item) for item in state["items"]})
        ]
        conclusions = {}
        coverage = sorted(state.get("coverage", []), key=canonical)
        for claim in sorted(set(claims)):
            relations = {i["relation"] for i in items if i["claim"] == claim}
            conclusions[claim] = (
                "disputed"
                if len(relations) == 2
                else "insufficient"
                if any(
                    not entry["complete"]
                    for entry in coverage
                    if entry["claim"] == claim
                )
                else "supported"
                if relations == {"supports"}
                else "contradicted"
                if relations
                else "insufficient"
            )
        return {
            "report": {
                "extraction_protocol": EXTRACTION_PROTOCOL,
                "coverage": coverage,
                "claims": conclusions,
                "evidence": items,
                "source_versions": [
                    {
                        "document_id": d["document_id"],
                        "version": d["version"],
                        "digest": d["digest"],
                    }
                    for d in state["documents"]
                ],
            }
        }

    graph = StateGraph(ResearchState)
    graph.add_node("load", lambda state: {"documents": documents})
    graph.add_node("extract", worker)
    graph.add_node("sequential", sequential)
    graph.add_node("aggregate", aggregate)
    graph.add_edge(START, "load")
    graph.add_conditional_edges("load", route, ["extract", "sequential"])
    graph.add_edge("extract", "aggregate")
    graph.add_edge("sequential", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


@router.post("/research/reports", response_model=ResearchReportRecord)
async def report(body: ReportRequest, request: Request):
    permission(request, "submit")
    async with locked(
        request.app.state.pool, f"research:{body.user_id}:{body.notebook_id}"
    ) as conn:
        documents = await sources(conn, body.user_id, body.notebook_id)
    result = await build_research_graph(
        documents,
        body.claims,
        body.parallel,
        getattr(request.app.state, "research_generator", None),
    ).ainvoke({"items": []}, {"max_concurrency": 3, "recursion_limit": 8})
    value = result["report"]
    fingerprint = digest(canonical(value))
    async with locked(request.app.state.pool, f"report:{body.report_id}") as conn:
        old = await conn.fetchrow(
            "SELECT * FROM ext_research_reports WHERE report_id=$1", body.report_id
        )
        if old:
            owned(request, old)
            if old["digest"] != fingerprint or (old["user_id"], old["notebook_id"]) != (
                body.user_id,
                body.notebook_id,
            ):
                raise HTTPException(
                    409, "Report ID already contains a different result"
                )
            return {**dict(old), "report": json.loads(old["report"])}
        row = await conn.fetchrow(
            "INSERT INTO ext_research_reports(report_id,notebook_id,user_id,evidence_digest,digest,report,claims) VALUES($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb) RETURNING *",
            body.report_id,
            body.notebook_id,
            body.user_id,
            snapshot(documents),
            fingerprint,
            canonical(value),
            canonical(sorted(set(body.claims))),
        )
    return {**dict(row), "report": value}


@router.post("/research/reports/{report_id}/review", response_model=AcceptedResult)
async def review(report_id: UUID, body: ReportReview, request: Request):
    permission(request, "review")
    reviewer = actor(request)
    async with request.app.state.pool.acquire() as conn:
        row = owned(
            request,
            await conn.fetchrow(
                "SELECT * FROM ext_research_reports WHERE report_id=$1", report_id
            ),
        )
    async with locked(
        request.app.state.pool, f"research:{row['user_id']}:{row['notebook_id']}"
    ) as conn:
        row = await conn.fetchrow(
            "SELECT * FROM ext_research_reports WHERE report_id=$1 FOR UPDATE",
            report_id,
        )
        if body.digest != row["digest"]:
            raise HTTPException(409, "Review must bind exact report")
        if row["reviewer"] is not None and (row["reviewer"], row["accepted"]) != (
            reviewer,
            body.accept,
        ):
            raise HTTPException(409, "Report already decided")
        if (
            body.accept
            and snapshot(await sources(conn, row["user_id"], row["notebook_id"]))
            != row["evidence_digest"]
        ):
            raise HTTPException(409, "Sources changed; regenerate and review")
        if (
            body.accept
            and json.loads(row["report"]).get("extraction_protocol")
            != EXTRACTION_PROTOCOL
        ):
            raise HTTPException(
                409, "Extraction protocol changed; regenerate and review"
            )
        await conn.execute(
            "UPDATE ext_research_reports SET reviewer=$2,accepted=$3 WHERE report_id=$1",
            report_id,
            reviewer,
            body.accept,
        )
    return {"accepted": body.accept}


@router.get("/research/{notebook_id}/assertions", response_model=ResearchRecall)
async def assertions(notebook_id: UUID, user_id: str, request: Request):
    async with locked(
        request.app.state.pool, f"research:{user_id}:{notebook_id}"
    ) as conn:
        current = snapshot(await sources(conn, user_id, notebook_id))
        rows = await conn.fetch(
            "SELECT report_id,report FROM ext_research_reports WHERE notebook_id=$1 AND user_id=$2 AND accepted AND evidence_digest=$3 AND report->>'extraction_protocol'=$4 ORDER BY report_id",
            notebook_id,
            user_id,
            current,
            EXTRACTION_PROTOCOL,
        )
    # Contradictions and unknowns remain in reports, never promoted as facts.
    return {
        "assertions": [
            {
                "report_id": str(row["report_id"]),
                "claim": claim,
                "evidence": [
                    e
                    for e in json.loads(row["report"])["evidence"]
                    if e["claim"] == claim
                ],
            }
            for row in rows
            for claim, status in json.loads(row["report"])["claims"].items()
            if status == "supported"
        ]
    }
