"""Append-only observations; current projections preserve arrival history."""

import json
from datetime import timedelta
from typing import TypedDict
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, AwareDatetime, model_validator
from langgraph.graph import START, END, StateGraph
from .common import StrictModel, canonical, digest, locked, owned, permission

from .responses import EventRecord, TimelineResult, SummaryResult

router = APIRouter()


class Event(StrictModel):
    event_id: UUID
    incident_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=128)
    source_revision: int = Field(ge=1)
    occurred_at: AwareDatetime
    content: str = Field(max_length=20000)
    supersedes: UUID | None = None
    revokes: UUID | None = None

    @model_validator(mode="after")
    def single_link(self):
        if self.supersedes and self.revokes:
            raise ValueError("Choose correction or revocation")
        return self


class Scope(StrictModel):
    user_id: str = Field(min_length=1, max_length=128)


@router.post(
    "/incidents/events", response_model=EventRecord, response_model_exclude_none=True
)
async def ingest(body: Event, request: Request):
    permission(request, "submit")
    fingerprint = digest(canonical(body.model_dump()))
    async with locked(
        request.app.state.pool, f"incident:{body.user_id}:{body.incident_id}"
    ) as conn:
        old = await conn.fetchrow(
            "SELECT * FROM ext_incident_events WHERE event_id=$1", body.event_id
        )
        if old:
            owned(request, old)
            if old["digest"] != fingerprint:
                raise HTTPException(409, "Event ID already contains different evidence")
            return dict(old)
        # Source revisions are immutable; new independent events need new source
        # IDs. Corrections/revocations explicitly link the current source head.
        source = await conn.fetchrow(
            "SELECT * FROM ext_incident_events WHERE user_id=$1 AND incident_id=$2 AND source_id=$3 ORDER BY source_revision DESC LIMIT 1",
            body.user_id,
            body.incident_id,
            body.source_id,
        )
        link = body.supersedes or body.revokes
        if link:
            prior = owned(
                request,
                await conn.fetchrow(
                    "SELECT * FROM ext_incident_events WHERE event_id=$1", link
                ),
            )
            if (prior["user_id"], prior["incident_id"], prior["source_id"]) != (
                body.user_id,
                body.incident_id,
                body.source_id,
            ):
                raise HTTPException(
                    403, "Linked event must belong to the same incident and source"
                )
            if (
                not source
                or source["event_id"] != link
                or body.source_revision != source["source_revision"] + 1
            ):
                raise HTTPException(409, "Link must extend the current source revision")
        elif source or body.source_revision != 1:
            raise HTTPException(
                409, "New sources start at revision 1; revisions require a link"
            )
        row = await conn.fetchrow(
            "INSERT INTO ext_incident_events(event_id,incident_id,user_id,source_id,source_revision,occurred_at,content,digest,supersedes,revokes) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING *",
            body.event_id,
            body.incident_id,
            body.user_id,
            body.source_id,
            body.source_revision,
            body.occurred_at,
            body.content,
            fingerprint,
            body.supersedes,
            body.revokes,
        )
    return dict(row)


class TimelineState(TypedDict, total=False):
    history: list[dict]
    timeline: list[dict]
    evidence_digest: str
    summary: dict


def build_timeline_graph(load):
    async def retrieve(state):
        return {"history": await load()}

    def project(state):
        events = state["history"]
        removed = {str(e[k]) for e in events for k in ("supersedes", "revokes") if e[k]}
        active = [
            e for e in events if str(e["event_id"]) not in removed and not e["revokes"]
        ]
        active.sort(key=lambda e: (e["occurred_at"], str(e["event_id"])))
        timeline = [
            {
                **e,
                "clock_conflict": e["occurred_at"]
                > e["received_at"] + timedelta(minutes=5),
            }
            for e in active
        ]
        return {
            "timeline": timeline,
            "evidence_digest": digest(
                canonical([(str(e["event_id"]), e["digest"]) for e in events])
            ),
        }

    def summarize(state):
        return {
            "summary": {
                "observations": [
                    {
                        "event_id": str(e["event_id"]),
                        "text": e["content"],
                        "source_id": e["source_id"],
                        "source_revision": e["source_revision"],
                        "clock_conflict": e["clock_conflict"],
                    }
                    for e in state["timeline"]
                ],
                "hypotheses": [],
                "causal_claim": None,
            }
        }

    graph = StateGraph(TimelineState)
    for name, node in (
        ("retrieve", retrieve),
        ("project", project),
        ("summarize", summarize),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "project")
    graph.add_edge("project", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


async def history(conn, user, incident):
    return [
        dict(e)
        for e in await conn.fetch(
            "SELECT * FROM ext_incident_events WHERE user_id=$1 AND incident_id=$2 ORDER BY arrival_order",
            user,
            incident,
        )
    ]


@router.get("/incidents/{incident_id}/timeline", response_model=TimelineResult)
async def timeline(incident_id: UUID, user_id: str, request: Request):
    async with request.app.state.pool.acquire() as conn:

        async def load():
            return await history(conn, user_id, incident_id)

        return await build_timeline_graph(load).ainvoke({}, {"recursion_limit": 8})


@router.post("/incidents/{incident_id}/summary", response_model=SummaryResult)
async def summarize(incident_id: UUID, body: Scope, request: Request):
    permission(request, "submit")
    async with locked(
        request.app.state.pool, f"incident:{body.user_id}:{incident_id}"
    ) as conn:

        async def load():
            return await history(conn, body.user_id, incident_id)

        result = await build_timeline_graph(load).ainvoke({}, {"recursion_limit": 8})
        await conn.execute(
            "INSERT INTO ext_incident_summaries VALUES($1,$2,$3,$4::jsonb) ON CONFLICT(user_id,incident_id) DO UPDATE SET evidence_digest=EXCLUDED.evidence_digest,summary=EXCLUDED.summary",
            body.user_id,
            incident_id,
            result["evidence_digest"],
            canonical(result["summary"]),
        )
    return {"summary": result["summary"], "stale": False}


@router.get("/incidents/{incident_id}/summary", response_model=SummaryResult)
async def stored_summary(incident_id: UUID, user_id: str, request: Request):
    async with locked(
        request.app.state.pool, f"incident:{user_id}:{incident_id}"
    ) as conn:
        row = await conn.fetchrow(
            "SELECT * FROM ext_incident_summaries WHERE user_id=$1 AND incident_id=$2",
            user_id,
            incident_id,
        )
        if not row:
            raise HTTPException(404, "No stored summary")
        events = await history(conn, user_id, incident_id)
        fingerprint = digest(
            canonical([(str(e["event_id"]), e["digest"]) for e in events])
        )
    return {
        "summary": json.loads(row["summary"]),
        "stale": fingerprint != row["evidence_digest"],
    }
