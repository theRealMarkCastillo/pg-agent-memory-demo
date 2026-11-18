"""Revision-bound review and restartable local-file effects.

The host controls its workspace. Writers use this service; arbitrary concurrent
OS writers are outside the file reconciliation contract. No external effects.
"""

import difflib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID, uuid4
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
from .common import (
    StrictModel,
    actor,
    canonical,
    digest,
    locked,
    effect_locked,
    owned,
    permission,
)

from .responses import ProposalRecord, RunResult

router = APIRouter()


class Proposal(StrictModel):
    proposal_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    target: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
    base_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    content: str = Field(max_length=100_000)


class Edit(StrictModel):
    revision: int = Field(ge=1)
    content: str = Field(max_length=100_000)


class Review(StrictModel):
    revision: int = Field(ge=1)
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["approved", "rejected"]


class Empty(StrictModel):
    pass


class ApprovalState(TypedDict, total=False):
    proposal_id: str
    revision: int
    status: str


def target_path(root: Path, user: str, target: str) -> Path:
    # Each owner has a distinct directory; target cannot choose ownership/path.
    folder = root / digest(user)
    if folder.is_symlink():
        raise HTTPException(409, "Workspace directory is a symlink")
    folder.mkdir(exist_ok=True)
    path = folder / target
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise HTTPException(409, "Target must be an ordinary file")
    return path


def read_target(path):
    return path.read_text() if path.exists() else ""


def replace_target(path, content):
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as file:
            name = file.name
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


async def fetch(pool, proposal_id):
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT * FROM ext_proposals WHERE proposal_id=$1", proposal_id
        )


async def fresh_evidence(conn, row):
    evidence = json.loads(row["evidence"])
    if evidence:
        # Domain-specific checks are registered in the support module.
        from .support import validate_snapshot

        await validate_snapshot(conn, row["user_id"], evidence)


def build_graph(pool, root, saver):
    async def propose(state):
        row = await fetch(pool, UUID(state["proposal_id"]))
        return {"revision": row["revision"], "status": row["status"]}

    async def review(state):
        row = await fetch(pool, UUID(state["proposal_id"]))
        interrupt(
            {
                "proposal_id": state["proposal_id"],
                "revision": row["revision"],
                "digest": row["digest"],
                "content": row["content"],
            }
        )
        # Resume contents never grant authority. The application persisted review.
        row = await fetch(pool, UUID(state["proposal_id"]))
        if row["status"] == "pending":
            raise HTTPException(409, "Review is still pending")
        return {"revision": row["revision"], "status": row["status"]}

    async def apply(state):
        proposal_id = UUID(state["proposal_id"])
        row = await fetch(pool, proposal_id)
        # Serialize different proposals for the same destination as well as retries.
        async with effect_locked(
            pool, f"file:{row['user_id']}:{row['target']}"
        ) as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ext_proposals WHERE proposal_id=$1",
                proposal_id,
            )
            if row["status"] not in ("approved", "applying"):
                return {"status": row["status"]}
            path = target_path(root, row["user_id"], row["target"])
            current = digest(read_target(path))
            if current not in (row["base_digest"], row["digest"]):
                raise HTTPException(409, "Base changed; create a new proposal")
            # Commit intent before file I/O. A later edit/cancel cannot erase
            # an ambiguous attempt, even if another proposal changes the file.
            # Hold the support freshness lock through the actual write, using
            # a separate transaction after the intent has been made durable.
            async with conn.transaction():
                if current != row["digest"]:
                    await fresh_evidence(conn, row)
                await conn.execute(
                    "UPDATE ext_proposals SET status='applying',effect_started_at=COALESCE(effect_started_at,clock_timestamp()) WHERE proposal_id=$1",
                    proposal_id,
                )
            if current != row["digest"]:
                async with conn.transaction():
                    await fresh_evidence(conn, row)
                    replace_target(path, row["content"])
                    crash_after_replace(proposal_id)
                    await acknowledge(conn, proposal_id)
            else:
                # A completed local effect is an observation, not a new action.
                # Policy changes after that effect must not erase its history.
                await acknowledge(conn, proposal_id)
        return {"status": "applied"}

    graph = StateGraph(ApprovalState)
    graph.add_node("proposal", propose)
    graph.add_node("review", review)
    graph.add_node("apply", apply)
    graph.add_edge(START, "proposal")
    graph.add_edge("proposal", "review")
    graph.add_edge("review", "apply")
    graph.add_edge("apply", END)
    return graph.compile(checkpointer=saver)


def crash_after_replace(proposal_id):
    # Test-only host configuration, never a request/model argument.
    marker = os.getenv("EXTENSION_CRASH_AFTER_REPLACE_MARKER")
    if marker and not Path(marker).exists():
        Path(marker).write_text(str(proposal_id))
        os._exit(86)


async def acknowledge(conn, proposal_id):
    await conn.execute(
        "UPDATE ext_proposals SET status='applied',applied_at=clock_timestamp() WHERE proposal_id=$1",
        proposal_id,
    )


def require_unstarted(request, row):
    if row["effect_started_at"] is not None:
        raise HTTPException(
            409, "Effect started; resume to reconcile before any further change"
        )
    # Older, pre-intent checkpoints can already have an unacknowledged file.
    if row["status"] == "approved" and row["digest"] != row["base_digest"]:
        path = target_path(request.app.state.workspace, row["user_id"], row["target"])
        if digest(read_target(path)) == row["digest"]:
            raise HTTPException(
                409, "Destination matches approved content; resume to reconcile"
            )


async def create_proposal(request, body, evidence=None):
    actor(request)
    permission(request, "submit")
    pool = request.app.state.pool
    async with locked(pool, f"proposal:{body.proposal_id}") as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM ext_proposals WHERE proposal_id=$1", body.proposal_id
        )
        if existing:
            owned(request, existing)
            if (
                existing["user_id"],
                existing["target"],
                existing["base_digest"],
                existing["content"],
                json.loads(existing["evidence"]),
            ) != (
                body.user_id,
                body.target,
                body.base_digest,
                body.content,
                evidence or {},
            ):
                raise HTTPException(409, "Proposal ID already has different content")
            return dict(existing)
        path = target_path(request.app.state.workspace, body.user_id, body.target)
        if digest(read_target(path)) != body.base_digest:
            raise HTTPException(409, "Base digest does not match destination")
        row = await conn.fetchrow(
            "INSERT INTO ext_proposals(proposal_id,user_id,target,base_digest,content,digest,thread_id,effect_key,evidence) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb) RETURNING *",
            body.proposal_id,
            body.user_id,
            body.target,
            body.base_digest,
            body.content,
            digest(body.content),
            f"approval:{digest(body.user_id)}:{uuid4()}",
            uuid4(),
            canonical(evidence or {}),
        )
        await conn.execute(
            "INSERT INTO ext_proposal_revisions VALUES($1,1,$2,$3)",
            body.proposal_id,
            body.content,
            row["digest"],
        )
    return dict(row)


@router.post("/proposals", response_model=ProposalRecord)
async def submit(body: Proposal, request: Request):
    return await create_proposal(request, body)


@router.get("/proposals/{proposal_id}", response_model=ProposalRecord)
async def get_proposal(proposal_id: UUID, request: Request):
    row = owned(request, await fetch(request.app.state.pool, proposal_id))
    path = target_path(request.app.state.workspace, row["user_id"], row["target"])
    diff = "".join(
        difflib.unified_diff(
            read_target(path).splitlines(True),
            row["content"].splitlines(True),
            fromfile="base",
            tofile=f"revision-{row['revision']}",
        )
    )
    return {**dict(row), "diff": diff}


@router.post("/proposals/{proposal_id}/edit", response_model=ProposalRecord)
async def edit(proposal_id: UUID, body: Edit, request: Request):
    permission(request, "submit")
    async with locked(request.app.state.pool, f"graph:{proposal_id}") as conn:
        row = owned(
            request,
            await conn.fetchrow(
                "SELECT * FROM ext_proposals WHERE proposal_id=$1 FOR UPDATE",
                proposal_id,
            ),
        )
        require_unstarted(request, row)
        if row["revision"] != body.revision or row["status"] not in (
            "pending",
            "approved",
            "rejected",
        ):
            raise HTTPException(409, "Revision cannot be edited")
        if json.loads(row["evidence"]):
            raise HTTPException(
                409, "Regenerate evidence-backed drafts instead of editing claims"
            )
        row = await conn.fetchrow(
            "UPDATE ext_proposals SET content=$2,digest=$3,revision=revision+1,status='pending',thread_id=$4 WHERE proposal_id=$1 RETURNING *",
            proposal_id,
            body.content,
            digest(body.content),
            f"approval:{digest(row['user_id'])}:{uuid4()}",
        )
        await conn.execute(
            "INSERT INTO ext_proposal_revisions VALUES($1,$2,$3,$4)",
            proposal_id,
            row["revision"],
            row["content"],
            row["digest"],
        )
    return dict(row)


@router.post("/proposals/{proposal_id}/review", response_model=ProposalRecord)
async def decide(proposal_id: UUID, body: Review, request: Request):
    permission(request, "review")
    reviewer = actor(request)
    async with locked(request.app.state.pool, f"graph:{proposal_id}") as conn:
        row = owned(
            request,
            await conn.fetchrow(
                "SELECT * FROM ext_proposals WHERE proposal_id=$1 FOR UPDATE",
                proposal_id,
            ),
        )
        if row["revision"] != body.revision or row["digest"] != body.digest:
            raise HTTPException(409, "Review must bind the current revision and digest")
        previous = await conn.fetchrow(
            "SELECT * FROM ext_reviews WHERE proposal_id=$1 AND revision=$2",
            proposal_id,
            body.revision,
        )
        if previous:
            if (
                previous["decision"] != body.decision
                or previous["reviewer"] != reviewer
            ):
                raise HTTPException(409, "Revision already decided")
            return dict(row)
        if row["status"] != "pending":
            raise HTTPException(409, "Proposal is no longer pending")
        if body.decision == "approved":
            await fresh_evidence(conn, row)
        await conn.execute(
            "INSERT INTO ext_reviews(proposal_id,revision,digest,reviewer,decision) VALUES($1,$2,$3,$4,$5)",
            proposal_id,
            body.revision,
            body.digest,
            reviewer,
            body.decision,
        )
        row = await conn.fetchrow(
            "UPDATE ext_proposals SET status=$2 WHERE proposal_id=$1 RETURNING *",
            proposal_id,
            body.decision,
        )
    return dict(row)


@router.post(
    "/proposals/{proposal_id}/cancel",
    response_model=RunResult,
    response_model_exclude_none=True,
)
async def cancel(proposal_id: UUID, body: Empty, request: Request):
    permission(request, "submit")
    async with locked(request.app.state.pool, f"graph:{proposal_id}") as conn:
        row = owned(
            request,
            await conn.fetchrow(
                "SELECT * FROM ext_proposals WHERE proposal_id=$1 FOR UPDATE",
                proposal_id,
            ),
        )
        if row["status"] == "applied":
            raise HTTPException(409, "Already applied")
        require_unstarted(request, row)
        await conn.execute(
            "UPDATE ext_proposals SET status='cancelled' WHERE proposal_id=$1",
            proposal_id,
        )
    return {"status": "cancelled"}


@router.post(
    "/proposals/{proposal_id}/run",
    response_model=RunResult,
    response_model_exclude_none=True,
)
async def run(proposal_id: UUID, body: Empty, request: Request):
    permission(request, "run")
    # Request has no thread/config/context fields. Rehydrate identity from DB.
    async with locked(request.app.state.pool, f"graph:{proposal_id}") as conn:
        row = owned(
            request,
            await conn.fetchrow(
                "SELECT * FROM ext_proposals WHERE proposal_id=$1", proposal_id
            ),
        )
        if row["status"] in ("applied", "cancelled", "rejected"):
            return {"status": row["status"]}
        config = {"configurable": {"thread_id": row["thread_id"]}, "recursion_limit": 8}
        graph = request.app.state.approval_graph
        snapshot = await graph.aget_state(config)
        if snapshot.values:
            # A crashed node is retried with None; only paused interrupts resume.
            paused = any(task.interrupts for task in snapshot.tasks)
            if paused and row["status"] == "pending":
                return {"status": "pending", "revision": row["revision"]}
            value = Command(resume=True) if paused else None
        else:
            value = {"proposal_id": str(proposal_id)}
        result = await graph.ainvoke(value, config, durability="sync")
        if result.get("__interrupt__"):
            # The first invocation always exposes the review boundary, even if a
            # decision was stored before the graph started.
            return {"status": "pending", "revision": row["revision"]}
        return {"status": result["status"], "revision": result["revision"]}
