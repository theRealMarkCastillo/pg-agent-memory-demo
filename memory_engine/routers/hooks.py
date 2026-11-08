"""Scoped, idempotent session excerpts; no inferred facts or success claims."""

import json
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

router = APIRouter()
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Excerpt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]


class Scope(BaseModel):
    project_id: Identifier
    git_branch: Identifier


class Session(Scope):
    source: Identifier
    session_id: Identifier


class Capture(Session):
    turn_id: Identifier
    prompt: Excerpt
    response: Excerpt


class Search(Scope):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    limit: int = Field(default=3, ge=1, le=5)


@router.post("/memories")
async def capture(body: Capture, request: Request):
    async with request.app.state.pool.acquire() as conn:
        async with conn.transaction():
            # Serialize duplicate deliveries before testing an existing immutable row.
            key = [body.project_id, body.git_branch, body.source, body.session_id, body.turn_id]
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,3))", json.dumps(key))
            old = await conn.fetchrow(
                """SELECT memory_id,prompt,response FROM agent_hook_memories
                WHERE project_id=$1 AND git_branch=$2 AND source=$3 AND session_id=$4 AND turn_id=$5""", *key
            )
            if old:
                if (old["prompt"], old["response"]) != (body.prompt, body.response):
                    raise HTTPException(409, "Turn ID already has different content")
                return {"status": "duplicate", "memory_id": str(old["memory_id"])}
            memory_id = await conn.fetchval(
                """INSERT INTO agent_hook_memories
                (project_id,git_branch,source,session_id,turn_id,prompt,response)
                VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING memory_id""", *key, body.prompt, body.response
            )
    return {"status": "stored", "memory_id": str(memory_id)}


@router.post("/memories/search")
async def recall(body: Search, request: Request):
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT memory_id,source,session_id,turn_id,prompt,response,created_at,
                      ts_rank_cd(search_document,q) AS rank
               FROM agent_hook_memories, websearch_to_tsquery('english',$3) q
               WHERE project_id=$1 AND git_branch=$2 AND search_document @@ q
               ORDER BY rank DESC,created_at DESC,memory_id LIMIT $4""",
            body.project_id, body.git_branch, body.query, body.limit,
        )
    return {"memories": [dict(row, verified=False) for row in rows]}


@router.post("/memories/forget")
async def forget(body: Session, request: Request):
    async with request.app.state.pool.acquire() as conn:
        result = await conn.execute(
            """DELETE FROM agent_hook_memories
               WHERE project_id=$1 AND git_branch=$2 AND source=$3 AND session_id=$4""",
            body.project_id, body.git_branch, body.source, body.session_id,
        )
    return {"deleted": int(result.split()[-1])}
