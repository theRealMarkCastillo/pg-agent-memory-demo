from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional
import os
import json
from embedding import get_embedding_client

router = APIRouter()


class TrajectoryStore(BaseModel):
    agent_id: str
    goal_description: str
    action_sequence: list
    execution_result: str
    success_score: float = Field(ge=0, le=1)


class TrajectorySearch(BaseModel):
    goal_description: str
    min_success_score: float = Field(default=0.7, ge=0, le=1)
    min_similarity: float = Field(default=0.35, ge=-1, le=1)
    agent_id: Optional[str] = None


@router.post("/trajectories")
async def store_trajectory(traj: TrajectoryStore, request: Request):
    pool = request.app.state.pool

    emb_resp = await get_embedding_client().embeddings.create(
        input=traj.goal_description, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO task_trajectories (agent_id, goal_description, goal_embedding, action_sequence, execution_result, success_score)
            VALUES ($1, $2, $3::halfvec, $4::jsonb, $5, $6)
            """,
            traj.agent_id,
            traj.goal_description,
            json.dumps(embedding),
            json.dumps(traj.action_sequence),
            traj.execution_result,
            traj.success_score,
        )

    return {"status": "stored"}


@router.post("/trajectories/search")
async def search_trajectories(search: TrajectorySearch, request: Request):
    pool = request.app.state.pool

    emb_resp = await get_embedding_client().embeddings.create(
        input=search.goal_description, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT trajectory_id, agent_id, goal_description, action_sequence, execution_result, success_score,
                   1 - (goal_embedding <=> $1::halfvec) AS similarity
            FROM task_trajectories
            WHERE success_score >= $2 AND ($3::text IS NULL OR agent_id=$3)
              AND goal_embedding <=> $1::halfvec <= 1.0-$4::double precision
            ORDER BY goal_embedding <=> $1::halfvec
            LIMIT 5
            """,
            json.dumps(embedding),
            search.min_success_score,
            search.agent_id,
            search.min_similarity,
        )

    return [dict(r) for r in rows]
