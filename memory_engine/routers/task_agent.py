from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI
import os
import json

router = APIRouter()

embedding_client = AsyncOpenAI(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
)


class TrajectoryStore(BaseModel):
    agent_id: str
    goal_description: str
    action_sequence: list
    execution_result: str
    success_score: float


class TrajectorySearch(BaseModel):
    goal_description: str
    min_success_score: Optional[float] = 0.7


@router.post("/trajectories")
async def store_trajectory(traj: TrajectoryStore, request: Request):
    pool = request.app.state.pool

    emb_resp = await embedding_client.embeddings.create(
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

    emb_resp = await embedding_client.embeddings.create(
        input=search.goal_description, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT trajectory_id, agent_id, goal_description, action_sequence, execution_result, success_score,
                   1 - (goal_embedding <=> $1::halfvec) AS similarity
            FROM task_trajectories
            WHERE success_score >= $2
            ORDER BY goal_embedding <=> $1::halfvec
            LIMIT 5
            """,
            json.dumps(embedding),
            search.min_success_score,
        )

    return [dict(r) for r in rows]
