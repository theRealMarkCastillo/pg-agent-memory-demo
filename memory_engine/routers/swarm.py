from fastapi import APIRouter, Request
from pydantic import BaseModel
import json

router = APIRouter()


class BlackboardTask(BaseModel):
    workflow_id: str
    task_name: str
    payload: dict = {}


class TaskClaim(BaseModel):
    task_id: str
    agent_name: str


class TaskComplete(BaseModel):
    task_id: str
    payload: dict = {}


@router.post("/tasks")
async def create_task(task: BlackboardTask, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO swarm_blackboard (workflow_id, task_name, payload)
            VALUES ($1, $2, $3::jsonb)
            RETURNING task_id
            """,
            task.workflow_id,
            task.task_name,
            json.dumps(task.payload),
        )

    return {"task_id": str(row["task_id"])}


@router.post("/tasks/claim-next")
async def claim_next_pending(agent_name: str, workflow_id: str, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT task_id, task_name, payload
                FROM swarm_blackboard
                WHERE workflow_id = $1
                  AND status = 'PENDING'
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                workflow_id,
            )

            if not row:
                return {"status": "no_pending_tasks"}

            await conn.execute(
                """
                UPDATE swarm_blackboard
                SET status = 'IN_PROGRESS', assigned_agent = $2, updated_at = clock_timestamp()
                WHERE task_id = $1::uuid
                """,
                str(row["task_id"]),
                agent_name,
            )

    return {"status": "claimed", "task": dict(row)}


@router.post("/tasks/claim")
async def claim_task(claim: TaskClaim, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT task_id, task_name, payload
                FROM swarm_blackboard
                WHERE task_id = $1::uuid
                  AND status = 'PENDING'
                FOR UPDATE SKIP LOCKED
                """,
                claim.task_id,
            )

            if not row:
                return {"status": "already_claimed_or_not_found"}

            await conn.execute(
                """
                UPDATE swarm_blackboard
                SET status = 'IN_PROGRESS', assigned_agent = $2, updated_at = clock_timestamp()
                WHERE task_id = $1::uuid
                """,
                claim.task_id,
                claim.agent_name,
            )

    return {"status": "claimed", "task": dict(row)}


@router.post("/tasks/complete")
async def complete_task(task: TaskComplete, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE swarm_blackboard
            SET status = 'COMPLETED', payload = $2::jsonb, updated_at = clock_timestamp()
            WHERE task_id = $1::uuid
            """,
            task.task_id,
            json.dumps(task.payload),
        )

    return {"status": "completed"}


@router.get("/tasks/{workflow_id}")
async def list_workflow_tasks(workflow_id: str, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT task_id, task_name, assigned_agent, status, payload, updated_at
            FROM swarm_blackboard
            WHERE workflow_id = $1
            ORDER BY updated_at DESC
            """,
            workflow_id,
        )

    return [dict(r) for r in rows]
