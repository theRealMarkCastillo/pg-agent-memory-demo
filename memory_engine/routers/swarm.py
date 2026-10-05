from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from uuid import UUID
import json

router = APIRouter()


class BlackboardTask(BaseModel):
    workflow_id: str
    task_name: str
    payload: dict = Field(default_factory=dict)


class TaskClaim(BaseModel):
    task_id: UUID
    agent_name: str


class TaskComplete(TaskClaim):
    lease_token: UUID
    payload: dict = Field(default_factory=dict)


@router.post("/tasks")
async def create_task(task: BlackboardTask, request: Request):
    async with request.app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO swarm_blackboard(workflow_id,task_name,payload) VALUES($1,$2,$3::jsonb) RETURNING task_id",
            task.workflow_id,
            task.task_name,
            json.dumps(task.payload),
        )
    return {"task_id": str(row["task_id"])}


async def claim(request, agent_name, workflow_id=None, task_id=None):
    async with request.app.state.pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """SELECT task_id FROM swarm_blackboard
                WHERE ($1::text IS NULL OR workflow_id=$1) AND ($2::uuid IS NULL OR task_id=$2)
                AND (status='PENDING' OR (status='IN_PROGRESS' AND lease_until<clock_timestamp()))
                ORDER BY created_at,task_id LIMIT 1 FOR UPDATE SKIP LOCKED""",
                workflow_id,
                task_id,
            )
            if not row:
                return {
                    "status": "no_pending_tasks"
                    if task_id is None
                    else "already_claimed_or_not_found"
                }
            row = await conn.fetchrow(
                """UPDATE swarm_blackboard SET status='IN_PROGRESS',assigned_agent=$2,
                lease_token=gen_random_uuid(),lease_until=clock_timestamp()+interval '5 minutes',updated_at=clock_timestamp()
                WHERE task_id=$1 RETURNING task_id,task_name,payload,lease_token,lease_until""",
                row["task_id"],
                agent_name,
            )
    return {"status": "claimed", "task": dict(row)}


@router.post("/tasks/claim-next")
async def claim_next_pending(agent_name: str, workflow_id: str, request: Request):
    return await claim(request, agent_name, workflow_id=workflow_id)


@router.post("/tasks/claim")
async def claim_task(body: TaskClaim, request: Request):
    return await claim(request, body.agent_name, task_id=body.task_id)


@router.post("/tasks/complete")
async def complete_task(task: TaskComplete, request: Request):
    async with request.app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """UPDATE swarm_blackboard SET status='COMPLETED',payload=payload || $4::jsonb,updated_at=clock_timestamp()
            WHERE task_id=$1 AND assigned_agent=$2 AND lease_token=$3 AND status='IN_PROGRESS' AND lease_until>clock_timestamp()
            RETURNING task_id""",
            task.task_id,
            task.agent_name,
            task.lease_token,
            json.dumps(task.payload),
        )
    if not row:
        raise HTTPException(409, "Task is not owned by this active lease")
    return {"status": "completed"}


@router.get("/tasks/{workflow_id}")
async def list_workflow_tasks(workflow_id: str, request: Request):
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT task_id,task_name,assigned_agent,status,payload,updated_at,lease_until FROM swarm_blackboard WHERE workflow_id=$1 ORDER BY created_at,task_id",
            workflow_id,
        )
    return [dict(r) for r in rows]
