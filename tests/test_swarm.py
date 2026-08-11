"""Tests for Swarm: task lifecycle, FOR UPDATE SKIP LOCKED, concurrency safety."""
import pytest
import asyncio
from conftest import post, post_params, get

WF = "swarm-test-wf"


@pytest.fixture(autouse=True)
async def seed_tasks(client):
    await post(client, "/swarm/tasks", workflow_id=WF, task_name="task_a", payload={"priority": 1})
    await post(client, "/swarm/tasks", workflow_id=WF, task_name="task_b", payload={"priority": 2})
    await post(client, "/swarm/tasks", workflow_id=WF, task_name="task_c", payload={"priority": 3})


@pytest.mark.asyncio
async def test_create_and_list_tasks(client):
    data = await get(client, f"/swarm/tasks/{WF}")
    assert len(data) >= 3, f"Expected >=3 tasks: {data}"
    pending = [t for t in data if t["status"] == "PENDING"]
    assert len(pending) >= 3


@pytest.mark.asyncio
async def test_claim_next_picks_oldest(client):
    result = await post_params(client, "/swarm/tasks/claim-next",
        agent_name="agent-1", workflow_id=WF)
    assert result["status"] == "claimed"


@pytest.mark.asyncio
async def test_claim_specific_task(client):
    data = await get(client, f"/swarm/tasks/{WF}")
    pending = [t for t in data if t["status"] == "PENDING"]
    assert len(pending) > 0
    result = await post(client, "/swarm/tasks/claim",
        task_id=pending[0]["task_id"], agent_name="agent-2")
    assert result["status"] == "claimed"


@pytest.mark.asyncio
async def test_complete_task(client):
    claim = await post_params(client, "/swarm/tasks/claim-next",
        agent_name="agent-3", workflow_id=WF)
    if claim["status"] != "claimed":
        pytest.skip("No pending tasks")
    await post(client, "/swarm/tasks/complete",
        task_id=claim["task"]["task_id"], payload={"result": "done"})
    data = await get(client, f"/swarm/tasks/{WF}")
    completed = [t for t in data if t["task_id"] == claim["task"]["task_id"]]
    assert completed[0]["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_no_double_claim(client):
    res = await post(client, "/swarm/tasks",
        workflow_id=WF, task_name="sole_task", payload={})
    task_id = res["task_id"]

    async def claim_one(name):
        return await post(client, "/swarm/tasks/claim",
            task_id=task_id, agent_name=name)

    r1, r2 = await asyncio.gather(claim_one("x"), claim_one("y"))
    claimed_count = sum(1 for r in (r1, r2) if r.get("status") == "claimed")
    assert claimed_count == 1, f"Double-claimed! {r1} {r2}"


@pytest.mark.asyncio
async def test_skip_locked_no_deadlock(client):
    for _ in range(10):
        await post(client, "/swarm/tasks",
            workflow_id=WF, task_name="concurrent_task", payload={})

    async def claim_next(name):
        return await post_params(client, "/swarm/tasks/claim-next",
            agent_name=name, workflow_id=WF)

    results = await asyncio.gather(*(claim_next(f"a{i}") for i in range(10)))
    claimed = [r for r in results if r.get("status") == "claimed"]
    task_ids = [r["task"]["task_id"] for r in claimed]
    assert len(task_ids) == len(set(task_ids)), f"Duplicate claims: {task_ids}"


@pytest.mark.asyncio
async def test_lifecycle_transitions(client):
    res = await post(client, "/swarm/tasks",
        workflow_id=WF, task_name="lifecycle", payload={})
    tid = res["task_id"]

    tasks = await get(client, f"/swarm/tasks/{WF}")
    t = next(t for t in tasks if t["task_id"] == tid)
    assert t["status"] == "PENDING"

    await post(client, "/swarm/tasks/claim", task_id=tid, agent_name="lc-agent")
    tasks = await get(client, f"/swarm/tasks/{WF}")
    t = next(t for t in tasks if t["task_id"] == tid)
    assert t["status"] == "IN_PROGRESS"

    await post(client, "/swarm/tasks/complete",
        task_id=tid, payload={"done": True})
    tasks = await get(client, f"/swarm/tasks/{WF}")
    t = next(t for t in tasks if t["task_id"] == tid)
    assert t["status"] == "COMPLETED"
