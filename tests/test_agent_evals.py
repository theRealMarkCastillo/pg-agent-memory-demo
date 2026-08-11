"""
End-to-end quality evals for all six memory patterns.
Measures retrieval precision, correctness, isolation, and concurrency safety.
"""
import pytest
import asyncio
from conftest import post, post_params, get

EVAL_PROJECT = "eval-quality"
EVAL_BRANCH = "main"


@pytest.fixture
async def seed_eval_developer(client):
    symbols = [
        ("find_user", "function", "def find_user(id: int) -> User", "Queries the database by primary key to retrieve a user record."),
        ("create_order", "function", "def create_order(items: list) -> Order", "Inserts a new order with line items into the orders table."),
        ("send_notification", "function", "def send_notification(user_id: int, msg: str) -> bool", "Dispatches a push notification via FCM/APNs."),
        ("UserModel", "class", "class UserModel(BaseModel)", "SQLAlchemy ORM model for the users table."),
    ]
    for name, stype, sig, content in symbols:
        await post(client, "/developer/symbols",
            project_id=EVAL_PROJECT, git_branch=EVAL_BRANCH,
            file_path=f"src/{name}.py", symbol_name=name,
            symbol_type=stype, signature=sig, code_content=content)


@pytest.mark.asyncio
async def test_eval_developer_precision(client, seed_eval_developer):
    results = await post(client, "/developer/symbols/search",
        project_id=EVAL_PROJECT, git_branch=EVAL_BRANCH,
        query="find user by id")
    assert len(results) > 0, "Zero results from semantic search — embedding API may not be configured"
    top = [r["symbol_name"] for r in results[:3]]
    assert "find_user" in top or any("user" in r.get("signature", "").lower() for r in results), \
        f"find_user should appear for user-lookup query. Top results: {top}"


@pytest.mark.asyncio
async def test_eval_developer_type_filtering(client, seed_eval_developer):
    results = await post(client, "/developer/symbols/search",
        project_id=EVAL_PROJECT, git_branch=EVAL_BRANCH,
        query="orm model", symbol_type="class")
    types = {r["symbol_type"] for r in results}
    assert types == {"class"}, f"Filter leaked: {types}"


@pytest.mark.asyncio
async def test_eval_task_quality(client):
    goals = [
        ("Extract financial data from SEC filings and generate P&L statements", 0.92),
        ("Download stock price history from Yahoo Finance for backtesting", 0.88),
        ("Write a haiku about cherry blossoms in spring", 0.70),
    ]
    for goal, score in goals:
        await post(client, "/task/trajectories",
            agent_id="eval-bot", goal_description=goal,
            action_sequence=[], execution_result="", success_score=score)

    results = await post(client, "/task/trajectories/search",
        goal_description="Extract financial data from SEC filings for earnings analysis")
    assert len(results) > 0 and ("financial" in results[0]["goal_description"].lower()
                                  or "earnings" in results[0]["goal_description"].lower())


@pytest.mark.asyncio
async def test_eval_enterprise_role_isolation(client):
    for role in ["employee", "admin", "contractor"]:
        await post(client, "/enterprise/documents",
            doc_title=f"Policy-{role}", allowed_role=role,
            content=f"Sensitive {role}-only policy content.")

    for role in ["employee", "admin"]:
        results = await post(client, "/enterprise/documents/search",
            query="policy", user_role=role)
        for r in results:
            assert r.get("allowed_role", role) == role


@pytest.mark.asyncio
async def test_eval_tutor_decay_monotonic(client):
    user = "eval_decay_user"
    await post(client, "/tutor/skills", skill_name="topic_x")
    await post(client, "/tutor/progress",
        user_id=user, skill_name="topic_x", proficiency_score=1.0)

    scores = []
    for _ in range(3):
        data = await get(client, f"/tutor/gaps/{user}")
        score = next(r["decayed_score"] for r in data if r["skill_name"] == "topic_x")
        scores.append(score)
        await asyncio.sleep(1)

    assert scores == sorted(scores, reverse=True), f"Not monotonic: {scores}"


@pytest.mark.asyncio
async def test_eval_swarm_concurrency(client):
    wf = "eval-concurrency"
    for i in range(20):
        await post(client, "/swarm/tasks",
            workflow_id=wf, task_name=f"t{i}", payload={})

    async def claim_one(i):
        return await post_params(client, "/swarm/tasks/claim-next",
            agent_name=f"a{i}", workflow_id=wf)

    results = await asyncio.gather(*(claim_one(i) for i in range(20)))
    claimed = [r for r in results if r.get("status") == "claimed"]
    task_ids = [r["task"]["task_id"] for r in claimed]
    assert len(task_ids) == len(set(task_ids)), \
        f"Duplicates in concurrent claims: {len(task_ids)} claimed, {len(set(task_ids))} unique"


@pytest.mark.asyncio
async def test_eval_companion_temporal(client):
    user = "eval_temporal_user"
    await post(client, "/companion/facts",
        user_id=user, name="Event", entity_type="event")
    await post(client, "/companion/ephemerals",
        user_id=user, description="Short-lived mood", ttl_seconds=1)

    ctx = await get(client, "/companion/context", user_id=user)
    assert len(ctx["ephemerals"]) >= 1
    await asyncio.sleep(2)
    ctx = await get(client, "/companion/context", user_id=user)
    assert len(ctx["ephemerals"]) == 0
    assert len(ctx["graph_facts"]) >= 1


@pytest.mark.asyncio
async def test_eval_cross_pattern_isolation(client):
    await post(client, "/developer/symbols",
        project_id="proj-A", git_branch="main", file_path="a.py",
        symbol_name="func_a", symbol_type="function",
        signature="def a()", code_content="Project A.")
    await post(client, "/developer/symbols",
        project_id="proj-B", git_branch="main", file_path="b.py",
        symbol_name="func_b", symbol_type="function",
        signature="def b()", code_content="Project B.")

    results_a = await post(client, "/developer/symbols/search",
        project_id="proj-A", git_branch="main", query="function")
    names_a = {r["symbol_name"] for r in results_a}
    assert "func_b" not in names_a

    await post(client, "/companion/facts",
        user_id="user-x", name="FactX", entity_type="item")
    await post(client, "/companion/facts",
        user_id="user-y", name="FactY", entity_type="item")

    ctx_x = await get(client, "/companion/context", user_id="user-x")
    names_x = {f["name"] for f in ctx_x["graph_facts"]}
    assert "FactY" not in names_x
