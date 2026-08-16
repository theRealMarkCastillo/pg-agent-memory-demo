import asyncio
import os
import httpx
from agents.developer_agent import build_developer_graph_with_checkpointer
from agents.task_agent import build_task_graph_with_checkpointer
from agents.enterprise_agent import build_enterprise_graph_with_checkpointer
from agents.tutor_agent import build_tutor_graph_with_checkpointer
from agents.swarm_agent import build_swarm_graph_with_checkpointer
from agents.companion_agent import build_companion_graph_with_checkpointer

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")
SEED_RETRIES = 30
SEED_RETRY_DELAY = 2


async def seed_data(client: httpx.AsyncClient):
    for attempt in range(1, SEED_RETRIES + 1):
        try:
            await client.get(f"{MEMORY_ENGINE_URL}/health")
            break
        except Exception:
            if attempt == SEED_RETRIES:
                raise
            print(f"Waiting for memory engine... ({attempt}/{SEED_RETRIES})")
            await asyncio.sleep(SEED_RETRY_DELAY)

    await client.post(
        f"{MEMORY_ENGINE_URL}/developer/symbols",
        json={
            "project_id": "demo-project",
            "git_branch": "main",
            "file_path": "src/memory/store.py",
            "symbol_name": "embed_memory",
            "symbol_type": "function",
            "signature": "def embed_memory(text: str) -> list[float]",
            "code_content": "Converts raw text into a vector embedding using the configured model for semantic retrieval.",
        },
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/developer/symbols",
        json={
            "project_id": "demo-project",
            "git_branch": "main",
            "file_path": "src/memory/recall.py",
            "symbol_name": "recall_relevant",
            "symbol_type": "function",
            "signature": "def recall_relevant(query: str, top_k: int = 5) -> list",
            "code_content": "Retrieves the top_k most semantically similar memory chunks given a query string.",
        },
    )

    await client.post(
        f"{MEMORY_ENGINE_URL}/task/trajectories",
        json={
            "agent_id": "task-bot-1",
            "goal_description": "Scrape product prices from competitor websites and compile a CSV report",
            "action_sequence": [
                {"action": "fetch_url", "url": "https://competitor.com/products"},
                {"action": "parse_prices", "selector": ".price-tag"},
                {"action": "write_csv", "file": "report.csv"},
            ],
            "execution_result": "Successfully scraped 45 products and generated CSV report with pricing data.",
            "success_score": 0.95,
        },
    )

    await client.post(
        f"{MEMORY_ENGINE_URL}/enterprise/documents",
        json={
            "doc_title": "Data Access Policy",
            "allowed_role": "employee",
            "content": "Employees may access customer data for support purposes only. All access is logged and audited quarterly.",
        },
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/enterprise/documents",
        json={
            "doc_title": "Admin-Only Security Protocol",
            "allowed_role": "admin",
            "content": "Only administrators may modify user roles and access control lists. All changes require two-factor authentication.",
        },
    )

    await client.post(
        f"{MEMORY_ENGINE_URL}/tutor/skills",
        json={"skill_name": "python_basics"},
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/tutor/skills",
        json={"skill_name": "async_await", "parent_skill_name": "python_basics"},
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/tutor/skills",
        json={"skill_name": "database_design", "parent_skill_name": "python_basics"},
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/tutor/progress",
        json={
            "user_id": "learner_001",
            "skill_name": "python_basics",
            "proficiency_score": 0.8,
        },
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/tutor/progress",
        json={
            "user_id": "learner_001",
            "skill_name": "async_await",
            "proficiency_score": 0.3,
        },
    )

    await client.post(
        f"{MEMORY_ENGINE_URL}/swarm/tasks",
        json={
            "workflow_id": "wf-001",
            "task_name": "analyze_sentiment",
            "payload": {"text": "I love this product!"},
        },
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/swarm/tasks",
        json={
            "workflow_id": "wf-001",
            "task_name": "extract_entities",
            "payload": {"text": "I love this product!"},
        },
    )

    await client.post(
        f"{MEMORY_ENGINE_URL}/companion/facts",
        json={
            "user_id": "usr_anthony",
            "name": "Apartment Hunting",
            "entity_type": "goal",
        },
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/companion/facts",
        json={
            "user_id": "usr_anthony",
            "name": "Brooklyn",
            "entity_type": "location",
            "relationship_to": "Apartment Hunting",
            "relationship_type": "target_area",
        },
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/companion/ephemerals",
        json={
            "user_id": "usr_anthony",
            "description": "Feeling excited about the new apartment listings this week",
            "ttl_seconds": 86400,
        },
    )
    await client.post(
        f"{MEMORY_ENGINE_URL}/companion/episodes",
        json={
            "user_id": "usr_anthony",
            "content": "Talked about wanting a 2-bedroom apartment in Brooklyn with natural light and a view of the Manhattan skyline. Budget is around $3,500/month.",
        },
    )

    await client.post(
        f"{MEMORY_ENGINE_URL}/companion/backstory",
        json={
            "user_id": "usr_anthony",
            "name": "Iris",
            "backstory": [
                {"name": "Iris", "entity_type": "self",
                 "relationship_to": "a quiet coastal town", "relationship_type": "lives_in"},
                {"name": "Iris", "entity_type": "self",
                 "relationship_to": "long conversations", "relationship_type": "values"},
                {"name": "Iris", "entity_type": "self",
                 "relationship_to": "poetry", "relationship_type": "writes"},
            ],
            "shared": [
                {"name": "Iris", "entity_type": "self",
                 "relationship_to": "usr_anthony", "relationship_type": "trusts"},
            ],
        },
    )

    print("Seed data inserted.\n")


def _last_assistant_response(result: dict) -> str:
    messages = result.get("messages", [])
    for m in reversed(messages):
        if hasattr(m, "tool_calls") and m.tool_calls:
            continue
        if hasattr(m, "type") and m.type == "tool":
            continue
        if hasattr(m, "content") and m.content:
            return m.content
    return "[No response generated]"


async def run_demos():
    print("=" * 60)
    print("Agent Memory Patterns Demonstration")
    print("  (Tool Calling + Write-Back + Checkpointer)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        await seed_data(client)

    print("\n[Demo 1/6] Developer Agent — Code Symbol Search + Write-Back")
    print("-" * 40)
    dev_app = await build_developer_graph_with_checkpointer()
    result = await dev_app.ainvoke(
        {
            "project_id": "demo-project",
            "git_branch": "main",
            "query": "How do I embed text for semantic search?",
        },
        config={"configurable": {"thread_id": "dev-demo-1"}},
    )
    print(f"Symbols Found:\n{result.get('retrieved_symbols', 'N/A')}")
    print(f"Response:\n{_last_assistant_response(result)}")

    print("\n[Demo 1b] Developer — Multi-turn: writes a new function to workspace")
    print("-" * 40)
    result2 = await dev_app.ainvoke(
        {
            "project_id": "demo-project",
            "git_branch": "main",
            "query": "Write a file called /tmp/agent-workspace/batch_embed.py with a batch_embed function that wraps embed_memory. Then run it with python3 to verify.",
        },
        config={"configurable": {"thread_id": "dev-demo-1"}},
    )
    print(f"Response:\n{_last_assistant_response(result2)}")

    print("\n[Demo 2/6] Task Agent — Trajectory Recall + Real Execution")
    print("-" * 40)
    task_app = await build_task_graph_with_checkpointer()
    result = await task_app.ainvoke(
        {
            "agent_id": "task-bot-2",
            "goal": "Write a Python script to /tmp/agent-workspace that counts words in a sentence, then execute it to verify.",
        },
        config={"configurable": {"thread_id": "task-demo-1"}},
    )
    print(f"Past Trajectories:\n{result.get('past_trajectories', 'N/A')}")
    print(f"Plan/Result:\n{_last_assistant_response(result)}")

    print("\n[Demo 2b] Task Agent — Multi-turn: fetch a real URL")
    print("-" * 40)
    result2 = await task_app.ainvoke(
        {
            "agent_id": "task-bot-2",
            "goal": "Fetch https://httpbin.org/json and extract the slide titles from the response.",
        },
        config={"configurable": {"thread_id": "task-demo-1"}},
    )
    print(f"Plan/Result:\n{_last_assistant_response(result2)}")

    print("\n[Demo 3/6] Enterprise Agent — Role-Filtered Policy Search + Write-Back")
    print("-" * 40)
    ent_app = await build_enterprise_graph_with_checkpointer()
    result = await ent_app.ainvoke(
        {
            "user_role": "employee",
            "query": "Can I access customer data?",
        },
        config={"configurable": {"thread_id": "ent-demo-1"}},
    )
    print(f"Docs Found:\n{result.get('retrieved_docs', 'N/A')}")
    print(f"Response:\n{_last_assistant_response(result)}")

    print("\n[Demo 4/6] Tutor Agent — Skill Gap Assessment + Progress Update")
    print("-" * 40)
    tutor_app = await build_tutor_graph_with_checkpointer()
    result = await tutor_app.ainvoke(
        {
            "user_id": "learner_001",
            "topic": "async programming",
        },
        config={"configurable": {"thread_id": "tutor-demo-1"}},
    )
    print(f"Skill Gaps:\n{result.get('skill_gaps', 'N/A')}")
    print(f"Response:\n{_last_assistant_response(result)}")

    print("\n[Demo 4b] Tutor — Multi-turn: after learning, update progress")
    print("-" * 40)
    result2 = await tutor_app.ainvoke(
        {
            "user_id": "learner_001",
            "topic": "async programming review after practice",
        },
        config={"configurable": {"thread_id": "tutor-demo-1"}},
    )
    print(f"Response:\n{_last_assistant_response(result2)}")

    print("\n[Demo 5/6] Swarm Agent — Supervisor + Send API Parallel Fan-out")
    print("-" * 40)
    swarm_app = await build_swarm_graph_with_checkpointer()
    result = await swarm_app.ainvoke(
        {"workflow_id": "wf-001"},
        config={"configurable": {"thread_id": "swarm-demo-1"}},
    )
    print("Worker Reports (parallel fan-out):")
    for r in result.get("reports", []):
        print(f"  {r}")
    print(f"Final Blackboard:\n{result.get('blackboard_state', 'N/A')}")

    print("\n[Demo 6/6] Companion Agent — Relational Context + Write-Back")
    print("-" * 40)
    companion_app = await build_companion_graph_with_checkpointer()
    result = await companion_app.ainvoke(
        {
            "user_id": "usr_anthony",
            "user_message": "How are things looking for my apartment setup?",
        },
        config={"configurable": {"thread_id": "comp-demo-1"}},
    )
    print(f"Retrieved Context:\n{result.get('retrieved_context', 'N/A')}")
    print(f"Response:\n{_last_assistant_response(result)}")

    print("\n[Demo 6b] Companion — Multi-turn: remembers previous turn")
    print("-" * 40)
    result2 = await companion_app.ainvoke(
        {
            "user_id": "usr_anthony",
            "user_message": "Actually, I found a place in Williamsburg. Can you remember that?",
        },
        config={"configurable": {"thread_id": "comp-demo-1"}},
    )
    print(f"Response:\n{_last_assistant_response(result2)}")
    print(f"Extracted Memories:\n{result2.get('extracted_facts', 'N/A')}")

    print("\n[Demo 6c] Companion — Conflict Resolution: moved cities")
    print("-" * 40)
    result3 = await companion_app.ainvoke(
        {
            "user_id": "usr_anthony",
            "user_message": "I decided not to move to Brooklyn after all — I'm staying in Manhattan.",
        },
        config={"configurable": {"thread_id": "comp-demo-1"}},
    )
    print(f"Response:\n{_last_assistant_response(result3)}")
    print(f"Extracted Memories:\n{result3.get('extracted_facts', 'N/A')}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        ctx = await client.get(
            f"{MEMORY_ENGINE_URL}/companion/context",
            params={"user_id": "usr_anthony"},
        )
        facts = ctx.json().get("graph_facts", [])
    print(f"Memory State After Conflict Resolution:")
    for f in facts:
        rel = f" {f['relationship_type']} {f['related_to']}" if f.get("related_to") else ""
        print(f"  - {f['name']} ({f['entity_type']}){rel} [salience={f['salience']}]")

    print("\n[Demo 6d] Companion — Right to Forget")
    print("-" * 40)
    result4 = await companion_app.ainvoke(
        {
            "user_id": "usr_anthony",
            "user_message": "Please forget everything you remember about my apartment search.",
        },
        config={"configurable": {"thread_id": "comp-demo-1"}},
    )
    print(f"Response:\n{_last_assistant_response(result4)}")

    print("\n" + "=" * 60)
    print("All 12 demos complete! (6 patterns + multi-turn + supervisor fan-out + companion lifecycle)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demos())
