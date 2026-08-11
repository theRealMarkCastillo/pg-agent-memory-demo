import asyncio
import os
import httpx
from agents.developer_agent import build_developer_graph
from agents.task_agent import build_task_graph
from agents.enterprise_agent import build_enterprise_graph
from agents.tutor_agent import build_tutor_graph
from agents.swarm_agent import build_swarm_graph
from agents.companion_agent import build_companion_graph

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


async def seed_data():
    async with httpx.AsyncClient() as client:
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

    print("Seed data inserted.\n")


async def run_demos():
    print("=" * 60)
    print("Agent Memory Patterns Demonstration")
    print("=" * 60)

    await seed_data()

    print("\n[Demo 1/6] Developer Agent — Code Symbol Search")
    print("-" * 40)
    dev_app = build_developer_graph()
    result = await dev_app.ainvoke({
        "project_id": "demo-project",
        "git_branch": "main",
        "query": "How do I embed text for semantic search?",
    })
    print(f"Symbols Found:\n{result['retrieved_symbols']}")
    print(f"Response:\n{result['agent_response']}")

    print("\n[Demo 2/6] Task Agent — Trajectory Recall")
    print("-" * 40)
    task_app = build_task_graph()
    result = await task_app.ainvoke({
        "agent_id": "task-bot-2",
        "goal": "Scrape product data from a website and save as CSV",
    })
    print(f"Past Trajectories:\n{result['past_trajectories']}")
    print(f"Plan/Result:\n{result['plan']}")

    print("\n[Demo 3/6] Enterprise Agent — Role-Filtered Policy Search")
    print("-" * 40)
    ent_app = build_enterprise_graph()
    result = await ent_app.ainvoke({
        "user_role": "employee",
        "query": "Can I access customer data?",
    })
    print(f"Docs Found:\n{result['retrieved_docs']}")
    print(f"Response:\n{result['agent_response']}")

    print("\n[Demo 4/6] Tutor Agent — Skill Gap Assessment")
    print("-" * 40)
    tutor_app = build_tutor_graph()
    result = await tutor_app.ainvoke({
        "user_id": "learner_001",
        "topic": "async programming",
    })
    print(f"Skill Gaps:\n{result['skill_gaps']}")
    print(f"Response:\n{result['agent_response']}")

    print("\n[Demo 5/6] Swarm Agent — Blackboard Task Claiming")
    print("-" * 40)
    swarm_app = build_swarm_graph()
    result = await swarm_app.ainvoke({
        "workflow_id": "wf-001",
        "agent_name": "sentiment-bot",
    })
    print(f"Blackboard:\n{result['blackboard_state']}")
    print(f"Assigned:\n{result['assigned_task']}")
    print(f"Response:\n{result['agent_response']}")

    print("\n[Demo 6/6] Companion Agent — Relational Context Recall")
    print("-" * 40)
    companion_app = build_companion_graph()
    result = await companion_app.ainvoke({
        "user_id": "usr_anthony",
        "user_message": "How are things looking for my apartment setup?",
    })
    print(f"Retrieved Context:\n{result['retrieved_context']}")
    print(f"Response:\n{result['agent_response']}")

    print("\n" + "=" * 60)
    print("All 6 demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demos())
