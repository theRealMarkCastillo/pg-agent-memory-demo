import os
import httpx
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    workflow_id: str
    agent_name: str
    assigned_task: str
    blackboard_state: str
    agent_response: str


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.4,
)


async def fetch_blackboard(state: AgentState) -> AgentState:
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"{MEMORY_ENGINE_URL}/swarm/tasks/{state['workflow_id']}"
        )
        tasks = res.json()

        claim_res = await client.post(
            f"{MEMORY_ENGINE_URL}/swarm/tasks/claim-next",
            params={
                "agent_name": state["agent_name"],
                "workflow_id": state["workflow_id"],
            },
        )
        claim_data = claim_res.json()

    tasks_str = "\n".join(
        f"Task: {t['task_name']} | Status: {t['status']} | Agent: {t['assigned_agent']}"
        for t in tasks
    )

    if claim_data.get("status") == "claimed":
        assigned_str = str(claim_data["task"])
    else:
        assigned_str = "No pending tasks available"

    return {
        **state,
        "blackboard_state": tasks_str,
        "assigned_task": assigned_str,
    }


async def execute_task(state: AgentState) -> AgentState:
    prompt = f"""You are a swarm agent working on a shared blackboard. Process your assigned task.

Blackboard State:
{state['blackboard_state']}

Your Assigned Task:
{state['assigned_task']}

Execute the task and produce a result. Response:"""

    response = await llm.ainvoke(prompt)
    return {**state, "agent_response": response.content}


def build_swarm_graph():
    builder = StateGraph(AgentState)
    builder.add_node("fetch", fetch_blackboard)
    builder.add_node("execute", execute_task)
    builder.set_entry_point("fetch")
    builder.add_edge("fetch", "execute")
    builder.add_edge("execute", END)
    return builder.compile()
