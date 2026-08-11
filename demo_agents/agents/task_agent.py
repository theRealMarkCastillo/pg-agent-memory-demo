import os
import httpx
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    agent_id: str
    goal: str
    past_trajectories: str
    plan: str
    execution_result: str
    success_score: float


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.4,
)


async def recall_past_trajectories(state: AgentState) -> AgentState:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{MEMORY_ENGINE_URL}/task/trajectories/search",
            json={
                "goal_description": state["goal"],
                "min_success_score": 0.7,
            },
        )
        data = res.json()

    parts = []
    for t in data:
        parts.append(
            f"Goal: {t['goal_description']}\n"
            f"Actions: {t['action_sequence']}\n"
            f"Result: {t['execution_result']} (score: {t['success_score']})"
        )
    return {**state, "past_trajectories": "\n---\n".join(parts)}


async def plan_and_execute(state: AgentState) -> AgentState:
    prompt = f"""You are an autonomous task agent. Based on past successful trajectories, plan and execute.

Past Successful Trajectories:
{state['past_trajectories']}

Goal: {state['goal']}

Output a JSON plan with steps and a simulated result. Reply with:
Plan: <your step-by-step plan>
Result: <simulated execution result>
Success Score: <0.0 to 1.0>"""

    response = await llm.ainvoke(prompt)
    content = response.content

    score = 0.9
    return {
        **state,
        "plan": content,
        "execution_result": content,
        "success_score": score,
    }


def build_task_graph():
    builder = StateGraph(AgentState)
    builder.add_node("recall", recall_past_trajectories)
    builder.add_node("execute", plan_and_execute)
    builder.set_entry_point("recall")
    builder.add_edge("recall", "execute")
    builder.add_edge("execute", END)
    return builder.compile()
