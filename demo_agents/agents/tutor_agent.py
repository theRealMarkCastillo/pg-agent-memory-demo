import os
import httpx
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    user_id: str
    topic: str
    skill_gaps: str
    recommended_skill: str
    agent_response: str


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.5,
)


async def assess_skill_gaps(state: AgentState) -> AgentState:
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"{MEMORY_ENGINE_URL}/tutor/gaps/{state['user_id']}"
        )
        data = res.json()

    gaps_str = "\n".join(
        f"{s['skill_name']}: score={s['decayed_score']:.2f} [{s['status']}]"
        for s in data
    )
    return {**state, "skill_gaps": gaps_str}


async def recommend_and_respond(state: AgentState) -> AgentState:
    prompt = f"""You are an adaptive tutor. Based on the learner's skill gaps and decayed proficiency, recommend next steps.

Skill Gaps (with forgetting-curve decay):
{state['skill_gaps']}

Topic Request: {state['topic']}

Recommend the single most important skill to review and provide a brief lesson on it.
Response:"""

    response = await llm.ainvoke(prompt)
    return {**state, "agent_response": response.content}


def build_tutor_graph():
    builder = StateGraph(AgentState)
    builder.add_node("assess", assess_skill_gaps)
    builder.add_node("recommend", recommend_and_respond)
    builder.set_entry_point("assess")
    builder.add_edge("assess", "recommend")
    builder.add_edge("recommend", END)
    return builder.compile()
