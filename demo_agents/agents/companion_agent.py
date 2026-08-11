import os
import httpx
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    user_id: str
    user_message: str
    retrieved_context: str
    agent_response: str


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.7,
)


async def retrieve_companion_context(state: AgentState) -> AgentState:
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"{MEMORY_ENGINE_URL}/companion/context",
            params={"user_id": state["user_id"]},
        )
        data = res.json()

    facts = data.get("graph_facts", [])
    ephs = data.get("ephemerals", [])

    facts_str = "; ".join(
        f"{f['name']} ({f['entity_type']})"
        + (
            f" {f['relationship_type']} {f['related_to']}"
            if f.get("related_to")
            else ""
        )
        for f in facts
    )
    ephs_str = "; ".join(e["description"] for e in ephs)

    context_str = f"Active Facts: {facts_str}\nCurrent Mood/Events: {ephs_str}"
    return {**state, "retrieved_context": context_str}


async def generate_response(state: AgentState) -> AgentState:
    prompt = f"""You are an AI companion. Use the persistent memory context to personalize your response.

Memory Context:
{state['retrieved_context']}

User Message: {state['user_message']}
Response:"""

    response = await llm.ainvoke(prompt)
    return {**state, "agent_response": response.content}


def build_companion_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_companion_context)
    builder.add_node("generate", generate_response)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
