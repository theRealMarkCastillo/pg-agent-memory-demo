import os
import httpx
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    project_id: str
    git_branch: str
    query: str
    retrieved_symbols: str
    agent_response: str


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.3,
)


async def search_code_symbols(state: AgentState) -> AgentState:
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{MEMORY_ENGINE_URL}/developer/symbols/search",
            json={
                "project_id": state["project_id"],
                "git_branch": state["git_branch"],
                "query": state["query"],
            },
        )
        data = res.json()

    symbols_str = "\n".join(
        f"{s['symbol_name']} ({s['symbol_type']}) in {s['file_path']}: {s['signature']}"
        for s in data
    )
    return {**state, "retrieved_symbols": symbols_str}


async def generate_response(state: AgentState) -> AgentState:
    prompt = f"""You are a developer assistant. Use the codebase symbols below to help the user.

Relevant Symbols:
{state['retrieved_symbols']}

Query: {state['query']}
Response:"""

    response = await llm.ainvoke(prompt)
    return {**state, "agent_response": response.content}


def build_developer_graph():
    builder = StateGraph(AgentState)
    builder.add_node("search", search_code_symbols)
    builder.add_node("generate", generate_response)
    builder.set_entry_point("search")
    builder.add_edge("search", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
