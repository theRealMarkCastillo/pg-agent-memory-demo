import os
import httpx
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    user_role: str
    query: str
    retrieved_docs: str
    agent_response: str


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.3,
)


async def search_policy_docs(state: AgentState) -> AgentState:
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{MEMORY_ENGINE_URL}/enterprise/documents/search",
            json={
                "query": state["query"],
                "user_role": state["user_role"],
            },
        )
        data = res.json()

    docs_str = "\n---\n".join(
        f"Title: {d['doc_title']}\nContent: {d['content']}" for d in data
    )
    return {**state, "retrieved_docs": docs_str}


async def generate_response(state: AgentState) -> AgentState:
    prompt = f"""You are an enterprise knowledge agent. Answer based on policy documents accessible to the user.

Relevant Documents (access-filtered by role):
{state['retrieved_docs']}

User Query: {state['query']}
Response:"""

    response = await llm.ainvoke(prompt)
    return {**state, "agent_response": response.content}


def build_enterprise_graph():
    builder = StateGraph(AgentState)
    builder.add_node("search", search_policy_docs)
    builder.add_node("generate", generate_response)
    builder.set_entry_point("search")
    builder.add_edge("search", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
