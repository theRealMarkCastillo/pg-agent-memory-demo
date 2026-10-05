import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import ENTERPRISE_TOOLS
from .checkpointer import get_checkpointer
from .runtime import (
    auth_headers,
    refresh_messages,
    prompt_messages,
    ScopedToolNode,
    ScopedGraph,
)


class AgentState(TypedDict):
    user_role: str
    query: str
    retrieved_docs: str
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(ENTERPRISE_TOOLS)


async def search_policy_docs(state: AgentState):
    import httpx

    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers()) as client:
        res = await client.post(
            f"{os.getenv('MEMORY_ENGINE_URL', 'http://memory-engine:8000')}/enterprise/documents/search",
            json={
                "query": state["query"],
                "user_role": state["user_role"],
            },
        )
        res.raise_for_status()
        data = res.json()

    docs_str = "\n---\n".join(
        f"Title: {d['doc_title']}\nContent: {d['content']}" for d in data
    )

    system_content = (
        "You are an enterprise knowledge agent with role-gated access to policy documents.\n"
        "Use search_policy_documents to find relevant policies for the user's role.\n"
        "Use store_policy_document to persist new policy documents when authorized.\n"
        f"Current user role: {state['user_role']}\n\n"
        f"Accessible Policy Documents:\n{docs_str}"
    )

    messages = refresh_messages(
        system_content, state.get("messages", []), state["query"]
    )

    return {
        "retrieved_docs": docs_str,
        "messages": messages,
    }


async def agent_node(state: AgentState):
    if not state.get("retrieved_docs"):
        return {
            "messages": [
                AIMessage(
                    content="No current authorized policy matches this query. Please clarify the policy you need."
                )
            ]
        }
    response = await llm_with_tools.ainvoke(prompt_messages(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_enterprise_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("search", search_policy_docs)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ScopedToolNode(ENTERPRISE_TOOLS))

    builder.set_entry_point("search")
    builder.add_edge("search", "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")

    return ScopedGraph(
        builder.compile(checkpointer=checkpointer), "enterprise", checkpointer
    )


async def build_enterprise_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_enterprise_graph(cp)
