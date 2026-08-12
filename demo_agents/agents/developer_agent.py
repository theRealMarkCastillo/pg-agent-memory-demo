import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import DEVELOPER_TOOLS
from .checkpointer import get_checkpointer


class AgentState(TypedDict):
    project_id: str
    git_branch: str
    query: str
    retrieved_symbols: str
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(DEVELOPER_TOOLS)


async def search_code_symbols(state: AgentState):
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{os.getenv('MEMORY_ENGINE_URL', 'http://memory-engine:8000')}/developer/symbols/search",
            json={
                "project_id": state["project_id"],
                "git_branch": state["git_branch"],
                "query": state["query"],
            },
        )
        data = res.json()

    symbols_str = "\n".join(
        f"{s['symbol_name']} ({s['symbol_type']}) in {s['file_path']}: {s['signature']}\n  {s.get('code_content', '')}"
        for s in data
    )

    system_content = (
        "You are a developer assistant with access to workspace memory and a real sandbox.\n"
        "Use search_code_symbols to find relevant code, and store_code_symbol to persist new symbols.\n"
        "Use read_file and write_file for file operations in the workspace.\n"
        "Use execute_shell_command to run shell commands, scripts, and tests.\n"
        f"Current context — Project: {state['project_id']}, Branch: {state['git_branch']}\n\n"
        f"Relevant Code Symbols:\n{symbols_str}"
    )

    existing = state.get("messages", [])
    if existing:
        messages = list(existing)
        if getattr(messages[0], "type", "") == "system":
            messages[0] = SystemMessage(content=system_content)
        else:
            messages.insert(0, SystemMessage(content=system_content))
        messages.append(HumanMessage(content=state["query"]))
    else:
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=state["query"]),
        ]

    return {
        "retrieved_symbols": symbols_str,
        "messages": messages,
    }


async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_developer_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("search", search_code_symbols)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(DEVELOPER_TOOLS))

    builder.set_entry_point("search")
    builder.add_edge("search", "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


async def build_developer_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_developer_graph(cp)
