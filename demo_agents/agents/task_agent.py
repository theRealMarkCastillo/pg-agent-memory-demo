import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import TASK_TOOLS
from .checkpointer import get_checkpointer


class AgentState(TypedDict):
    agent_id: str
    goal: str
    past_trajectories: str
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.4,
)

llm_with_tools = llm.bind_tools(TASK_TOOLS)


async def recall_past_trajectories(state: AgentState):
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{os.getenv('MEMORY_ENGINE_URL', 'http://memory-engine:8000')}/task/trajectories/search",
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
    trajectories_str = "\n---\n".join(parts)

    system_content = (
        "You are an autonomous task agent with real execution capabilities.\n"
        "Use search_trajectories to recall similar past tasks, then plan and execute.\n"
        "Use execute_shell_command to run scripts and data processing.\n"
        "Use fetch_url to make HTTP requests to APIs and websites.\n"
        "Use read_file and write_file for data file operations.\n"
        "After completing a task, use store_trajectory to record it in memory for future recall.\n"
        f"Your agent ID: {state['agent_id']}\n\n"
        f"Past Successful Trajectories:\n{trajectories_str}"
    )

    existing = state.get("messages", [])
    if existing:
        messages = list(existing)
        if getattr(messages[0], "type", "") == "system":
            messages[0] = SystemMessage(content=system_content)
        else:
            messages.insert(0, SystemMessage(content=system_content))
        messages.append(HumanMessage(content=f"Goal: {state['goal']}\nPlan, execute, and store the result."))
    else:
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Goal: {state['goal']}\nPlan, execute, and store the result."),
        ]

    return {
        "past_trajectories": trajectories_str,
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


def build_task_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("recall", recall_past_trajectories)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TASK_TOOLS))

    builder.set_entry_point("recall")
    builder.add_edge("recall", "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


async def build_task_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_task_graph(cp)
