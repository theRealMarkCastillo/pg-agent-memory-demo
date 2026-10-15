import os
from datetime import datetime, timezone
import operator
from typing import TypedDict, Annotated
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import Send
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import SWARM_TOOLS
from .checkpointer import get_checkpointer
from .runtime import auth_headers, bound_args, prompt_messages, ScopedGraph

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class SwarmState(TypedDict):
    workflow_id: str
    pending_tasks: list[dict]
    worker_task: dict
    agent_name: str
    reports: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]
    final_summary: str
    blackboard_state: str


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.4,
)

llm_with_tools = llm.bind_tools(SWARM_TOOLS)
TOOLS_BY_NAME = {t.name: t for t in SWARM_TOOLS}

SPECIALTY_MAP = {
    "analyze_sentiment": "sentiment-bot",
    "extract_entities": "entity-bot",
    "summarize_text": "summary-bot",
}


async def supervisor_node(state: SwarmState):
    """Read the shared blackboard and collect every PENDING task for fan-out."""
    workflow_id = state["workflow_id"]
    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers()) as client:
        res = await client.get(f"{MEMORY_ENGINE_URL}/swarm/tasks/{workflow_id}")
        res.raise_for_status()
        tasks = res.json()

    now = datetime.now(timezone.utc)
    pending = [
        t
        for t in tasks
        if t.get("status") == "PENDING"
        or (
            t.get("status") == "IN_PROGRESS"
            and t.get("lease_until")
            and datetime.fromisoformat(t["lease_until"]) < now
        )
    ]
    return {"pending_tasks": pending}


def assign_tasks(state: SwarmState):
    """Fan out one Send per pending task; each Send launches a worker in parallel."""
    pending = state.get("pending_tasks", [])
    if not pending:
        return "aggregate"
    return [
        Send(
            "worker",
            {
                "workflow_id": state["workflow_id"],
                "worker_task": t,
                "agent_name": SPECIALTY_MAP.get(t.get("task_name"), "general-bot"),
            },
        )
        for t in pending
    ]


async def worker_node(state: SwarmState):
    """A specialist worker: claim its assigned task, execute it, and report back."""
    task = state.get("worker_task") or {}
    task_id = task.get("task_id")
    task_name = task.get("task_name", "next_available_task")
    agent_name = state.get("agent_name", "worker-bot")
    workflow_id = state["workflow_id"]

    if task_id:
        assignment_block = (
            f"Your assigned task: {task_name} (id: {task_id})\n"
            f"Payload: {task.get('payload', {})}\n\n"
            "Do the following, in order:\n"
            "1. Claim YOUR task with claim_task, using the exact task_id above.\n"
        )
    else:
        assignment_block = (
            "No specific task_id was pre-assigned.\n\n"
            "Do the following, in order:\n"
            "1. Claim the next pending task in the workflow using claim_next_task.\n"
        )

    system_content = (
        "You are a specialist worker in a multi-agent swarm. A supervisor has fanned "
        "work out to you in parallel, and other workers are handling their own tasks.\n"
        f"Your name: {agent_name}\n"
        f"Workflow: {workflow_id}\n"
        f"{assignment_block}"
        "2. Execute the task using execute_shell_command or fetch_url as appropriate.\n"
        "3. Mark it complete with complete_swarm_task and a concise result summary.\n"
        "4. End with a one-line report of what you did.\n"
        "Do not claim tasks assigned to other workers."
    )
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Process task '{task_name}' now."),
    ]

    for _ in range(8):
        response = await llm_with_tools.ainvoke(prompt_messages(messages))
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            break
        for tc in response.tool_calls:
            tool = TOOLS_BY_NAME.get(tc["name"])
            if tool is None:
                result = f"Unknown tool '{tc['name']}'."
            else:
                try:
                    args = bound_args(state, tc["args"])
                    if tc["name"] == "claim_next_task":
                        if task.get("task_id"):
                            raise ValueError("Workers must claim their assigned task")
                        args["workflow_id"] = workflow_id
                        args["agent_name"] = agent_name
                    if tc["name"] in ("claim_task", "complete_swarm_task") and task.get(
                        "task_id"
                    ):
                        args["task_id"] = task["task_id"]
                    result = await tool.ainvoke(args)
                except Exception as e:
                    result = f"Tool error: {e}"
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )

    report = ""
    for m in reversed(messages):
        if getattr(m, "tool_calls", None):
            continue
        if getattr(m, "type", "") == "tool":
            continue
        if getattr(m, "content", None):
            report = m.content
            break

    summary_line = f"[{agent_name}] {task.get('task_name', 'task')}: {report}"
    return {
        "reports": [summary_line],
        "messages": [AIMessage(content=summary_line)],
    }


async def aggregate_node(state: SwarmState):
    """Collect worker reports and snapshot the final blackboard state."""
    workflow_id = state["workflow_id"]
    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers()) as client:
        res = await client.get(f"{MEMORY_ENGINE_URL}/swarm/tasks/{workflow_id}")
        res.raise_for_status()
        tasks = res.json()

    board = "\n".join(
        f"Task: {t['task_name']} | Status: {t['status']} | Agent: {t.get('assigned_agent', 'unassigned')}"
        for t in tasks
    )
    reports = state.get("reports", [])
    summary = (
        "\n".join(f"- {r}" for r in reports) if reports else "(no tasks to process)"
    )
    return {"final_summary": summary, "blackboard_state": board}


def build_swarm_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(SwarmState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("worker", worker_node)
    builder.add_node("aggregate", aggregate_node)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", assign_tasks, ["worker", "aggregate"])
    builder.add_edge("worker", "aggregate")
    builder.add_edge("aggregate", END)

    return ScopedGraph(
        builder.compile(checkpointer=checkpointer), "swarm", checkpointer
    )


async def build_swarm_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_swarm_graph(cp)
