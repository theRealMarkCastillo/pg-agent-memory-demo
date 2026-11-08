import os
from datetime import datetime, timezone
import operator
from typing import Annotated
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Send
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import SWARM_TOOLS
from .checkpointer import get_checkpointer
from .runtime import (
    AgentContext,
    auth_headers,
    bound_args,
    prompt_messages,
    ScopedGraph,
)

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class SwarmState(MessagesState):
    workflow_id: str
    claim_mode: str  # "assigned" (supervisor pins task IDs) | "pull" (workers race via SKIP LOCKED)
    pending_tasks: list[dict]
    worker_task: dict
    agent_name: str
    reports: Annotated[list[str], operator.add]
    final_summary: str
    blackboard_state: str


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    timeout=60,
    max_retries=2,
    temperature=0.4,
)

WORKER_TOOLS = [
    t for t in SWARM_TOOLS if t.name not in {"claim_task", "claim_next_task"}
]
llm_with_tools = llm.bind_tools(WORKER_TOOLS)
TOOLS_BY_NAME = {t.name: t for t in WORKER_TOOLS}

SPECIALTY_MAP = {
    "analyze_sentiment": "sentiment-bot",
    "extract_entities": "entity-bot",
    "summarize_text": "summary-bot",
}
POOL_WORKER = "general-bot"


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
    """Fan out one Send per pending task; each Send launches a worker in parallel.

    In "pull" mode the supervisor only sizes the worker pool. Workers are not told
    which task to take: each one calls claim_next_task, and FOR UPDATE SKIP LOCKED
    hands every concurrent claimer a different row."""
    pending = state.get("pending_tasks", [])
    if not pending:
        return "aggregate"
    if state.get("claim_mode") == "pull":
        return [
            Send(
                "worker",
                {
                    "workflow_id": state["workflow_id"],
                    "worker_task": {},
                    "agent_name": POOL_WORKER,
                },
            )
            for _ in pending
        ]
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


def _worker_report(agent_name, task_name, report):
    line = f"[{agent_name}] {task_name}: {report}"
    return {"reports": [line], "messages": [AIMessage(content=line)]}


async def worker_node(state: SwarmState):
    """Claim in application code before the model can dispatch any execution."""
    assigned = state.get("worker_task") or {}
    agent_name = state.get("agent_name", "worker-bot")
    workflow_id = state["workflow_id"]
    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers()) as client:
        if assigned.get("task_id"):
            res = await client.post(
                f"{MEMORY_ENGINE_URL}/swarm/tasks/claim",
                json={"task_id": assigned["task_id"], "agent_name": agent_name},
            )
        else:
            res = await client.post(
                f"{MEMORY_ENGINE_URL}/swarm/tasks/claim-next",
                params={"workflow_id": workflow_id, "agent_name": agent_name},
            )
        res.raise_for_status()
        claim = res.json()
        if claim.get("status") != "claimed":
            return _worker_report(
                agent_name,
                assigned.get("task_name", "task"),
                "Not claimed; execution skipped.",
            )
        task = claim["task"]
        task_name = task["task_name"]
        lease = {
            "task_id": task["task_id"],
            "agent_name": agent_name,
            "lease_token": task["lease_token"],
        }
        messages = [
            SystemMessage(
                content=(
                    "You are a specialist worker. The application has already claimed your task.\n"
                    f"Your name: {agent_name}\nWorkflow: {workflow_id}\n"
                    f"Your assigned task: {task_name} (id: {task['task_id']})\n"
                    f"Payload: {task.get('payload', {})}\n"
                    "Execute the task using execute_shell_command or fetch_url as appropriate. "
                    "Then call complete_swarm_task with a concise result summary and end with a report. "
                    "The application supplies task identity and lease credentials at dispatch."
                )
            ),
            HumanMessage(content=f"Process task '{task_name}' now."),
        ]
        completed = False
        for _ in range(8):
            response = await llm_with_tools.ainvoke(prompt_messages(messages))
            messages.append(response)
            if not getattr(response, "tool_calls", None):
                break
            for tc in response.tool_calls:
                tool = TOOLS_BY_NAME.get(tc["name"])
                status = "success"
                if tool is None:
                    result, status = f"Unknown tool '{tc['name']}'.", "error"
                elif completed:
                    result, status = (
                        "Task is already completed; further tools are disabled.",
                        "error",
                    )
                else:
                    # Fail closed on stale leases and transport failures. This is a
                    # pre-dispatch check, not exactly-once execution of remote effects.
                    owned = await client.post(
                        f"{MEMORY_ENGINE_URL}/swarm/tasks/validate-lease", json=lease
                    )
                    if owned.status_code == 409:
                        return _worker_report(
                            agent_name, task_name, "Lease lost; execution stopped."
                        )
                    owned.raise_for_status()
                    try:
                        args = bound_args(
                            dict(state, agent_name=agent_name), tc["args"]
                        )
                        if tc["name"] == "complete_swarm_task":
                            args.update(lease)
                        result = await tool.ainvoke(args)
                        if tc["name"] == "complete_swarm_task":
                            completed = True
                    except Exception as exc:
                        result, status = f"Tool error: {exc}", "error"
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        status=status,
                    )
                )

        report = next(
            (
                m.text
                for m in reversed(messages)
                if m.type == "ai" and not getattr(m, "tool_calls", None)
            ),
            "No final report.",
        )
        if not completed:
            report = "Incomplete (no successful completion recorded). " + str(report)
        return _worker_report(agent_name, task_name, report)


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
    builder = StateGraph(SwarmState, context_schema=AgentContext)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("worker", worker_node)
    builder.add_node("aggregate", aggregate_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", assign_tasks, ["worker", "aggregate"])
    builder.add_edge("worker", "aggregate")
    builder.add_edge("aggregate", END)

    return ScopedGraph(
        builder.compile(checkpointer=checkpointer), "swarm", checkpointer
    )


async def build_swarm_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_swarm_graph(cp)
