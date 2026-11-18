"""Trusted scope binding and bounded prompt construction for agent graphs."""

import hashlib
import os
from contextvars import ContextVar
from dataclasses import dataclass
import httpx

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

_memory_generation = ContextVar("memory_generation", default=None)

SCOPE_KEYS = (
    "user_id",
    "project_id",
    "git_branch",
    "user_role",
    "workflow_id",
    "agent_id",
    "agent_name",
)
DEFAULT_RECURSION_LIMIT = 32


@dataclass(frozen=True)
class AgentContext:
    """Application-owned scope for one run, separate from checkpointed state.

    Scope still appears in state for retrieval and backwards-compatible callers,
    but tool dispatch uses this immutable snapshot rather than mutable state.
    The caller must authenticate/authorize these inputs at the application edge.
    """

    scope: tuple[tuple[str, str], ...]

    @classmethod
    def from_input(cls, state):
        return cls(tuple((key, state[key]) for key in SCOPE_KEYS if key in state))


def auth_headers():
    token = os.environ.get("MEMORY_API_TOKEN", "")
    if not token:
        raise RuntimeError("MEMORY_API_TOKEN is required")
    headers = {"Authorization": f"Bearer {token}"}
    generation = _memory_generation.get()
    if generation is not None:
        headers["X-Memory-Generation"] = str(generation)
    return headers


def refresh_messages(system_content, existing, user_text):
    # IDs are stable so add_messages replaces the prior system prompt.
    old = next((m for m in existing if m.type == "system"), None)
    return [
        SystemMessage(
            content=system_content[:20000], id=old.id if old else "memory-system"
        ),
        HumanMessage(content=user_text),
    ]


def prompt_messages(messages, max_chars=32000):
    """Select complete conversational turns; never truncate a tool-call/result pair."""
    systems = [m for m in messages if m.type == "system"]
    turns = []
    for m in messages:
        if m.type == "system":
            continue
        if m.type == "human" or not turns:
            turns.append([])
        turns[-1].append(m)
    selected = []
    budget = max_chars - sum(len(str(m.content)) for m in systems[-1:])
    for turn in reversed(turns):
        size = sum(
            len(str(m.content)) + len(str(getattr(m, "tool_calls", []))) for m in turn
        )
        if size > budget:
            if not selected:
                raise ValueError("Current turn exceeds the context budget")
            break
        selected = turn + selected
        budget -= size
    return systems[-1:] + selected


def bound_args(state, args):
    result = dict(args)
    for key in SCOPE_KEYS:
        if key in state and key in result:
            result[key] = state[key]
    if "allowed_role" in result and "user_role" in state:
        result["allowed_role"] = state["user_role"]
    return result


class ScopedToolNode:
    def __init__(self, tools):
        # v1 defaults to catching argument errors only. Preserve the demo's
        # recoverable tool-error messages; cancellation/interrupts still propagate.
        self.node = ToolNode(tools, handle_tool_errors=Exception)

    async def __call__(
        self, state, config: RunnableConfig, runtime: Runtime[AgentContext]
    ):
        last = state["messages"][-1]
        scope = dict(runtime.context.scope)
        calls = [dict(tc, args=bound_args(scope, tc["args"])) for tc in last.tool_calls]
        for tc in calls:
            if tc["name"] == "search_trajectories" and "agent_id" in scope:
                tc["args"]["agent_id"] = scope["agent_id"]
        # Deletion is exclusive within a tool batch. Never dispatch sibling writes.
        forget = next(
            (c for c in calls if c["name"] == "forget_companion_memory"), None
        )
        if forget:
            response = await self.node.ainvoke(
                dict(
                    state, messages=[last.model_copy(update={"tool_calls": [forget]})]
                ),
                config,
                runtime=runtime,
            )
            result = response["messages"][0]
            failed = getattr(result, "status", None) == "error" or not str(
                result.content
            ).startswith("Memory forgotten")
            if failed:
                return {
                    "messages": response["messages"]
                    + [
                        ToolMessage(
                            content="Cancelled because deletion failed",
                            tool_call_id=c["id"],
                        )
                        for c in calls
                        if c is not forget
                    ]
                }
            return {
                "forgotten": True,
                "messages": response["messages"]
                + [
                    ToolMessage(
                        content="Cancelled by forget request", tool_call_id=c["id"]
                    )
                    for c in calls
                    if c is not forget
                ]
                + [
                    AIMessage(
                        content="Your stored memory has been deleted. Memory recording is disabled until you explicitly resume it."
                    )
                ],
            }
        return await self.node.ainvoke(
            dict(state, messages=[last.model_copy(update={"tool_calls": calls})]),
            config,
            runtime=runtime,
        )


class ScopedGraph:
    """Namespace checkpoints using application-owned input scope, never tool arguments."""

    def __init__(self, graph, domain, saver=None):
        self.graph, self.domain, self.saver = graph, domain, saver

    def get_graph(self, **kwargs):
        return self.graph.get_graph(**kwargs)

    async def ainvoke(self, state, config=None, **kwargs):
        if "context" in kwargs:
            raise ValueError(
                "ScopedGraph constructs context from application-owned input"
            )
        context = AgentContext.from_input(state)
        config = dict(config or {})
        # LangGraph v1's default is 1000 steps. Bound autonomous tool loops here.
        config.setdefault("recursion_limit", DEFAULT_RECURSION_LIMIT)
        config.setdefault("max_concurrency", 5)
        settings = dict(config.get("configurable", {}))
        if self.domain == "companion":
            identity = state["user_id"]
            settings["memory_user_id"] = identity
            from urllib.parse import quote

            async with httpx.AsyncClient(headers=auth_headers(), timeout=30) as client:
                response = await client.get(
                    os.environ.get("MEMORY_ENGINE_URL", "http://memory-engine:8000")
                    + "/companion/memory/"
                    + quote(identity, safe="")
                    + "/state"
                )
                response.raise_for_status()
                settings["memory_generation"] = response.json()["generation"]
        else:
            import json

            identity = json.dumps(
                {
                    k: state[k]
                    for k in (
                        "user_id",
                        "project_id",
                        "git_branch",
                        "user_role",
                        "workflow_id",
                        "agent_id",
                    )
                    if k in state
                },
                sort_keys=True,
            )
        thread = settings.get("thread_id", "default")
        settings["thread_id"] = (
            f"{self.domain}:{hashlib.sha256(identity.encode()).hexdigest()}:{thread}"
        )
        config["configurable"] = settings
        token = _memory_generation.set(settings.get("memory_generation"))
        try:
            # Persist each superstep before dispatching the next. This does not
            # make external tool effects exactly-once on checkpoint replay.
            if self.saver is not None:
                kwargs.setdefault("durability", "sync")
            result = await self.graph.ainvoke(state, config, context=context, **kwargs)
        finally:
            _memory_generation.reset(token)
        if result.get("forgotten"):
            if self.saver:
                await self.saver.adelete_thread(settings["thread_id"])
            return {
                "user_id": identity,
                "forgotten": True,
                "messages": [
                    AIMessage(
                        content="Your stored memory has been deleted. Memory recording is disabled until you explicitly resume it."
                    )
                ],
            }
        return result
