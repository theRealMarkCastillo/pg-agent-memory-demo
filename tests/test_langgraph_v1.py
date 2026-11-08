"""Exercise the v1 runtime contracts with real graphs and scripted model output."""

import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_agents"))

from agents.runtime import AgentContext, ScopedGraph, ScopedToolNode


def test_v1_content_blocks_are_read_as_text(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-no-external-calls")
    monkeypatch.setenv("LLM_MODEL_NAME", "test")
    from agents.companion_agent import _last_assistant_content
    import main as demo_main

    message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "internal reasoning"},
            {"type": "text", "text": "The visible answer."},
        ]
    )
    assert _last_assistant_content([message]) == "The visible answer."
    assert (
        demo_main._last_assistant_response({"messages": [message]})
        == "The visible answer."
    )


@pytest.mark.parametrize(
    "module,builder,inputs,tool_name,tool_args",
    [
        (
            "developer_agent",
            "build_developer_graph",
            {"project_id": "v1-project", "git_branch": "main", "query": "embed"},
            "search_code_symbols",
            {"project_id": "forged", "git_branch": "forged", "query": "embed"},
        ),
        (
            "task_agent",
            "build_task_graph",
            {"agent_id": "v1-bot", "goal": "recall"},
            "search_trajectories",
            {"agent_id": "forged", "goal_description": "recall"},
        ),
        (
            "tutor_agent",
            "build_tutor_graph",
            {"user_id": "v1-learner", "topic": "python"},
            "get_skill_gaps",
            {"user_id": "forged"},
        ),
    ],
)
async def test_graph_tool_cycle_on_v1(
    client, monkeypatch, module, builder, inputs, tool_name, tool_args
):
    monkeypatch.setenv("LLM_API_KEY", "test-no-external-calls")
    monkeypatch.setenv("LLM_MODEL_NAME", "test")
    agent = importlib.import_module("agents." + module)

    class Model:
        async def ainvoke(self, messages):
            results = [m for m in messages if m.type == "tool"]
            if not results:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "call", "name": tool_name, "args": tool_args}],
                )
            assert results[-1].status == "success"
            return AIMessage(content="done")

    monkeypatch.setattr(agent, "llm_with_tools", Model())
    result = await getattr(agent, builder)().ainvoke(inputs)
    assert result["messages"][-1].text == "done"


async def test_runtime_scope_survives_state_mutation_and_concurrent_runs():
    @tool
    async def scoped_user(user_id: str) -> str:
        """Return the bound user for the test."""
        await asyncio.sleep(0)
        return user_id

    class State(MessagesState):
        user_id: str

    async def forge_state(state):
        return {
            "user_id": "forged",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "scope",
                            "name": "scoped_user",
                            "args": {"user_id": "model-forged"},
                        }
                    ],
                )
            ],
        }

    builder = StateGraph(State, context_schema=AgentContext)
    builder.add_node("forge", forge_state)
    builder.add_node("tools", ScopedToolNode([scoped_user]))
    builder.add_edge(START, "forge")
    builder.add_edge("forge", "tools")
    builder.add_edge("tools", END)
    graph = ScopedGraph(builder.compile(), "test")
    results = await asyncio.gather(
        *(graph.ainvoke({"user_id": user}) for user in ("alice", "bob"))
    )
    assert [r["messages"][-1].content for r in results] == ["alice", "bob"]
    with pytest.raises(ValueError, match="constructs context"):
        await graph.ainvoke({"user_id": "alice"}, context=AgentContext(()))


async def test_loop_budget_is_explicit_and_caller_can_lower_it():
    calls = []

    def loop(state):
        calls.append(1)
        return {}

    builder = StateGraph(MessagesState, context_schema=AgentContext)
    builder.add_node("loop", loop)
    builder.add_edge(START, "loop")
    builder.add_edge("loop", "loop")
    graph = ScopedGraph(builder.compile(), "test")
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke({})
    assert len(calls) == 32
    calls.clear()
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke({}, {"recursion_limit": 3})
    assert len(calls) == 3


async def test_tool_errors_return_matching_tool_message():
    @tool
    async def fail() -> str:
        """Fail to exercise the explicit v1 error policy."""
        raise ValueError("test failure")

    builder = StateGraph(MessagesState, context_schema=AgentContext)
    builder.add_node("tools", ScopedToolNode([fail]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = ScopedGraph(builder.compile(), "test")
    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "failure", "name": "fail", "args": {}}],
                )
            ]
        }
    )
    message = result["messages"][-1]
    assert message.status == "error" and message.tool_call_id == "failure"


async def test_postgres_pending_fanout_resumes_with_upstream_serializer(
    client, monkeypatch
):
    from agents import checkpointer as cp
    from langgraph.types import Send

    monkeypatch.setattr(cp, "_checkpointer_instance", None)
    monkeypatch.setattr(cp, "_setup_lock", asyncio.Lock())
    saver = await cp.get_checkpointer()
    try:
        import operator
        from typing import Annotated, TypedDict

        class State(TypedDict):
            reports: Annotated[list[str], operator.add]

        builder = StateGraph(State)
        builder.add_node("supervisor", lambda state: {})
        builder.add_node("worker", lambda state: {"reports": [state["task"]]})
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            lambda state: [
                Send("worker", {"task": "one"}),
                Send("worker", {"task": "two"}),
            ],
            ["worker"],
        )
        builder.add_edge("worker", END)
        config = {"configurable": {"thread_id": "swarm:v1-pending-fanout"}}
        graph = builder.compile(checkpointer=saver, interrupt_after=["supervisor"])
        await graph.ainvoke({"reports": []}, config, durability="sync")
        # Rebuild the graph to force a database read, rather than continuing an
        # in-process task. Both pending Send payloads must survive serialization.
        resumed = builder.compile(checkpointer=saver)
        snapshot = await resumed.aget_state(config)
        assert len(snapshot.tasks) == 2
        result = await resumed.ainvoke(None, config, durability="sync")
        assert sorted(result["reports"]) == ["one", "two"]
    finally:
        await saver.adelete_thread("swarm:v1-pending-fanout")
        await cp.close_checkpointer()
    assert (
        cp._checkpointer_instance
        is cp._pool_instance
        is cp._guard_pool_instance
        is None
    )
