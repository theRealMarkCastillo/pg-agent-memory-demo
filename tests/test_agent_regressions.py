import asyncio
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages
from conftest import post, get

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_agents"))


@pytest.fixture
def agents(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-no-external-calls")
    monkeypatch.setenv("LLM_MODEL_NAME", "test")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1")
    return SimpleNamespace(
        enterprise=importlib.import_module("agents.enterprise_agent"),
        companion=importlib.import_module("agents.companion_agent"),
        tools=importlib.import_module("agents.tools"),
        runtime=importlib.import_module("agents.runtime"),
    )


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    async def ainvoke(self, messages):
        self.inputs.append(messages)
        return next(self.outputs)


async def test_scope_bound_through_actual_tool_node(client, agents, monkeypatch):
    for role in ("employee", "admin"):
        await post(
            client,
            "/enterprise/documents",
            doc_title=role,
            allowed_role=role,
            content=role + " access policy",
        )
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call",
                        "name": "search_policy_documents",
                        "args": {"query": "access policy", "user_role": "admin"},
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr(agents.enterprise, "llm_with_tools", model)
    await agents.enterprise.build_enterprise_graph().ainvoke(
        {"query": "access policy", "user_role": "employee"}
    )
    tools = [m.content for m in model.inputs[-1] if m.type == "tool"]
    assert tools and "employee" in tools[0] and "admin" not in tools[0]


async def test_forget_suppresses_extraction_and_purges_checkpoints(
    client, agents, monkeypatch
):
    from agents import checkpointer as cp

    monkeypatch.setattr(cp, "_checkpointer_instance", None)
    monkeypatch.setattr(cp, "_setup_lock", asyncio.Lock())
    saver = await cp.get_checkpointer()
    try:
        model = ScriptedModel(
            [
                AIMessage(content="remembered"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "forget",
                            "name": "forget_companion_memory",
                            "args": {"user_id": "other-user"},
                        }
                    ],
                ),
            ]
        )
        extraction = ScriptedModel([AIMessage(content="{}")])
        monkeypatch.setattr(agents.companion, "llm_with_tools", model)
        monkeypatch.setattr(agents.companion, "extraction_llm", extraction)
        graph = agents.companion.build_companion_graph(saver)
        config = {"configurable": {"thread_id": "same-thread"}}
        await graph.ainvoke(
            {"user_id": "alice", "user_message": "private diagnosis"}, config
        )
        result = await graph.ainvoke(
            {"user_id": "alice", "user_message": "Please forget private diagnosis"},
            config,
        )
        assert result["forgotten"] and "private diagnosis" not in str(result)
        assert len(extraction.inputs) == 1
        assert not (await get(client, "/companion/context", user_id="alice"))[
            "graph_facts"
        ]
        from test_review_regressions import connect

        conn = await connect()
        try:
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM companion_episodes WHERE user_id='alice'"
                )
                == 0
            )
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                assert (
                    await conn.fetchval(
                        f"SELECT count(*) FROM {table} WHERE thread_id LIKE 'companion:%'"
                    )
                    == 0
                )
        finally:
            await conn.close()
    finally:
        await cp._pool_instance.close()
        await cp._guard_pool_instance.close()
        cp._checkpointer_instance = None


def test_system_refresh_replaces_and_bounds(agents):
    messages = add_messages(
        [],
        [
            SystemMessage(content="old memory"),
            HumanMessage(content="old user"),
            AIMessage(content="old answer"),
        ],
    )
    updated = add_messages(
        messages,
        agents.runtime.refresh_messages("new memory", messages, "next question"),
    )
    systems = [m for m in updated if m.type == "system"]
    assert len(systems) == 1 and systems[0].content == "new memory"
    bounded = agents.runtime.prompt_messages(updated, max_chars=30)
    assert bounded[-1].content == "next question"
    assert len(bounded) == 2


def test_workspace_escape_and_symlinks_rejected(agents, monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("private")
    monkeypatch.setattr(agents.tools, "WORKSPACE_DIR", str(root))
    for path in ("../outside", str(outside)):
        with pytest.raises((ValueError, OSError)):
            agents.tools.read_file.invoke({"path": path})
        with pytest.raises((ValueError, OSError)):
            agents.tools.write_file.invoke({"path": path, "content": "changed"})
    (root / "link").symlink_to(outside)
    with pytest.raises(OSError):
        agents.tools.read_file.invoke({"path": "link"})
    agents.tools.write_file.invoke({"path": "nested/ok", "content": "works"})
    assert agents.tools.read_file.invoke({"path": "nested/ok"}) == "works"
    assert outside.read_text() == "private"
    monkeypatch.delenv("ENABLE_SHELL_SANDBOX", raising=False)
    assert "disabled" in agents.tools.execute_shell_command.invoke(
        {"command": "pwd", "working_dir": "/"}
    )


def test_shell_uses_isolated_container_and_no_credentials(
    agents, monkeypatch, tmp_path
):
    monkeypatch.setenv("ENABLE_SHELL_SANDBOX", "1")
    monkeypatch.setattr(agents.tools, "WORKSPACE_DIR", str(tmp_path / "work"))
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(agents.tools._subprocess, "run", run)
    monkeypatch.setattr(
        agents.tools,
        "run_container",
        lambda args: (calls.append(args) or "exit code: 0"),
    )
    agents.tools.execute_shell_command.invoke({"command": "echo ok"})
    command = calls[0]
    assert (
        command[:2] == ["docker", "run"]
        and command[command.index("--network") + 1] == "none"
    )
    assert "--read-only" in command and "--cap-drop" in command and "--user" in command
    assert not any("LLM_API_KEY" in x or "POSTGRES_PASSWORD" in x for x in command)
    assert calls[-1][:3] == ["docker", "rm", "-f"]


@pytest.mark.skipif(
    os.getenv("RUN_DOCKER_SANDBOX_TEST") != "1",
    reason="Requires the isolated Docker test runner",
)
def test_real_container_hides_host_secrets_and_files(agents, monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_SHELL_SANDBOX", "1")
    monkeypatch.setenv("LLM_API_KEY", "secret-must-not-cross-boundary")
    monkeypatch.setattr(agents.tools, "WORKSPACE_DIR", str(tmp_path / "work"))
    outside = tmp_path / "outside"
    outside.write_text("host private file")
    command = (
        "python -c \"import os,socket; assert 'LLM_API_KEY' not in os.environ; assert not os.path.exists('"
        + str(outside)
        + "'); assert os.getcwd()=='/workspace'; open('result.txt','w').write('isolated'); print('isolated')\""
    )
    result = agents.tools.execute_shell_command.invoke({"command": command})
    assert "exit code: 0" in result and "isolated" in result
    assert (tmp_path / "work/result.txt").read_text() == "isolated"


async def test_swarm_graph_persists_parallel_claims(client, agents, monkeypatch):
    import re
    from agents import swarm_agent, checkpointer as cp

    monkeypatch.setattr(cp, "_checkpointer_instance", None)
    monkeypatch.setattr(cp, "_setup_lock", asyncio.Lock())
    for name in ("analyze_sentiment", "extract_entities"):
        await post(
            client, "/swarm/tasks", workflow_id="workflow", task_name=name, payload={}
        )

    class WorkerModel:
        async def ainvoke(self, messages):
            system = messages[0].content
            task_id = re.search(r"\(id: ([^)]+)\)", system).group(1)
            agent_name = re.search(r"Your name: (\S+)", system).group(1)
            results = [m for m in messages if m.type == "tool"]
            if not results:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "claim",
                            "name": "claim_task",
                            "args": {"task_id": task_id, "agent_name": agent_name},
                        }
                    ],
                )
            if len(results) == 1:
                lease = re.search(r"Lease token: (\S+)", results[0].content).group(1)
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "complete",
                            "name": "complete_swarm_task",
                            "args": {
                                "task_id": task_id,
                                "agent_name": agent_name,
                                "lease_token": lease,
                                "result_summary": "done",
                            },
                        }
                    ],
                )
            return AIMessage(content="done")

    monkeypatch.setattr(swarm_agent, "llm_with_tools", WorkerModel())
    saver = await cp.get_checkpointer()
    try:
        result = await swarm_agent.build_swarm_graph(saver).ainvoke(
            {"workflow_id": "workflow"}, {"configurable": {"thread_id": "parallel"}}
        )
        assert len(result["reports"]) == 2
        assert all(
            t["status"] == "COMPLETED"
            for t in await get(client, "/swarm/tasks/workflow")
        )
    finally:
        await cp._pool_instance.close()
        await cp._guard_pool_instance.close()
        cp._checkpointer_instance = None
