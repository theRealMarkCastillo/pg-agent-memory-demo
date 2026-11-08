"""Protocol fixtures and real PostgreSQL tests for agent session hooks."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agent_hooks", ROOT / "tools/agent_hooks.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def capture(**overrides):
    return dict(project_id="allowed", git_branch="main", source="claude",
                session_id="session-1", turn_id="turn-1",
                prompt="PostgreSQL lease expiry", response="Use a lease token and test expiry.", **overrides)


async def test_capture_concurrent_retries_and_conflict(client):
    results = await asyncio.gather(*[client.post("/hooks/memories", json=capture()) for _ in range(8)])
    assert all(r.status_code == 200 for r in results)
    assert len({r.json()["memory_id"] for r in results}) == 1
    assert sum(r.json()["status"] == "stored" for r in results) == 1
    changed = capture()
    changed["response"] = "Different response"
    assert (await client.post("/hooks/memories", json=changed)).status_code == 409


async def test_recall_scope_provenance_and_forget(client):
    for branch, project, response in (("main", "allowed", "expected"),
                                       ("other", "allowed", "wrong branch"),
                                       ("main", "other", "wrong project")):
        body = capture()
        body.update(git_branch=branch, project_id=project, response=response)
        assert (await client.post("/hooks/memories", json=body)).status_code == 200
    query = dict(project_id="allowed", git_branch="main", query="lease")
    result = (await client.post("/hooks/memories/search", json=query)).json()["memories"]
    assert len(result) == 1
    assert result[0]["response"] == "expected"
    assert result[0]["verified"] is False
    assert result[0]["source"] == "claude"
    assert result[0]["session_id"] == "session-1"
    session = {k: capture()[k] for k in ("project_id", "git_branch", "source", "session_id")}
    assert (await client.post("/hooks/memories/forget", json=session)).json() == {"deleted": 1}
    assert (await client.post("/hooks/memories/search", json=query)).json() == {"memories": []}


async def test_hook_auth_and_validation(client):
    headers = {"Authorization": "Bearer " + os.environ["MEMORY_EMPLOYEE_TEST_TOKEN"]}
    assert (await client.post("/hooks/memories", json=capture(), headers=headers)).status_code == 200
    for field in ("project_id", "git_branch"):
        body = capture()
        body[field] = "forbidden"
        assert (await client.post("/hooks/memories", json=body, headers=headers)).status_code == 403
        body = {k: body[k] for k in ("project_id", "git_branch")}
        body["query"] = "lease"
        assert (await client.post("/hooks/memories/search", json=body, headers=headers)).status_code == 403
    assert (await client.post("/hooks/memories", json=capture(), headers={"Authorization": "Bearer bad"})).status_code == 401
    body = capture()
    body["response"] = "x" * 12001
    assert (await client.post("/hooks/memories", json=body)).status_code == 422
    body = capture()
    del body["project_id"]
    assert (await client.post("/hooks/memories", json=body)).status_code == 422


@pytest.fixture
def config(tmp_path):
    repo = tmp_path / "workspace"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    return dict(enabled=True, engine_url="http://127.0.0.1:8001", project_id="allowed",
                workspace_root=str(repo), state_dir=str(tmp_path / "state"))


def event(config, name, **fields):
    return dict(hook_event_name=name, cwd=config["workspace_root"], session_id="session", **fields)


@pytest.mark.parametrize("source", ["claude", "codex", "hermes", "generic"])
def test_protocol_round_trip(source, config):
    calls = []
    def transport(cfg, path, body):
        calls.append((path, body))
        return {"memories": [{"source": "codex", "prompt": "lease", "response": "old context"}]}
    before, after = bridge.EVENTS[source]
    if source == "hermes":
        start = event(config, before, extra={"turn_id": "turn", "user_message": "lease"})
        end = event(config, after, extra={"turn_id": "turn", "user_message": "lease", "assistant_response": "done"})
    else:
        start = event(config, before, prompt="lease")
        end = event(config, after, last_assistant_message="done")
        if source == "codex":
            start["turn_id"] = end["turn_id"] = "turn"
    output = bridge.handle(source, config, start, transport)
    assert "unverified data" in json.dumps(output)
    if source in ("claude", "codex"):
        assert output["hookSpecificOutput"]["hookEventName"] == before
    else:
        assert "context" in output
    assert bridge.handle(source, config, end, transport) == {}
    bridge.handle(source, config, end, transport)
    assert calls[-1] == calls[-2]  # Same ingestion identity on Stop redelivery.
    assert calls[-1][1]["prompt"] == "lease"
    assert calls[-1][1]["response"] == "done"
    assert calls[-1][1]["git_branch"] == "main"
    assert (Path(config["state_dir"]).stat().st_mode & 0o777) == 0o700


def test_no_capture_without_prompt_or_for_other_workspace(config, tmp_path):
    def forbidden(*args):
        pytest.fail("Unexpected HTTP request")
    assert bridge.handle("claude", config, event(config, "Stop", last_assistant_message="done"), forbidden) == {}
    assert bridge.handle("claude", config, event(config, "SubagentStop", last_assistant_message="done"), forbidden) == {}
    payload = event(config, "UserPromptSubmit", prompt="private")
    payload["cwd"] = str(tmp_path)
    assert bridge.handle("claude", config, payload, forbidden) == {}
    payload["cwd"] = config["workspace_root"]
    config["enabled"] = False
    assert bridge.handle("claude", config, payload, forbidden) == {}


def test_native_turn_correlation_and_recall_failure(config):
    def broken(*args):
        raise TimeoutError()
    for turn in ("first", "second"):
        with pytest.raises(TimeoutError):
            bridge.handle("codex", config, event(config, "UserPromptSubmit", prompt=turn, turn_id=turn), broken)
    captured = []
    bridge.handle("codex", config, event(config, "Stop", turn_id="first", last_assistant_message="finished"),
                  lambda cfg, path, body: captured.append(body))
    assert captured[0]["prompt"] == "first"


@pytest.mark.parametrize("raw", ["not json", "[]", '{"hook_event_name":"Stop"}', "x" * (bridge.MAX_INPUT + 1)],
                         ids=["invalid-json", "array", "missing-config", "oversized"])
def test_cli_fails_open_without_content_in_diagnostics(tmp_path, raw):
    result = subprocess.run([sys.executable, str(ROOT / "tools/agent_hooks.py"), "run", "--agent", "codex",
                             "--config", str(tmp_path / "missing.json")], input=raw, text=True, capture_output=True)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {}
    assert len(result.stderr) < 200


def test_config_generation_and_output_bound(config):
    for source in ("claude", "codex"):
        rendered = json.loads(bridge.render_config(source, "/tmp/path with spaces.json"))
        assert set(rendered["hooks"]) == {"UserPromptSubmit", "Stop"}
        assert shlex.split(rendered["hooks"]["Stop"][0]["hooks"][0]["command"])[-1] == str(Path("/tmp/path with spaces.json").resolve())
    rendered = bridge.render_config("hermes", "/tmp/config.json")
    assert "pre_llm_call:" in rendered and "post_llm_call:" in rendered
    output = bridge.context_output("hermes", "pre_llm_call", {"memories": [{"response": "x" * 20000}]})
    assert len(output["context"]) <= 6000


async def test_bridge_cli_to_live_engine_cross_agent_recall(client, config, tmp_path):
    config["engine_url"] = os.environ["MEMORY_ENGINE_TEST_URL"]
    config["token_env"] = "MEMORY_EMPLOYEE_TEST_TOKEN"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    def invoke(source, payload):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/agent_hooks.py"), "run", "--agent", source,
             "--config", str(path)], input=json.dumps(payload), text=True, capture_output=True, timeout=10,
        )
        assert result.returncode == 0 and result.stderr == ""
        return json.loads(result.stdout)
    await asyncio.to_thread(invoke, "codex", event(config, "UserPromptSubmit", prompt="PostgreSQL lease", turn_id="one"))
    await asyncio.to_thread(invoke, "codex", event(config, "Stop", last_assistant_message="Persist lease tokens.", turn_id="one"))
    recalled = await asyncio.to_thread(invoke, "claude", event(config, "UserPromptSubmit", prompt="lease"))
    assert "Persist lease tokens." in recalled["hookSpecificOutput"]["additionalContext"]
    recalled = await asyncio.to_thread(invoke, "hermes", event(config, "pre_llm_call", extra={"turn_id": "two", "user_message": "lease"}))
    assert "Persist lease tokens." in recalled["context"]


def test_remote_plaintext_url_rejected_before_token_lookup(config):
    config["engine_url"] = "http://remote.example"
    with pytest.raises(ValueError, match="HTTPS"):
        bridge.request(config, "/hooks/memories/search", {})


def test_long_session_id_is_rejected_without_aliasing(config):
    payload = event(config, "UserPromptSubmit", prompt="lease")
    payload["session_id"] = "x" * 257
    with pytest.raises(ValueError, match="identifier"):
        bridge.handle("codex", config, payload)
