import os
import json
import httpx
from langchain_core.tools import tool

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


def _post(path: str, body: dict | None = None, params: dict | None = None):
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{MEMORY_ENGINE_URL}{path}",
            json=body,
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


def _get(path: str, params: dict | None = None):
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{MEMORY_ENGINE_URL}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


def _delete(path: str):
    with httpx.Client(timeout=30.0) as client:
        resp = client.delete(f"{MEMORY_ENGINE_URL}{path}")
        resp.raise_for_status()
        return resp.json()


# ── Developer Pattern Tools ──────────────────────────────────────────────


@tool
def search_code_symbols(project_id: str, git_branch: str, query: str, symbol_type: str | None = None) -> str:
    """Search code symbols in the developer workspace memory. Returns matching
    functions, classes, and other code symbols with their signatures and file paths."""
    body = {
        "project_id": project_id,
        "git_branch": git_branch,
        "query": query,
    }
    if symbol_type:
        body["symbol_type"] = symbol_type
    results = _post("/developer/symbols/search", body=body)
    if not results:
        return "No matching code symbols found."
    lines = []
    for s in results:
        lines.append(f"{s['symbol_name']} ({s['symbol_type']}) in {s['file_path']}: {s['signature']}\n  {s.get('code_content', '')}")
    return "\n---\n".join(lines)


@tool
def store_code_symbol(project_id: str, git_branch: str, file_path: str, symbol_name: str, symbol_type: str, signature: str, code_content: str) -> str:
    """Store a new or updated code symbol in the developer workspace memory."""
    _post("/developer/symbols", body={
        "project_id": project_id,
        "git_branch": git_branch,
        "file_path": file_path,
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "signature": signature,
        "code_content": code_content,
    })
    return f"Symbol '{symbol_name}' stored in project {project_id}."


# ── Task Agent Pattern Tools ─────────────────────────────────────────────


@tool
def search_trajectories(goal_description: str, min_success_score: float = 0.7) -> str:
    """Search past task trajectories for similar goals. Returns successful past executions
    with their action sequences and results."""
    results = _post("/task/trajectories/search", body={
        "goal_description": goal_description,
        "min_success_score": min_success_score,
    })
    if not results:
        return "No matching past trajectories found."
    lines = []
    for t in results:
        lines.append(
            f"Goal: {t['goal_description']}\n"
            f"Actions: {t['action_sequence']}\n"
            f"Result: {t['execution_result']} (score: {t['success_score']})"
        )
    return "\n---\n".join(lines)


@tool
def store_trajectory(agent_id: str, goal_description: str, action_sequence: str, execution_result: str, success_score: float) -> str:
    """Store a completed task trajectory in memory for future recall."""
    try:
        actions = json.loads(action_sequence)
    except (json.JSONDecodeError, TypeError):
        actions = [{"action": action_sequence}]
    _post("/task/trajectories", body={
        "agent_id": agent_id,
        "goal_description": goal_description,
        "action_sequence": actions,
        "execution_result": execution_result,
        "success_score": success_score,
    })
    return f"Trajectory stored: '{goal_description}' (score: {success_score})."


# ── Enterprise Agent Pattern Tools ───────────────────────────────────────


@tool
def search_policy_documents(query: str, user_role: str) -> str:
    """Search enterprise policy documents filtered by the user's access role.
    Only returns documents the user is authorized to see."""
    results = _post("/enterprise/documents/search", body={
        "query": query,
        "user_role": user_role,
    })
    if not results:
        return "No authorized documents found for this role."
    lines = []
    for d in results:
        lines.append(f"Title: {d['doc_title']}\nContent: {d['content']}")
    return "\n---\n".join(lines)


@tool
def store_policy_document(doc_title: str, allowed_role: str, content: str) -> str:
    """Store a new enterprise policy document with role-based access control."""
    _post("/enterprise/documents", body={
        "doc_title": doc_title,
        "allowed_role": allowed_role,
        "content": content,
    })
    return f"Policy document '{doc_title}' stored for role '{allowed_role}'."


# ── Tutor Agent Pattern Tools ────────────────────────────────────────────


@tool
def get_skill_gaps(user_id: str) -> str:
    """Get the current skill gaps for a learner, including decayed proficiency
    scores based on the Ebbinghaus forgetting curve."""
    results = _get(f"/tutor/gaps/{user_id}")
    if not results:
        return "No skills found for this user."
    lines = []
    for s in results:
        lines.append(
            f"Skill: {s['skill_name']} | "
            f"Decayed Score: {s['decayed_score']:.3f} | "
            f"Status: {s['status']}"
        )
    return "\n".join(lines)


@tool
def update_skill_progress(user_id: str, skill_name: str, proficiency_score: float) -> str:
    """Update a learner's proficiency score for a skill. Higher scores indicate mastery.
    This resets the forgetting curve decay timer."""
    _post("/tutor/progress", body={
        "user_id": user_id,
        "skill_name": skill_name,
        "proficiency_score": proficiency_score,
    })
    return f"Updated '{skill_name}' proficiency to {proficiency_score} for user {user_id}."


# ── Swarm Agent Pattern Tools ───────────────────────────────────────────


@tool
def list_workflow_tasks(workflow_id: str) -> str:
    """List all tasks on the swarm blackboard for a workflow, including their status."""
    results = _get(f"/swarm/tasks/{workflow_id}")
    if not results:
        return "No tasks found for this workflow."
    lines = []
    for t in results:
        lines.append(
            f"Task: {t['task_name']} | Status: {t['status']} | "
            f"Agent: {t.get('assigned_agent', 'unassigned')} | ID: {t['task_id']}"
        )
    return "\n".join(lines)


@tool
def claim_next_task(agent_name: str, workflow_id: str) -> str:
    """Claim the oldest pending task from the swarm blackboard using lock-free
    coordination (SKIP LOCKED)."""
    result = _post("/swarm/tasks/claim-next", params={
        "agent_name": agent_name,
        "workflow_id": workflow_id,
    })
    if result.get("status") == "claimed":
        task = result["task"]
        return (
            f"Task claimed: {task['task_name']} (ID: {task['task_id']})\n"
            f"Payload: {task['payload']}"
        )
    return "No pending tasks available to claim."


@tool
def claim_task(task_id: str, agent_name: str) -> str:
    """Claim a specific task from the swarm blackboard by its task ID using lock-free
    coordination (SKIP LOCKED). Returns the claimed task details, or an error if it
    was already claimed by another worker."""
    result = _post("/swarm/tasks/claim", body={
        "task_id": task_id,
        "agent_name": agent_name,
    })
    if result.get("status") == "claimed":
        task = result["task"]
        return (
            f"Task claimed: {task['task_name']} (ID: {task['task_id']})\n"
            f"Payload: {task['payload']}"
        )
    return f"Could not claim task {task_id}: {result.get('status')}."


@tool
def complete_swarm_task(task_id: str, result_summary: str) -> str:
    """Mark a claimed swarm task as completed and store the result."""
    _post("/swarm/tasks/complete", body={
        "task_id": task_id,
        "payload": {"result": result_summary},
    })
    return f"Task {task_id} marked as completed."


# ── Companion Agent Pattern Tools ────────────────────────────────────────


@tool
def get_companion_context(user_id: str, query: str | None = None) -> str:
    """Get the persistent memory context for a user, including relationship
    graph facts and current ephemeral state. Optionally pass a query to rank
    facts by relevance to that query."""
    params = {"user_id": user_id}
    if query:
        params["query"] = query
    results = _get("/companion/context", params=params)
    facts = results.get("graph_facts", [])
    ephs = results.get("ephemerals", [])

    facts_str = "; ".join(
        f"{f['name']} ({f['entity_type']})"
        + (f" {f['relationship_type']} {f['related_to']}" if f.get("related_to") else "")
        for f in facts
    )
    ephs_str = "; ".join(e["description"] for e in ephs)
    return f"Active Facts: {facts_str}\nCurrent Mood/Events: {ephs_str}"


@tool
def search_episodic_memory(user_id: str, query: str) -> str:
    """Search the user's episodic conversation memory for semantically similar past exchanges."""
    results = _post("/companion/context/search", params={
        "user_id": user_id,
        "query": query,
    })
    if not results:
        return "No relevant past conversations found."
    return "\n---\n".join(r["content"] for r in results)


@tool
def store_companion_episode(user_id: str, content: str) -> str:
    """Store a new episodic memory of a conversation or interaction with the user."""
    result = _post("/companion/episodes", body={
        "user_id": user_id,
        "content": content,
    })
    return f"Episode stored ({result['chunks_stored']} chunks)."


@tool
def store_companion_fact(user_id: str, name: str, entity_type: str, relationship_to: str | None = None, relationship_type: str | None = None) -> str:
    """Store a fact about the user in the relationship graph (e.g., preferences, goals, people)."""
    body = {
        "user_id": user_id,
        "name": name,
        "entity_type": entity_type,
    }
    if relationship_to and relationship_type:
        body["relationship_to"] = relationship_to
        body["relationship_type"] = relationship_type
    _post("/companion/facts", body=body)
    return f"Fact stored: {name} ({entity_type})."


@tool
def store_companion_ephemeral(user_id: str, description: str, ttl_seconds: int = 3600) -> str:
    """Store a temporary, expiring piece of context (e.g., current mood, ongoing activity)."""
    _post("/companion/ephemerals", body={
        "user_id": user_id,
        "description": description,
        "ttl_seconds": ttl_seconds,
    })
    return f"Ephemeral context stored (TTL: {ttl_seconds}s)."


@tool
def terminate_companion_relationship(user_id: str, name: str, relationship_to: str | None = None, relationship_type: str | None = None) -> str:
    """Terminate a relationship in the companion's memory graph when a fact is no
    longer true (e.g., the user moved away, changed jobs, or ended a goal)."""
    body = {"user_id": user_id, "name": name}
    if relationship_to and relationship_type:
        body["relationship_to"] = relationship_to
        body["relationship_type"] = relationship_type
    result = _post("/companion/facts/terminate", body=body)
    return f"Relationship terminated: {name} (status: {result.get('status')})."


@tool
def forget_companion_memory(user_id: str) -> str:
    """Permanently forget all memory about a user (facts, episodes, ephemerals).
    Use when the user requests deletion of their remembered information."""
    result = _delete(f"/companion/memory/{user_id}")
    return f"Memory forgotten (status: {result.get('status')})."


# ── Real Execution Tools ──────────────────────────────────────────────────

import subprocess as _subprocess

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/tmp/agent-workspace")


@tool
def execute_shell_command(command: str, working_dir: str | None = None) -> str:
    """Execute a shell command in the agent workspace. Returns stdout, stderr,
    and exit code. Commands are sandboxed to the workspace directory. Use for
    running scripts, data processing, git operations, etc."""
    import pathlib

    cwd = working_dir or WORKSPACE_DIR
    pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)

    try:
        result = _subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        parts = []
        if result.stdout.strip():
            parts.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            parts.append(f"stderr:\n{result.stderr.strip()}")
        parts.append(f"exit code: {result.returncode}")
        return "\n".join(parts)
    except _subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."


@tool
def fetch_url(url: str, method: str = "GET", headers: str | None = None, body: str | None = None) -> str:
    """Make an HTTP request to a URL. Returns status code, headers, and body.
    Use for scraping, API calls, health checks. Specify method (GET/POST),
    optional JSON headers, and optional request body."""
    import json as _json

    parsed_headers = _json.loads(headers) if headers else {}
    parsed_body = body.encode() if body else None

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.request(
            method=method.upper(),
            url=url,
            headers=parsed_headers,
            content=parsed_body,
        )
        content_type = resp.headers.get("content-type", "")
        body_text = resp.text[:4000]
        if "json" in content_type:
            try:
                body_text = _json.dumps(resp.json(), indent=2)
            except Exception:
                pass

    return f"HTTP {resp.status_code}\nBody:\n{body_text}"


@tool
def read_file(path: str) -> str:
    """Read the contents of a file in the agent workspace. Returns file content
    or an error if the file doesn't exist."""
    import pathlib

    full_path = pathlib.Path(path)
    if not full_path.is_absolute():
        full_path = pathlib.Path(WORKSPACE_DIR) / path

    try:
        content = full_path.read_text()
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"
        return content
    except FileNotFoundError:
        return f"File not found: {full_path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the agent workspace. Creates parent directories
    if needed. Returns the full path of the written file."""
    import pathlib

    full_path = pathlib.Path(path)
    if not full_path.is_absolute():
        full_path = pathlib.Path(WORKSPACE_DIR) / path

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    return f"File written: {full_path} ({len(content)} bytes)"


# ── Tool collections per pattern ─────────────────────────────────────────

DEVELOPER_TOOLS = [search_code_symbols, store_code_symbol, execute_shell_command, read_file, write_file]
TASK_TOOLS = [search_trajectories, store_trajectory, execute_shell_command, fetch_url, read_file, write_file]
ENTERPRISE_TOOLS = [search_policy_documents, store_policy_document]
TUTOR_TOOLS = [get_skill_gaps, update_skill_progress]
SWARM_TOOLS = [list_workflow_tasks, claim_next_task, claim_task, complete_swarm_task, execute_shell_command, fetch_url]
COMPANION_TOOLS = [get_companion_context, search_episodic_memory, store_companion_episode, store_companion_fact, store_companion_ephemeral, terminate_companion_relationship, forget_companion_memory]
