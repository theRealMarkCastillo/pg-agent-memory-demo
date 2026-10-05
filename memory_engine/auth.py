"""Bearer credentials map to server-owned scopes; request fields never grant access."""

import hmac
import json
import os
from uuid import UUID
from contextlib import asynccontextmanager
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def credentials():
    configured = json.loads(os.getenv("MEMORY_AUTH_TOKENS", "{}"))
    admin = os.getenv("MEMORY_ADMIN_TOKEN")
    if admin:
        configured[admin] = {"admin": True}
    if not configured or any(len(token) < 32 for token in configured):
        raise RuntimeError(
            "Configure MEMORY_AUTH_TOKENS or MEMORY_ADMIN_TOKEN with tokens of at least 32 characters"
        )
    for principal in configured.values():
        if not isinstance(principal, dict) or not isinstance(
            principal.get("admin", False), bool
        ):
            raise RuntimeError(
                "Credential grants must be objects with a boolean admin flag"
            )
    return configured


def require_scope(principal, field, value):
    if principal.get("admin"):
        return
    key = {
        "user_id": "users",
        "user_role": "roles",
        "allowed_role": "roles",
        "workflow_id": "workflows",
        "agent_id": "agents",
        "agent_name": "agents",
    }[field]
    if value not in principal.get(key, []):
        raise HTTPException(403, f"Unauthorized {field}")


async def authorize(request: Request, bearer=Depends(_bearer)):
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    principal = next(
        (
            p
            for secret, p in request.app.state.credentials.items()
            if scheme.lower() == "bearer" and hmac.compare_digest(secret, token)
        ),
        None,
    )
    if principal is None:
        raise HTTPException(
            401, "Bearer credential required", headers={"WWW-Authenticate": "Bearer"}
        )
    request.state.principal = principal
    domain = request.url.path.split("/")[1]
    read = request.method == "GET" or request.url.path.endswith("/search")
    permission = f"{domain}:{'read' if read else 'write'}"
    if not principal.get("admin") and permission not in principal.get(
        "permissions", []
    ):
        raise HTTPException(403, f"Missing {permission}")
    try:
        body = (
            await request.json() if request.method in ("POST", "PUT", "PATCH") else {}
        )
    except (ValueError, UnicodeDecodeError):
        body = {}
    values = (
        dict(request.query_params)
        | (body if isinstance(body, dict) else {})
        | request.path_params
    )
    for field in (
        "user_id",
        "user_role",
        "allowed_role",
        "workflow_id",
        "agent_id",
        "agent_name",
    ):
        if field in values:
            require_scope(principal, field, values[field])
    if "project_id" in values and not isinstance(values["project_id"], str):
        raise HTTPException(422, "project_id must be a string")
    if (
        domain == "tutor"
        and request.url.path == "/tutor/skills"
        and not principal.get("admin")
        and "tutor:manage-skills" not in principal.get("permissions", [])
    ):
        raise HTTPException(
            403, "Shared curriculum changes require tutor:manage-skills"
        )
    if "project_id" in values and not principal.get("admin"):
        if values.get("git_branch") not in principal.get("projects", {}).get(
            values["project_id"], []
        ):
            raise HTTPException(403, "Unauthorized project or branch")
    if (
        domain == "task"
        and read
        and not principal.get("admin")
        and not values.get("agent_id")
    ):
        raise HTTPException(403, "Trajectory retrieval requires an agent scope")
    if domain == "swarm" and values.get("task_id"):
        try:
            task_id = UUID(str(values["task_id"]))
        except ValueError:
            raise HTTPException(422, "Invalid task UUID")
        async with request.app.state.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT workflow_id FROM swarm_blackboard WHERE task_id=$1",
                task_id,
            )
        if row:
            require_scope(principal, "workflow_id", row["workflow_id"])
    if domain == "companion" and not read and values.get("user_id"):
        async with request.app.state.pool.acquire() as conn:
            async with conn.transaction():
                # Serialize deletion/resumption with ingestion and checkpoint publication.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                    values["user_id"],
                )
                row = await conn.fetchrow(
                    "SELECT enabled,generation FROM companion_memory_state WHERE user_id=$1",
                    values["user_id"],
                )
                enabled = row["enabled"] if row else True
                generation = row["generation"] if row else 0
                expected = request.headers.get("x-memory-generation")
                if expected is not None and expected != str(generation):
                    raise HTTPException(409, "Stale memory generation")
                if (
                    enabled is False
                    and request.method != "DELETE"
                    and not request.url.path.endswith("/resume")
                ):
                    raise HTTPException(
                        409,
                        "Memory is disabled after forgetting; explicitly resume before writing",
                    )
                request.state.connection = conn
                yield
    else:
        yield


@asynccontextmanager
async def connection(request):
    if getattr(request.state, "connection", None) is not None:
        yield request.state.connection
    else:
        async with request.app.state.pool.acquire() as conn:
            yield conn
