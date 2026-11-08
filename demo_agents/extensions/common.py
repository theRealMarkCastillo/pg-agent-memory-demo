"""Application-owned identity and canonical evidence; never model authority."""

import hashlib
import json
from contextlib import asynccontextmanager
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict
from memory_engine.auth import require_scope


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def permission(request, action):
    principal = request.state.principal
    if not principal.get("admin") and f"extensions:{action}" not in principal.get(
        "permissions", []
    ):
        raise HTTPException(403, f"Missing extensions:{action}")


def actor(request):
    principal = request.state.principal
    if principal.get("admin"):
        return "admin"
    subject = principal.get("subject")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(403, "Credential requires a server-configured subject")
    return subject


def owned(request, row):
    if row is None:
        raise HTTPException(404, "Record not found")
    require_scope(request.state.principal, "user_id", row["user_id"])
    return row


@asynccontextmanager
async def locked(pool, namespace):
    """Cross-process lock; transaction automatically releases on host death."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,0))", namespace
            )
            yield conn


@asynccontextmanager
async def effect_locked(pool, namespace):
    """Hold a destination lock across the intent and acknowledgement commits.

    The session lock disappears when the host/connection dies. Pool release also
    resets advisory locks; explicit unlock keeps the normal path easy to audit.
    This pool must be separate from API requests waiting for graph locks.
    """
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock(hashtextextended($1,0))", namespace)
        try:
            yield conn
        finally:
            await conn.execute(
                "SELECT pg_advisory_unlock(hashtextextended($1,0))", namespace
            )
