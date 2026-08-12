import os
import logging
from typing import Any
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Send
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_checkpointer_instance = None
_pool_instance = None  # retained to prevent GC of the connection pool


class SendSafeAsyncPostgresSaver(AsyncPostgresSaver):
    """AsyncPostgresSaver that serializes the Send API's `pending_sends`.

    LangGraph stores `Send` objects in `checkpoint["pending_sends"]`, which the
    stock `aput` writes via psycopg's Jsonb adapter (a plain `json.dumps`) and
    crashes with "Object of type Send is not JSON serializable" (upstream issue
    langchain-ai/langgraph#6456). We convert Send -> JSON-safe dict on write and
    reconstruct Send objects on read.
    """

    @staticmethod
    def _encode_pending_sends(checkpoint: dict[str, Any]) -> None:
        sends = checkpoint.get("pending_sends")
        if not sends:
            return
        checkpoint["pending_sends"] = [
            {"node": s.node, "arg": s.arg} for s in sends
        ]

    @staticmethod
    def _decode_pending_sends(checkpoint: dict[str, Any]) -> None:
        sends = checkpoint.get("pending_sends")
        if not sends:
            return
        checkpoint["pending_sends"] = [
            Send(s["node"], s["arg"]) if isinstance(s, dict) else s for s in sends
        ]

    async def aput(self, config, checkpoint, metadata, new_versions):
        self._encode_pending_sends(checkpoint)
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def _load_checkpoint_tuple(self, value):
        tup = await super()._load_checkpoint_tuple(value)
        self._decode_pending_sends(tup.checkpoint)
        return tup


def _pg_conn_string():
    user = os.getenv("POSTGRES_USER", "agent_user")
    password = os.getenv("POSTGRES_PASSWORD", "agent_password")
    database = os.getenv("POSTGRES_DB", "agent_memory_db")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgres://{user}:{password}@{host}:{port}/{database}"


async def get_checkpointer():
    global _checkpointer_instance, _pool_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    conn_string = _pg_conn_string()
    _pool_instance = AsyncConnectionPool(
        conn_string,
        min_size=1,
        max_size=5,
        open=True,
        kwargs={"autocommit": True},
    )
    _checkpointer_instance = SendSafeAsyncPostgresSaver(_pool_instance)
    await _checkpointer_instance.setup()
    logger.info("Checkpointer initialized and tables created")
    return _checkpointer_instance
