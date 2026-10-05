import os
import logging
import asyncio
from contextlib import asynccontextmanager
from psycopg.conninfo import make_conninfo
from typing import Any
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Send
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_checkpointer_instance = None
_setup_lock = asyncio.Lock()
_guard_pool_instance = None
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
        checkpoint["pending_sends"] = [{"node": s.node, "arg": s.arg} for s in sends]

    @staticmethod
    def _decode_pending_sends(checkpoint: dict[str, Any]) -> None:
        sends = checkpoint.get("pending_sends")
        if not sends:
            return
        checkpoint["pending_sends"] = [
            Send(s["node"], s["arg"]) if isinstance(s, dict) else s for s in sends
        ]

    @asynccontextmanager
    async def memory_guard(self, config):
        settings = config.get("configurable", {})
        user = settings.get("memory_user_id")
        if not user:
            yield True
            return
        async with self.guard_pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (user,)
                )
                cur = await conn.execute(
                    "SELECT enabled,generation FROM companion_memory_state WHERE user_id=%s",
                    (user,),
                )
                row = await cur.fetchone()
                yield (row is None and settings.get("memory_generation", 0) == 0) or (
                    row is not None
                    and row[0]
                    and row[1] == settings.get("memory_generation")
                )

    async def aget_tuple(self, config):
        async with self.memory_guard(config) as allowed:
            return await super().aget_tuple(config) if allowed else None

    async def aput(self, config, checkpoint, metadata, new_versions):
        async with self.memory_guard(config) as allowed:
            if not allowed:
                return config
            checkpoint = dict(checkpoint)
            self._encode_pending_sends(checkpoint)
            saved = await super().aput(config, checkpoint, metadata, new_versions)
            saved["configurable"] = {
                **config.get("configurable", {}),
                **saved["configurable"],
            }
            return saved

    async def aput_writes(self, config, writes, task_id, task_path=""):
        async with self.memory_guard(config) as allowed:
            if allowed:
                await super().aput_writes(config, writes, task_id, task_path)

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
    return make_conninfo(
        user=user, password=password, host=host, port=port, dbname=database
    )


async def get_checkpointer():
    global _checkpointer_instance, _pool_instance, _guard_pool_instance
    async with _setup_lock:
        if _checkpointer_instance is not None:
            return _checkpointer_instance
        pool = AsyncConnectionPool(
            _pg_conn_string(),
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True},
        )
        guard = AsyncConnectionPool(
            _pg_conn_string(),
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True},
        )
        try:
            await pool.open()
            await guard.open()
            saver = SendSafeAsyncPostgresSaver(pool)
            saver.guard_pool = guard
            await saver.setup()
        except BaseException:
            await pool.close()
            await guard.close()
            raise
        _pool_instance, _guard_pool_instance, _checkpointer_instance = (
            pool,
            guard,
            saver,
        )
        return saver
