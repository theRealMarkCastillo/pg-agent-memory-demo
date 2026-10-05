import os
from pathlib import Path
import asyncpg
from pgvector.asyncpg import register_vector


async def get_db_pool():
    async def init(conn):
        await register_vector(conn)

    async def setup(conn):
        # Pool reset clears session SETs; setup runs on every checkout.
        await conn.execute("SET hnsw.iterative_scan = strict_order")

    pool = await asyncpg.create_pool(
        user=os.getenv("POSTGRES_USER", "agent_user"),
        password=os.getenv("POSTGRES_PASSWORD", "agent_password"),
        database=os.getenv("POSTGRES_DB", "agent_memory_db"),
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        init=init,
        setup=setup,
    )
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(62180431)")
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY)"
                )
                for file in sorted(
                    Path(__file__).with_name("migrations").glob("*.sql")
                ):
                    if not await conn.fetchval(
                        "SELECT 1 FROM schema_migrations WHERE name=$1", file.name
                    ):
                        await conn.execute(file.read_text())
                        await conn.execute(
                            "INSERT INTO schema_migrations VALUES ($1)", file.name
                        )
    except BaseException:
        await pool.close()
        raise
    return pool
