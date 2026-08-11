import os
import asyncpg
from pgvector.asyncpg import register_vector

async def get_db_pool():
    user = os.getenv("POSTGRES_USER", "agent_user")
    password = os.getenv("POSTGRES_PASSWORD", "agent_password")
    database = os.getenv("POSTGRES_DB", "agent_memory_db")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")

    conn_str = f"postgres://{user}:{password}@{host}:{port}/{database}"

    async def init(conn):
        await register_vector(conn)

    return await asyncpg.create_pool(conn_str, init=init)
