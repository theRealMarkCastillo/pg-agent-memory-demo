"""Forget all companion memory for one or more users.

Usage:
    python tools/cleanup_replay.py --user-id trace-replay-full trace-replay-v2
    python tools/cleanup_replay.py --all-replay      # forget every user starting with 'trace-replay'
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import asyncpg
import httpx

MEMORY_ENGINE_URL = "http://localhost:8001"


def load_env(env_path: Path) -> dict:
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


async def replay_user_ids(env: dict) -> list[str]:
    """Discover user_ids matching 'trace-replay%' from Postgres."""
    conn = await asyncpg.connect(
        user=env.get("POSTGRES_USER", "agent_user"),
        password=env.get("POSTGRES_PASSWORD", "agent_password"),
        database=env.get("POSTGRES_DB", "agent_memory_db"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(env.get("POSTGRES_PORT", "5434")),
    )
    try:
        tables = [
            "companion_graph_nodes",
            "companion_episodes",
            "companion_ephemerals",
        ]
        ids: set[str] = set()
        for table in tables:
            rows = await conn.fetch(
                f"SELECT DISTINCT user_id FROM {table} WHERE user_id LIKE 'trace-replay%'"
            )
            ids.update(r["user_id"] for r in rows)
        return sorted(ids)
    finally:
        await conn.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", default=MEMORY_ENGINE_URL)
    parser.add_argument("--user-id", nargs="+", help="user ids to forget")
    parser.add_argument("--all-replay", action="store_true",
                        help="forget every user whose id starts with 'trace-replay'")
    args = parser.parse_args()

    user_ids = list(args.user_id or [])
    if args.all_replay:
        env = load_env(Path(".env"))
        user_ids = await replay_user_ids(env)
        print(f"Found {len(user_ids)} replay users: {user_ids}")

    if not user_ids:
        if args.all_replay:
            print("No replay users to clean up.")
            return 0
        parser.error("provide --user-id or --all-replay")

    async with httpx.AsyncClient(base_url=args.engine_url, timeout=30.0) as client:
        for uid in user_ids:
            resp = await client.delete(f"/companion/memory/{uid}")
            if resp.status_code == 200:
                print(f"forgotten: {uid} -> {resp.json()}")
            else:
                print(f"FAILED {uid}: {resp.status_code} {resp.text}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
