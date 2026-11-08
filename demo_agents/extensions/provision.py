"""Explicit local analytics provisioning. Never exposed through an HTTP route."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import asyncpg
from .common import digest


async def provision(users, seed=False):
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DB"],
    )
    try:
        logins = {}
        for user in users:
            role = "ext_reader_" + digest(os.environ["POSTGRES_DB"] + ":" + user)[:24]
            password = secrets.token_hex(32)
            # Generated identifiers/passwords contain only this fixed alphabet.
            async with conn.transaction():
                if not await conn.fetchval(
                    "SELECT 1 FROM pg_roles WHERE rolname=$1", role
                ):
                    await conn.execute(
                        f'CREATE ROLE "{role}" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS'
                    )
                await conn.execute(f"ALTER ROLE \"{role}\" PASSWORD '{password}'")
                await conn.execute(
                    f"ALTER ROLE \"{role}\" SET statement_timeout='250ms'"
                )
                await conn.execute(
                    f'ALTER ROLE "{role}" SET default_transaction_read_only=on'
                )
                await conn.execute(f'GRANT agent_ext_reader TO "{role}"')
                await conn.execute(
                    "INSERT INTO ext_analytics_identities VALUES($1,$2) ON CONFLICT(role_name) DO UPDATE SET user_id=EXCLUDED.user_id",
                    role,
                    user,
                )
                if seed:
                    await conn.execute(
                        "INSERT INTO ext_customers VALUES($1,1,'west'),($1,2,'east') ON CONFLICT DO NOTHING",
                        user,
                    )
                    await conn.execute(
                        "INSERT INTO ext_orders VALUES($1,1,1,10,'paid','private'),($1,2,2,20,'open','private'),($1,3,1,30,'paid','private') ON CONFLICT DO NOTHING",
                        user,
                    )
            logins[user] = {"role": role, "password": password}
        return logins
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-fixtures", action="store_true")
    args = parser.parse_args()
    logins = asyncio.run(provision(args.user, args.seed_fixtures))
    # A separate private file; never print or put database passwords in examples.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(json.dumps(logins))


if __name__ == "__main__":
    main()
