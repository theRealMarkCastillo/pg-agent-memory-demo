"""Run the complete deterministic suite with disposable PostgreSQL and fake embeddings.

Use --engine-python to run the engine in its separately pinned environment.
The caller's .env and remote model credentials are never loaded.
"""

import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-python", default=sys.executable)
    parser.add_argument("pytest_args", nargs="*")
    args = parser.parse_args()
    name = "memory-test-" + secrets.token_hex(5)
    env = dict(os.environ)
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME", "LIVE_LLM_EVAL"):
        env.pop(k, None)
    password = secrets.token_hex(24)
    admin = secrets.token_hex(24)
    employee = secrets.token_hex(24)
    env.update(
        POSTGRES_HOST="127.0.0.1",
        POSTGRES_USER="postgres",
        POSTGRES_PASSWORD=password,
        POSTGRES_DB="memory_test_" + secrets.token_hex(5),
        MEMORY_ADMIN_TOKEN=admin,
        MEMORY_API_TOKEN=admin,
        MEMORY_EMPLOYEE_TEST_TOKEN=employee,
        MEMORY_TEST_RESET="1",
        RUN_DOCKER_SANDBOX_TEST="1",
        PYTHONDONTWRITEBYTECODE="1",
        EMBEDDING_MODEL_NAME="deterministic-logic-fixture",
        MEMORY_AUTH_TOKENS=json.dumps(
            {
                employee: {
                    "users": ["alice"],
                    "roles": ["employee"],
                    "projects": {"allowed": ["main"]},
                    "agents": ["worker"],
                    "workflows": ["allowed"],
                    "permissions": [
                        "companion:read",
                        "companion:write",
                        "enterprise:read",
                        "developer:read",
                        "task:read",
                        "swarm:read",
                        "swarm:write",
                    ],
                }
            }
        ),
    )
    server = None
    try:
        run(
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "-e",
            "POSTGRES_PASSWORD=" + password,
            "-e",
            "POSTGRES_DB=" + env["POSTGRES_DB"],
            "-p",
            "127.0.0.1::5432",
            "pgvector/pgvector:pg16",
            stdout=subprocess.DEVNULL,
        )
        port = (
            subprocess.check_output(["docker", "port", name, "5432/tcp"], text=True)
            .strip()
            .rsplit(":", 1)[1]
        )
        env["POSTGRES_PORT"] = port
        for _ in range(100):
            if (
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        name,
                        "psql",
                        "-h",
                        "127.0.0.1",
                        "-U",
                        "postgres",
                        "-d",
                        env["POSTGRES_DB"],
                        "-c",
                        "SELECT 1",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            ):
                break
            time.sleep(0.1)
        with (ROOT / "postgres/init.sql").open() as schema:
            run(
                "docker",
                "exec",
                "-i",
                name,
                "psql",
                "-U",
                "postgres",
                "-d",
                env["POSTGRES_DB"],
                "-v",
                "ON_ERROR_STOP=1",
                stdin=schema,
                stdout=subprocess.DEVNULL,
            )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            api_port = sock.getsockname()[1]
        env["MEMORY_ENGINE_TEST_URL"] = env["MEMORY_ENGINE_URL"] = (
            f"http://127.0.0.1:{api_port}"
        )
        log = ROOT / "tests/.engine-test.log"
        with log.open("w") as output:
            server = subprocess.Popen(
                [
                    args.engine_python,
                    "-m",
                    "uvicorn",
                    "server:app",
                    "--app-dir",
                    str(ROOT / "tests"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(api_port),
                    "--log-level",
                    "error",
                ],
                env=env,
                stdout=output,
                stderr=output,
            )
            ready = False
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError(log.read_text())
                try:
                    urllib.request.urlopen(
                        env["MEMORY_ENGINE_TEST_URL"] + "/health", timeout=1
                    )
                    ready = True
                    break
                except Exception:
                    time.sleep(0.1)
            if not ready:
                raise RuntimeError("Test server failed to start: " + log.read_text())
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--tb=short",
                    str(ROOT / "tests"),
                    *args.pytest_args,
                ],
                env=env,
                cwd=ROOT,
            )
            if result.returncode:
                print("Engine diagnostics:", log.read_text()[-12000:])
            return result.returncode
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    sys.exit(main())
