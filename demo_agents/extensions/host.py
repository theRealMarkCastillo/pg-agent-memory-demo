"""Real disposable graph-host process; restart preserves only PG and workspace."""

import json
import os
from pathlib import Path
import socket
import secrets
import subprocess
import sys
import time
import urllib.request

WRITER = secrets.token_hex(32)
REVIEWER = secrets.token_hex(32)
OTHER = secrets.token_hex(32)


class ExtensionHost:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, EXTENSION_WORKSPACE=str(workspace))
        grants = json.loads(self.env.get("MEMORY_AUTH_TOKENS", "{}"))
        for token, subject, user, actions in (
            (WRITER, "writer-alice", "alice", ["submit", "run"]),
            (REVIEWER, "reviewer-01", "alice", ["review", "promote", "run"]),
            (OTHER, "writer-bob", "bob", ["submit", "run", "review", "promote"]),
        ):
            grants[token] = {
                "subject": subject,
                "users": [user],
                "permissions": ["extensions:" + a for a in ["read", "write", *actions]],
            }
        self.env["MEMORY_AUTH_TOKENS"] = json.dumps(grants)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = None

    def start(self):
        self.log = (self.workspace / "host.log").open("a")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "demo_agents.extensions.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "error",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=self.env,
            stdout=self.log,
            stderr=self.log,
        )
        for _ in range(150):
            if self.process.poll() is not None:
                raise AssertionError((self.workspace / "host.log").read_text())
            try:
                urllib.request.urlopen(self.url + "/health", timeout=0.2)
                return self
            except Exception:
                time.sleep(0.1)
        self.stop()
        raise AssertionError((self.workspace / "host.log").read_text())

    def stop(self, kill=False):
        if self.process and self.process.poll() is None:
            self.process.kill() if kill else self.process.terminate()
            self.process.wait(timeout=10)
        if getattr(self, "log", None):
            self.log.close()


def headers(token=WRITER):
    return {"Authorization": f"Bearer {token}"}
