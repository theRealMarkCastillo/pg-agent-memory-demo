"""Create separate random admin and scoped demo credentials without overwriting existing files."""

import json
import os
from pathlib import Path
import secrets


def main():
    admin, client = secrets.token_hex(32), secrets.token_hex(32)
    grants = {
        "users": ["usr_anthony", "learner_001"],
        "roles": ["employee"],
        "projects": {"demo-project": ["main"]},
        "workflows": ["wf-001"],
        "agents": [
            "task-bot-1",
            "task-bot-2",
            "sentiment-bot",
            "entity-bot",
            "summary-bot",
            "general-bot",
        ],
        "permissions": [
            "developer:read",
            "developer:write",
            "task:read",
            "task:write",
            "enterprise:read",
            "tutor:read",
            "tutor:write",
            "swarm:read",
            "swarm:write",
            "companion:read",
            "companion:write",
        ],
    }
    content = f"MEMORY_ADMIN_TOKEN={admin}\nMEMORY_API_TOKEN={client}\nMEMORY_AUTH_TOKENS='{json.dumps({client: grants})}'\n"
    path = Path(".env.auth")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(content)
    print(
        "Created .env.auth. Keep this file private; load it alongside .env with Docker Compose."
    )


if __name__ == "__main__":
    main()
