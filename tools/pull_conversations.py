"""Pull all companion conversation traces from a LangSmith project.

Paginates backward through time (end_time cursor), dedupes by trace_id, and
saves full trace exports for traces that carry conversation content
(LangGraph, MasterGraphExecution, or RunnableSequence with an embedded
CONVERSATION: block). Skips audit/safety-only noise.

Usage:
    python tools/pull_conversations.py --project eidolon-prod --out /tmp/conv
    python tools/pull_conversations.py --session 4784de55-... --out /tmp/conv --max-pages 80
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.smith.langchain.com"
KEY_HEADER = "x-api-key"


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


def api_post(path: str, api_key: str, body: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{BASE_URL}{path}",
            data=json.dumps(body).encode(),
            headers={KEY_HEADER: api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def api_get(path: str, api_key: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}", headers={KEY_HEADER: api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def resolve_session(api_key: str, project: str) -> str:
    for s in api_get("/sessions", api_key):
        if s.get("name") == project:
            return s["id"]
    raise SystemExit(f"ERROR: project '{project}' not found")


def has_conversation(runs: list) -> bool:
    for r in runs:
        name = r.get("name", "")
        if name in ("LangGraph", "MasterGraphExecution"):
            return True
        if name == "RunnableSequence":
            for m in (r.get("inputs") or {}).get("input", []):
                if "CONVERSATION:" in str(m.get("content", "")):
                    return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="project name")
    parser.add_argument("--session", default=None, help="session uuid (overrides --project)")
    parser.add_argument("--out", default="traces", help="output directory")
    parser.add_argument("--max-pages", type=int, default=80, help="max pagination pages")
    parser.add_argument("--end-time", default="2026-08-16T12:00:00Z", help="start cursor")
    parser.add_argument("--min-time", default="2026-07-01T00:00:00Z", help="stop at this age")
    args = parser.parse_args()

    env = load_env(Path(".env"))
    api_key = os.getenv("LANGSMITH_API_KEY") or env.get("LANGSMITH_API_KEY")
    if not api_key or api_key == "your-langsmith-api-key-here":
        print("ERROR: set LANGSMITH_API_KEY in .env")
        return 1

    session = args.session or (
        resolve_session(api_key, args.project) if args.project else env.get("LANGSMITH_PROJECT")
    )
    if not session:
        print("ERROR: provide --session, --project, or LANGSMITH_PROJECT")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = {f.stem for f in out_dir.glob("*.json")}
    files = len(seen)
    top_count = 0
    end_time = args.end_time

    print(f"Pulling from session {session} into {out_dir}/ (have {files} already)")
    for page in range(args.max_pages):
        body = {"session": [session], "limit": 100}
        if end_time:
            body["end_time"] = end_time
        runs = api_post("/runs/query", api_key, body).get("runs", [])
        if not runs:
            break
        oldest = runs[-1].get("start_time")

        for r in runs:
            tid = r.get("trace_id") or r.get("id")
            if tid in seen:
                continue
            seen.add(tid)
            top_count += 1
            try:
                exp = api_post(
                    "/runs/query",
                    api_key,
                    {
                        "session": [session],
                        "filter": f'and(eq(trace_id, "{tid}"))',
                        "limit": 100,
                    },
                ).get("runs", [])
            except Exception:
                continue
            if not has_conversation(exp):
                continue
            path = out_dir / f"{tid}.json"
            path.write_text(json.dumps({"trace_id": tid, "runs": exp}, indent=2, default=str))
            files += 1
            print(f"  saved {tid[:12]} ({len(exp)} runs)")
            time.sleep(0.5)

        print(f"page {page}: cursor->{oldest} (unique seen {top_count}, saved {files})")
        if not oldest or oldest < args.min_time:
            break
        end_time = oldest
        time.sleep(1)

    print(f"DONE: {files} conversation traces in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
