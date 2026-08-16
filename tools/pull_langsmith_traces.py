"""Pull LangSmith traces for a project and save full run exports to ./traces.

Usage:
    python tools/pull_langsmith_traces.py --limit 10
    python tools/pull_langsmith_traces.py --project my-project --limit 5 --out /tmp/traces

Reads LANGSMITH_API_KEY and LANGSMITH_PROJECT from .env (or environment).
Stdlib only - no langsmith SDK required.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
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


def api_get(path: str, api_key: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    if params:
        qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(url, headers={KEY_HEADER: api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


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


def resolve_project_id(api_key: str, project: str) -> str:
    sessions = api_get("/sessions", api_key)
    for s in sessions:
        if s.get("name") == project:
            return s["id"]
    raise SystemExit(f"ERROR: project '{project}' not found. Available: "
                     f"{', '.join(s['name'] for s in sessions)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="max top-level runs to export")
    parser.add_argument("--project", default=None, help="override LANGSMITH_PROJECT")
    parser.add_argument("--out", default="traces", help="output directory")
    parser.add_argument("--name", default=None, help="only export top-level runs with this name (e.g. LangGraph)")
    parser.add_argument("--run-type", default=None, help="only export top-level runs of this type (e.g. llm, chain)")
    args = parser.parse_args()

    env = load_env(Path(".env"))
    api_key = os.getenv("LANGSMITH_API_KEY") or env.get("LANGSMITH_API_KEY")
    project = args.project or os.getenv("LANGSMITH_PROJECT") or env.get("LANGSMITH_PROJECT")

    if not api_key or api_key == "your-langsmith-api-key-here":
        print("ERROR: set LANGSMITH_API_KEY in .env (or export it)")
        return 1
    if not project:
        print("ERROR: set LANGSMITH_PROJECT in .env (or pass --project)")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing up to {args.limit} top-level runs in project '{project}'...")
    session_id = resolve_project_id(api_key, project)
    result = api_post("/runs/query", api_key, {"session": [session_id], "limit": args.limit})
    runs = result.get("runs", [])
    if not runs:
        print(f"No runs found in project '{project}'.")
        return 1

    if args.name:
        runs = [r for r in runs if r.get("name") == args.name]
    if args.run_type:
        runs = [r for r in runs if r.get("run_type") == args.run_type]

    seen: set[str] = set()
    unique: list[dict] = []
    for r in runs:
        trace_id = r.get("trace_id") or r.get("id")
        if trace_id in seen:
            continue
        seen.add(trace_id)
        unique.append(r)
    runs = unique

    print(f"Exporting {len(runs)} unique traces...")
    for run in runs:
        run_id = run.get("id")
        run_type = run.get("run_type")
        name = run.get("name")
        trace_id = run.get("trace_id")
        try:
            export = _fetch_trace(api_key, session_id, str(trace_id or run_id))
            time.sleep(1)
        except Exception as e:
            print(f"  export failed for {run_id}: {e}")
            continue
        path = out_dir / f"{run_id}.json"
        path.write_text(json.dumps(export, indent=2, default=str))
        print(f"  [{run_type}] {name} ({len(export.get('runs', []))} runs) -> {path}")

    print(f"\nDone. Exports in {out_dir}/")
    return 0


def _fetch_trace(api_key: str, session_id: str, trace_id: str) -> dict:
    """Fetch every run in a trace (top-level + children) keyed by run id."""
    result = api_post(
        "/runs/query",
        api_key,
        {
            "session": [session_id],
            "filter": f'and(eq(trace_id, "{trace_id}"))',
            "limit": 100,
        },
    )
    runs = result.get("runs", [])
    if not runs:
        # fall back to the top-level run itself
        return {"runs": [api_get(f"/runs/{trace_id}", api_key)]}
    return {"trace_id": trace_id, "runs": runs}


if __name__ == "__main__":
    sys.exit(main())
