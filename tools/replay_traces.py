"""Replay LangSmith trace conversations through the companion memory engine.

For each (user, assistant) turn parsed from the traces, this script:
  1. Runs the same extraction prompt used by the companion agent's
     ``extract_memory`` node (demo_agents/agents/companion_agent.py).
  2. POSTs extracted facts / terminated edges / episodes to the memory engine.
  3. Optionally evaluates recall by querying /companion/context.

Usage:
    python tools/replay_traces.py                     # replay into memory engine
    python tools/replay_traces.py --traces-dir traces --user-id replay-demo
    python tools/replay_traces.py --dry-run           # just show what would be stored
    python tools/replay_traces.py --max-turns 5       # limit turns per trace

Reads LLM_* and EMBEDDING_* from .env like the rest of the stack.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from parse_traces import parse_trace_dir

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "demo_agents" / "agents")
)
from companion_extraction import (  # noqa: E402
    EXTRACTION_PROMPT,
    PREDICATE_VOCAB,
    build_extraction_prompt,
    parse_extraction_response,
    strip_code_fence as _strip_code_fence,
)

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://localhost:8001")


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


def get_openai_client(env: dict):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=env.get("LLM_BASE_URL"),
        api_key=env.get("LLM_API_KEY"),
        timeout=110.0,
        max_retries=1,
    )


async def extract_turn(
    client, model: str, user_msg: str, agent_msg: str, timeout: float = 90.0
) -> dict:
    """Run the companion extraction prompt for a single turn."""
    prompt = build_extraction_prompt(user_msg, agent_msg)

    async def _call():
        return await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

    try:
        response = await asyncio.wait_for(_call(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        return {}
    content = response.choices[0].message.content or ""
    return parse_extraction_response(content)


async def store_extraction(engine, user_id: str, data: dict):
    """POST extracted facts (user/self/shared), terminations, and episode."""
    results = {"facts": 0, "terminated": 0, "episodes": 0}
    episode_content = data.get("episode_content")
    if not episode_content:
        raise ValueError("Extraction must retain its source conversation")
    episode = await engine.post(
        "/companion/episodes", json={"user_id": user_id, "content": episode_content}
    )
    episode.raise_for_status()
    episode_id = episode.json()["episode_id"]
    results["episodes"] = 1
    for t in data.get("terminated_edges", []) or []:
        if not t.get("name"):
            continue
        r = await engine.post(
            "/companion/facts/terminate",
            json={
                "user_id": user_id,
                "name": t["name"],
                "relationship_to": t.get("relationship_to"),
                "relationship_type": t.get("relationship_type"),
                "subject": t.get("subject", "user"),
            },
        )
        r.raise_for_status()
        if r.status_code == 200:
            results["terminated"] += 1

    for subject, key in (
        ("user", "user_facts"),
        ("self", "self_facts"),
        ("shared", "shared_facts"),
    ):
        for f in data.get(key, []) or []:
            if not f.get("name"):
                continue
            r = await engine.post(
                "/companion/facts",
                json={
                    "user_id": user_id,
                    "name": f["name"],
                    "entity_type": f.get("entity_type") or "entity",
                    "relationship_to": f.get("relationship_to"),
                    "relationship_type": f.get("relationship_type"),
                    "subject": subject,
                    "source_episode_id": episode_id,
                    "valence": f.get("valence", 0.0),
                    "intensity": f.get("intensity", 0.5),
                },
            )
            r.raise_for_status()
            if r.status_code == 200:
                results["facts"] += 1
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces-dir", default="traces", help="directory of trace exports"
    )
    parser.add_argument(
        "--user-id", default="trace-replay", help="memory engine user_id"
    )
    parser.add_argument("--dry-run", action="store_true", help="print instead of store")
    parser.add_argument(
        "--max-turns", type=int, default=0, help="limit turns per trace (0=all)"
    )
    args = parser.parse_args()

    env = load_env(Path(".env"))
    model = env.get("LLM_MODEL_NAME") or "gpt-4o-mini"
    llm = get_openai_client(env) if not args.dry_run else None
    engine_url = os.getenv("MEMORY_ENGINE_URL", MEMORY_ENGINE_URL)

    parsed = parse_trace_dir(args.traces_dir)
    if not parsed:
        print(f"No conversations found in {args.traces_dir}/")
        return 1
    print(f"Parsed {len(parsed)} traces with conversations.")

    engine = httpx.AsyncClient(
        base_url=engine_url,
        timeout=30.0,
        headers={"Authorization": "Bearer " + os.environ["MEMORY_API_TOKEN"]},
    )
    try:
        total = {"facts": 0, "terminated": 0, "episodes": 0}
        for turns, meta in parsed:
            print(f"=== {meta['file']} [{meta.get('shape')}] turns={len(turns)} ===")
            if args.max_turns:
                turns = turns[: args.max_turns]
            if args.dry_run:
                for i, turn in enumerate(turns):
                    print(f"  turn {i}: user={turn.user[:60]!r}")
                continue
            results = await _process_turns(llm, engine, model, args.user_id, turns)
            for k in total:
                total[k] += results[k]
            print(f"  -> {results}")
        print(f"\nTotal stored: {total}")
    finally:
        await engine.aclose()
    return 0


async def _process_turns(llm, engine, model, user_id, turns, concurrency: int = 4):
    """Extract concurrently, then commit strictly in source order."""
    sem = asyncio.Semaphore(concurrency)

    async def extract(turn):
        async with sem:
            data = await extract_turn(llm, model, turn.user, turn.assistant)
            # Retain the original evidence, not only a lossy generated summary.
            data["episode_content"] = f"User: {turn.user}\nCompanion: {turn.assistant}"
            return data

    extracted = await asyncio.gather(*(extract(turn) for turn in turns))
    results = {"facts": 0, "terminated": 0, "episodes": 0}
    for data in extracted:
        stored = await store_extraction(engine, user_id, data)
        for key in results:
            results[key] += stored[key]
    return results


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
