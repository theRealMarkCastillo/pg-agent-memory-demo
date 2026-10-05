"""Evaluate companion memory recall after replaying traces.

Queries the memory engine and reports:
  - all stored graph facts / episodes / ephemerals for a user
  - a set of probe queries run through /companion/context (relevance-ranked)
    to check whether stored facts surface at the top of retrieval

Usage:
    python tools/eval_recall.py --user-id trace-replay-test
    python tools/eval_recall.py --user-id trace-replay-test --probes-file probes.json
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path

import httpx

MEMORY_ENGINE_URL = "http://localhost:8001"

DEFAULT_PROBES = [
    "What do you know about my hobbies?",
    "What do you know about where I live?",
    "What do you know about my job or work?",
    "What do you know about my pets?",
    "What do you know about my family?",
    "What do you know about my goals?",
]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-id", default="trace-replay", help="memory engine user_id"
    )
    parser.add_argument("--engine-url", default=MEMORY_ENGINE_URL)
    parser.add_argument(
        "--probes-file", default=None, help="JSON file with a list of probe queries"
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--min-recall", type=float, default=1.0)
    args = parser.parse_args()
    evaluated = []

    probes = DEFAULT_PROBES
    if args.probes_file:
        probes = json.loads(Path(args.probes_file).read_text())

    async with httpx.AsyncClient(
        base_url=args.engine_url,
        timeout=30.0,
        headers={"Authorization": "Bearer " + os.environ["MEMORY_API_TOKEN"]},
    ) as client:
        context = await client.get(
            "/companion/context", params={"user_id": args.user_id}
        )
        context.raise_for_status()
        data = context.json()

        facts = data.get("graph_facts", [])
        ephs = data.get("ephemerals", [])

        print(f"=== Memory for user '{args.user_id}' ===")
        print(f"Graph facts: {len(facts)}")
        for f in facts:
            rel = (
                f" {f['relationship_type']} {f['related_to']}"
                if f.get("related_to")
                else ""
            )
            print(
                f"  - {f['name']} ({f['entity_type']})[salience={f.get('salience')}]{rel}"
            )

        print(f"\nEphemerals: {len(ephs)}")
        for e in ephs:
            print(f"  - {e['description'][:80]} (expires {e['expires_at']})")

        print("\n=== Probe queries (top fact per query) ===")
        for probe in probes:
            q = probe["query"] if isinstance(probe, dict) else probe
            ranked = await client.get(
                "/companion/context",
                params={"user_id": args.user_id, "query": q, "limit": args.limit},
            )
            ranked.raise_for_status()
            top = ranked.json().get("graph_facts", [])[: args.limit]
            if isinstance(probe, dict) and probe.get("expected"):
                from benchmark_extraction import score_facts

                predictions = [
                    dict(f, relationship_to=f.get("related_to")) for f in top
                ]
                tp, fp, fn = score_facts(predictions, probe["expected"])
                recall = tp / (tp + fn)
                precision = tp / (tp + fp) if tp + fp else 0
                evaluated.append(recall)
                print(
                    f"Recall@{args.limit}={recall:.3f} Precision@{args.limit}={precision:.3f}"
                )
            print(f"Q: {q}")
            for f in top:
                print(f"   [{f.get('relevance')}] {f['name']} ({f['entity_type']})")
            if not top:
                print("   (no facts retrieved)")

    if not evaluated:
        print("Inspection only: provide labeled expected facts to measure recall.")
    return 1 if evaluated and min(evaluated) < args.min_recall else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
