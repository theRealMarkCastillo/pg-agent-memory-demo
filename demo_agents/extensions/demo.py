"""Repeatable offline HTTP/graph/PostgreSQL extension demonstration.

Run through `make extensions-demo` for disposable infrastructure. No provider
requests are made here; `evaluate --mode live` is an explicit separate command.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile
from uuid import NAMESPACE_URL, uuid5
import asyncpg
import httpx
from .common import digest
from .host import ExtensionHost, REVIEWER, headers
from .provision import provision
from .evaluate import evaluate


def identity(name):
    return str(uuid5(NAMESPACE_URL, "pg-agent-memory-extension-demo:" + name))


async def post(client, path, data, review=False, expected=200):
    response = await client.post(
        "/extensions" + path,
        json=data,
        headers=headers(REVIEWER) if review else headers(),
    )
    if response.status_code != expected:
        raise AssertionError(f"{path}: {response.status_code}: {response.text}")
    return response.json()


async def accept(client, proposal):
    path = "/proposals/" + proposal["proposal_id"]
    result = await post(client, path + "/run", {})
    if result["status"] == "applied":
        return result
    await post(
        client,
        path + "/review",
        {
            "revision": proposal["revision"],
            "digest": proposal["digest"],
            "decision": "approved",
        },
        review=True,
    )
    result = await post(client, path + "/run", {})
    assert result["status"] == "applied"
    return result


async def demonstrate(client):
    proposal = await post(
        client,
        "/proposals",
        {
            "proposal_id": identity("file"),
            "user_id": "alice",
            "target": "reviewed.txt",
            "base_digest": digest(""),
            "content": "Approved local change.\n",
        },
    )
    await accept(client, proposal)

    policy = {
        "document_id": identity("policy"),
        "user_id": "alice",
        "version": 1,
        "refund_days": 30,
        "content": "Refunds are available within 30 days.",
        "valid_until": "2100-01-01T00:00:00Z",
    }
    await post(client, "/support/policies", policy, review=True)
    assertion = {
        "assertion_id": identity("assertion-wrong"),
        "user_id": "alice",
        "case_id": identity("case"),
        "purchase_age_days": 40,
        "source": "Purchase age: 40 days.",
    }
    await post(client, "/support/assertions", assertion)
    await post(
        client,
        "/support/assertions",
        dict(
            assertion,
            assertion_id=identity("assertion-corrected"),
            purchase_age_days=3,
            source="Purchase age: 3 days.",
            supersedes=assertion["assertion_id"],
        ),
    )
    stale_id = identity("support-stale")
    prior = await client.get("/extensions/proposals/" + stale_id)
    request = {
        "proposal_id": stale_id,
        "user_id": "alice",
        "document_id": policy["document_id"],
        "case_id": assertion["case_id"],
    }
    if prior.status_code == 404:
        stale = (await post(client, "/support/drafts", request))["proposal"]
    else:
        prior.raise_for_status()
        stale = prior.json()
    await post(
        client,
        "/support/policies",
        dict(
            policy,
            version=2,
            refund_days=14,
            content="Refunds are available within 14 days.",
        ),
        review=True,
    )
    await post(
        client,
        f"/proposals/{stale_id}/review",
        {
            "revision": stale["revision"],
            "digest": stale["digest"],
            "decision": "approved",
        },
        review=True,
        expected=409,
    )
    supported = await post(
        client,
        "/support/drafts",
        dict(request, proposal_id=identity("support-current")),
    )
    assert supported["answer"]["claim"] == "eligible"
    await accept(client, supported["proposal"])
    abstention = await post(
        client,
        "/support/drafts",
        dict(
            request,
            proposal_id=identity("support-empty"),
            case_id=identity("missing-case"),
        ),
    )
    assert abstention["answer"]["claim"] == "abstain"

    attempt = {
        "attempt_id": identity("procedure-valid"),
        "user_id": "alice",
        "source_csv": "name,quantity\npear,2\napple,1\n",
        "artifact": '[{"name":"apple","quantity":1},{"name":"pear","quantity":2}]',
        "procedure": "Parse CSV, validate integer quantities, sort by name, emit JSON objects.",
    }
    valid = await post(client, "/procedures/attempts", attempt)
    assert valid["verified"]
    await post(client, f"/procedures/{attempt['attempt_id']}/promote", {}, review=True)
    failed = await post(
        client,
        "/procedures/attempts",
        dict(attempt, attempt_id=identity("procedure-failed"), artifact="[]"),
    )
    assert not failed["verified"]
    await post(
        client,
        f"/procedures/{failed['attempt_id']}/promote",
        {},
        review=True,
        expected=409,
    )

    incident = identity("incident")
    event = {
        "event_id": identity("recovered"),
        "incident_id": incident,
        "user_id": "alice",
        "source_id": "recovery-log",
        "source_revision": 1,
        "occurred_at": "2026-01-01T01:00:00Z",
        "content": "Synthetic service recovered",
    }
    await post(client, "/incidents/events", event)
    await post(
        client,
        "/incidents/events",
        dict(
            event,
            event_id=identity("outage"),
            source_id="outage-log",
            occurred_at="2026-01-01T00:00:00Z",
            content="Synthetic outage observed",
        ),
    )
    timeline = (
        await client.get(
            f"/extensions/incidents/{incident}/timeline", params={"user_id": "alice"}
        )
    ).json()
    assert [e["event_id"] for e in timeline["timeline"]] == [
        identity("outage"),
        identity("recovered"),
    ]
    await post(client, f"/incidents/{incident}/summary", {"user_id": "alice"})

    notebook = identity("notebook")
    for name in ("cache-a", "cache-b"):
        await post(
            client,
            "/research/documents",
            {
                "document_id": identity(name),
                "notebook_id": notebook,
                "user_id": "alice",
                "version": 1,
                "content": (
                    Path(__file__).parent / "fixtures" / (name + ".txt")
                ).read_text(),
            },
        )
    report_request = {
        "report_id": identity("research-parallel"),
        "notebook_id": notebook,
        "user_id": "alice",
        "claims": ["cache_reduces_latency", "replicas_improve_availability"],
    }
    report = await post(client, "/research/reports", report_request)
    baseline = await post(
        client,
        "/research/reports",
        dict(report_request, report_id=identity("research-sequential"), parallel=False),
    )
    assert report["report"] == baseline["report"]
    assert report["report"]["claims"]["cache_reduces_latency"] == "disputed"
    await post(
        client,
        f"/research/reports/{report['report_id']}/review",
        {"digest": report["digest"], "accept": True},
        review=True,
    )
    assertions = (
        await client.get(
            f"/extensions/research/{notebook}/assertions", params={"user_id": "alice"}
        )
    ).json()["assertions"]
    assert [a["claim"] for a in assertions] == ["replicas_improve_availability"]
    analytics = await post(
        client,
        "/analytics/query",
        {
            "run_id": identity("analytics"),
            "user_id": "alice",
            "plan": {"operation": "sum_amount", "status": "paid"},
        },
    )
    assert analytics["summary"]["observed_rows"] == [{"total": 40}]
    return {
        "approval": "applied",
        "support": "corrected, cited, stale rejected, empty abstained",
        "procedure": "verified promoted; failed denied",
        "incident": "late event ordered; arrival retained",
        "research": "parallel equals sequential; disagreement retained; reviewed support reusable",
        "analytics": "restricted tenant paid total = 40",
    }


async def counts():
    conn = await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DB"],
    )
    tables = [
        "ext_proposals",
        "ext_reviews",
        "ext_attempts",
        "ext_incident_events",
        "ext_research_documents",
        "ext_research_reports",
        "ext_analytics_runs",
    ]
    try:
        return {
            table: await conn.fetchval(f"SELECT count(*) FROM {table}")
            for table in tables
        }
    finally:
        await conn.close()


async def main_async(repeat=2):
    with tempfile.TemporaryDirectory(prefix="pg-extension-demo-") as directory:
        root = Path(directory)
        config = root / "analytics-logins.json"
        config.write_text(json.dumps(await provision(["alice", "bob"], seed=True)))
        config.chmod(0o600)
        host = ExtensionHost(root)
        host.env["EXTENSION_ANALYTICS_CONFIG"] = str(config)
        try:
            host.start()
            previous_counts = previous_files = None
            async with httpx.AsyncClient(
                base_url=host.url, headers=headers(), timeout=30
            ) as client:
                for index in range(repeat):
                    outcomes = await demonstrate(client)
                    current_counts = await counts()
                    files = {
                        str(p): (p.stat().st_mtime_ns, digest(p.read_text()))
                        for p in (root / digest("alice")).iterdir()
                    }
                    if previous_counts is not None:
                        assert current_counts == previous_counts, (
                            "Repeat duplicated accepted records"
                        )
                        assert files == previous_files, "Repeat rewrote local effects"
                    previous_counts, previous_files = current_counts, files
                    print(
                        json.dumps(
                            {
                                "mode": "offline",
                                "run": index + 1,
                                "outcomes": outcomes,
                                "counts": current_counts,
                            },
                            sort_keys=True,
                        )
                    )
                recall = await client.get(
                    "/extensions/procedures", params={"user_id": "alice"}
                )
                recall.raise_for_status()
                evaluation = await evaluate(
                    "scripted", [r["procedure"] for r in recall.json()["procedures"]]
                )
                assert all(
                    task["verified"]
                    for condition in evaluation["procedure_conditions"]
                    for task in condition["tasks"]
                )
                assert (
                    evaluation["research_conditions"][0]["report"]
                    == evaluation["research_conditions"][1]["report"]
                )
                assert evaluation["support_accepted"]
                print(
                    json.dumps(
                        {
                            "scripted_evaluation": "passed",
                            "memory_origin": evaluation["memory_origin"],
                            "live_quality": "unverified",
                        }
                    )
                )
        finally:
            host.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=2, choices=range(1, 6))
    asyncio.run(main_async(parser.parse_args().repeat))
