"""
Tests for Task/Trajectory memory pattern: store, similarity search, success_score filtering.
"""
import pytest
from conftest import post


SEED_TRAJECTORIES = [
    {
        "agent_id": "bot-1",
        "goal_description": "Scrape product prices from e-commerce site",
        "action_sequence": [{"action": "fetch_url"}, {"action": "parse_html"}, {"action": "save_csv"}],
        "execution_result": "Scraped 100 products successfully.",
        "success_score": 0.95,
    },
    {
        "agent_id": "bot-2",
        "goal_description": "Summarize legal contract clauses",
        "action_sequence": [{"action": "read_pdf"}, {"action": "extract_clauses"}, {"action": "generate_summary"}],
        "execution_result": "Generated summary of 15 clauses.",
        "success_score": 0.85,
    },
    {
        "agent_id": "bot-3",
        "goal_description": "Scrape prices from competitor site but failed due to captcha",
        "action_sequence": [{"action": "fetch_url"}, {"action": "captcha_detected"}],
        "execution_result": "Blocked by captcha after 3 attempts.",
        "success_score": 0.15,
    },
    {
        "agent_id": "bot-4",
        "goal_description": "Extract text from scanned PDF invoices",
        "action_sequence": [{"action": "ocr_scan"}, {"action": "extract_fields"}],
        "execution_result": "Extracted 45 invoices.",
        "success_score": 0.60,
    },
]


@pytest.fixture(autouse=True)
async def seed_trajectories(client):
    for t in SEED_TRAJECTORIES:
        await post(client, "/task/trajectories", **t)


@pytest.mark.asyncio
async def test_similarity_search_ranks_relevant(client):
    data = await post(client, "/task/trajectories/search",
        goal_description="scrape product prices from a website")
    assert len(data) > 0
    goals = [r["goal_description"] for r in data]
    assert "Scrape product prices from e-commerce site" in goals


@pytest.mark.asyncio
async def test_success_score_filter_excludes_failures(client):
    data = await post(client, "/task/trajectories/search",
        goal_description="scrape prices")
    scores = [r["success_score"] for r in data]
    assert all(s >= 0.7 for s in scores), f"Low-score trajectories leaked: {scores}"


@pytest.mark.asyncio
async def test_lower_threshold_includes_more(client):
    strict = await post(client, "/task/trajectories/search",
        goal_description="scrape", min_success_score=0.7)
    loose = await post(client, "/task/trajectories/search",
        goal_description="scrape", min_success_score=0.0)
    assert len(loose) >= len(strict)


@pytest.mark.asyncio
async def test_similarity_score_present(client):
    data = await post(client, "/task/trajectories/search",
        goal_description="extract text from documents")
    for r in data:
        assert "similarity" in r


@pytest.mark.asyncio
async def test_different_domains_dont_cross_contaminate(client):
    data = await post(client, "/task/trajectories/search",
        goal_description="scrape e-commerce product pricing data")
    results_text = " ".join(str(r) for r in data)
    assert "legal" not in results_text.lower()
