"""Tests for Enterprise: role-gated access, RRF ranking, full-text + vector hybrid search."""
import pytest
from conftest import post

SEED_DOCS = {
    "employee": [
        {"doc_title": "Data Access Policy", "allowed_role": "employee",
         "content": "Employees may access customer data for support purposes only. All access is logged and audited quarterly."},
        {"doc_title": "Remote Work Guidelines", "allowed_role": "employee",
         "content": "Employees may work remotely up to 3 days per week. Core hours are 10am-3pm."},
    ],
    "admin": [
        {"doc_title": "Admin Security Protocol", "allowed_role": "admin",
         "content": "Only administrators may modify user roles and access control lists. Two-factor authentication required."},
    ],
    "contractor": [
        {"doc_title": "Contractor NDA Terms", "allowed_role": "contractor",
         "content": "Contractors must sign an NDA before accessing any internal systems."},
    ],
}


@pytest.fixture(autouse=True)
async def seed_enterprise(client):
    for docs in SEED_DOCS.values():
        for doc in docs:
            await post(client, "/enterprise/documents", **doc)


@pytest.mark.asyncio
async def test_role_gated_access_employee(client):
    data = await post(client, "/enterprise/documents/search",
        query="access control customer data", user_role="employee")
    titles = [r["doc_title"] for r in data]
    assert "Data Access Policy" in titles or "Remote Work Guidelines" in titles
    assert "Admin Security Protocol" not in titles


@pytest.mark.asyncio
async def test_role_gated_access_admin(client):
    data = await post(client, "/enterprise/documents/search",
        query="two-factor authentication user roles", user_role="admin")
    titles = [r["doc_title"] for r in data]
    assert "Admin Security Protocol" in titles


@pytest.mark.asyncio
async def test_role_gated_access_contractor(client):
    data = await post(client, "/enterprise/documents/search",
        query="NDA terms access", user_role="contractor")
    titles = [r["doc_title"] for r in data]
    assert "Contractor NDA Terms" in titles
    for title in titles:
        assert "Admin" not in title, f"Admin doc leaked: {title}"


@pytest.mark.asyncio
async def test_rrf_scoring_present(client):
    data = await post(client, "/enterprise/documents/search",
        query="customer data access", user_role="employee")
    for r in data:
        assert "rrf_score" in r


@pytest.mark.asyncio
async def test_full_text_keyword_match(client):
    data = await post(client, "/enterprise/documents/search",
        query="remote work guidelines", user_role="employee")
    titles = [r["doc_title"] for r in data]
    assert "Remote Work Guidelines" in titles


@pytest.mark.asyncio
async def test_empty_role_returns_nothing(client):
    data = await post(client, "/enterprise/documents/search",
        query="access", user_role="nonexistent_role")
    assert len(data) == 0
