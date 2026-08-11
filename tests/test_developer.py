"""
Tests for Developer memory pattern: pg_trgm fuzzy search, halfvec HNSW semantic search, branch/project filtering.
"""
import pytest
from conftest import post, client

PROJECT = "eval-project"
BRANCH = "main"

SEED_SYMBOLS = [
    {
        "project_id": PROJECT, "git_branch": BRANCH,
        "file_path": "src/search/embed.py",
        "symbol_name": "embed_text",
        "symbol_type": "function",
        "signature": "def embed_text(text: str) -> list[float]",
        "code_content": "Converts raw text into a vector embedding for semantic retrieval.",
    },
    {
        "project_id": PROJECT, "git_branch": BRANCH,
        "file_path": "src/search/recall.py",
        "symbol_name": "recall_similar",
        "symbol_type": "function",
        "signature": "def recall_similar(query: str, top_k: int = 10) -> list",
        "code_content": "Finds the most semantically similar documents given a query.",
    },
    {
        "project_id": PROJECT, "git_branch": BRANCH,
        "file_path": "src/config.py",
        "symbol_name": "Config",
        "symbol_type": "class",
        "signature": "class Config(BaseSettings)",
        "code_content": "Application configuration loaded from environment variables.",
    },
    {
        "project_id": PROJECT, "git_branch": "feature/semantic",
        "file_path": "src/search/hybrid.py",
        "symbol_name": "hybrid_search",
        "symbol_type": "function",
        "signature": "def hybrid_search(query: str, filters: dict) -> list",
        "code_content": "Combines vector similarity and keyword matching for hybrid search.",
    },
]


@pytest.fixture(autouse=True)
async def seed_developer(client):
    for sym in SEED_SYMBOLS:
        await post(client, "/developer/symbols", **sym)


@pytest.mark.asyncio
async def test_semantic_search_retrieves_relevant(client):
    data = await post(client, "/developer/symbols/search",
        project_id=PROJECT, git_branch=BRANCH,
        query="embed text for semantic search")
    assert len(data) > 0
    names = [r["symbol_name"] for r in data]
    assert "embed_text" in names, f"embed_text not found: {names}"


@pytest.mark.asyncio
async def test_fuzzy_trgm_matches_typos(client):
    data = await post(client, "/developer/symbols/search",
        project_id=PROJECT, git_branch=BRANCH, query="recal")
    assert len(data) > 0, "Search returned no results for 'recal'"


@pytest.mark.asyncio
async def test_branch_isolation(client):
    data = await post(client, "/developer/symbols/search",
        project_id=PROJECT, git_branch=BRANCH, query="hybrid search")
    names = [r["symbol_name"] for r in data]
    assert "hybrid_search" not in names


@pytest.mark.asyncio
async def test_symbol_type_filter(client):
    data = await post(client, "/developer/symbols/search",
        project_id=PROJECT, git_branch=BRANCH, query="config", symbol_type="class")
    names = [r["symbol_name"] for r in data]
    assert "Config" in names
    assert all(r["symbol_type"] == "class" for r in data)


@pytest.mark.asyncio
async def test_semantic_search_ranking(client):
    data = await post(client, "/developer/symbols/search",
        project_id=PROJECT, git_branch=BRANCH, query="embedding text conversion")
    if len(data) >= 2:
        scores = [r.get("similarity", 0) for r in data]
        assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"
