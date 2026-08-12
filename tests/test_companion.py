"""Tests for Companion: graph facts, TTL ephemerals, episodic search, user isolation."""
import pytest
import asyncio
from conftest import post, post_params, get, delete

USER = "companion_test_user"


@pytest.fixture(autouse=True)
async def seed_facts(client):
    await post(client, "/companion/facts",
        user_id=USER, name="Alice", entity_type="person")
    await post(client, "/companion/facts",
        user_id=USER, name="Brooklyn", entity_type="location",
        relationship_to="Alice", relationship_type="lives_in")
    await post(client, "/companion/facts",
        user_id=USER, name="Photography", entity_type="hobby",
        relationship_to="Alice", relationship_type="enjoys")


@pytest.mark.asyncio
async def test_graph_facts_retrieved(client):
    data = await get(client, "/companion/context", user_id=USER)
    names = {f["name"] for f in data["graph_facts"]}
    assert "Alice" in names
    assert "Brooklyn" in names
    assert "Photography" in names


@pytest.mark.asyncio
async def test_relationship_edges_present(client):
    data = await get(client, "/companion/context", user_id=USER)
    brooklyn = next(f for f in data["graph_facts"] if f["name"] == "Brooklyn")
    assert brooklyn["relationship_type"] == "lives_in"
    assert brooklyn["related_to"] == "Alice"


@pytest.mark.asyncio
async def test_episode_creation_and_search(client):
    await post(client, "/companion/episodes",
        user_id=USER, content="Alice visited the Brooklyn Bridge and took photos.")
    await asyncio.sleep(1)
    data = await post_params(client, "/companion/context/search",
        user_id=USER, query="brooklyn bridge photos")
    assert len(data) > 0, "Episodic search returned no results"


@pytest.mark.asyncio
async def test_ephemeral_ttl_expiration(client):
    await post(client, "/companion/ephemerals",
        user_id=USER, description="Alice is excited about the weekend", ttl_seconds=1)
    data = await get(client, "/companion/context", user_id=USER)
    descs = [e["description"] for e in data["ephemerals"]]
    assert any("excited" in d for d in descs)
    await asyncio.sleep(2)
    data = await get(client, "/companion/context", user_id=USER)
    descs = [e["description"] for e in data["ephemerals"]]
    assert not any("excited" in d for d in descs), f"TTL didn't expire: {descs}"


@pytest.mark.asyncio
async def test_user_isolation(client):
    await post(client, "/companion/facts",
        user_id="other_user", name="Bob", entity_type="person")
    data = await get(client, "/companion/context", user_id=USER)
    names = {f["name"] for f in data["graph_facts"]}
    assert "Bob" not in names


@pytest.mark.asyncio
async def test_episodic_search_ranking(client):
    await post(client, "/companion/episodes",
        user_id=USER, content="Alice loves taking photos of the Brooklyn Bridge at sunset.")
    await asyncio.sleep(1)
    data = await post_params(client, "/companion/context/search",
        user_id=USER, query="Alice brooklyn")
    if len(data) >= 2:
        scores = [r["similarity"] for r in data]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_salience_bumps_on_remention(client):
    user = "salience_user"
    await post(client, "/companion/facts",
        user_id=user, name="Guitar", entity_type="hobby")
    await post(client, "/companion/facts",
        user_id=user, name="Guitar", entity_type="hobby")
    data = await get(client, "/companion/context", user_id=user)
    guitar = next(f for f in data["graph_facts"] if f["name"] == "Guitar")
    assert guitar["salience"] >= 2.0, f"Salience not bumped: {guitar}"


@pytest.mark.asyncio
async def test_terminate_relationship(client):
    user = "terminate_user"
    await post(client, "/companion/facts",
        user_id=user, name="Brooklyn", entity_type="location",
        relationship_to="Home", relationship_type="lives_in")
    await post(client, "/companion/facts/terminate",
        user_id=user, name="Brooklyn", relationship_to="Home",
        relationship_type="lives_in")

    data = await get(client, "/companion/context", user_id=user)
    brooklyn = next(f for f in data["graph_facts"] if f["name"] == "Brooklyn")
    assert brooklyn["related_to"] is None or brooklyn["relationship_type"] is None, \
        f"Edge not terminated: {brooklyn}"


@pytest.mark.asyncio
async def test_forget_user_memory(client):
    user = "forget_user"
    await post(client, "/companion/facts",
        user_id=user, name="Secret", entity_type="fact")
    await post(client, "/companion/episodes",
        user_id=user, content="A private conversation.")
    await post(client, "/companion/ephemerals",
        user_id=user, description="temporary mood", ttl_seconds=3600)

    data = await get(client, "/companion/context", user_id=user)
    assert len(data["graph_facts"]) >= 1

    await delete(client, f"/companion/memory/{user}")

    data = await get(client, "/companion/context", user_id=user)
    assert len(data["graph_facts"]) == 0
    assert len(data["ephemerals"]) == 0


@pytest.mark.asyncio
async def test_context_relevance_ranking_and_limit(client):
    user = "rank_user"
    await post(client, "/companion/facts",
        user_id=user, name="Photography", entity_type="hobby")
    await post(client, "/companion/facts",
        user_id=user, name="Cooking", entity_type="hobby")
    await post(client, "/companion/facts",
        user_id=user, name="Hiking", entity_type="hobby")

    data = await get(client, "/companion/context", user_id=user, limit=2)
    assert len(data["graph_facts"]) == 2, f"Limit not applied: {data['graph_facts']}"

    ranked = await get(client, "/companion/context", user_id=user, query="outdoors nature trails")
    top = ranked["graph_facts"][0]["name"]
    assert top == "Hiking", f"Relevance ranking failed, top was {top}"
