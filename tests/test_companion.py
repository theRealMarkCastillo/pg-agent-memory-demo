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


# --- Self-model tests: the companion builds a model of ITSELF (subject='self')
# --- plus shared relationship facts (subject='shared'), separate from user facts.


@pytest.mark.asyncio
async def test_self_facts_stored_and_retrieved(client):
    user = "self_model_user"
    await post(client, "/companion/facts",
        user_id=user, name="Vesper", entity_type="self",
        relationship_to="a quiet coastal town", relationship_type="lives_in",
        subject="self")
    await post(client, "/companion/facts",
        user_id=user, name="Vesper", entity_type="self",
        relationship_to="poetry", relationship_type="writes",
        subject="self")

    data = await get(client, "/companion/context", user_id=user)
    self_names = {f["name"] for f in data["self_facts"]}
    assert "Vesper" in self_names
    assert len(data["self_facts"]) >= 2


@pytest.mark.asyncio
async def test_shared_facts_separate_from_user_facts(client):
    user = "shared_model_user"
    await post(client, "/companion/facts",
        user_id=user, name="Vesper", entity_type="self",
        relationship_to="usr", relationship_type="trusts",
        subject="shared")
    await post(client, "/companion/facts",
        user_id=user, name="usr", entity_type="person",
        relationship_to="Seattle", relationship_type="lives_in",
        subject="user")

    data = await get(client, "/companion/context", user_id=user)
    assert len(data["shared_facts"]) >= 1, "shared_facts missing"
    assert len(data["graph_facts"]) >= 1, "user facts missing"
    shared_types = {f["entity_type"] for f in data["shared_facts"]}
    # self/shared facts must not leak into user facts bucket
    assert "self" not in {f["entity_type"] for f in data["graph_facts"]}


@pytest.mark.asyncio
async def test_subject_isolation_same_name(client):
    """The same entity name can exist under different subjects without merging."""
    user = "subject_iso_user"
    await post(client, "/companion/facts",
        user_id=user, name="Vesper", entity_type="self", subject="self")
    await post(client, "/companion/facts",
        user_id=user, name="Vesper", entity_type="hobby",
        relationship_to="music", relationship_type="loves", subject="shared")

    data = await get(client, "/companion/context", user_id=user)
    self_has = any(f["name"] == "Vesper" for f in data["self_facts"])
    shared_has = any(f["name"] == "Vesper" for f in data["shared_facts"])
    assert self_has and shared_has


@pytest.mark.asyncio
async def test_backstory_seeding(client):
    user = "backstory_user"
    resp = await post(client, "/companion/backstory",
        user_id=user, name="Iris",
        backstory=[
            {"name": "Iris", "entity_type": "self",
             "relationship_to": "a quiet coastal town", "relationship_type": "lives_in"},
            {"name": "Iris", "entity_type": "self",
             "relationship_to": "long conversations", "relationship_type": "values"},
        ],
        shared=[
            {"name": "Iris", "entity_type": "self",
             "relationship_to": user, "relationship_type": "trusts"},
        ])
    assert resp["status"] == "seeded"

    data = await get(client, "/companion/context", user_id=user)
    self_rel = {f["relationship_type"] for f in data["self_facts"] if f["name"] == "Iris"}
    assert "lives_in" in self_rel
    assert "values" in self_rel
    shared_rel = {f["relationship_type"] for f in data["shared_facts"] if f["name"] == "Iris"}
    assert "trusts" in shared_rel


@pytest.mark.asyncio
async def test_backstory_idempotent(client):
    """Re-seeding the same backstory should not error or explode fact counts."""
    user = "backstory_idem"
    body = dict(
        user_id=user, name="Iris",
        backstory=[{"name": "Iris", "entity_type": "self",
                    "relationship_to": "a quiet coastal town", "relationship_type": "lives_in"}],
        shared=[],
    )
    await post(client, "/companion/backstory", **body)
    await post(client, "/companion/backstory", **body)

    data = await get(client, "/companion/context", user_id=user)
    iris = [f for f in data["self_facts"] if f["name"] == "Iris"]
    assert len(iris) == 1, f"Backstory duplicated Iris nodes: {len(iris)}"


# --- Valence / intensity / provenance: emotional weight + fact sourcing.


@pytest.mark.asyncio
async def test_fact_valence_and_intensity_stored(client):
    user = "valence_user"
    await post(client, "/companion/facts",
        user_id=user, name="user", entity_type="person",
        relationship_to="Luna", relationship_type="HAS_PET",
        valence=-0.9, intensity=0.9)

    data = await get(client, "/companion/context", user_id=user)
    luna = next(f for f in data["graph_facts"] if f.get("related_to") == "Luna")
    assert luna["valence"] < -0.5, f"Valence not stored: {luna}"
    assert luna["intensity"] > 0.5, f"Intensity not stored: {luna}"


@pytest.mark.asyncio
async def test_valence_clamped(client):
    user = "valence_clamp"
    await post(client, "/companion/facts",
        user_id=user, name="user", entity_type="person",
        relationship_to="thing", relationship_type="LIKES",
        valence=5.0, intensity=3.0)

    data = await get(client, "/companion/context", user_id=user)
    f = next(x for x in data["graph_facts"] if x.get("related_to") == "thing")
    assert f["valence"] <= 1.0, f"Valence not clamped: {f}"
    assert f["intensity"] <= 1.0, f"Intensity not clamped: {f}"


@pytest.mark.asyncio
async def test_fact_provenance_trace(client):
    user = "prov_user"
    ep = await post(client, "/companion/episodes",
        user_id=user, content="Luna the cat passed away this week. Really hard.")
    ep_id = ep["episode_id"]
    await post(client, "/companion/facts",
        user_id=user, name="user", entity_type="person",
        relationship_to="Luna", relationship_type="HAS_PET",
        valence=-0.9, source_episode_id=ep_id)

    prov = await get(client, "/companion/facts/provenance",
        user_id=user, name="user", relationship_to="Luna",
        relationship_type="HAS_PET")
    assert prov["sources"], "No provenance sources found"
    s = prov["sources"][0]
    assert s["source_episode_id"] == ep_id
    assert "passed away" in s["source_episode_content"]
    assert s["valence"] < -0.5


@pytest.mark.asyncio
async def test_fact_provenance_handles_relation_normalization(client):
    """Provenance query with UPPER_SNAKE rel should match stored lowercase."""
    user = "prov_norm"
    await post(client, "/companion/facts",
        user_id=user, name="user", entity_type="person",
        relationship_to="Sushi", relationship_type="LOVES")
    prov = await get(client, "/companion/facts/provenance",
        user_id=user, name="user", relationship_to="Sushi",
        relationship_type="LOVES")
    assert len(prov["sources"]) >= 1, f"Normalization mismatch: {prov}"


@pytest.mark.asyncio
async def test_fact_without_provenance_returns_empty_sources(client):
    user = "prov_none"
    await post(client, "/companion/facts",
        user_id=user, name="user", entity_type="person",
        relationship_to="Seattle", relationship_type="LIVES_IN")
    prov = await get(client, "/companion/facts/provenance",
        user_id=user, name="user", relationship_to="Seattle",
        relationship_type="LIVES_IN")
    assert len(prov["sources"]) == 1
    assert prov["sources"][0]["source_episode_id"] is None
