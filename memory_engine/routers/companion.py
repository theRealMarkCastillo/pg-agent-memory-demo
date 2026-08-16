from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import os
import json
import re
import unicodedata
from embedding import get_embedding_client

router = APIRouter()


def normalize_name(name: str) -> str:
    """Canonical key for entity resolution: lowercase, strip accents + punctuation."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


_ALIASES = {
    "user": {"user", "me", "i", "myself", "self"},
    "companion": {"companion", "ai", "ai companion", "ai companion character"},
}


def resolve_entity(name: str) -> str:
    """Return a canonical display name for a resolved entity."""
    key = normalize_name(name)
    for canonical, aliases in _ALIASES.items():
        if key in aliases:
            return canonical
    return name


# Canonical relationship predicates (lowercase snake_case). The LLM extractor
# emits free-form relations ("has title", "holds_title", "has alias"); fold the
# noise into a single canonical predicate per relation type.
_RELATION_SYNONYMS = {
    # residence / location
    "lives in": "lives_in",
    "living in": "lives_in",
    "resides in": "lives_in",
    "current location": "lives_in",
    "from": "from",
    "originally from": "from",
    "comes from": "from",
    "moved to": "moved_to",
    "relocated to": "moved_to",
    # occupation
    "has job": "works_as",
    "works as": "works_as",
    "is a": "works_as",
    "is author": "works_as",
    "is a writer": "works_as",
    "writes": "works_as",
    "works at": "works_at",
    "employed at": "works_at",
    # relationships
    "married to": "married_to",
    "parent of": "parent_of",
    "sibling of": "sibling_of",
    "friend of": "friend_of",
    "friends with": "friend_of",
    "has pet": "has_pet",
    "romantic partner of": "romantic_partner_of",
    "is romantically involved with": "romantic_partner_of",
    "romantically involved with": "romantic_partner_of",
    "has romantic feelings for": "romantic_partner_of",
    "in romantic relationship with": "romantic_partner_of",
    "partner of": "romantic_partner_of",
    # preference / affinity
    "like": "likes",
    "love": "loves",
    "dislike": "dislikes",
    "hate": "hates",
    "enjoy": "enjoys",
    "interested in": "interested_in",
    "prefer": "prefers",
    "prefers to be called": "prefers",
    "is passionate about": "values",
    "passionate about": "values",
    "dedicated to": "values",
    "committed to": "values",
    "values authenticity": "values",
    # usage
    "use": "uses",
    "uses for image generation": "uses",
    # identity / naming
    "identifies as": "identifies_as",
    "has title": "identifies_as",
    "holds title": "identifies_as",
    "has alias": "identifies_as",
    "alias of": "identifies_as",
    "also known as": "identifies_as",
    "known as": "identifies_as",
    "has name": "has_name",
    "is named": "has_name",
    # goals / desires
    "wants": "wants_to",
    "want": "wants_to",
    "desires": "wants_to",
    "wants emotional vulnerability": "wants_to",
    "wants emotional vulnerability from": "wants_to",
    "plans to": "plans_to",
    "planning to": "plans_to",
    # traits / skills
    "good at": "skilled_at",
    "skilled at": "skilled_at",
    # spending
    "spends money on": "spends_money_on",
    "pays for": "spends_money_on",
    "willing to spend significant money on": "spends_money_on",
    # hobby
    "has hobby": "has_hobby",
}


def normalize_relation(rel: str | None) -> str | None:
    """Fold a free-form relationship type into its canonical predicate."""
    if not rel:
        return rel
    key = normalize_name(rel)
    if key in _RELATION_SYNONYMS:
        return _RELATION_SYNONYMS[key]
    # fall back to a normalized snake_case form of the original text
    parts = key.split()
    if parts:
        return "_".join(parts)
    return rel


class EpisodeCreate(BaseModel):
    user_id: str
    content: str


class GraphFact(BaseModel):
    user_id: str
    name: str
    entity_type: str
    relationship_to: Optional[str] = None
    relationship_type: Optional[str] = None
    subject: str = "user"  # 'user' | 'self' | 'shared'
    valence: float = 0.0     # -1.0 .. +1.0
    intensity: float = 0.5   # 0.0 .. 1.0
    source_episode_id: Optional[str] = None  # provenance


class BackstoryFact(BaseModel):
    name: str
    entity_type: str = "self"
    relationship_to: Optional[str] = None
    relationship_type: Optional[str] = None
    valence: float = 0.0
    intensity: float = 0.5


class EphemeralCreate(BaseModel):
    user_id: str
    description: str
    ttl_seconds: int = 3600


class FactTerminate(BaseModel):
    user_id: str
    name: str
    relationship_to: Optional[str] = None
    relationship_type: Optional[str] = None
    subject: str = "user"


@router.post("/episodes")
async def create_episode(ep: EpisodeCreate, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO companion_episodes (user_id, content) VALUES ($1, $2) RETURNING episode_id",
            ep.user_id,
            ep.content,
        )
        episode_id = row["episode_id"]

        chunks = _chunk_text(ep.content, chunk_size=500)
        for chunk in chunks:
            emb_resp = await get_embedding_client().embeddings.create(
                input=chunk, model=os.getenv("EMBEDDING_MODEL_NAME")
            )
            embedding = emb_resp.data[0].embedding
            await conn.execute(
                """
                INSERT INTO companion_chunks (episode_id, user_id, content, embedding)
                VALUES ($1, $2, $3, $4::halfvec)
                """,
                episode_id,
                ep.user_id,
                chunk,
                json.dumps(embedding),
            )

    return {"episode_id": str(episode_id), "chunks_stored": len(chunks)}


@router.post("/facts")
async def add_graph_fact(fact: GraphFact, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        await _store_fact_conn(conn, fact)

    return {"status": "stored"}


async def _upsert_node(conn, user_id: str, name: str, entity_type: str, subject: str = "user",
                       existing_name: str | None = None, bump_salience: bool = False) -> dict | None:
    """Insert a node, or merge into an existing node with the same normalized name."""
    key = normalize_name(name)

    # Try to reuse an existing node with the same normalized name (any casing/spelling).
    existing = await conn.fetchrow(
        """
        SELECT node_id, name, entity_type, salience
        FROM companion_graph_nodes
        WHERE user_id = $1 AND normalize_name_key = $2 AND subject = $3
        ORDER BY salience DESC
        LIMIT 1
        """,
        user_id,
        key,
        subject,
    )

    if existing:
        if bump_salience:
            await conn.execute(
                "UPDATE companion_graph_nodes SET salience = salience + 1 WHERE node_id = $1",
                existing["node_id"],
            )
        return {
            "node_id": existing["node_id"],
            "name": existing["name"],
            "entity_type": existing["entity_type"],
            "salience": existing["salience"],
        }

    row = await conn.fetchrow(
        """
        INSERT INTO companion_graph_nodes (user_id, name, entity_type, salience, normalize_name_key, subject)
        VALUES ($1, $2, $3, 1.0, $4, $5)
        ON CONFLICT (user_id, normalize_name_key, subject)
        DO UPDATE SET entity_type = EXCLUDED.entity_type,
                      salience = companion_graph_nodes.salience + 1
        RETURNING node_id, name, entity_type, salience
        """,
        user_id,
        name,
        entity_type,
        key,
        subject,
    )
    if row:
        return dict(row)

    # Fallback: the (name, entity_type, subject) unique constraint may have
    # fired instead (e.g. same name/type under a different normalization). Fetch
    # and bump that node.
    existing = await conn.fetchrow(
        """
        SELECT node_id, name, entity_type, salience
        FROM companion_graph_nodes
        WHERE user_id = $1 AND name = $2 AND entity_type = $3 AND subject = $4
        LIMIT 1
        """,
        user_id,
        name,
        entity_type,
        subject,
    )
    if existing:
        if bump_salience:
            await conn.execute(
                "UPDATE companion_graph_nodes SET salience = salience + 1 WHERE node_id = $1",
                existing["node_id"],
            )
        return {
            "node_id": existing["node_id"],
            "name": existing["name"],
            "entity_type": existing["entity_type"],
            "salience": existing["salience"],
        }
    return None


def _normalize_subject(subject: str) -> str:
    key = subject.strip().lower()
    if key in ("user", "self", "shared"):
        return key
    return "user"


@router.post("/facts/terminate")
async def terminate_relationship(req: FactTerminate, request: Request):
    pool = request.app.state.pool

    name = resolve_entity(req.name)
    relationship_to = resolve_entity(req.relationship_to) if req.relationship_to else None
    relationship_type = normalize_relation(req.relationship_type)
    subject = _normalize_subject(req.subject)

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            """
            UPDATE companion_graph_edges e
            SET status = 'INACTIVE', valid_until = clock_timestamp()
            WHERE e.user_id = $1
              AND e.source_node_id IN (
                  SELECT node_id FROM companion_graph_nodes
                  WHERE user_id = $1 AND normalize_name_key = $2 AND subject = $5
              )
              AND ($3::varchar IS NULL OR e.target_node_id IN (
                  SELECT node_id FROM companion_graph_nodes
                  WHERE user_id = $1 AND normalize_name_key = $3 AND subject = $5
              ))
              AND ($4::varchar IS NULL OR e.relationship_type = $4)
              AND e.subject = $5
              AND e.status = 'ACTIVE'
            RETURNING 1
            """,
            req.user_id,
            normalize_name(name),
            normalize_name(relationship_to) if relationship_to else None,
            relationship_type,
            subject,
        )

    terminated = result is not None
    return {"status": "terminated" if terminated else "no_active_edge"}


@router.delete("/memory/{user_id}")
async def forget_user_memory(user_id: str, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM companion_chunks WHERE user_id = $1", user_id
            )
            await conn.execute(
                "DELETE FROM companion_episodes WHERE user_id = $1", user_id
            )
            await conn.execute(
                "DELETE FROM companion_graph_edges WHERE user_id = $1", user_id
            )
            await conn.execute(
                "DELETE FROM companion_graph_nodes WHERE user_id = $1", user_id
            )
            await conn.execute(
                "DELETE FROM companion_ephemerals WHERE user_id = $1", user_id
            )

    return {"status": "forgotten"}


@router.post("/ephemerals")
async def add_ephemeral(eph: EphemeralCreate, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO companion_ephemerals (user_id, description, expires_at)
            VALUES ($1, $2, clock_timestamp() + ($3 || ' seconds')::interval)
            """,
            eph.user_id,
            eph.description,
            str(eph.ttl_seconds),
        )

    return {"status": "stored"}


@router.get("/context")
async def get_companion_context(
    user_id: str,
    request: Request,
    query: Optional[str] = None,
    limit: Optional[int] = None,
):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        facts = await conn.fetch(
            """
            SELECT n.name, n.entity_type, n.salience, n.subject,
                   e.relationship_type, target.name AS related_to,
                   e.valence, e.intensity, e.source_episode_id
            FROM companion_graph_nodes n
            LEFT JOIN companion_graph_edges e ON n.node_id = e.source_node_id
                AND e.user_id = $1 AND e.status = 'ACTIVE'
                AND (e.valid_until IS NULL OR e.valid_until > clock_timestamp())
            LEFT JOIN companion_graph_nodes target ON e.target_node_id = target.node_id
            WHERE n.user_id = $1
            """,
            user_id,
        )

        ephemerals = await conn.fetch(
            """
            SELECT description, expires_at
            FROM companion_ephemerals
            WHERE user_id = $1 AND expires_at > clock_timestamp()
            ORDER BY expires_at ASC
            """,
            user_id,
        )

    def _fact_dict(r):
        return {
            "name": r["name"],
            "entity_type": r["entity_type"],
            "salience": r["salience"],
            "subject": r["subject"],
            "relationship_type": r["relationship_type"],
            "related_to": r["related_to"],
            "valence": float(r["valence"] or 0.0),
            "intensity": float(r["intensity"] or 0.5),
            "source_episode_id": str(r["source_episode_id"]) if r["source_episode_id"] else None,
        }

    user_facts = [_fact_dict(r) for r in facts if r["subject"] == "user"]
    self_facts = [_fact_dict(r) for r in facts if r["subject"] == "self"]
    shared_facts = [_fact_dict(r) for r in facts if r["subject"] == "shared"]

    # Relevance ranking applies per group, keyed by the query.
    if query:
        if user_facts:
            user_facts = await _rank_facts_by_relevance(query, user_facts)
        if self_facts:
            self_facts = await _rank_facts_by_relevance(query, self_facts)
        if shared_facts:
            shared_facts = await _rank_facts_by_relevance(query, shared_facts)

    if limit is not None and limit > 0:
        user_facts = user_facts[:limit]
        self_facts = self_facts[:limit]
        shared_facts = shared_facts[:limit]

    return {
        "graph_facts": user_facts,          # backward-compatible: facts about the user
        "self_facts": self_facts,           # the companion's model of itself
        "shared_facts": shared_facts,       # relationship facts (growing together)
        "ephemerals": [
            {"description": r["description"], "expires_at": str(r["expires_at"])}
            for r in ephemerals
        ],
    }


@router.get("/facts/provenance")
async def fact_provenance(
    user_id: str,
    request: Request,
    name: str,
    relationship_to: Optional[str] = None,
    relationship_type: Optional[str] = None,
    subject: str = "user",
):
    """Trace a fact back to the episode(s) it was inferred from.

    Matches edges on the (normalized) source entity and returns the source
    episode content so you can see exactly where a memory came from.
    """
    pool = request.app.state.pool
    subj = _normalize_subject(subject)
    rel_key = normalize_name(name)
    target_key = normalize_name(relationship_to) if relationship_to else None
    rel_norm = normalize_relation(relationship_type) if relationship_type else None

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.relationship_type, e.valence, e.intensity, e.status, e.valid_until,
                   ep.episode_id, ep.content AS episode_content, ep.created_at
            FROM companion_graph_edges e
            JOIN companion_graph_nodes s ON s.node_id = e.source_node_id
            LEFT JOIN companion_episodes ep ON ep.episode_id = e.source_episode_id
            WHERE e.user_id = $1
              AND s.normalize_name_key = $2
              AND s.subject = $3
              AND ($4::varchar IS NULL OR e.relationship_type = $4)
              AND ($5::varchar IS NULL OR e.target_node_id IN (
                  SELECT node_id FROM companion_graph_nodes
                  WHERE user_id = $1 AND normalize_name_key = $5
              ))
            ORDER BY e.edge_id
            """,
            user_id,
            rel_key,
            subj,
            rel_norm,
            target_key,
        )

    result = []
    for r in rows:
        result.append({
            "relationship_type": r["relationship_type"],
            "valence": float(r["valence"] or 0.0),
            "intensity": float(r["intensity"] or 0.5),
            "status": r["status"],
            "valid_until": str(r["valid_until"]) if r["valid_until"] else None,
            "source_episode_id": str(r["episode_id"]) if r["episode_id"] else None,
            "source_episode_content": r["episode_content"],
            "inferred_at": str(r["created_at"]) if r["created_at"] else None,
        })
    return {"fact": name, "sources": result}


async def _rank_facts_by_relevance(query: str, facts: list) -> list:
    """Rank graph facts by cosine similarity between the query embedding and each
    fact's name embedding, blended with salience."""
    emb_client = get_embedding_client()

    query_resp = await emb_client.embeddings.create(
        input=query, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    query_vec = query_resp.data[0].embedding

    names = [f["name"] for f in facts]
    name_resp = await emb_client.embeddings.create(
        input=names, model=os.getenv("EMBEDDING_MODEL_NAME")
    )

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    for fact, emb in zip(facts, name_resp.data):
        similarity = _cos(query_vec, emb.embedding)
        fact["relevance"] = round(similarity, 4)
        fact["_score"] = similarity + 0.1 * fact.get("salience", 1.0)

    facts.sort(key=lambda f: f.get("_score", 0.0), reverse=True)
    for f in facts:
        f.pop("_score", None)
    return facts


@router.post("/context/search")
async def search_episodic_context(
    user_id: str, query: str, request: Request, limit: int = 5
):
    pool = request.app.state.pool

    emb_resp = await get_embedding_client().embeddings.create(
        input=query, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content, 1 - (embedding <=> $1::halfvec) AS similarity
            FROM companion_chunks
            WHERE user_id = $2
            ORDER BY embedding <=> $1::halfvec
            LIMIT $3
            """,
            json.dumps(embedding),
            user_id,
            limit,
        )

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Backstory seeding: pre-populate the companion's SELF-model per user so the
# graph starts with a persona/backstory and grows from conversation. Re-runs
# are idempotent (facts upsert, salience bumps).
# ---------------------------------------------------------------------------
class BackstoryRequest(BaseModel):
    user_id: str
    name: str
    backstory: list[BackstoryFact]  # facts about the companion itself (subject='self')
    shared: list[BackstoryFact] = []  # relationship facts (subject='shared')


@router.post("/backstory")
async def seed_backstory(req: BackstoryRequest, request: Request):
    pool = request.app.state.pool
    stored = 0

    async with pool.acquire() as conn:
        # The companion's self-node is created by the first backstory fact; no
        # separate pre-seed needed (avoids conflicting unique constraints).
        for fact in req.backstory:
            f = GraphFact(
                user_id=req.user_id,
                name=fact.name,
                entity_type=fact.entity_type,
                relationship_to=fact.relationship_to,
                relationship_type=fact.relationship_type,
                subject="self",
            )
            await _store_fact_conn(conn, f)
            stored += 1

        for fact in req.shared:
            f = GraphFact(
                user_id=req.user_id,
                name=fact.name,
                entity_type=fact.entity_type,
                relationship_to=fact.relationship_to,
                relationship_type=fact.relationship_type,
                subject="shared",
            )
            await _store_fact_conn(conn, f)
            stored += 1

    return {"status": "seeded", "facts_stored": stored, "companion_name": req.name}


async def _store_fact_conn(conn, fact: GraphFact) -> None:
    """Shared fact-storage helper (add_graph_fact body extracted for reuse)."""
    name = resolve_entity(fact.name)
    relationship_to = resolve_entity(fact.relationship_to) if fact.relationship_to else None
    relationship_type = normalize_relation(fact.relationship_type)
    subject = _normalize_subject(fact.subject)
    valence = max(-1.0, min(1.0, float(fact.valence or 0.0)))
    intensity = max(0.0, min(1.0, float(fact.intensity if fact.intensity is not None else 0.5)))
    source_episode_id = fact.source_episode_id or None

    source = await _upsert_node(
        conn, fact.user_id, name, fact.entity_type, subject=subject,
        existing_name=name, bump_salience=True,
    )
    if relationship_to and relationship_type:
        target = await _upsert_node(
            conn, fact.user_id, relationship_to, "entity", subject=subject,
            existing_name=relationship_to, bump_salience=False,
        )
        if target:
            await conn.execute(
                """
                INSERT INTO companion_graph_edges
                    (user_id, source_node_id, target_node_id, relationship_type, subject, valence, intensity, source_episode_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (user_id, source_node_id, target_node_id, relationship_type, subject)
                DO UPDATE SET status = 'ACTIVE', valid_until = NULL,
                              valence = EXCLUDED.valence,
                              intensity = EXCLUDED.intensity,
                              source_episode_id = COALESCE(EXCLUDED.source_episode_id, companion_graph_edges.source_episode_id)
                """,
                fact.user_id,
                source["node_id"],
                target["node_id"],
                relationship_type,
                subject,
                valence,
                intensity,
                source_episode_id,
            )


def _chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks or [text]
