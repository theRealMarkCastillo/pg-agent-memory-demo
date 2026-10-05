from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import os
import json
import unicodedata
import hashlib
from auth import connection
from embedding import get_embedding_client

router = APIRouter()


def normalize_name(name: str) -> str:
    """Canonical key for entity resolution: lowercase, strip accents + punctuation."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = " ".join("".join(c if c.isalnum() else " " for c in text).split())
    return text.strip()


_ALIASES = {
    "user": {"user", "me", "i", "myself", "self"},
    "companion": {"companion", "ai", "ai companion", "ai companion character"},
}


def resolve_entity(name: str) -> str:
    """Return a canonical display name for a resolved entity."""
    key = normalize_name(name)
    if not key:
        raise HTTPException(422, "Empty canonical entity name")
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
    content: str = Field(min_length=1, max_length=100000)
    ingestion_id: Optional[str] = None


class GraphFact(BaseModel):
    user_id: str
    name: str
    entity_type: str
    relationship_to: Optional[str] = None
    relationship_type: Optional[str] = None
    subject: str = "user"  # 'user' | 'self' | 'shared'
    valence: float = 0.0  # -1.0 .. +1.0
    intensity: float = 0.5  # 0.0 .. 1.0
    source_episode_id: Optional[str] = None  # provenance

    @field_validator("name", "relationship_to")
    @classmethod
    def meaningful_name(cls, value):
        if value is not None and (not normalize_name(value) or len(value) > 100):
            raise ValueError(
                "Entity names must contain letters or numbers and fit in 100 characters"
            )
        return value


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
    ttl_seconds: int = Field(default=3600, ge=1, le=2592000)


class FactTerminate(BaseModel):
    user_id: str
    name: str
    relationship_to: Optional[str] = None
    relationship_type: Optional[str] = None
    subject: str = "user"


@router.post("/episodes")
async def create_episode(ep: EpisodeCreate, request: Request):
    ingestion_id = ep.ingestion_id or hashlib.sha256(ep.content.encode()).hexdigest()
    async with connection(request) as conn:
        existing = await conn.fetchrow(
            "SELECT episode_id,content FROM companion_episodes WHERE user_id=$1 AND ingestion_id=$2",
            ep.user_id,
            ingestion_id,
        )
        if existing:
            if existing["content"] != ep.content:
                raise HTTPException(
                    409, "Ingestion ID already belongs to different content"
                )
            count = await conn.fetchval(
                "SELECT count(*) FROM companion_chunks WHERE episode_id=$1",
                existing["episode_id"],
            )
            return {"episode_id": str(existing["episode_id"]), "chunks_stored": count}
        chunks = _chunk_text(ep.content, chunk_size=500)
        # Complete external work before publishing any part of the episode.
        resp = await get_embedding_client().embeddings.create(
            input=chunks, model=os.getenv("EMBEDDING_MODEL_NAME")
        )
        embeddings = sorted(resp.data, key=lambda item: item.index)
        if len(embeddings) != len(chunks):
            raise HTTPException(502, "Incomplete embedding response")
        async with conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO companion_episodes(user_id,content,ingestion_id) VALUES($1,$2,$3) RETURNING episode_id",
                ep.user_id,
                ep.content,
                ingestion_id,
            )
            episode_id = row["episode_id"]
            for chunk, embedding in zip(chunks, embeddings):
                await conn.execute(
                    "INSERT INTO companion_chunks(episode_id,user_id,content,embedding) VALUES($1,$2,$3,$4::halfvec)",
                    episode_id,
                    ep.user_id,
                    chunk,
                    json.dumps(embedding.embedding),
                )
    return {"episode_id": str(episode_id), "chunks_stored": len(chunks)}


@router.post("/facts")
async def add_graph_fact(fact: GraphFact, request: Request):
    async with connection(request) as conn:
        await _store_fact_conn(conn, fact)

    return {"status": "stored"}


async def _upsert_node(
    conn,
    user_id: str,
    name: str,
    entity_type: str,
    subject: str = "user",
    existing_name: str | None = None,
    bump_salience: bool = False,
) -> dict | None:
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
                "UPDATE companion_graph_nodes SET salience = salience + 1, is_subject=true WHERE node_id = $1",
                existing["node_id"],
            )
        return {
            "node_id": existing["node_id"],
            "name": existing["name"],
            "entity_type": existing["entity_type"],
            "salience": existing["salience"],
        }

    vector = await _embed(name)
    row = await conn.fetchrow(
        """
        INSERT INTO companion_graph_nodes (user_id, name, entity_type, salience, normalize_name_key, subject, embedding, is_subject)
        VALUES ($1, $2, $3, 1.0, $4, $5, $6::halfvec, $7)
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
        vector,
        bump_salience,
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
                "UPDATE companion_graph_nodes SET salience = salience + 1, is_subject=true WHERE node_id = $1",
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
    name = resolve_entity(req.name)
    relationship_to = (
        resolve_entity(req.relationship_to) if req.relationship_to else None
    )
    relationship_type = normalize_relation(req.relationship_type)
    subject = _normalize_subject(req.subject)

    async with connection(request) as conn:
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
    async with connection(request) as conn:
        async with conn.transaction():
            if await conn.fetchval("SELECT to_regclass('checkpoints')"):
                legacy = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM checkpoints WHERE thread_id !~ '^[a-z]+:[0-9a-f]{64}:')"
                )
                if legacy:
                    raise HTTPException(
                        409,
                        "Unowned legacy checkpoints remain; an administrator must migrate or explicitly purge them before forgetting can be guaranteed",
                    )
            await conn.execute(
                "INSERT INTO companion_memory_state(user_id,enabled) VALUES($1,false) ON CONFLICT(user_id) DO UPDATE SET enabled=false,generation=companion_memory_state.generation+1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM companion_graph_edges WHERE user_id=$1", user_id
            )
            await conn.execute(
                "DELETE FROM companion_episodes WHERE user_id=$1", user_id
            )
            await conn.execute(
                "DELETE FROM companion_graph_nodes WHERE user_id=$1", user_id
            )
            await conn.execute(
                "DELETE FROM companion_ephemerals WHERE user_id=$1", user_id
            )
            prefix = "companion:" + hashlib.sha256(user_id.encode()).hexdigest() + ":"
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                if await conn.fetchval("SELECT to_regclass($1)", table):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE left(thread_id,length($1))=$1",
                        prefix,
                    )
    return {"status": "forgotten", "memory_enabled": False}


@router.post("/memory/{user_id}/resume")
async def resume_memory(user_id: str, request: Request):
    """Explicit application/user opt-in. Never exposed as a model tool."""
    async with connection(request) as conn:
        await conn.execute(
            "INSERT INTO companion_memory_state(user_id,enabled) VALUES($1,true) ON CONFLICT(user_id) DO UPDATE SET enabled=true,generation=companion_memory_state.generation+1",
            user_id,
        )
    return {"memory_enabled": True}


@router.post("/ephemerals")
async def add_ephemeral(eph: EphemeralCreate, request: Request):
    async with connection(request) as conn:
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
    limit: int = Query(default=15, ge=1, le=50),
):
    vector = await _embed(query) if query else None
    buckets = {}
    async with connection(request) as conn:
        for subject, key in [
            ("user", "graph_facts"),
            ("self", "self_facts"),
            ("shared", "shared_facts"),
        ]:
            rows = await conn.fetch(
                """
                WITH edge_candidates AS MATERIALIZED (
                    SELECT e.* FROM companion_graph_edges e
                    WHERE e.user_id=$1 AND e.subject=$2 AND e.status='ACTIVE'
                      AND (e.valid_until IS NULL OR e.valid_until>clock_timestamp())
                    ORDER BY e.embedding <=> $3::halfvec NULLS LAST,e.edge_id LIMIT 200
                ), node_candidates AS MATERIALIZED (
                    SELECT n.* FROM companion_graph_nodes n WHERE n.user_id=$1 AND n.subject=$2 AND n.is_subject
                      AND NOT EXISTS(SELECT 1 FROM companion_graph_edges e WHERE e.source_node_id=n.node_id AND e.status='ACTIVE')
                    ORDER BY n.embedding <=> $3::halfvec NULLS LAST,n.node_id LIMIT 100
                ), facts AS (
                    SELECT n.name,n.entity_type,n.salience,n.subject,e.relationship_type,t.name AS related_to,
                           e.valence,e.intensity,e.source_episode_id,1-(e.embedding <=> $3::halfvec) AS relevance
                    FROM edge_candidates e JOIN companion_graph_nodes n ON n.node_id=e.source_node_id
                    JOIN companion_graph_nodes t ON t.node_id=e.target_node_id
                    UNION ALL
                    SELECT name,entity_type,salience,subject,NULL,NULL,0,0.5,NULL,1-(embedding <=> $3::halfvec)
                    FROM node_candidates
                ) SELECT * FROM facts
                ORDER BY COALESCE(relevance,0)+0.01*LEAST(salience,5) DESC,name,relationship_type,related_to LIMIT $4
                """,
                user_id,
                subject,
                vector,
                limit,
            )
            buckets[key] = [dict(r) for r in rows]
        ephs = await conn.fetch(
            "SELECT description,expires_at FROM companion_ephemerals WHERE user_id=$1 AND expires_at>clock_timestamp() ORDER BY expires_at LIMIT 20",
            user_id,
        )
        enabled = await conn.fetchval(
            "SELECT enabled FROM companion_memory_state WHERE user_id=$1", user_id
        )
    return buckets | {
        "ephemerals": [dict(r) for r in ephs],
        "memory_enabled": enabled is not False,
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
    subj = _normalize_subject(subject)
    rel_key = normalize_name(resolve_entity(name))
    target_key = (
        normalize_name(resolve_entity(relationship_to)) if relationship_to else None
    )
    rel_norm = normalize_relation(relationship_type) if relationship_type else None

    async with connection(request) as conn:
        rows = await conn.fetch(
            """
            SELECT e.relationship_type, e.valence, e.intensity, e.status, e.valid_until,
                   ep.episode_id, ep.content AS episode_content, ep.created_at
            FROM companion_graph_edges e
            JOIN companion_graph_nodes s ON s.node_id = e.source_node_id
            LEFT JOIN companion_fact_sources fs ON fs.edge_id=e.edge_id AND fs.user_id=e.user_id
            LEFT JOIN companion_episodes ep ON ep.episode_id=fs.episode_id AND ep.user_id=e.user_id
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
        result.append(
            {
                "relationship_type": r["relationship_type"],
                "valence": float(r["valence"] or 0.0),
                "intensity": float(
                    r["intensity"] if r["intensity"] is not None else 0.5
                ),
                "status": r["status"],
                "valid_until": str(r["valid_until"]) if r["valid_until"] else None,
                "source_episode_id": str(r["episode_id"]) if r["episode_id"] else None,
                "source_episode_content": r["episode_content"],
                "inferred_at": str(r["created_at"]) if r["created_at"] else None,
            }
        )
    return {"fact": name, "sources": result}


async def _embed(text: str) -> str:
    response = await get_embedding_client().embeddings.create(
        input=text, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    return json.dumps(response.data[0].embedding)


@router.post("/context/search")
async def search_episodic_context(
    user_id: str,
    query: str,
    request: Request,
    limit: int = Query(default=5, ge=1, le=50),
):
    emb_resp = await get_embedding_client().embeddings.create(
        input=query, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with connection(request) as conn:
        rows = await conn.fetch(
            """
            SELECT episode_id, content, 1 - (embedding <=> $1::halfvec) AS similarity
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
    stored = 0

    async with connection(request) as conn:
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
                valence=fact.valence,
                intensity=fact.intensity,
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
                valence=fact.valence,
                intensity=fact.intensity,
            )
            await _store_fact_conn(conn, f)
            stored += 1

    return {"status": "seeded", "facts_stored": stored, "companion_name": req.name}


async def _store_fact_conn(conn, fact: GraphFact) -> None:
    """Shared fact-storage helper (add_graph_fact body extracted for reuse)."""
    name = resolve_entity(fact.name)
    relationship_to = (
        resolve_entity(fact.relationship_to) if fact.relationship_to else None
    )
    relationship_type = normalize_relation(fact.relationship_type)
    subject = _normalize_subject(fact.subject)
    valence = max(-1.0, min(1.0, float(fact.valence or 0.0)))
    intensity = max(
        0.0, min(1.0, float(fact.intensity if fact.intensity is not None else 0.5))
    )
    source_episode_id = fact.source_episode_id or None

    if source_episode_id:
        from uuid import UUID

        try:
            episode_uuid = UUID(source_episode_id)
        except ValueError:
            raise HTTPException(422, "Invalid source episode ID")
        if not await conn.fetchval(
            "SELECT 1 FROM companion_episodes WHERE user_id=$1 AND episode_id=$2",
            fact.user_id,
            episode_uuid,
        ):
            raise HTTPException(403, "Source episode does not belong to this user")
    else:
        episode_uuid = None
    vector = (
        await _embed(
            f"{name} {relationship_type.replace(chr(95), chr(32))} {relationship_to}"
        )
        if relationship_to and relationship_type
        else None
    )
    async with conn.transaction():
        source = await _upsert_node(
            conn,
            fact.user_id,
            name,
            fact.entity_type,
            subject=subject,
            bump_salience=True,
        )
        if relationship_to and relationship_type:
            target = await _upsert_node(
                conn, fact.user_id, relationship_to, "entity", subject=subject
            )
            # Preserve prior emotional state as a closed version when it changes.
            await conn.execute(
                """UPDATE companion_graph_edges SET status='INACTIVE',valid_until=clock_timestamp()
                WHERE user_id=$1 AND source_node_id=$2 AND target_node_id=$3 AND relationship_type=$4 AND subject=$5
                  AND status='ACTIVE' AND (valence IS DISTINCT FROM $6 OR intensity IS DISTINCT FROM $7)""",
                fact.user_id,
                source["node_id"],
                target["node_id"],
                relationship_type,
                subject,
                valence,
                intensity,
            )
            row = await conn.fetchrow(
                """INSERT INTO companion_graph_edges
                (user_id,source_node_id,target_node_id,relationship_type,subject,valence,intensity,source_episode_id,embedding)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::halfvec)
                ON CONFLICT(user_id,source_node_id,target_node_id,relationship_type,subject) WHERE status='ACTIVE'
                DO UPDATE SET embedding=EXCLUDED.embedding
                RETURNING edge_id""",
                fact.user_id,
                source["node_id"],
                target["node_id"],
                relationship_type,
                subject,
                valence,
                intensity,
                episode_uuid,
                vector,
            )
            if episode_uuid:
                await conn.execute(
                    "INSERT INTO companion_fact_sources(edge_id,user_id,episode_id) VALUES($1,$2,$3) ON CONFLICT DO NOTHING",
                    row["edge_id"],
                    fact.user_id,
                    episode_uuid,
                )


def _chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks or [text]


@router.get("/memory/{user_id}/state")
async def memory_state(user_id: str, request: Request):
    async with connection(request) as conn:
        row = await conn.fetchrow(
            "SELECT enabled,generation FROM companion_memory_state WHERE user_id=$1",
            user_id,
        )
    return dict(row) if row else {"enabled": True, "generation": 0}


@router.post("/memory/{user_id}/reindex")
async def reindex_memory(user_id: str, request: Request):
    """Backfill persisted assertion vectors after a schema or embedding-model migration."""
    async with connection(request) as conn:
        nodes = await conn.fetch(
            "SELECT node_id,name FROM companion_graph_nodes WHERE user_id=$1", user_id
        )
        edges = await conn.fetch(
            """SELECT e.edge_id,n.name,e.relationship_type,t.name AS target
            FROM companion_graph_edges e JOIN companion_graph_nodes n ON n.node_id=e.source_node_id
            JOIN companion_graph_nodes t ON t.node_id=e.target_node_id WHERE e.user_id=$1""",
            user_id,
        )
        for node in nodes:
            await conn.execute(
                "UPDATE companion_graph_nodes SET embedding=$2::halfvec WHERE node_id=$1",
                node["node_id"],
                await _embed(node["name"]),
            )
        for edge in edges:
            text = f"{edge['name']} {edge['relationship_type'].replace('_', ' ')} {edge['target']}"
            await conn.execute(
                "UPDATE companion_graph_edges SET embedding=$2::halfvec WHERE edge_id=$1",
                edge["edge_id"],
                await _embed(text),
            )
    return {"nodes_indexed": len(nodes), "facts_indexed": len(edges)}


class LegacyPurge(BaseModel):
    confirmation: str


@router.post("/legacy-checkpoints/purge")
async def purge_legacy_checkpoints(body: LegacyPurge, request: Request):
    """Explicit administrative cleanup; deliberately absent from model tools."""
    if not request.state.principal.get("admin"):
        raise HTTPException(403, "Administrator credential required")
    if body.confirmation != "delete unowned checkpoint history":
        raise HTTPException(
            422, "Explicit legacy-history deletion confirmation required"
        )
    async with connection(request) as conn:
        async with conn.transaction():
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                if await conn.fetchval("SELECT to_regclass($1)", table):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id !~ '^[a-z]+:[0-9a-f]{{64}}:'"
                    )
    return {"status": "legacy checkpoints deleted"}
