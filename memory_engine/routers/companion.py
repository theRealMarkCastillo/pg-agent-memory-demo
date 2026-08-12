from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import os
import json
from embedding import get_embedding_client

router = APIRouter()


class EpisodeCreate(BaseModel):
    user_id: str
    content: str


class GraphFact(BaseModel):
    user_id: str
    name: str
    entity_type: str
    relationship_to: Optional[str] = None
    relationship_type: Optional[str] = None


class EphemeralCreate(BaseModel):
    user_id: str
    description: str
    ttl_seconds: int = 3600


class FactTerminate(BaseModel):
    user_id: str
    name: str
    relationship_to: Optional[str] = None
    relationship_type: Optional[str] = None


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
        source = await conn.fetchrow(
            """
            INSERT INTO companion_graph_nodes (user_id, name, entity_type, salience)
            VALUES ($1, $2, $3, 1.0)
            ON CONFLICT (user_id, name, entity_type)
            DO UPDATE SET entity_type = EXCLUDED.entity_type,
                          salience = companion_graph_nodes.salience + 1
            RETURNING node_id, salience
            """,
            fact.user_id,
            fact.name,
            fact.entity_type,
        )

        if fact.relationship_to and fact.relationship_type:
            target = await conn.fetchrow(
                "SELECT node_id FROM companion_graph_nodes WHERE user_id = $1 AND name = $2",
                fact.user_id,
                fact.relationship_to,
            )
            if not target:
                target = await conn.fetchrow(
                    """
                    INSERT INTO companion_graph_nodes (user_id, name, entity_type)
                    VALUES ($1, $2, 'entity')
                    ON CONFLICT (user_id, name, entity_type) DO NOTHING
                    RETURNING node_id
                    """,
                    fact.user_id,
                    fact.relationship_to,
                )
            if target:
                await conn.execute(
                    """
                    INSERT INTO companion_graph_edges (user_id, source_node_id, target_node_id, relationship_type)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                    """,
                    fact.user_id,
                    source["node_id"],
                    target["node_id"],
                    fact.relationship_type,
                )

    return {"status": "stored"}


@router.post("/facts/terminate")
async def terminate_relationship(req: FactTerminate, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            """
            UPDATE companion_graph_edges e
            SET status = 'INACTIVE', valid_until = clock_timestamp()
            WHERE e.user_id = $1
              AND e.source_node_id IN (
                  SELECT node_id FROM companion_graph_nodes WHERE user_id = $1 AND name = $2
              )
              AND ($3::varchar IS NULL OR e.target_node_id IN (
                  SELECT node_id FROM companion_graph_nodes WHERE user_id = $1 AND name = $3
              ))
              AND ($4::varchar IS NULL OR e.relationship_type = $4)
              AND e.status = 'ACTIVE'
            RETURNING 1
            """,
            req.user_id,
            req.name,
            req.relationship_to,
            req.relationship_type,
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
            SELECT n.name, n.entity_type, n.salience, e.relationship_type, target.name AS related_to
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

    graph_facts = [
        {
            "name": r["name"],
            "entity_type": r["entity_type"],
            "salience": r["salience"],
            "relationship_type": r["relationship_type"],
            "related_to": r["related_to"],
        }
        for r in facts
    ]

    if query and graph_facts:
        graph_facts = await _rank_facts_by_relevance(query, graph_facts)

    if limit is not None and limit > 0:
        graph_facts = graph_facts[:limit]

    return {
        "graph_facts": graph_facts,
        "ephemerals": [
            {"description": r["description"], "expires_at": str(r["expires_at"])}
            for r in ephemerals
        ],
    }


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


def _chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks or [text]
