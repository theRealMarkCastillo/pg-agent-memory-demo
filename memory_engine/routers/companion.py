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
            INSERT INTO companion_graph_nodes (user_id, name, entity_type)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, name, entity_type) DO UPDATE SET entity_type = $3
            RETURNING node_id
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
async def get_companion_context(user_id: str, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        facts = await conn.fetch(
            """
            SELECT n.name, n.entity_type, e.relationship_type, target.name AS related_to
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

    return {
        "graph_facts": [
            {
                "name": r["name"],
                "entity_type": r["entity_type"],
                "relationship_type": r["relationship_type"],
                "related_to": r["related_to"],
            }
            for r in facts
        ],
        "ephemerals": [
            {"description": r["description"], "expires_at": str(r["expires_at"])}
            for r in ephemerals
        ],
    }


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
