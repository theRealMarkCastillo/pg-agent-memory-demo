from fastapi import APIRouter, Request
from pydantic import BaseModel, AwareDatetime
from typing import Optional
import os
import json
from embedding import get_embedding_client

router = APIRouter()


class DocumentUpsert(BaseModel):
    doc_title: str
    allowed_role: str
    content: str
    valid_until: Optional[AwareDatetime] = None


class DocumentSearch(BaseModel):
    query: str
    user_role: str


@router.post("/documents")
async def upsert_document(doc: DocumentUpsert, request: Request):
    pool = request.app.state.pool

    emb_resp = await get_embedding_client().embeddings.create(
        input=doc.content, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,1))",
                json.dumps([doc.allowed_role, doc.doc_title]),
            )
            current = await conn.fetchrow(
                "SELECT * FROM enterprise_documents WHERE doc_title=$1 AND allowed_role=$2 AND status='ACTIVE'",
                doc.doc_title,
                doc.allowed_role,
            )
            if (
                current
                and current["content"] == doc.content
                and current["valid_until"] == doc.valid_until
            ):
                return {"status": "stored"}
            await conn.execute(
                "UPDATE enterprise_documents SET status='ARCHIVED' WHERE doc_title=$1 AND allowed_role=$2 AND status='ACTIVE'",
                doc.doc_title,
                doc.allowed_role,
            )
            await conn.execute(
                """INSERT INTO enterprise_documents(doc_title,allowed_role,content,embedding,valid_until)
                                  VALUES($1,$2,$3,$4::halfvec,$5)""",
                doc.doc_title,
                doc.allowed_role,
                doc.content,
                json.dumps(embedding),
                doc.valid_until,
            )

    return {"status": "stored"}


@router.post("/documents/search")
async def search_documents(search: DocumentSearch, request: Request):
    pool = request.app.state.pool

    emb_resp = await get_embedding_client().embeddings.create(
        input=search.query, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH vectors AS MATERIALIZED (
                SELECT doc_id,embedding <=> $1::halfvec AS distance FROM enterprise_documents
                WHERE status='ACTIVE' AND allowed_role=$3 AND valid_from<=clock_timestamp()
                  AND (valid_until IS NULL OR valid_until>clock_timestamp())
                ORDER BY embedding <=> $1::halfvec LIMIT 50
            ), lexical AS MATERIALIZED (
                SELECT doc_id,ts_rank(tsv,plainto_tsquery('english',$2)) AS score
                FROM enterprise_documents
                WHERE status='ACTIVE' AND allowed_role=$3 AND valid_from<=clock_timestamp()
                  AND (valid_until IS NULL OR valid_until>clock_timestamp())
                  AND tsv @@ plainto_tsquery('english',$2)
                ORDER BY score DESC,doc_id LIMIT 50
            ), ranks AS (
                SELECT doc_id,1.0/(60+row_number() OVER(ORDER BY distance,doc_id)) AS score
                FROM vectors WHERE distance<0.5
                UNION ALL
                SELECT doc_id,1.0/(60+row_number() OVER(ORDER BY score DESC,doc_id)) FROM lexical
            ), fused AS (SELECT doc_id,sum(score) AS rrf_score FROM ranks GROUP BY doc_id)
            SELECT d.doc_id,d.doc_title,d.content,d.allowed_role,f.rrf_score,
                   1-(d.embedding <=> $1::halfvec) AS vec_score,
                   ts_rank(d.tsv,plainto_tsquery('english',$2)) AS text_score
            FROM fused f JOIN enterprise_documents d USING(doc_id)
            ORDER BY f.rrf_score DESC,d.doc_id LIMIT 10
            """,
            json.dumps(embedding),
            search.query,
            search.user_role,
        )

    return [dict(r) for r in rows]


@router.delete("/documents")
async def archive_document(doc_title: str, allowed_role: str, request: Request):
    async with request.app.state.pool.acquire() as conn:
        await conn.execute(
            "UPDATE enterprise_documents SET status='ARCHIVED' WHERE doc_title=$1 AND allowed_role=$2 AND status='ACTIVE'",
            doc_title,
            allowed_role,
        )
    return {"status": "archived"}
