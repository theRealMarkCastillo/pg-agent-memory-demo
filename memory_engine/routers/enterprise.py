from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import os
import json
from embedding import get_embedding_client

router = APIRouter()


class DocumentUpsert(BaseModel):
    doc_title: str
    allowed_role: str
    content: str
    valid_until: Optional[str] = None


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
        await conn.execute(
            """
            INSERT INTO enterprise_documents (doc_title, allowed_role, content, embedding, valid_until)
            VALUES ($1, $2, $3, $4::halfvec, $5::timestamptz)
            """,
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
            SELECT doc_id, doc_title, content,
                   (1 - (embedding <=> $1::halfvec)) AS vec_score,
                   ts_rank(tsv, plainto_tsquery('english', $2)) AS text_score,
                   COALESCE(1 - (embedding <=> $1::halfvec), 0) + COALESCE(ts_rank(tsv, plainto_tsquery('english', $2)), 0) AS rrf_score
            FROM enterprise_documents
            WHERE status = 'ACTIVE'
              AND allowed_role = $3
              AND (valid_until IS NULL OR valid_until > clock_timestamp())
              AND (embedding <=> $1::halfvec < 0.5 OR tsv @@ plainto_tsquery('english', $2))
            ORDER BY rrf_score DESC
            LIMIT 10
            """,
            json.dumps(embedding),
            search.query,
            search.user_role,
        )

    return [dict(r) for r in rows]
