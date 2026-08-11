from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI
import os
import json

router = APIRouter()

embedding_client = AsyncOpenAI(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
)


class SymbolUpsert(BaseModel):
    project_id: str
    git_branch: str
    file_path: str
    symbol_name: str
    symbol_type: str
    signature: str
    code_content: str


class SymbolSearch(BaseModel):
    project_id: str
    git_branch: str
    query: str
    symbol_type: Optional[str] = None


@router.post("/symbols")
async def upsert_symbol(symbol: SymbolUpsert, request: Request):
    pool = request.app.state.pool

    emb_resp = await embedding_client.embeddings.create(
        input=symbol.code_content, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dev_code_symbols (project_id, git_branch, file_path, symbol_name, symbol_type, signature, code_content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::halfvec)
            """,
            symbol.project_id,
            symbol.git_branch,
            symbol.file_path,
            symbol.symbol_name,
            symbol.symbol_type,
            symbol.signature,
            symbol.code_content,
            json.dumps(embedding),
        )

    return {"status": "stored"}


@router.post("/symbols/search")
async def search_symbols(search: SymbolSearch, request: Request):
    pool = request.app.state.pool

    emb_resp = await embedding_client.embeddings.create(
        input=search.query, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT symbol_name, symbol_type, file_path, signature, code_content,
                   1 - (embedding <=> $1::halfvec) AS similarity
            FROM dev_code_symbols
            WHERE project_id = $2
              AND git_branch = $3
              AND ($4::varchar IS NULL OR symbol_type = $4)
              AND symbol_name % $5
            ORDER BY similarity DESC
            LIMIT 10
            """,
            json.dumps(embedding),
            search.project_id,
            search.git_branch,
            search.symbol_type,
            search.query,
        )

    return [dict(r) for r in rows]
