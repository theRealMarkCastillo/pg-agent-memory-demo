from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import json
from embedding import get_embedding_client

router = APIRouter()


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

    emb_resp = await get_embedding_client().embeddings.create(
        input=f"{symbol.symbol_name} {symbol.signature} {symbol.code_content}",
        model=os.getenv("EMBEDDING_MODEL_NAME"),
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,2))",
                json.dumps([symbol.project_id, symbol.git_branch, symbol.file_path]),
            )
            await conn.execute(
                """
                INSERT INTO dev_code_symbols (project_id, git_branch, file_path, symbol_name, symbol_type, signature, code_content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::halfvec)
                ON CONFLICT(project_id,git_branch,file_path,symbol_name,symbol_type)
                DO UPDATE SET signature=EXCLUDED.signature,code_content=EXCLUDED.code_content,
                              embedding=EXCLUDED.embedding,created_at=clock_timestamp()
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

    emb_resp = await get_embedding_client().embeddings.create(
        input=search.query, model=os.getenv("EMBEDDING_MODEL_NAME")
    )
    embedding = emb_resp.data[0].embedding

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH vectors AS MATERIALIZED (
                SELECT symbol_id,embedding <=> $1::halfvec AS distance
                FROM dev_code_symbols
                WHERE project_id=$2 AND git_branch=$3 AND ($4::varchar IS NULL OR symbol_type=$4)
                ORDER BY embedding <=> $1::halfvec LIMIT 50
            ), lexical AS MATERIALIZED (
                SELECT symbol_id,similarity(symbol_name,$5) AS score
                FROM dev_code_symbols
                WHERE project_id=$2 AND git_branch=$3 AND ($4::varchar IS NULL OR symbol_type=$4)
                  AND (symbol_name % $5 OR symbol_name ILIKE $5 || '%')
                ORDER BY score DESC,symbol_id LIMIT 50
            ), ranks AS (
                SELECT symbol_id,1.0/(60+row_number() OVER(ORDER BY distance,symbol_id)) AS score
                FROM vectors WHERE distance <= 0.8
                UNION ALL
                SELECT symbol_id,1.0/(60+row_number() OVER(ORDER BY score DESC,symbol_id)) FROM lexical
            ), fused AS (SELECT symbol_id,sum(score) AS rrf_score FROM ranks GROUP BY symbol_id)
            SELECT d.symbol_name,d.symbol_type,d.file_path,d.signature,d.code_content,
                   1-(d.embedding <=> $1::halfvec) AS similarity,f.rrf_score
            FROM fused f JOIN dev_code_symbols d USING(symbol_id)
            ORDER BY f.rrf_score DESC,d.symbol_id LIMIT 10
            """,
            json.dumps(embedding),
            search.project_id,
            search.git_branch,
            search.symbol_type,
            search.query,
        )

    return [dict(r) for r in rows]


@router.delete("/symbols")
async def delete_symbols(
    project_id: str, git_branch: str, file_path: str, request: Request
):
    """Remove the indexed symbols of a deleted source file within its branch."""
    async with request.app.state.pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM dev_code_symbols WHERE project_id=$1 AND git_branch=$2 AND file_path=$3",
            project_id,
            git_branch,
            file_path,
        )
    return {"status": "deleted"}


class FileSymbol(BaseModel):
    symbol_name: str
    symbol_type: str
    signature: str
    code_content: str


class FileReconcile(BaseModel):
    project_id: str
    git_branch: str
    file_path: str
    symbols: list[FileSymbol]


@router.post("/symbols/reconcile")
async def reconcile_file(body: FileReconcile, request: Request):
    """Atomically replace a file's index, including removal of renamed/deleted symbols."""
    identities = [(s.symbol_name, s.symbol_type) for s in body.symbols]
    if len(set(identities)) != len(identities):
        raise HTTPException(
            422, "Use unique, fully qualified symbol names within a file"
        )
    vectors = []
    if body.symbols:
        response = await get_embedding_client().embeddings.create(
            input=[
                f"{s.symbol_name} {s.signature} {s.code_content}" for s in body.symbols
            ],
            model=os.getenv("EMBEDDING_MODEL_NAME"),
        )
        vectors = sorted(response.data, key=lambda item: item.index)
        if len(vectors) != len(body.symbols):
            raise HTTPException(502, "Incomplete embedding response")
    async with request.app.state.pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,2))",
                json.dumps([body.project_id, body.git_branch, body.file_path]),
            )
            await conn.execute(
                "DELETE FROM dev_code_symbols WHERE project_id=$1 AND git_branch=$2 AND file_path=$3",
                body.project_id,
                body.git_branch,
                body.file_path,
            )
            for symbol, vector in zip(body.symbols, vectors):
                await conn.execute(
                    """INSERT INTO dev_code_symbols(project_id,git_branch,file_path,symbol_name,symbol_type,signature,code_content,embedding)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8::halfvec)""",
                    body.project_id,
                    body.git_branch,
                    body.file_path,
                    symbol.symbol_name,
                    symbol.symbol_type,
                    symbol.signature,
                    symbol.code_content,
                    json.dumps(vector.embedding),
                )
    return {"status": "reconciled", "symbols_stored": len(body.symbols)}
