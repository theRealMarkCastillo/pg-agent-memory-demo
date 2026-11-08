"""Separate extension graph host: python -m uvicorn demo_agents.extensions.app:app."""

import os
import asyncpg
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from memory_engine.auth import authorize, credentials
from memory_engine.db import get_db_pool
from demo_agents.agents.checkpointer import get_checkpointer, close_checkpointer
from .approval import router, build_graph
from .support import router as support_router
from .procedures import router as procedure_router
from .incidents import router as incident_router
from .research import router as research_router
from .analytics import router as analytics_router, open_pools


@asynccontextmanager
async def lifespan(app):
    app.state.credentials = credentials()
    root = Path(os.environ["EXTENSION_WORKSPACE"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    app.state.workspace = root
    app.state.pool = await get_db_pool()
    try:
        saver = await get_checkpointer()
        # Waiting API locks must never consume the connections nodes need to
        # finish the run that releases those locks.
        app.state.approval_pool = await get_db_pool(min_size=1, max_size=5)
        app.state.approval_graph = build_graph(app.state.approval_pool, root, saver)
        app.state.analytics_pools = await open_pools()
        yield
    finally:
        if getattr(app.state, "approval_pool", None) is not None:
            await app.state.approval_pool.close()
        for pool in getattr(app.state, "analytics_pools", {}).values():
            await pool.close()
        await close_checkpointer()
        await app.state.pool.close()


app = FastAPI(title="Verifiable agent extensions", lifespan=lifespan)


@app.exception_handler(asyncpg.UniqueViolationError)
async def duplicate_identifier(request, error):
    return JSONResponse(
        status_code=409,
        content={"detail": "Identifier or source revision already exists"},
    )


app.include_router(router, prefix="/extensions", dependencies=[Depends(authorize)])
app.include_router(
    support_router, prefix="/extensions", dependencies=[Depends(authorize)]
)
app.include_router(
    procedure_router, prefix="/extensions", dependencies=[Depends(authorize)]
)
app.include_router(
    incident_router, prefix="/extensions", dependencies=[Depends(authorize)]
)
app.include_router(
    research_router, prefix="/extensions", dependencies=[Depends(authorize)]
)
app.include_router(
    analytics_router, prefix="/extensions", dependencies=[Depends(authorize)]
)


@app.get("/health")
async def health():
    return {"status": "ok"}
