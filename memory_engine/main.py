from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from db import get_db_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await get_db_pool()
    yield
    await app.state.pool.close()

app = FastAPI(title="Unified Agent Memory Engine", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

from routers import developer, task_agent, enterprise, tutor, swarm, companion

app.include_router(developer.router, prefix="/developer")
app.include_router(task_agent.router, prefix="/task")
app.include_router(enterprise.router, prefix="/enterprise")
app.include_router(tutor.router, prefix="/tutor")
app.include_router(swarm.router, prefix="/swarm")
app.include_router(companion.router, prefix="/companion")
