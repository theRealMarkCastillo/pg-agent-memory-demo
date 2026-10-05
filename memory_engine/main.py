from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from db import get_db_pool
from auth import authorize, credentials
from fastapi.responses import JSONResponse
from openai import OpenAIError
from routers import developer, task_agent, enterprise, tutor, swarm, companion


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.credentials = credentials()
    app.state.pool = await get_db_pool()
    yield
    await app.state.pool.close()


app = FastAPI(title="Unified Agent Memory Engine", lifespan=lifespan)


@app.exception_handler(OpenAIError)
async def embedding_failure(request, exc):
    return JSONResponse(
        status_code=502,
        content={
            "detail": "Embedding provider failed; retry with the same ingestion ID"
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(
    developer.router, dependencies=[Depends(authorize)], prefix="/developer"
)
app.include_router(task_agent.router, dependencies=[Depends(authorize)], prefix="/task")
app.include_router(
    enterprise.router, dependencies=[Depends(authorize)], prefix="/enterprise"
)
app.include_router(tutor.router, dependencies=[Depends(authorize)], prefix="/tutor")
app.include_router(swarm.router, dependencies=[Depends(authorize)], prefix="/swarm")
app.include_router(
    companion.router, dependencies=[Depends(authorize)], prefix="/companion"
)
