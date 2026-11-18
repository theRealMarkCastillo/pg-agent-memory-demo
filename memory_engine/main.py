import asyncio
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from db import get_db_pool
from auth import authorize, credentials
from embedding import embedding_provider, get_embedding_client
from fastapi.responses import JSONResponse
from openai import OpenAIError
from routers import developer, task_agent, enterprise, tutor, swarm, companion, hooks


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.credentials = credentials()
    app.state.pool = await get_db_pool()
    if embedding_provider() == "local":
        # Load the model before serving instead of blocking the first request.
        await asyncio.to_thread(get_embedding_client)
    yield
    await app.state.pool.close()


app = FastAPI(title="Unified Agent Memory Engine", lifespan=lifespan)
app.include_router(hooks.router, dependencies=[Depends(authorize)], prefix="/hooks")


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
