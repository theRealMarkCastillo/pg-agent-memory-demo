# pg-agent-memory-demo

A Docker Compose stack demonstrating six agent memory patterns using **PostgreSQL** (pgvector + pg_trgm), **LangGraph**, and any **OpenAI-compatible API** (OpenAI, Ollama, vLLM, OpenRouter, Gemini, etc.).

## Overview

Three containerized services running on an isolated bridge network:

| Service | Role | Host Port |
|---------|------|-----------|
| `postgres` | PostgreSQL 16 with `vector`, `pg_trgm`, `uuid-ossp` extensions | `5434` |
| `memory-engine` | FastAPI REST API — DDL, ingestion, state transitions, retrieval queries | `8001` |
| `demo-agents` | LangGraph test runner — 6 agents hitting the memory engine + LLM API | — |

## Six Memory Patterns

| # | Pattern | Postgres Mechanism |
|---|---------|--------------------|
| 1 | **Developer Workspace** | `pg_trgm` GIN indexes + halfvec HNSW on code symbols |
| 2 | **Autonomous Task** | Trajectory similarity search filtered by `success_score` |
| 3 | **Enterprise Knowledge** | Reciprocal Rank Fusion (`pgvector` + `tsvector`) with role-gated access |
| 4 | **Adaptive Tutor** | Skill tree with forgetting-curve decay computed in SQL |
| 5 | **Multi-Agent Swarm** | `FOR UPDATE SKIP LOCKED` on shared blackboard tasks |
| 6 | **AI Companion** | Bitemporal graph facts + TTL ephemerals, status invalidation |

## Quick Start

```bash
cp .env.example .env
# Edit .env with your API credentials
docker compose up --build
```

Once running:
- **API docs:** http://localhost:8001/docs
- **Health check:** `curl http://localhost:8001/health`

The `demo-agents` container seeds test data then runs all six memory patterns and prints results to its logs:
```bash
docker logs -f demo-agent-runner
```

## Configuration

All settings are in `.env`:

```ini
LLM_BASE_URL=https://api.openai.com/v1      # OpenAI-compatible endpoint
LLM_API_KEY=sk-...                           # Your API key
LLM_MODEL_NAME=gpt-4o-mini                   # Chat model
EMBEDDING_BASE_URL=https://api.openai.com/v1 # Embedding endpoint
EMBEDDING_API_KEY=sk-...                     # Embedding API key
EMBEDDING_MODEL_NAME=text-embedding-3-small  # Embedding model
```

Works with any OpenAI-compatible provider. For local models via Ollama:
```ini
LLM_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
```

## Project Structure

```
├── docker-compose.yml
├── .env.example
├── postgres/
│   └── init.sql              # Schema + indexes (11 tables)
├── memory_engine/
│   ├── Dockerfile
│   ├── main.py               # FastAPI app
│   ├── db.py                 # asyncpg pool
│   └── routers/
│       ├── developer.py      # Symbol upsert + pg_trgm search
│       ├── task_agent.py     # Trajectory store + similarity search
│       ├── enterprise.py     # RRF hybrid search
│       ├── tutor.py          # Skill tree + decayed proficiency
│       ├── swarm.py          # Blackboard with SKIP LOCKED
│       └── companion.py      # Graph facts + ephemerals + chunk retrieval
└── demo_agents/
    ├── Dockerfile
    ├── main.py               # Seeds data, runs all 6 demos
    └── agents/
        ├── developer_agent.py
        ├── task_agent.py
        ├── enterprise_agent.py
        ├── tutor_agent.py
        ├── swarm_agent.py
        └── companion_agent.py
```
