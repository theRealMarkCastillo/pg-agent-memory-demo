# pg-agent-memory-demo: A Reference Implementation of Agent Memory Patterns

A Docker Compose stack demonstrating six canonical agent memory patterns using **PostgreSQL** (pgvector + pg_trgm), **LangGraph**, and any **OpenAI-compatible API** (OpenAI, Ollama, vLLM, OpenRouter, Gemini).

---

## Academic Context

Modern LLM-based agents face a fundamental challenge: **LLMs are stateless**. Every inference call is independent, bounded by a context window, and loses all information from prior interactions. This creates three problems:

1. **Continuity Gap** — Agents cannot persist context, preferences, or history across sessions.
2. **Knowledge Decay** — Agents cannot learn from prior successes or failures.
3. **Scalability Wall** — Embedding all relevant history in every prompt saturates context windows.

While vector databases (Pinecone, Weaviate, Qdrant) and agent frameworks (LangGraph, CrewAI, AutoGen) have made embedding-based retrieval accessible, **the database is the memory** — and the choice of storage primitives, indexing strategies, and retrieval algorithms fundamentally shapes what an agent can remember and how it can reason.

This project demonstrates that a single PostgreSQL instance, enhanced with pgvector and pg_trgm, can serve as the memory substrate for six fundamentally different agent memory architectures — each with distinct storage schemas, retrieval strategies, and coordination mechanisms.

For an in-depth analysis, see **[GUIDE.md](GUIDE.md)** — a comprehensive taxonomy covering memory architecture, compare/contrast analysis, LangGraph integration, and production considerations.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DOCKER COMPOSE NETWORK                           │
│                                                                          │
│  ┌──────────────────┐      ┌──────────────────┐     ┌─────────────────┐ │
│  │   demo-agents    │      │  memory-engine   │     │    postgres     │ │
│  │   (LangGraph)    │─────>│    (FastAPI)     │────>│  (pgvector +    │ │
│  │                  │      │                  │     │   pg_trgm)      │ │
│  └────────┬─────────┘      └──────────────────┘     └─────────────────┘ │
└───────────┼──────────────────────────────────────────────────────────────┘
            │
            ▼ (External HTTP)
  ┌──────────────────────────────────────────┐
  │  OpenAI-Compatible LLM & Embedding API   │
  └──────────────────────────────────────────┘
```

| Service | Role | Host Port |
|---------|------|-----------|
| `postgres` | PostgreSQL 16 with `vector`, `pg_trgm`, `uuid-ossp` | `5434` |
| `memory-engine` | FastAPI REST API for ingestion, retrieval, and state transitions | `8001` |
| `demo-agents` | LangGraph test runner with 6 agent implementations | — |

---

## Six Memory Patterns

Each pattern addresses a distinct agent archetype with different memory requirements. The following table maps each pattern to its storage primitives, retrieval algorithm, and the cognitive function it analogizes:

| # | Archetype | Memory Type | PostgreSQL Primitive | Retrieval Algorithm | Cognitive Function |
|---|-----------|------------|---------------------|-------------------|-------------------|
| 1 | **Developer Assistant** | Symbolic Code Memory | halfvec HNSW + pg_trgm GIN | Cosine ANN + trigram hybrid | Semantic recall + pattern matching |
| 2 | **Autonomous Task Agent** | Episodic Trajectory Memory | halfvec HNSW + REAL filter | Similarity search with success gate | Experience-based planning |
| 3 | **Enterprise Knowledge Agent** | Policy & Audit Memory | halfvec HNSW + tsvector GIN | Reciprocal Rank Fusion (RRF) | Role-gated knowledge retrieval |
| 4 | **Adaptive Tutor** | Skill Tree Memory | CTE + decay function | Forgetting-curve gap analysis | Spaced repetition modeling |
| 5 | **Multi-Agent Swarm** | Blackboard Memory | FOR UPDATE SKIP LOCKED | Lock-free task claiming | Distributed coordination |
| 6 | **AI Companion** | Relational Identity Memory | Graph + TTL + chunk HNSW | Multi-domain context merge | Autobiographical memory |

For detailed analysis including compare/contrast with alternative approaches, production considerations, and LangGraph integration patterns, see **[GUIDE.md](GUIDE.md)**.

---

## Quick Start

```bash
cp .env.example .env
# Edit .env with your API credentials
docker compose up --build
```

Once running:
- **API docs:** http://localhost:8001/docs
- **Health check:** `curl http://localhost:8001/health`
- **Demo output:** `docker logs -f demo-agent-runner`

The `demo-agents` container seeds test data, then runs all six memory patterns sequentially and prints results.

---

## Configuration

All settings are in `.env`:

```ini
LLM_BASE_URL=https://api.openai.com/v1      # Any OpenAI-compatible endpoint
LLM_API_KEY=sk-...                           # Your API key
LLM_MODEL_NAME=gpt-4o-mini                   # Chat model (or ollama/mistral, etc.)
EMBEDDING_BASE_URL=https://api.openai.com/v1 # May differ from LLM endpoint
EMBEDDING_API_KEY=sk-...                     # Embedding API key
EMBEDDING_MODEL_NAME=text-embedding-3-small  # Must match halfvec dim in init.sql
EMBEDDING_DIM=1536                           # Documentation only — hardcoded in SQL
```

For local models via Ollama:
```ini
LLM_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL_NAME=llama3.1
```

---

## Testing and Evaluation

50 tests across 8 modules validate functional correctness, retrieval quality, and concurrency safety:

```bash
make up && make test        # Start engine + run full suite (~45s)
make test-unit              # Integration tests only (fast)
make test-eval              # Quality evals only
make ci                     # Build + start + test + cleanup
```

| Module | Tests | Validates |
|--------|-------|-----------|
| `test_schema` | 7 | API health, all router endpoints exist |
| `test_developer` | 5 | Semantic search ranking, branch isolation, type filter |
| `test_task` | 5 | Trajectory recall ranking, success_score filter |
| `test_enterprise` | 6 | RBAC enforcement, RRF scoring, FTS matching |
| `test_tutor` | 6 | Skill completeness, decay monotonicity, gap threshold |
| `test_swarm` | 7 | SKIP LOCKED concurrency, task lifecycle, no-double-claim |
| `test_companion` | 6 | Graph facts, TTL expiration, user isolation |
| `test_agent_evals` | 8 | Cross-pattern isolation, 20-agent parallel swarm, temporal correctness |

---

## Project Structure

```
├── docker-compose.yml          # 3-service stack
├── .env.example                # Configuration template
├── Makefile                    # up, down, test, test-eval, ci, clean
├── GUIDE.md                    # Educational deep-dive & taxonomy
├── RUNBOOK.md                  # Operations & troubleshooting
├── postgres/
│   └── init.sql                # 11 tables, HNSW + GIN indexes, extensions
├── memory_engine/
│   ├── Dockerfile
│   ├── main.py                 # FastAPI application
│   ├── db.py                   # asyncpg connection pool
│   ├── embedding.py            # Shared OpenAI embedding client
│   └── routers/
│       ├── developer.py        # Symbol CRUD + hybrid search
│       ├── task_agent.py        # Trajectory store + similarity search
│       ├── enterprise.py       # RRF + RBAC document search
│       ├── tutor.py            # Skill tree + decayed proficiency query
│       ├── swarm.py            # Blackboard task lifecycle + SKIP LOCKED
│       └── companion.py        # Graph facts + ephemerals + episodic chunks
├── demo_agents/
│   ├── Dockerfile
│   ├── main.py                 # Seeds data, runs all 6 demos
│   └── agents/
│       ├── developer_agent.py
│       ├── task_agent.py
│       ├── enterprise_agent.py
│       ├── tutor_agent.py
│       ├── swarm_agent.py
│       └── companion_agent.py
└── tests/
    ├── conftest.py             # Shared fixtures + HTTP helpers
    ├── pytest.ini              # asyncio mode configuration
    └── test_*.py               # 50 tests across 8 modules
```

---

## Limitations and Roadmap

### Current Limitations

- **Agent graphs are linear pipelines** (2 nodes each) — they demonstrate the memory retrieval layer but not advanced LangGraph patterns (conditional routing, loops, sub-agents, Send API)
- **Read-only agents** — agents retrieve memory but never write back; no feedback loops
- **No tool calling** — agents simulate execution via LLM; no real function calls
- **Single-invocation** — each demo runs once; no multi-turn conversation loops
- **No checkpointing** — agent state is not persisted across invocations

### Production-Ready Enhancements

| Enhancement | Patterns Affected | Description |
|-------------|------------------|-------------|
| **Bidirectional memory** | All | Agents write back facts, trajectories, and proficiency updates |
| **Conditional routing** | Enterprise, Tutor | Guard against empty retrievals; route to fallback nodes |
| **Sub-agents + Send API** | Swarm | Compose supervisor + worker subgraphs with parallel fan-out |
| **Checkpointer** | All | Persist conversation state across invocations |
| **Tool calling** | Task, Developer | Execute real HTTP requests, shell commands, code execution |
| **Context window management** | Companion | Rank and prune facts when the prompt approaches token limits |

See **[GUIDE.md](GUIDE.md)** for detailed architectural analysis of each pattern and production guidance.

---

## References

- **pgvector**: https://github.com/pgvector/pgvector
- **LangGraph**: https://github.com/langchain-ai/langgraph
- **HNSW Index**: Malkov & Yashunin (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI.*
- **Reciprocal Rank Fusion**: Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods." *SIGIR.*
- **Ebbinghaus Forgetting Curve**: Ebbinghaus, H. (1885). *Über das Gedächtnis.*
- **SKIP LOCKED**: PostgreSQL 9.5+ documentation. "SELECT … FOR UPDATE SKIP LOCKED."
