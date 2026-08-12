# Runbook

## Prerequisites

- Docker (with Compose v2)
- An OpenAI-compatible API endpoint and key (OpenAI, Ollama, vLLM, OpenRouter, etc.)

## Setup

```bash
cp .env.example .env
```

Edit `.env` and provide your API credentials. The stack works with any OpenAI-compatible provider.

## Launch

```bash
docker compose up --build
```

On first run Docker builds both service images. Subsequent runs are faster — use `docker compose up` without `--build` unless code changes.

## Verify

```bash
curl http://localhost:8001/health
# {"status":"ok"}
```

Open http://localhost:8001/docs for the Swagger UI.

## Run Demos

The demo runner executes automatically on container start. It runs **12 demos**: the 6 base patterns plus 6 multi-turn and real execution demos (shell commands, HTTP requests, file operations). View output:

```bash
docker logs -f demo-agent-runner
```

If the runner exits before the memory engine is ready, restart it:

```bash
docker compose restart demo-agents
```

### Demo Breakdown

| # | Pattern | Type |
|---|---------|------|
| 1 | Developer | Symbol search + write-back |
| 1b | Developer | Multi-turn: writes and executes a Python file |
| 2 | Task | Trajectory recall + real shell execution |
| 2b | Task | Multi-turn: fetches a real URL (httpbin.org) |
| 3 | Enterprise | Role-filtered policy search + write-back |
| 4 | Tutor | Skill gap assessment + progress update |
| 4b | Tutor | Multi-turn: after-learning progress update |
| 5 | Swarm | Supervisor + Send API parallel fan-out (2 workers) |
| 6 | Companion | Relational context recall + write-back |
| 6b | Companion | Multi-turn: remembers previous conversation |
| 6c | Companion | Conflict resolution: auto-terminates stale facts |
| 6d | Companion | Right to forget |

## Tear Down

```bash
docker compose down -v   # removes containers, network, and volume
docker compose down      # keeps the volume (data persists)
```

## Common Issues

### Port conflicts

| Port | Used by | Fix |
|------|---------|-----|
| 5434 | PostgreSQL host port | Edit `docker-compose.yml`, change `${POSTGRES_PORT:-5434}:5432` to a free port |
| 8001 | Memory engine host port | Edit `docker-compose.yml`, change `8001:8000` to a free port |

Then update `POSTGRES_PORT` in `.env` if you changed it.

### Connection refused from demo-agents

The `demo-agents` container depends on `memory-engine` but the FastAPI server may take a few seconds to bind. If you see connection errors, wait 5 seconds and restart:

```bash
docker compose restart demo-agents
```

### pgvector extension not found

Ensure you're using the `pgvector/pgvector:pg16` image (not plain `postgres:16`). The `docker-compose.yml` already references the correct image.

### Checkpointer connection failure

The demo-agents container connects directly to Postgres for the LangGraph checkpointer. If you see `psycopg` connection errors:

```bash
docker compose logs postgres | grep -i "ready to accept"
```

Ensure the postgres container is healthy before starting demo-agents.

### Out of memory during HNSW index builds

Reduce `embedding_dim` in `.env` or lower `maintenance_work_mem` by adding to `postgres` environment:
```yaml
postgres:
  environment:
    PG_MAINTENANCE_WORK_MEM: "64MB"
```

## Schema Reference

All tables are created by `postgres/init.sql` at first launch. Tables per pattern:

| Pattern | Tables |
|---------|--------|
| Developer | `dev_code_symbols` |
| Task | `task_trajectories` |
| Enterprise | `enterprise_documents` |
| Tutor | `tutor_skills`, `tutor_user_progress` |
| Swarm | `swarm_blackboard` |
| Companion | `companion_episodes`, `companion_chunks`, `companion_graph_nodes`, `companion_graph_edges`, `companion_ephemerals` |
| Checkpointer | `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` |

## Ports Reference

| Service | Internal (container) | External (host) |
|---------|---------------------|-----------------|
| PostgreSQL | 5432 | 5434 |
| Memory Engine | 8000 | 8001 |

Inter-service communication uses the internal network (`agent-memory-net` bridge), so only the host ports are configurable.

## Agent Architecture

Five of the six agents use a 3-node LangGraph graph with tool calling:

```
entry node → agent (LLM + bound tools) → [conditional: tool_calls?] → tools → agent → END
```

The Swarm agent uses a supervisor + Send API topology instead:

```
supervisor → [Send per pending task] → worker (claim → execute → complete) → aggregate → END
```

| Pattern | Entry Node | Tools | Checkpointer |
|---------|-----------|-------|-------------|
| Developer | `search_code_symbols` | 5 (search + store + shell + read + write) | Yes |
| Task | `recall_past_trajectories` | 6 (search + store + shell + HTTP + read + write) | Yes |
| Enterprise | `search_policy_docs` | 2 (search + store) | Yes |
| Tutor | `assess_skill_gaps` | 2 (get gaps + update progress) | Yes |
| Swarm | `supervisor` (Send API) | 6 (list + claim-next + claim + complete + shell + HTTP) | Yes |
| Companion | `retrieve_companion_context` | 7 (context + search + episode + fact + ephemeral + terminate + forget) | Yes |

## Useful Database Queries

Exec into the postgres container:

```bash
docker compose exec postgres psql -U agent_user -d agent_memory_db
```

Inspect tables:
```sql
\dt              -- list all tables (14 total: 11 pattern + 3 checkpointer)
SELECT COUNT(*) FROM companion_chunks;
SELECT * FROM swarm_blackboard ORDER BY updated_at DESC LIMIT 5;
SELECT skill_name, decayed_score FROM ( /* paste tutor gap query from init.sql */ );
```

View checkpointed agent state:
```sql
SELECT thread_id, checkpoint_id FROM checkpoints ORDER BY thread_id;
SELECT * FROM checkpoint_blobs WHERE thread_id = 'dev-demo-1';
```

View HNSW index size:
```sql
SELECT pg_size_pretty(pg_relation_size('idx_dev_symbols_hnsw'));
```
