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

Companion graph columns:

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `companion_graph_nodes` | `user_id`, `name`, `entity_type`, `salience`, `normalize_name_key`, `subject` | `subject` ∈ `user`/`self`/`shared`; unique on `(user_id, name, entity_type, subject)` |
| `companion_graph_edges` | `user_id`, `source_node_id`, `target_node_id`, `relationship_type`, `status`, `valid_until`, `subject`, `valence`, `intensity`, `source_episode_id` | `valence` −1…1, `intensity` 0…1; `source_episode_id` = provenance link |
| `companion_episodes` | `episode_id`, `user_id`, `content` | Linked from edges for provenance |

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

The Companion graph has four nodes — the standard `retrieve → agent → tools` loop plus a **3-scope extraction node**:

```
retrieve (user + self + shared facts, common ground) → agent → [tools] → extract (user_facts / self_facts / shared_facts + valence + provenance) → END
```

The extraction node creates the episode first, then posts every fact (user/self/shared) with its subject, emotional `valence`/`intensity`, and a `source_episode_id` so each fact is traceable to where it was learned. A backstory (`POST /companion/backstory`) pre-seeds the self-model per user.

## Companion Memory Tooling (`tools/`)

The `tools/` directory contains a standalone pipeline for testing companion memory with real LangSmith traces or synthetic data. It requires `httpx` + `openai` (a `.venv` at the repo root works):

```bash
python -m venv .venv && .venv/bin/pip install httpx openai asyncpg pytest pytest-asyncio
```

**Synthetic (deterministic, no external API):**
```bash
.venv/bin/python tools/generate_synthetic.py --count 20 --out traces_synthetic
.venv/bin/python tools/benchmark_extraction.py --traces-dir traces_synthetic --all
.venv/bin/python tools/cleanup_replay.py --all-replay
```

**Real LangSmith traces (needs `LANGSMITH_API_KEY` in `.env`):**
```bash
.venv/bin/python tools/pull_conversations.py --project eidolon-prod --out /tmp/conv
.venv/bin/python tools/parse_traces.py /tmp/conv            # inspect parsed turns
.venv/bin/python tools/benchmark_extraction.py --traces-dir /tmp/conv --sample 30
```

Tool overview:

| Tool | Purpose |
|------|---------|
| `pull_langsmith_traces.py` | Pull top-level LangSmith traces for a project |
| `pull_conversations.py` | Paginated, resumable pull of conversation traces (dedupes by trace_id) |
| `parse_traces.py` | Parse LangGraph / RunnableSequence / MasterGraph traces into turns |
| `replay_traces.py` | Replay conversations through companion extraction → memory engine |
| `eval_recall.py` | Probe `/companion/context` for recall |
| `benchmark_extraction.py` | Precision/recall/F1 vs production ground-truth facts |
| `generate_synthetic.py` | Generate synthetic conversations with known ground truth |
| `cleanup_replay.py` | Forget replay/test users |

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

Inspect companion self-model / shared / provenance:
```sql
-- every fact grouped by subject
SELECT subject, count(*) FROM companion_graph_nodes GROUP BY subject;

-- the companion's self-model (subject='self')
SELECT n.name, e.relationship_type, e.valence, e.intensity
FROM companion_graph_nodes n
JOIN companion_graph_edges e ON n.node_id = e.source_node_id
WHERE n.user_id = 'usr_anthony' AND n.subject = 'self' AND e.status = 'ACTIVE';

-- facts with emotional weight
SELECT source.name, e.relationship_type, e.valence, e.intensity
FROM companion_graph_edges e
JOIN companion_graph_nodes source ON source.node_id = e.source_node_id
WHERE e.user_id = 'usr_anthony' AND e.valence < -0.5;

-- provenance: fact → source episode
SELECT s.name, e.relationship_type, ep.content AS source_episode
FROM companion_graph_edges e
JOIN companion_graph_nodes s ON s.node_id = e.source_node_id
LEFT JOIN companion_episodes ep ON ep.episode_id = e.source_episode_id
WHERE e.user_id = 'usr_anthony' AND e.source_episode_id IS NOT NULL;
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
