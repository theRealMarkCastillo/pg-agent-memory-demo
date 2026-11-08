> Current authentication, migration, deletion, execution, and verification instructions: [REMEDIATION.md](REMEDIATION.md). API requests require a scoped bearer token; health checks remain public on loopback.

# Runbook

## Prerequisites

- Docker (with Compose v2)
- A tool-capable chat endpoint and an embedding endpoint whose output matches `halfvec(1536)`; both must support the client requests used here
- Python 3.10+ for the agent packages (the Docker images use Python 3.12); separate local environments for engine and tests

## Setup

```bash
cp .env.example .env
```

Edit `.env` and provide your API credentials. Compatibility depends on the selected provider and model: verify tool calling, message formats, and embedding dimensions. `EMBEDDING_DIM` documents the schema dimension; it does not alter the database type or configure the embedding request.

## Launch

```bash
python tools/configure_auth.py
make build up
make demo   # runs `make seed` first
```

On first run Docker builds both application images. Use `make up` on later starts and `make build` after code or dependency changes. Make targets load both `.env` and `.env.auth`. For direct Compose commands use `docker compose --env-file .env --env-file .env.auth ...`.

## Verify

```bash
curl http://localhost:8001/health
# {"status":"ok"}
```

Open http://localhost:8001/docs for the Swagger UI.

## Run Demos

The demo agents use LangGraph 1.2.17 and LangChain Core/OpenAI 1.7.x. After updating `demo_agents/requirements.txt`, run `make build` before `make demo`. The PostgreSQL checkpointer applies its schema migrations on startup; keep a backup for valued databases and use fresh thread IDs. See [LangGraph upgrade details and verification limits](GUIDE.md#upgrade-and-verification-boundary). The default graph budget is 32 supersteps; a `GraphRecursionError` means the run exceeded it. Application callers can pass a top-level `recursion_limit` in the invocation config.

`make demo` runs `make seed` first. Seeding forgets and explicitly resumes the demo companion user (Demo 6d disables that user's memory), skips duplicate trajectories, and each run uses fresh checkpoint threads. It creates new swarm task rows; completed rows remain as history, and an interrupted run can leave unfinished tasks for a later run. This supports repeating the demonstration without `make clean`, but it is not a full database reset. The Compose runner leaves shell execution disabled, so the demos verify files with `read_file` instead of running them; configure the local Docker sandbox explicitly for shell execution. See [REMEDIATION.md](REMEDIATION.md) for authentication, migrations, and test environments. It runs **12 scenarios** across six patterns. Output streams to the terminal running `make demo`; that command uses a temporary one-off container removed on exit. To rerun, use `make demo` again after the engine is healthy, rather than restarting a removed container. Seeding resets the companion demo user on each run.

### Demo Breakdown

| # | Pattern | Type |
|---|---------|------|
| 1 | Developer | Symbol search + write-back |
| 1b | Developer | Multi-turn: writes a Python file and reads it back |
| 2 | Task | Trajectory recall (low-score run filtered out) + file write/verify |
| 2b | Task | Multi-turn: fetches a real URL (httpbin.org) |
| 3 | Enterprise | Role-filtered policy search + write-back |
| 4 | Tutor | Skill gap assessment + progress update |
| 4b | Tutor | Multi-turn: after-learning progress update |
| 5 | Swarm | Pull mode: parallel workers race `claim_next_task`; SKIP LOCKED gives each a different task |
| 6 | Companion | Relational context recall + write-back |
| 6b | Companion | Multi-turn: remembers previous conversation |
| 6c | Companion | Conflict resolution: `lives_in Manhattan` supersedes `lives_in Brooklyn` in the engine |
| 6d | Companion | Right to forget: deletes companion records/checkpoints for the user and disables companion recording until resumed |

## Tear Down

```bash
make down    # preserves data
make clean   # explicitly removes the database volume
```

## Common Issues

### Port conflicts

| Port | Used by | Fix |
|------|---------|-----|
| 5434 | PostgreSQL host port | Edit `docker-compose.yml`, change `${POSTGRES_PORT:-5434}:5432` to a free port |
| 8001 | Memory engine host port | Edit `docker-compose.yml`, change `8001:8000` to a free port |
| 8002 | Optional extension host on loopback | Choose another `uvicorn --port` and update extension clients; this host is not a Compose service |

Then update `POSTGRES_PORT` in `.env` if you changed it.

### Connection refused from demo-agents

The `demo-agents` container depends on `memory-engine`, but dependency ordering is not application readiness. Confirm the engine is healthy, inspect its logs, and rerun the one-off demo:

```bash
curl --fail http://localhost:8001/health
docker compose --env-file .env --env-file .env.auth logs memory-engine
make demo
```

### pgvector extension not found

Ensure you're using the `pgvector/pgvector:pg16` image (not plain `postgres:16`). The `docker-compose.yml` already references the correct image.

### Checkpointer connection failure

The demo-agents container connects directly to Postgres for the LangGraph checkpointer. If you see `psycopg` connection errors:

```bash
docker compose --env-file .env --env-file .env.auth logs postgres
```

Ensure the postgres container is healthy before starting demo-agents.

### Out of memory during HNSW index builds

Inspect the index build and database memory budget first. PostgreSQL server settings must be configured as server settings, not arbitrary environment variable names. For example, merge this into the existing service definition:

```yaml
services:
  postgres:
    command: ["postgres", "-c", "maintenance_work_mem=64MB"]
```

This changes the server's maintenance memory allowance and may slow index construction; measure the result. Changing only `EMBEDDING_DIM` has no effect on `halfvec(1536)`. Changing embedding dimensions requires a coordinated schema/index migration, matching embeddings, and reingestion/reindexing. See [embedding-space operating contracts](REMEDIATION.md).

## Schema Reference

Domain tables start in `postgres/init.sql`, with versioned upgrades in `memory_engine/migrations/`. The hook integration adds `agent_hook_memories` through its migration. LangGraph checkpoint tables and checkpoint migration metadata are managed by the saver's `setup()`, not by `init.sql`. Main tables per pattern:

| Pattern | Tables |
|---------|--------|
| Developer | `dev_code_symbols` |
| Task | `task_trajectories` |
| Enterprise | `enterprise_documents` |
| Tutor | `tutor_skills`, `tutor_user_progress` |
| Swarm | `swarm_blackboard` |
| Companion | `companion_memory_state`, `companion_episodes`, `companion_chunks`, `companion_graph_nodes`, `companion_graph_edges`, `companion_fact_sources`, `companion_ephemerals` |
| Agent hooks | `agent_hook_memories` (migration-managed) |
| Approval and support | `ext_proposals`, `ext_proposal_revisions`, `ext_reviews`, `ext_policies`, `ext_assertions` |
| Procedures and incidents | `ext_attempts`, `ext_incident_events`, `ext_incident_summaries` |
| Research | `ext_research_documents`, `ext_research_reports` |
| Analytics | `ext_analytics_identities`, `ext_customers`, `ext_orders`, `ext_analytics_runs` |
| Checkpointer | `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` (saver-managed; migration metadata omitted) |

Migrations `004`–`010` add the extensions. `009` requires role-creating database privileges; `010` adds durable approval intent. Stop older extension hosts before upgrading and use the [startup/upgrade procedure](REMEDIATION.md#start-or-upgrade). The local stack uses an administrator for migrations and application access; privilege separation remains a deployment task.

Companion graph columns:

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `companion_memory_state` | `user_id`, `enabled`, `generation`, `embedding_space` | Controls right-to-forget fencing and explicit opt-in resume |
| `companion_graph_nodes` | `user_id`, `name`, `entity_type`, `salience`, `normalize_name_key`, `subject`, `embedding`, `is_subject` | `subject` ∈ `user`/`self`/`shared`; unique on `(user_id, normalize_name_key, subject)` |
| `companion_graph_edges` | `user_id`, `source_node_id`, `target_node_id`, `relationship_type`, `status`, `valid_from`, `valid_until`, `recorded_at`, `subject`, `valence`, `intensity`, `source_episode_id`, `embedding` | `valence` −1…1, `intensity` 0…1; partial unique index on active edges |
| `companion_fact_sources` | `edge_id`, `user_id`, `episode_id` | Multi-episode provenance join table |
| `companion_episodes` | `episode_id`, `user_id`, `content`, `ingestion_id` | Idempotent episode ingestion linked from edges for provenance |

## Ports Reference

| Service | Internal (container) | External (host) |
|---------|---------------------|-----------------|
| PostgreSQL | 5432 | 5434 |
| Memory Engine | 8000 | 8001 |

Inter-service communication uses the internal network (`agent-memory-net` bridge), so only the host ports are configurable.

## Agent Architecture

Four agents use three nodes with a tool-calling loop. The Companion adds an extraction node after its final answer; forgetting exits before extraction:

```
entry node → agent (LLM + bound tools) → [conditional: tool_calls?] → tools → agent → END
```

The Swarm agent uses a supervisor + Send API topology instead:

```
supervisor → [Send per pending task] → worker (claim → execute → complete) → aggregate → END
```

With `claim_mode="assigned"` the supervisor pins each worker to a task ID. With `claim_mode="pull"` (used by Demo 5) workers get no task ID and call the claim-next endpoint, so `FOR UPDATE SKIP LOCKED` decides who gets which row. In both modes application code claims before invoking the model, holds the lease token, and validates it before each tool dispatch; the LLM never copies it.

| Pattern | Entry Node | Tools | Checkpointer |
|---------|-----------|-------|-------------|
| Developer | `search_code_symbols` | 5 (search + store + shell + read + write) | Yes |
| Task | `recall_past_trajectories` | 6 (search + store + shell + HTTP + read + write) | Yes |
| Enterprise | `search_policy_docs` | 2 (search + store) | Yes |
| Tutor | `assess_skill_gaps` | 2 (get gaps + update progress) | Yes |
| Swarm | `supervisor` (Send API) | 4 model tools (list + complete + shell + HTTP); application claims | Yes |
| Companion | `retrieve_companion_context` | 7 (context + search + episode + fact + ephemeral + terminate + forget) | Yes |

The Companion graph has four nodes — the standard `retrieve → agent → tools` loop plus a **3-scope extraction node**:

```
retrieve (user + self + shared facts, common ground) → agent → [tools] → extract (user_facts / self_facts / shared_facts + valence + provenance) → END
```

The extraction node creates the episode first, then posts every fact (user/self/shared) with its subject, emotional `valence`/`intensity`, and a `source_episode_id` so each fact is traceable to where it was recorded. A backstory (`POST /companion/backstory`) pre-seeds the self-model per user.

## Companion Memory Tooling (`tools/`)

Use the pinned `.venv-tests` environment from [REMEDIATION.md](REMEDIATION.md#verification). Separate these operations by their effects:

**Offline generation and parsing:**

```bash
.venv-tests/bin/python tools/generate_synthetic.py --count 20 --out /tmp/pg-memory-synthetic
.venv-tests/bin/python tools/parse_traces.py /tmp/pg-memory-synthetic
```

**Model-dependent extraction benchmark:** this reads the configured model settings from `.env`, calls the provider, and scores against labels. It does not replay records into the memory engine, so it does not require database cleanup.

```bash
.venv-tests/bin/python tools/benchmark_extraction.py --traces-dir /tmp/pg-memory-synthetic --all
```

**Trace export and inspection:** configure the LangSmith API key/project in `.env`. Exported conversations may contain personal or proprietary data; use a permitted dataset and apply its retention policy.

```bash
.venv-tests/bin/python tools/pull_conversations.py --out /tmp/pg-memory-traces
.venv-tests/bin/python tools/parse_traces.py /tmp/pg-memory-traces
.venv-tests/bin/python tools/benchmark_extraction.py --traces-dir /tmp/pg-memory-traces --sample 30
```

**Optional replay, retrieval inspection, and deletion:** these contact the engine. Export an application-authorized `MEMORY_API_TOKEN` for the specific replay user; the default demo token does not authorize arbitrary users. On the host, export `MEMORY_ENGINE_URL=http://localhost:8001`, rather than using the inter-container hostname from `.env`. The model settings still come from `.env`.

```bash
.venv-tests/bin/python tools/replay_traces.py --traces-dir /tmp/pg-memory-synthetic --user-id trace-replay-workshop --max-turns 3
.venv-tests/bin/python tools/eval_recall.py --user-id trace-replay-workshop
.venv-tests/bin/python tools/cleanup_replay.py --user-id trace-replay-workshop
```

Unlabeled recall probes only inspect results. To measure recall, supply labeled probes as described in [verification](REMEDIATION.md#verification). Cleanup deletes this user's companion records/checkpoints and disables companion recording. Extension records/checkpoints and files, hook excerpts, exported files, backups, and provider-held copies require their own lifecycle handling. Trace-derived labels can measure model agreement rather than verified truth.

Tool overview:

| Tool | Purpose |
|------|---------|
| `pull_langsmith_traces.py` | Pull top-level LangSmith traces for a project |
| `pull_conversations.py` | Paginated, resumable pull of conversation traces (dedupes by trace_id) |
| `parse_traces.py` | Parse LangGraph / RunnableSequence / MasterGraph traces into turns |
| `replay_traces.py` | Replay conversations through companion extraction → memory engine |
| `eval_recall.py` | Probe `/companion/context` for recall |
| `benchmark_extraction.py` | Precision/recall/F1 against supplied labels; inspect label origin |
| `generate_synthetic.py` | Generate synthetic conversations with controlled labels |
| `cleanup_replay.py` | Forget replay/test users |

## Useful Database Queries

These queries are for a trusted local operator with database access. They bypass API-level authorization and must not become model tools. Use your configured database user/name if you changed the defaults:

```bash
docker compose --env-file .env --env-file .env.auth exec postgres psql -U agent_user -d agent_memory_db
```

Inspect tables (the set varies with migrations and saver initialization):

```sql
\dt
SELECT COUNT(*) FROM companion_chunks;
SELECT * FROM swarm_blackboard ORDER BY updated_at DESC LIMIT 5;
SELECT s.skill_name, p.proficiency_score, p.last_reviewed_at
FROM tutor_user_progress p
JOIN tutor_skills s ON s.skill_id = p.skill_id
WHERE p.user_id = 'learner_001';
```

Inspect companion persona assertions and provenance. Demo 6d deletes `usr_anthony`'s companion records, so these return no companion records after a completed run unless you reseed:

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

Inspect checkpoint identifiers and blob sizes without dumping conversation content. Thread IDs are namespaced and scope-hashed, so the raw demo label is not the database key:

```sql
SELECT thread_id, checkpoint_id FROM checkpoints ORDER BY thread_id;
SELECT thread_id, channel, octet_length(blob) AS bytes
FROM checkpoint_blobs WHERE thread_id LIKE 'developer:%' LIMIT 10;
```

View HNSW index size:

```sql
SELECT pg_size_pretty(pg_relation_size('idx_dev_symbols_hnsw'));
```


## Teaching and extension material

Use the [workshop path](GUIDE.md#workshop-path) to connect each demo to its source, expected observation, and failure case. The [extension curriculum](USE_CASES.md) now contains six runnable modules. Follow [EXTENSIONS.md](EXTENSIONS.md) for disposable demonstrations, service setup, scoped credentials, restricted analytics roles, and optional live evaluation. Approval/resume is a separate service; the original scoped wrapper continues to accept new turns. Its local-file recovery proof does not establish exactly-once remote effects. The [final review](DESIGN_REVIEW.md) records 207 passing tests and the design boundaries of the reviewed working tree.

If a restarted proposal remains `applying`, use its authorized `/run` endpoint and the [recovery table](EXTENSIONS.md#recovery-and-limits). Edits/cancellation return 409 once an effect has started. Preserve the proposal when a later file write or stale evidence prevents automatic reconciliation; do not reset its status to conceal the attempt. Research reports created before `complete-labelled-evidence/2` must be regenerated under a new report ID and reviewed before trusted recall.
