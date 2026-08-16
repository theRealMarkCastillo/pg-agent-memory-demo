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
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DOCKER COMPOSE NETWORK                              │
│                                                                              │
│  ┌───────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │   demo-agents     │◄────►│  memory-engine   │◄────►│    postgres      │  │
│  │   (LangGraph)     │      │    (FastAPI)     │      │  (pgvector +     │  │
│  │  tool calling +   │      │                  │      │   pg_trgm +      │  │
│  │  checkpointer     │      │                  │      │  checkpointer)   │  │
│  └────────┬──────────┘      └──────────────────┘      └──────────────────┘  │
└───────────┼──────────────────────────────────────────────────────────────────┘
            │
            ▼ (External HTTP)
  ┌──────────────────────────────────────────┐
  │  OpenAI-Compatible LLM & Embedding API   │
  └──────────────────────────────────────────┘
```

| Service | Role | Host Port |
|---------|------|-----------|
| `postgres` | PostgreSQL 16 with `vector`, `pg_trgm`, `uuid-ossp` + LangGraph checkpoint tables | `5434` |
| `memory-engine` | FastAPI REST API for ingestion, retrieval, and state transitions | `8001` |
| `demo-agents` | LangGraph agent runner with tool calling, write-back, and checkpointer | — |

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
| 6 | **AI Companion** | Relational Identity Memory | Graph + TTL + chunk HNSW | Multi-domain context merge | Autobiographical memory (user + self + shared) |

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

The `demo-agents` container seeds test data, then runs 12 demos across all six patterns — including multi-turn conversations and real execution (shell commands, HTTP requests, file operations).

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

Optional — LangSmith access for the trace tooling (`tools/`):
```ini
LANGSMITH_API_KEY=ls-...       # smith.langchain.com API key
LANGSMITH_PROJECT=eidolon-prod # Project/session to pull traces from
```

---

## Testing and Evaluation

74 tests across 9 modules validate functional correctness, retrieval quality, concurrency safety, and the companion memory pipeline:

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
| `test_companion` | 20 | Graph facts, self-model, shared facts, backstory, TTL, provenance, valence, user isolation |
| `test_synthetic` | 10 | Synthetic corpus generation, ground-truth matching, LLM extraction (LLM-gated), common-ground detection |
| `test_agent_evals` | 8 | Cross-pattern isolation, 20-agent parallel swarm, temporal correctness |

For testing the companion memory with real or synthetic data, see the **`tools/` pipeline** below.

---

## Project Structure

```
├── docker-compose.yml          # 3-service stack
├── .env.example                # Configuration template
├── Makefile                    # up, down, test, test-eval, ci, clean
├── GUIDE.md                    # Educational deep-dive & taxonomy
├── RUNBOOK.md                  # Operations & troubleshooting
├── postgres/
│   └── init.sql                # 14 tables (11 pattern + 3 checkpointer), HNSW + GIN indexes
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
│       └── companion.py        # Graph facts + self-model + provenance + valence
├── demo_agents/
│   ├── Dockerfile
│   ├── main.py                 # Seeds data, runs 12 demos (6 patterns + multi-turn + supervisor fan-out + companion lifecycle)
│   └── agents/
│       ├── __init__.py
│       ├── checkpointer.py     # Singleton AsyncPostgresSaver (Send-safe subclass)
│       ├── tools.py            # 19 memory tools + 4 real execution tools
│       ├── developer_agent.py  # 3-node graph: search → agent → tools
│       ├── task_agent.py       # 3-node graph: recall → agent → tools
│       ├── enterprise_agent.py # 3-node graph: search → agent → tools
│       ├── tutor_agent.py      # 3-node graph: assess → agent → tools
│       ├── swarm_agent.py      # supervisor + Send API fan-out → workers → aggregate
│       └── companion_agent.py  # retrieve → agent → tools + 3-scope fact extraction node
├── tools/
│   ├── pull_langsmith_traces.py  # Pull top-level LangSmith traces for a project
│   ├── pull_conversations.py     # Paginated pull of conversation traces (resumable)
│   ├── parse_traces.py           # Parse LangGraph / RunnableSequence / MasterGraph traces into turns
│   ├── replay_traces.py          # Replay parsed conversations through companion extraction → engine
│   ├── eval_recall.py            # Query /companion/context + probe queries for recall
│   ├── benchmark_extraction.py   # Score our extractor vs production ground-truth facts
│   ├── generate_synthetic.py     # Generate synthetic conversations + known ground truth
│   └── cleanup_replay.py         # Forget replay/test users from the memory engine
└── tests/
    ├── conftest.py             # Shared fixtures + HTTP helpers
    ├── pytest.ini              # asyncio mode configuration
    └── test_*.py               # 74 tests across 9 modules
```

---

## Companion Memory Design

The Companion pattern (Pattern 6) is the most feature-rich: it models **three subjects** in one graph — the user, the AI companion itself, and their shared relationship — so the memory is balanced and grows together.

### Three-Subject Memory (`subject` column)

Every graph node and edge carries a `subject`: `user`, `self`, or `shared`.

| Subject | What it stores | Example |
|---------|---------------|---------|
| `user` | Durable facts the user revealed | `user LIVES_IN Seattle` |
| `self` | The companion's model of **itself** — extracted from its own outputs | `Vesper VALUES honesty` |
| `shared` | Relationship facts binding both — "growing together" | `Vesper TRUSTS user`, `we SHARE_RITUAL morning coffee` |

`/companion/context` returns all three groups (`graph_facts`, `self_facts`, `shared_facts`), each relevance-ranked. The agent's system prompt is built **dynamically** from the self-model + shared + user facts each turn, so responses depend on memory state rather than a static personality script:

```
About YOU (your self-model, backstory and how you've grown): ...
About your RELATIONSHIP (shared facts, growing together): ...
Things you and the user have in common: pizza, hiking
```

### Backstory Seeding

`POST /companion/backstory` pre-populates the self-model per user (companion name, traits, history) plus initial shared facts. It is idempotent — re-seeding upserts rather than duplicating. The demo seeds Iris's backstory at startup.

### Common-Interest Detection

The retrieve node cross-references `user_facts` against `self_facts` on affinity predicates (`LOVES`, `ENJOYS`, `PLAYS`, `INTERESTED_IN`, ...) with normalized entity matching, and surfaces shared interests in the prompt — so the companion can naturally say *"hey, we both like pizza!"*.

### Emotional Scoring (valence + intensity)

Edges carry `valence` (−1.0 sad/loss … +1.0 joyful) and `intensity` (0.0…1.0), emitted by the extractor and stored. The graph *remembers* emotional weight after the turn ends — e.g. a pet's death is stored as `HAS_PET Luna` with `valence=-0.9`, so future retrievals know to be gentle. Values are clamped server-side.

### Fact Provenance

Edges link to the episode they were inferred from via `source_episode_id`. `GET /companion/facts/provenance?name=...` traces any fact back to its exact source conversation, answering *"where did we learn that?"*. Full-text/semantic search finds *conversations*; provenance finds *which fact came from which message*.

### Entity & Relation Normalization

The engine canonicalizes entities (accent/case/punctuation-insensitive keys + alias map: `me`/`I` → `user`) and relation predicates (UPPER_SNAKE vocabulary aligned with production). Facts fold into existing nodes/edges instead of fragmenting — the same entity under different spellings or the same relation under different phrasings becomes a single canonical row.

---

## Memory Testing Pipeline (`tools/`)

The `tools/` directory provides a full pipeline for testing the companion memory against **real LangSmith traces** or **synthetic data with known ground truth**:

| Tool | Purpose |
|------|---------|
| `pull_langsmith_traces.py` | Pull top-level LangSmith traces for a project |
| `pull_conversations.py` | Paginated, resumable pull of conversation traces |
| `parse_traces.py` | Parse LangGraph / RunnableSequence / MasterGraph traces into `(user, assistant)` turns |
| `replay_traces.py` | Replay parsed conversations through the companion extraction → memory engine |
| `eval_recall.py` | Query `/companion/context` + probe queries to measure recall |
| `benchmark_extraction.py` | Score our extractor against production ground-truth facts (precision/recall/F1) |
| `generate_synthetic.py` | Generate synthetic conversations with **known** ground-truth facts (user/self/shared) |
| `cleanup_replay.py` | Forget replay/test users from the engine |

**Real-trace workflow:**
```bash
python tools/pull_conversations.py --project eidolon-prod --out /tmp/conv
python tools/benchmark_extraction.py --traces-dir /tmp/conv --sample 30
python tools/cleanup_replay.py --all-replay
```

**Synthetic workflow (deterministic, no API):**
```bash
python tools/generate_synthetic.py --count 50 --out traces_synthetic
python tools/benchmark_extraction.py --traces-dir traces_synthetic --all
```

Synthetic data is the primary regression corpus (controlled ground truth, CI-safe); real traces are a periodic reality-check for extractor drift.

---

## Capabilities

This project implements three key agent capabilities on top of the memory engine:

### Tool Calling + Write-Back
All six agents use LangGraph's `ToolNode` with bound `ChatOpenAI` models. Each agent has 2–6 tools including:
- **Memory tools**: search and write-back to the memory engine (e.g., `search_code_symbols` + `store_code_symbol`)
- **Real execution tools**: `execute_shell_command`, `fetch_url`, `read_file`, `write_file` (available to Developer, Task, and Swarm agents)

The LLM decides when to call a tool (search memory, write back a result, execute a command), and the graph loops between the agent and tools nodes until the agent produces a final response.

### Bidirectional Memory
Agents both read from and write to the memory engine. For example:
- **Task Agent**: recalls past trajectories, executes a task, then stores the new trajectory with a success score
- **Swarm Agent**: claims a task from the blackboard, processes it, and marks it complete
- **Companion Agent**: retrieves user context, responds, then extracts and stores **user, self, and shared facts** with emotional valence and provenance

### Checkpointer (State Persistence)
Every agent graph is compiled with a `PostgresSaver` checkpointer backed by a connection pool. Conversation state is persisted to PostgreSQL across invocations via `thread_id` configs. This enables:
- Multi-turn conversations where the LLM sees previous messages
- Fresh memory context injected each turn (updated system prompts)
- Resumable agent runs after failures

---

## Limitations and Roadmap

### Current Limitations

- **Most agent graphs are linear 3-node pipelines** — they use tool calling with conditional routing; only the Swarm pattern uses a supervisor + Send API fan-out topology
- **No conditional routing** — agents don't yet guard against empty retrievals by routing to fallback/deny nodes
- **No context window management** — the Companion agent ranks facts by salience + relevance, but does not yet prune by token budget
- **Tools use synchronous HTTP** — real execution tools block the event loop; fine for a demo, not for production concurrency

### Production-Ready Enhancements

| Enhancement | Patterns Affected | Description | Status |
|-------------|------------------|-------------|--------|
| **Bidirectional memory** | All | Agents write back facts, trajectories, and proficiency updates | Done |
| **Tool calling** | All | Real HTTP requests, shell commands, file operations, code execution | Done |
| **Checkpointer** | All | Persist conversation state across invocations via Postgres | Done |
| **Conditional routing** | Enterprise, Tutor | Guard against empty retrievals; route to fallback nodes | Future |
| **Sub-agents + Send API** | Swarm | Supervisor fans out to parallel worker subgraphs via Send API | Done |
| **Context window management** | Companion | Rank facts by salience + query relevance; prune by limit | Done |
| **AI self-model** | Companion | Companion builds a model of itself (subject=`self`), dynamic persona from memory state | Done |
| **Shared relationship memory** | Companion | `subject=shared` facts capture the growing bond; common-interest surfacing | Done |
| **Emotional scoring** | Companion | `valence`/`intensity` on facts; the graph remembers emotional weight | Done |
| **Fact provenance** | Companion | Edges link to source episodes; trace any fact to where it was learned | Done |
| **Entity/relation normalization** | Companion | Canonical entities + UPPER_SNAKE predicates fold instead of fragmenting | Done |
| **Synthetic + trace evaluation** | Companion | `tools/` pipeline: replay real traces or synthetic ground truth; precision/recall | Done |

See **[GUIDE.md](GUIDE.md)** for detailed architectural analysis of each pattern and production guidance.

---

## References

- **pgvector**: https://github.com/pgvector/pgvector
- **LangGraph**: https://github.com/langchain-ai/langgraph
- **HNSW Index**: Malkov & Yashunin (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI.*
- **Reciprocal Rank Fusion**: Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods." *SIGIR.*
- **Ebbinghaus Forgetting Curve**: Ebbinghaus, H. (1885). *Über das Gedächtnis.*
- **SKIP LOCKED**: PostgreSQL 9.5+ documentation. "SELECT … FOR UPDATE SKIP LOCKED."
