# Agent Memory Patterns: A Taxonomy and Implementation Guide

## Table of Contents
1. [The Memory Problem in AI Agents](#the-memory-problem-in-ai-agents)
2. [A Taxonomy of Agent Memory](#a-taxonomy-of-agent-memory)
3. [Pattern 1: Developer Workspace Memory](#pattern-1-developer-workspace-memory)
4. [Pattern 2: Autonomous Task Memory](#pattern-2-autonomous-task-memory)
5. [Pattern 3: Enterprise Knowledge Memory](#pattern-3-enterprise-knowledge-memory)
6. [Pattern 4: Adaptive Tutor Memory](#pattern-4-adaptive-tutor-memory)
7. [Pattern 5: Multi-Agent Swarm Memory](#pattern-5-multi-agent-swarm-memory)
8. [Pattern 6: Relational Companion Memory](#pattern-6-relational-companion-memory)
9. [Comparative Analysis](#comparative-analysis)
10. [LangGraph Integration Analysis](#langgraph-integration-analysis)
11. [Going Further: Production Considerations](#going-further-production-considerations)

---

## The Memory Problem in AI Agents

### Why Agent Memory Matters

Modern AI agents face a fundamental challenge: LLMs are stateless. Every inference call is independent, bounded by a context window (typically 4K–128K tokens), and loses all information from previous interactions. This creates three problems:

1. **Continuity Gap**: Agents cannot persist user context, preferences, or conversation history across sessions.
2. **Knowledge Decay**: Agents cannot remember what they learned from prior successes or failures.
3. **Scalability Wall**: Embedding all relevant history in every prompt saturates context windows and degrades output quality with noise.

Agent memory systems address these by externalizing state into a persistent store and providing retrieval mechanisms that return precisely the information needed for the current interaction. This project implements six distinct memory patterns, each addressing a different agent archetype's memory requirements.

### What This Project Is — And What It Is Not

**This project is** a reference implementation demonstrating how the same underlying infrastructure (PostgreSQL with pgvector and pg_trgm) can support six fundamentally different memory architectures. It runs as a self-contained Docker stack that you can inspect, modify, and benchmark. Agents use **tool calling** (LangGraph `ToolNode`) for both retrieving and writing back to memory, **real execution tools** (shell, HTTP, file operations), and a **Postgres checkpointer** for persistent multi-turn conversations.

**This project is not** a production-grade agent framework. The LangGraph agents are intentionally focused on memory patterns — most use a 3-node tool-calling graph (retrieve → agent → tools) rather than deep hierarchies, and the Companion adds a fourth 3-scope extraction node. A production system would add sub-agent orchestration (supervisor/worker with Send API), context window management, and conditional routing for edge cases.

---

## A Taxonomy of Agent Memory

Agent memory can be categorized along two axes: **temporal scope** and **structural complexity**.

| Temporal Scope | Description | Examples |
|----------------|-------------|----------|
| **Working** | Transient, conversation-scoped | Current task, recent messages |
| **Short-term** | Session-scoped, expires | Ephemeral moods, active blackboard tasks |
| **Long-term** | Persistent, accumulates over time | User facts, learned skills, codebase symbols |
| **Archival** | Immutable, auditable | Policy documents, trajectory logs |

| Structural Complexity | Description | Examples |
|------------------------|-------------|----------|
| **Flat** | Key-value or unlabeled chunks | Episodic text chunks, raw vectors |
| **Graph** | Entities with typed relationships | Knowledge graphs, skill trees |
| **Temporal** | Time-versioned or decaying | Forgetting curves, bitemporal facts |
| **Access-controlled** | Role-gated with audit trails | Enterprise policies, RBAC documents |

The six patterns in this project span this taxonomy:

| Pattern | Temporal Scope | Structural Complexity | Primary DB Mechanism |
|---------|---------------|----------------------|---------------------|
| Developer Workspace | Long-term | Flat + Vector | HNSW + pg_trgm |
| Autonomous Task | Long-term + Archival | Flat + Metadata | HNSW + score filter |
| Enterprise Knowledge | Long-term + Archival | Access-controlled + Flat | RRF (vector + FTS) + role gate |
| Adaptive Tutor | Long-term | Graph + Temporal | Skill tree + decay SQL |
| Multi-Agent Swarm | Short-term | Flat + Locked | FOR UPDATE SKIP LOCKED |
| AI Companion | Long-term + Short-term | Graph + Temporal | Bitemporal facts + TTL |

---

## Pattern 1: Developer Workspace Memory

### Use Case
A coding assistant that understands a codebase at the symbol level — functions, classes, methods, their signatures, and their semantic purpose. It must retrieve relevant code across branches and projects, tolerate partial queries and typos, and rank results by relevance.

### Memory Architecture

**Storage**: The `dev_code_symbols` table stores each code symbol as a row with project/git-branch metadata, a human-readable signature, and the symbol's semantic description encoded as a `halfvec(1536)` embedding.

**Retrieval Strategy**: Hybrid search combining:
- **Cosine distance** via HNSW index on the `embedding` halfvec column (semantic similarity)
- **Trigram similarity** via GIN index on `symbol_name` (fuzzy name matching for typos like "recal" → "recall_similar")

**Filtering**: Project + branch isolation ensures the agent doesn't cross-contaminate between different git repositories or feature branches.

**Why SQL and not a vector database?** For developer tools, the metadata filtering (project, branch, symbol type) is as important as the vector search. PostgreSQL lets you combine `WHERE project_id = $1 AND git_branch = $2` with `ORDER BY embedding <=> $3` in a single query, eliminating the need for a separate metadata store.

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **pg_trgm + HNSW (this project)** | Combined symbolic + semantic search in one query; no external service | Trgm threshold must be tuned per use case; halfvec trades precision for speed | Small-to-medium codebases (<10K symbols) where SQL-native ops suffice |
| **Dedicated vector DB (Pinecone/Weaviate/Qdrant)** | Better recall at scale (>1M vectors); built-in re-ranking | Separate service adds complexity; metadata filtering can be limited | Large monorepos with millions of symbols |
| **Full LLM context dump** | Zero infrastructure; trivial to implement | Context window exhaustion; high token costs; no persistence across sessions | Prototypes and throwaway scripts |
| **RAG with chunked source files** | Captures surrounding context; good for "how does this work" questions | Poor for "which function does X" type symbol-level queries | Documentation Q&A, code explanation |

### Agent Implementation

The LangGraph agent uses a **3-node tool-calling graph**: `search_code_symbols` → `agent` → `tools` (with conditional loop back to agent). The search node retrieves relevant symbols, the agent node invokes the LLM with bound tools, and the tools node executes tool calls. The LLM can choose to:

- Answer directly based on retrieved symbols
- Call `search_code_symbols` again with different parameters
- Call `store_code_symbol` to persist a new symbol (write-back)
- Call `read_file`, `write_file`, or `execute_shell_command` for real file and shell operations

**Tools available**: `search_code_symbols`, `store_code_symbol`, `execute_shell_command`, `read_file`, `write_file`

**What changed from the original**: The agent now supports bidirectional memory (write-back), real file/shell execution, and multi-turn conversations via the Postgres checkpointer.

---

## Pattern 2: Autonomous Task Memory

### Use Case
An agent that autonomously executes multi-step tasks (web scraping, report generation, data pipeline orchestration). When given a new goal, it should recall how it succeeded at similar goals in the past, adapt those strategies, and actually execute them.

### Memory Architecture

**Storage**: The `task_trajectories` table logs completed task executions as serialized action sequences (`JSONB`), the original goal description, an embedding of the goal for similarity search, the execution result, and a success score (0.0–1.0).

**Retrieval Strategy**: Semantic similarity search on goal embeddings with a minimum success score filter (`WHERE success_score >= 0.7`). This ensures the agent learns from successful trajectories and ignores failures. If a new goal is "scrape product prices from competitor site," it will retrieve the trajectory of the last successful web scraping task as a few-shot example.

**Why success_score filtering matters**: Without it, the agent would retrieve the most similar trajectory regardless of outcome — potentially learning from failures. The score filter creates a "quality gate" that only surfaces proven strategies.

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **Trajectory recall + real tools (this project)** | Learns from past successes; actually executes tasks via shell/HTTP | Requires manual or programmatic scoring; shell access adds security concerns | Structured tasks with measurable outcomes |
| **ReAct / Tool-using agents** | Dynamic tool selection; no historical dependency | No learning across episodes; each task starts from scratch | Tasks requiring diverse tool chains |
| **Fine-tuned agent models** | Compresses experience into model weights; fast inference | Expensive to retrain; catastrophic forgetting | High-volume, repetitive task domains |
| **Reflexion / self-critique loops** | Self-improving via verbal feedback; no external scoring needed | Can amplify biases; requires strong base model | OpenAI/exploratory domains without clear success metrics |

### Agent Implementation

The LangGraph agent uses a **3-node tool-calling graph**: `recall_past_trajectories` → `agent` → `tools`. The recall node searches for similar successful trajectories, the agent node plans and decides on tools, and the tools node executes. The LLM can:

- Call `search_trajectories` again if more history is needed
- Call `execute_shell_command` to run data processing scripts
- Call `fetch_url` to make real HTTP requests (e.g., scraping, API calls)
- Call `read_file` / `write_file` for data I/O
- Call `store_trajectory` to record the completed task in memory for future recall

**Tools available**: `search_trajectories`, `store_trajectory`, `execute_shell_command`, `fetch_url`, `read_file`, `write_file`

**What changed from the original**: The agent no longer simulates execution via the LLM. It has real execution tools and writes completed trajectories back to the memory engine, creating a genuine feedback loop.

---

## Pattern 3: Enterprise Knowledge Memory

### Use Case
An internal knowledge base agent serving employees across an organization. Different roles (admin, employee, contractor) must have access to different subsets of documents. The system must enforce access control at the retrieval level — not just hide results in the UI, but never retrieve them in the first place.

### Memory Architecture

**Storage**: The `enterprise_documents` table stores documents with `allowed_role`, `status` (ACTIVE/ARCHIVED), temporal validity (`valid_from`/`valid_until`), full-text search vectors (`tsvector`), and semantic embeddings.

**Retrieval Strategy**: Reciprocal Rank Fusion (RRF) combining:
- **Vector similarity** via HNSW index (semantic matches)
- **Full-text search** via GIN index on `tsvector` (keyword matches)
- **Role-based access** via `WHERE allowed_role = $3`
- **Temporal validity** via `WHERE (valid_until IS NULL OR valid_until > now())`

The RRF formula `vec_score + text_score` ensures that documents matching by either vector or keyword surface in the results, with documents matching both ranking highest.

**Why RRF over weighted blending?** Weighted blending requires tuning weights per query domain. RRF is parameter-free and robust across diverse query types — from exact policy lookups ("what's the password rotation policy") to open-ended searches ("what are my remote work options").

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **RRF + RBAC (this project)** | Zero-parameter hybrid ranking; access enforced at retrieval | Two separate indexes to maintain; FTS quality depends on language config | Regulated industries, multi-tenant SaaS, internal knowledge bases |
| **Pure vector search** | Simple; good for semantic similarity | Misses exact keyword matches ("NDA" → "Non-Disclosure Agreement") | Q&A over unstructured text |
| **Pure FTS (Elasticsearch)** | Excellent keyword precision; faceted search | No semantic understanding; misses conceptually similar but lexically different docs | E-commerce search, log search |
| **Graph RAG (Neo4j + LLM)** | Captures document relationships and citations | Complex infrastructure; slow ingestion | Legal document analysis, research literature |

### Agent Implementation

The LangGraph agent uses a **3-node tool-calling graph**: `search_policy_docs` → `agent` → `tools`. The search node passes the user's role and query to the memory engine, which filters documents server-side. The LLM can use `search_policy_documents` for additional lookups or `store_policy_document` to write back new policies.

**Tools available**: `search_policy_documents`, `store_policy_document`

**What changed from the original**: The agent now supports write-back (storing new policy documents) and multi-turn conversations via the checkpointer.

---

## Pattern 4: Adaptive Tutor Memory

### Use Case
An AI tutor that tracks a learner's proficiency across a skill tree, applies a forgetting curve to model knowledge decay over time, and identifies the most impactful skill gaps to address next.

### Memory Architecture

**Storage**: Two tables form a directed acyclic graph:
- `tutor_skills`: Nodes in the skill tree with optional `parent_skill_id` for hierarchical relationships (e.g., `calculus` requires `algebra` which requires `math_basics`)
- `tutor_user_progress`: Per-user proficiency scores (0.0–1.0) with `last_reviewed_at` timestamps

**Retrieval Strategy**: The gap assessment query computes a decayed score using the Ebbinghaus forgetting curve:

```sql
COALESCE(proficiency_score, 0.0) *
POWER(0.95, seconds_since_review / 86400.0) AS decayed_score
```

This formula models the exponential decay of memory: a skill mastered at 1.0 drops to 0.95 after one day, 0.90 after two days, and approaches 0.0 over time. Skills below 0.5 are classified as "gaps" — the tutor should prioritize these.

**Why compute decay in SQL?** Moving the decay computation to the database layer means the agent doesn't need to retrieve all user scores and recalculate. The query returns pre-computed decayed scores, and the agent can focus on pedagogical decisions.

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **Ebbinghaus decay in SQL (this project)** | Familiar, mathematically grounded model; efficient server-side computation | Fixed decay constant (0.95) — doesn't adapt to individual learning rates | Structured curricula with clear skill hierarchies |
| **Spaced Repetition (Anki/SM-2)** | Proven algorithm with per-card intervals; adaptive difficulty | Single-card focus; no hierarchical skill modeling | Flashcard learning, language vocabulary |
| **Bayesian Knowledge Tracing** | Probabilistic; models guess/slip parameters | Requires labeled problem-response data; complex to tune | Educational assessment platforms |
| **LLM-only assessment** | Flexible; can evaluate open-ended responses | No persistent model of knowledge state; inconsistent scoring | Qualitative feedback, essay grading |

### Agent Implementation

The LangGraph agent uses a **3-node tool-calling graph**: `assess_skill_gaps` → `agent` → `tools`. It queries the memory engine's `/tutor/gaps/{user_id}` endpoint, which returns all skills with their decayed scores. The LLM can use `get_skill_gaps` to refresh the assessment and **`update_skill_progress` to write back updated proficiency** after the learner demonstrates mastery.

**Tools available**: `get_skill_gaps`, `update_skill_progress`

**What changed from the original**: The agent now closes the feedback loop — after teaching and assessing, it writes updated proficiency scores back to memory. Multi-turn conversations via the checkpointer enable iterative learning sessions.

---

## Pattern 5: Multi-Agent Swarm Memory

### Use Case
A coordinated team of specialized agents (e.g., sentiment analysis, entity extraction, summarization) working on a shared workflow. Each agent independently claims pending tasks from a shared blackboard, executes its specialty using real tools, and marks tasks complete. No agent should claim a task that another agent already took.

### Memory Architecture

**Storage**: The `swarm_blackboard` table is a distributed task queue. Each row represents a task with `status` (PENDING → IN_PROGRESS → COMPLETED), an optional `assigned_agent`, and a `payload` (JSONB) for task-specific data.

**Coordination Strategy**: PostgreSQL's `FOR UPDATE SKIP LOCKED` clause provides lock-free task claiming:

```sql
SELECT task_id, task_name, payload
FROM swarm_blackboard
WHERE workflow_id = $1 AND status = 'PENDING'
ORDER BY created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED
```

When multiple agents execute this query simultaneously:
1. PostgreSQL acquires a row-level lock on the first PENDING row for one agent
2. Other agents **skip** the locked row and move to the next available one
3. Each agent updates their claimed row to `IN_PROGRESS` within the same transaction
4. No polling, no external lock manager, no race conditions

**Why SKIP LOCKED over a message queue (RabbitMQ/Kafka)?** For agents that already need a database for state persistence, using the same database as a lightweight task queue eliminates an additional infrastructure dependency. This pattern works well for moderate throughput (<1K tasks/second). For high-throughput streaming, a dedicated message broker is preferable.

### Swarm vs. Supervisor vs. Hierarchical Multi-Agent

This is a critical distinction that the current demo label ("Swarm Agent") oversimplifies:

| Pattern | Coordination Mechanism | LangGraph Primitive | This Demo |
|---------|----------------------|-------------------|-----------|
| **Swarm (this demo)** | Shared blackboard + autonomous pull; a supervisor decomposes work and fans out to parallel specialist workers | `StateGraph` + `Send` API for parallel fan-out; each worker is a tool-calling agent claiming its assigned task | **Hybrid** — LangGraph-level fan-out + database-level SKIP LOCKED claiming |
| **Supervisor** | Central orchestrator that decomposes work and assigns to specialist sub-agents | `StateGraph` with subgraphs compiled into nodes; supervisor node routes via conditional edges | Partially implemented (single-level supervisor + Send) |
| **Hierarchical** | Tree of supervisors, each managing a team of specialists with escalation paths | Nested `StateGraph` instances with `Send` API for fan-out to parallel workers | **Not implemented** |

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **SKIP LOCKED blackboard (this project)** | Co-located with app DB; no external queue; transactional consistency | Not suitable for high throughput (>1K TPS); no message redelivery/replay | Moderate-scale multi-agent workflows with strong consistency needs |
| **RabbitMQ / Kafka** | Battle-tested; high throughput; persistence and replay | Additional infrastructure; eventual consistency model | High-volume event-driven agent systems |
| **LangGraph Send API** | Native multi-agent fan-out within the graph framework; no external queue needed | All workers share the same Python process; not horizontally scalable | Single-process agent systems; prototyping |
| **Celery / Temporal** | Horizontal scaling; retry policies; cron scheduling built in | Heavy infrastructure; complex deployment | Production workflow orchestration |

### Agent Implementation

The LangGraph agent uses a **supervisor + Send API fan-out** topology:

1. A `supervisor` node reads the shared blackboard and collects every PENDING task
2. A conditional edge returns one `Send` per task, dispatching parallel worker nodes
3. Each `worker` node is a tool-calling agent that claims its assigned task by ID, executes it with real tools, marks it complete, and reports back
4. An `aggregate` node merges worker reports (via a reducer) into a final summary and snapshots the blackboard

The workers run concurrently; safe claiming is still enforced at the database layer via `FOR UPDATE SKIP LOCKED`, so a worker can never claim a task that another worker already took.

**Tools available**: `list_workflow_tasks`, `claim_next_task`, `claim_task`, `complete_swarm_task`, `execute_shell_command`, `fetch_url`

**What changed from the original**: The swarm is no longer a sequence of independent agent invocations. A single supervisor invocation fans out to parallel specialist workers via the LangGraph Send API — combining LangGraph-level multi-agent topology with the DB-level SKIP LOCKED blackboard. Multi-agent concurrency is validated in the test suite (`test_skip_locked_no_deadlock`, `test_eval_swarm_concurrency`).

---

## Pattern 6: Relational Companion Memory

### Use Case
An AI companion or personal assistant that maintains a rich, evolving model of its user — their relationships, location, hobbies, goals, and emotional state. It must remember facts indefinitely, track temporary moods that expire, and recall past conversations to provide contextually aware responses.

### Memory Architecture

**Storage**: Five tables implementing three temporal domains:

| Domain | Tables | Persistence | Example |
|--------|--------|-------------|---------|
| **Graph (Persistent)** | `companion_graph_nodes`, `companion_graph_edges` | Until explicitly terminated (`valid_until`) | "Alice LIVES_IN Brooklyn" |
| **Episodic (Persistent)** | `companion_episodes`, `companion_chunks` | Permanent (archival) | "Talked about wanting a 2BR apartment" |
| **Ephemeral (Transient)** | `companion_ephemerals` | TTL-based expiration | "Feeling excited this week" (expires in 24h) |

**Graph Model**: Entities (`companion_graph_nodes`) are connected by typed, directed edges (`companion_graph_edges`). Edges can be terminated by setting `status='INACTIVE'` or `valid_until` — this implements bitemporal state, where a fact has both a real-world validity period and a database-level status flag. For example, "Alice WORKS_AT CompanyX" can be marked inactive when she changes jobs, but the historical fact remains queryable for past-date context.

**Three-Subject Model**: Every node and edge carries a `subject` column — `user`, `self`, or `shared`:

- **`user`** — durable facts the user revealed ("Alice LIVES_IN Brooklyn")
- **`self`** — the companion's model of **itself**, extracted from its own outputs ("Iris VALUES honesty", "Iris WRITES poetry"). A backstory endpoint (`POST /companion/backstory`) pre-seeds this per user so the companion starts with a persona and grows from conversation.
- **`shared`** — relationship facts belonging to both ("we SHARE_RITUAL morning coffee", "Vesper TRUSTS Alice"), the memory of the bond growing together.

Because both sides live in one graph, the retrieve node can **cross-reference** user vs self facts on affinity predicates and surface common ground — *"hey, we both like pizza!"*.

**Emotional Scoring + Provenance**: Edges carry `valence` (−1.0…+1.0) and `intensity` (0.0…1.0) so the graph remembers the emotional weight of a fact after the turn ends (a pet's death is `HAS_PET Luna` with `valence=-0.9`). Edges also link to their source episode via `source_episode_id`, enabling provenance queries (`/companion/facts/provenance`) that trace any fact back to where it was inferred.

**Ephemeral Model**: Short-lived states (moods, current activities, temporary preferences) use a TTL column (`expires_at`). The context retrieval query filters `WHERE expires_at > clock_timestamp()` — expired ephemerals are silently dropped. This avoids polluting the companion's long-term memory with transient states while still providing real-time contextual awareness.

**This is the most structurally complex memory pattern in the project**, combining three persistence strategies, three memory subjects, emotional metadata, and provenance in a unified retrieval endpoint.

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **Graph + Episodic + Ephemeral (this project)** | Models all three temporal domains; typed relationships enable reasoning; bitemporal edges support history | Complex schema; edge traversal can be expensive at scale | Personal AI companions, NPC dialogue systems, customer 360 views |
| **MemGPT / Letta** | OS-inspired memory hierarchy (core ↔ archival); automatic memory management | Fixed memory tiers; limited relationship modeling | Chat-oriented agents with long conversation histories |
| **Neo4j-native graph agents** | Native graph traversal; optimized for deep relationship chains | Separate infrastructure; embedding generation is external | Social network analysis, recommendation engines |
| **LangChain ConversationBufferMemory** | Dead simple; just stores recent messages | No persistence across restarts; no entity extraction; context window limits | Quick prototypes, single-session bots |

### Agent Implementation

The LangGraph agent uses a **4-node graph**: `retrieve_companion_context` → `agent` → `tools`, plus a `extract` node. The retrieve node fetches all graph facts (split into user / self / shared), active ephemerals, and computes common ground between the user and the companion's self-model. The LLM can use:

- `get_companion_context` to refresh the user's memory state
- `search_episodic_memory` to find semantically similar past conversations
- `store_companion_episode` to persist a conversation in episodic memory
- `store_companion_fact` to add new facts to the relationship graph
- `store_companion_ephemeral` to record temporary mood/context

**Tools available**: `get_companion_context`, `search_episodic_memory`, `store_companion_episode`, `store_companion_fact`, `store_companion_ephemeral`

**Dynamic persona from memory**: Instead of a static system prompt, the retrieve node assembles the prompt each turn from the three memory subjects — *"About YOU (your self-model...)", "About your RELATIONSHIP (shared facts...)", "Things you and the user have in common"* — so the companion's personality and behavior are driven by its accumulated self-model state.

**3-scope extraction**: After responding, the `extract` node runs a structured extraction that separates `user_facts`, `self_facts` (what the companion revealed about itself), and `shared_facts` (relationship), each with a canonical predicate (production-aligned UPPER_SNAKE vocabulary), emotional `valence`/`intensity`, and a `source_episode_id`. The episode is created first, so every fact links back to the conversation it was learned from.

**What changed from the original**: The companion is no longer a passive observer nor a one-sided model of the user — it maintains a balanced memory of both itself and the user, records the emotional weight of facts, tracks provenance, and surfaces shared ground as the relationship grows. Multi-turn conversations via the checkpointer preserve dialogue continuity.

---

## Comparative Analysis

### Memory Retrieval Latency Characteristics

| Pattern | Query Type | Expected Latency | Scaling Bottleneck |
|---------|-----------|-----------------|-------------------|
| Developer | HNSW ANN + trgm filter | <10ms | Index build time on symbol insert |
| Task | HNSW ANN + range filter | <5ms | HNSW graph size at >1M trajectories |
| Enterprise | RRF (vector + FTS) + joins | <20ms | tsvector GIN index size |
| Tutor | Aggregation query (CTE) | <5ms | Unlikely (small cardinality) |
| Swarm | SKIP LOCKED SELECT + UPDATE | <5ms (contended) | Contention at >100 concurrent agents |
| Companion | Multiple independent queries | <15ms | Graph edge traversal at >10K nodes |

### Agent Complexity Comparison

| Dimension | Developer | Task | Enterprise | Tutor | Swarm | Companion |
|-----------|-----------|------|------------|-------|-------|-----------|
| Storage tables | 1 | 1 | 1 | 2 | 1 | 5 |
| Indexes | 2 (HNSW + GIN) | 1 (HNSW) | 2 (HNSW + GIN) | 0 | 0 | 1 (HNSW) |
| Query complexity | Medium | Simple | High | Medium | Medium | High |
| Agent graph nodes | 3 | 3 | 3 | 3 | 3 | 3 |
| Tools available | 5 | 6 | 2 | 2 | 5 | 5 |
| State transitions | Read + Write | Read + Write | Read + Write | Read + Write | Read + Write | Read + Write |
| Multi-agent capable | — | — | — | — | Yes (DB level) | — |
| Real execution | Shell, file | Shell, HTTP, file | — | — | Shell, HTTP | — |

### When to Use Which Pattern

| If your agent needs… | Use this pattern | Because… |
|----------------------|-----------------|----------|
| To search and understand code across projects | Developer Workspace | Branched symbol storage with hybrid semantic+symbolic search + file/shell tools |
| To learn from past successes and actually execute tasks | Autonomous Task | Trajectory recall with success score quality gate + real HTTP/shell execution |
| To serve different internal audiences with access-controlled knowledge | Enterprise Knowledge | RRF ranking with server-side RBAC enforcement |
| To personalize education with adaptive skill tracking | Adaptive Tutor | Skill tree with Ebbinghaus forgetting curve in SQL + write-back feedback loop |
| To coordinate multiple specialist agents on shared workflows | Multi-Agent Swarm | Supervisor + Send API fan-out with SKIP LOCKED blackboard claiming + execution tools |
| To build a relationship with a user over time with persistent memory | AI Companion | Three-domain memory (graph + episodic + TTL ephemeral) + bidirectional write-back |

---

## LangGraph Integration Analysis

### Current State

Every LangGraph agent in this project is a **3-node tool-calling pipeline** with conditional routing (the Companion adds a fourth extraction node):

```
entry node → agent (LLM + tools) → [conditional: tool_calls?] → tools → agent → END
```

The entry node retrieves memory context from the API, the agent node invokes the LLM with bound tools, and the tools node (`ToolNode`) executes tool calls. A conditional edge checks for `tool_calls` on the last message — if tools were called, the graph loops back to the agent for follow-up; otherwise it ends. The Companion graph branches to a final `extract` node that runs structured 3-scope extraction (user/self/shared) with emotional valence and provenance before ending.

**This project uses**:
- **ToolNode** — tool calling and function execution via `langgraph.prebuilt`
- **Conditional edges** — branching based on whether the LLM requested tools
- **Loops/cycles** — agent → tools → agent iteration for multi-step tool sequences
- **Checkpointer** — Postgres-backed state persistence via `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`
- **Bidirectional memory** — all agents can both read from and write to the memory engine
- **Send API** — the Swarm pattern fans out from a supervisor to parallel worker nodes
- **Post-response extraction** — the Companion writes back structured, typed memory (including self-model facts) after each turn

**This project does not yet use**:
- **Subgraphs** — no composed agent hierarchies (the Swarm supervisor uses nodes, not compiled subgraphs)
- **Interrupt/Command** — no human-in-the-loop or dynamic routing

### What a Production-Grade Implementation Would Add

For each agent to become production-ready using LangGraph's full capabilities:

**Developer Agent**: Add AST-based symbol auto-discovery (tool that parses Python/JS files), incremental index updates, and a conditional edge that checks "did we find relevant symbols?" before routing.

**Task Agent**: Already has real execution tools. Add a reflexion loop (`planning` → `execution` → `evaluation` → `replan` via conditional edges) and automatic success scoring of executed tasks.

**Enterprise Agent**: Add a conditional guardrail node (`search` → `has_results?` → `generate` or `deny`), implement citation chaining via tool calls, and add an audit logging node that records every retrieval.

**Tutor Agent**: Implement prerequisite-aware routing via conditional edges (if `algebra` is a gap, route to `algebra_lesson` before `calculus_lesson`), and adaptive decay rates per learner.

**Swarm Agent**: Already has a supervisor + Send API fan-out. A further production upgrade would compile each worker as a subgraph, add a merge/rollback policy for failed workers, and support nested hierarchies (supervisors of supervisors) with escalation paths.

**Companion Agent**: Add a fact extraction node that analyzes the conversation and writes new graph nodes/edges, an importance scorer that curates the context window when facts exceed limits, and a personality modulator that adjusts tone based on ephemeral emotional state.

### Why the Focus on Infrastructure First

Building the memory engine first — before adding complex agent graph logic — follows the principle of separating **storage** from **computation**. The PostgreSQL schemas, indexes, and retrieval queries in this project are reusable regardless of the agent framework on top. You could replace LangGraph with a different orchestration layer (CrewAI, AutoGen, raw async Python) and the memory engine would still serve all six patterns.

---

## Going Further: Production Considerations

### Observability
Add LangSmith/LangFuse callbacks to trace retrieval-to-generation pipelines. Log every memory engine query with timing, result count, and error codes.

### Testing and Evaluation
The 54-test suite in `tests/` validates functional correctness. For production, add:
- **Retrieval quality evals**: Precision@K and NDCG for each pattern using labeled query→document pairs
- **Concurrency stress tests**: 100+ parallel swarm agents claiming tasks under load
- **Decay accuracy tests**: Pre-compute expected decayed scores and assert within tolerance
- **Role isolation penetration tests**: Verify that no combination of queries leaks cross-role documents

### Scaling
- For >1M vectors in the developer/task patterns, consider IVF instead of HNSW for faster index builds
- For the companion graph, consider adding recursive CTEs for deeper relationship traversal (friend-of-friend queries)
- For the swarm blackboard, add a `claimed_at` timestamp and a reaper process that releases tasks stalled in `IN_PROGRESS` beyond a timeout
