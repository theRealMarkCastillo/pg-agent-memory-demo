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

**This project is** a reference implementation demonstrating how the same underlying infrastructure (PostgreSQL with pgvector and pg_trgm) can support six fundamentally different memory architectures. It runs as a self-contained Docker stack that you can inspect, modify, and benchmark.

**This project is not** a production-grade agent framework. The LangGraph agents are intentionally minimal (linear 2-node pipelines) to keep the focus on the memory engine — the storage, indexing, retrieval, and state transition logic in PostgreSQL. A production system would add multi-turn conversation loops, conditional routing, sub-agents, tool calling, and checkpointing on top of the memory layer demonstrated here.

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

The LangGraph agent is a linear 2-node pipeline: `search_code_symbols` → `generate_response`. It retrieves relevant symbols from the memory engine, formats them into a prompt, and asks the LLM to explain or use them.

**Limitations in the current implementation**: The agent does not write back new symbols (e.g., when the user adds a function), has no file system access (it can't actually read source files), and uses a hardcoded `temperature=0.3`. A production version would add: file ingestion pipelines, symbol auto-discovery via AST parsing, incremental index updates, and a chat loop with tool calling.

---

## Pattern 2: Autonomous Task Memory

### Use Case
An agent that autonomously executes multi-step tasks (web scraping, report generation, data pipeline orchestration). When given a new goal, it should recall how it succeeded at similar goals in the past and adapt those strategies.

### Memory Architecture

**Storage**: The `task_trajectories` table logs completed task executions as serialized action sequences (`JSONB`), the original goal description, an embedding of the goal for similarity search, the execution result, and a success score (0.0–1.0).

**Retrieval Strategy**: Semantic similarity search on goal embeddings with a minimum success score filter (`WHERE success_score >= 0.7`). This ensures the agent learns from successful trajectories and ignores failures. If a new goal is "scrape product prices from competitor site," it will retrieve the trajectory of the last successful web scraping task as a few-shot example.

**Why success_score filtering matters**: Without it, the agent would retrieve the most similar trajectory regardless of outcome — potentially learning from failures. The score filter creates a "quality gate" that only surfaces proven strategies.

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **Trajectory recall + few-shot (this project)** | Learns from past successes; scores create quality gate | Requires manual or programmatic scoring; trajectory quality degrades if scoring is noisy | Structured tasks with measurable outcomes |
| **ReAct / Tool-using agents** | Dynamic tool selection; no historical dependency | No learning across episodes; each task starts from scratch | Tasks requiring diverse tool chains |
| **Fine-tuned agent models** | Compresses experience into model weights; fast inference | Expensive to retrain; catastrophic forgetting | High-volume, repetitive task domains |
| **Reflexion / self-critique loops** | Self-improving via verbal feedback; no external scoring needed | Can amplify biases; requires strong base model | OpenAI/exploratory domains without clear success metrics |

### Agent Implementation

The LangGraph agent: `recall_past_trajectories` → `plan_and_execute`. The recall node queries the memory engine for similar successful trajectories, and the plan node asks the LLM to generate a plan and a simulated result.

**Limitations**: The `success_score` in the demo is hardcoded to 0.9 — there is no real execution or feedback loop. The plan node simulates execution rather than actually running tools. A production version would: execute real tool calls, measure actual outcomes, write back new trajectories with real success scores, and implement iterative correction when plans fail.

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

The agent is `search_policy_docs` → `generate_response`. It passes the user's role and query to the memory engine, which filters documents server-side. The retrieved documents are formatted into a prompt and the LLM answers based only on the documents it's authorized to see.

**Limitations**: No audit logging of document access, no JWT/session-based auth verification (the role is trusted from input), and no handling of the "no documents found" case (the LLM hallucinates instead of returning "I don't have information about that"). A production version would: integrate with an identity provider (OAuth/OIDC), log every retrieval for compliance, emit structured citations, and implement access-denied guardrails.

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

The agent runs `assess_skill_gaps` → `recommend_and_respond`. It queries the memory engine's `/tutor/gaps/{user_id}` endpoint, which returns all skills with their decayed scores and gap/mastered classifications. The LLM receives this data and generates a personalized lesson focused on the weakest relevant skill.

**Limitations**: The agent never writes back updated proficiency after the lesson. The decay constant (0.95) is hardcoded. There's no adaptive routing — the agent doesn't skip review if proficiency is high, or route to prerequisites if a foundation gap is found. A production version would: close the feedback loop (assess → teach → reassess), adapt decay rates per learner, implement prerequisite-aware routing, and schedule review sessions proactively (push notifications for skills approaching gap threshold).

---

## Pattern 5: Multi-Agent Swarm Memory

### Use Case
A coordinated team of specialized agents (e.g., sentiment analysis, entity extraction, summarization) working on a shared workflow. Each agent independently claims pending tasks from a shared blackboard, executes its specialty, and marks tasks complete. No agent should claim a task that another agent already took.

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
| **Swarm (this demo)** | Shared blackboard + autonomous pull; agents are equal peers with no central coordinator | Multiple independent `StateGraph` instances reading/writing the same DB-backed state | **Database only** — the agent graph is a single-node invocation; there is no multi-agent topology in LangGraph |
| **Supervisor** | Central orchestrator that decomposes work and assigns to specialist sub-agents | `StateGraph` with subgraphs compiled into nodes; supervisor node routes via conditional edges | **Not implemented** |
| **Hierarchical** | Tree of supervisors, each managing a team of specialists with escalation paths | Nested `StateGraph` instances with `Send` API for fan-out to parallel workers | **Not implemented** |

The current demo demonstrates the *database-level* coordination pattern (SKIP LOCKED on a shared table) — a valid and production-tested approach. To demonstrate the *LangGraph-level* multi-agent pattern, a swarm demo would need:

1. A `supervisor_agent` node that decomposes a workflow into subtasks and writes them to the blackboard
2. Multiple `worker_agent` subgraphs (specialist nodes), each compiled as a `StateGraph` and invoked in parallel using LangGraph's `Send` API
3. A `merge_results` node that collects completed subtasks and synthesizes a final output

These patterns are additive — the database blackboard mechanism demonstrated here can serve as the persistence layer for all three coordination topologies.

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **SKIP LOCKED blackboard (this project)** | Co-located with app DB; no external queue; transactional consistency | Not suitable for high throughput (>1K TPS); no message redelivery/replay | Moderate-scale multi-agent workflows with strong consistency needs |
| **RabbitMQ / Kafka** | Battle-tested; high throughput; persistence and replay | Additional infrastructure; eventual consistency model | High-volume event-driven agent systems |
| **LangGraph Send API** | Native multi-agent fan-out within the graph framework; no external queue needed | All workers share the same Python process; not horizontally scalable | Single-process agent systems; prototyping |
| **Celery / Temporal** | Horizontal scaling; retry policies; cron scheduling built in | Heavy infrastructure; complex deployment | Production workflow orchestration |

### Agent Implementation

The LangGraph agent runs `fetch_blackboard` → `execute_task`. It GETs the blackboard state, claims the next pending task via `/swarm/tasks/claim-next`, and asks the LLM to process it.

**Limitations**: The demo invokes a single `StateGraph` instance — there is no actual multi-agent concurrency in the LangGraph layer. The "swarm" behavior is demonstrated at the database level (multiple agents *could* call `claim-next` in parallel and get unique tasks), but the demo runner only invokes one agent sequentially. A proper multi-agent demo would: spawn multiple `StateGraph` invocations in parallel (using `asyncio.gather`), each calling `claim-next` independently, and observe the SKIP LOCKED behavior preventing double-claims. The 50-test suite already validates this in `test_swarm.py::test_skip_locked_no_deadlock`.

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

**Ephemeral Model**: Short-lived states (moods, current activities, temporary preferences) use a TTL column (`expires_at`). The context retrieval query filters `WHERE expires_at > clock_timestamp()` — expired ephemerals are silently dropped. This avoids polluting the companion's long-term memory with transient states while still providing real-time contextual awareness.

**This is the most structurally complex memory pattern in the project**, combining three persistence strategies in a unified retrieval endpoint.

### Compare and Contrast

| Approach | Strength | Weakness | When to Use |
|----------|----------|----------|-------------|
| **Graph + Episodic + Ephemeral (this project)** | Models all three temporal domains; typed relationships enable reasoning; bitemporal edges support history | Complex schema; edge traversal can be expensive at scale | Personal AI companions, NPC dialogue systems, customer 360 views |
| **MemGPT / Letta** | OS-inspired memory hierarchy (core ↔ archival); automatic memory management | Fixed memory tiers; limited relationship modeling | Chat-oriented agents with long conversation histories |
| **Neo4j-native graph agents** | Native graph traversal; optimized for deep relationship chains | Separate infrastructure; embedding generation is external | Social network analysis, recommendation engines |
| **LangChain ConversationBufferMemory** | Dead simple; just stores recent messages | No persistence across restarts; no entity extraction; context window limits | Quick prototypes, single-session bots |

### Agent Implementation

The agent runs `retrieve_companion_context` → `generate_response`. It fetches all graph facts, active ephemerals, and optional episodic search results from the memory engine, structures them into a prompt, and generates a personalized response.

**Limitations**: The most significant gap is that the agent never writes back to memory. A companion that only reads memory and never updates it is an *observer*, not a *participant*. After the conversation, it should: extract new facts from the dialogue and insert graph nodes, log the conversation as a new episode, and update ephemeral states. Additionally, there's no context window management — with hundreds of facts, the prompt would overflow. Production companions need: bidirectional memory (read + write), automatic fact extraction from conversations, context pruning/ranking, and emotional state modulation in responses.

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
| Query complexity | Medium (filter + ANN) | Simple (ANN + range) | High (RRF + role + temporal) | Medium (CTE + decay) | Medium (lock + update) | High (3-domain join) |
| Agent graph nodes | 2 | 2 | 2 | 2 | 2 | 2 |
| State transitions | Read-only | Read-only | Read-only | Read-only | Read + Write | Read-only |
| Multi-agent capable | — | — | — | — | Yes (DB level) | — |

### When to Use Which Pattern

| If your agent needs… | Use this pattern | Because… |
|----------------------|-----------------|----------|
| To search and understand code across projects | Developer Workspace | Branched symbol storage with hybrid semantic+symbolic search |
| To learn from past successes and avoid repeating failures | Autonomous Task | Trajectory recall with success score quality gate |
| To serve different internal audiences with access-controlled knowledge | Enterprise Knowledge | RRF ranking with server-side RBAC enforcement |
| To personalize education with adaptive skill tracking | Adaptive Tutor | Skill tree with Ebbinghaus forgetting curve in SQL |
| To coordinate multiple specialist agents on shared workflows | Multi-Agent Swarm | Blackboard pattern with SKIP LOCKED for lock-free task claiming |
| To build a relationship with a user over time with persistent memory | AI Companion | Three-domain memory (graph + episodic + TTL ephemeral) |

---

## LangGraph Integration Analysis

### Current State

Every LangGraph agent in this project is a **linear 2-node pipeline**: a retrieval node fetches data from the memory engine, and a generation node passes it to the LLM. This was an intentional design choice to keep the focus on the memory infrastructure layer. No agent uses:

- **Conditional edges** — no branching based on state
- **Loops/cycles** — no iterative refinement or conversation turns
- **Subgraphs** — no composed agent hierarchies
- **Send API** — no parallel fan-out to workers
- **Interrupt/Command** — no human-in-the-loop or dynamic routing
- **Checkpointer** — no persistence of agent state across invocations
- **ToolNode** — no tool calling or function execution

This means every agent is a **single-shot RAG pipeline** rather than an autonomous, multi-step agent. The LangGraph framework is vastly underutilized.

### What a Production-Grade Implementation Would Add

For each agent to become production-ready using LangGraph's full capabilities:

**Developer Agent**: Add a `ChatOpenAI` tool-calling node that can create new symbols (write back to memory), a conditional edge that checks "did we find relevant symbols?" before routing to generation or to a fallback node, and a checkpointer for multi-turn sessions.

**Task Agent**: Implement `ToolNode` with real tools (HTTP requests, code execution), add a reflexion loop (`planning` → `execution` → `evaluation` → `replan` via conditional edges), and write completed trajectories back to the memory engine.

**Enterprise Agent**: Add a conditional guardrail node (`search` → `has_results?` → `generate` or `deny`), implement citation chaining via tool calls, and add an audit logging node that records every retrieval.

**Tutor Agent**: Implement a multi-turn loop with progress updates (write decaded scores back), add prerequisite routing via conditional edges (if `algebra` is a gap, route to `algebra_lesson` before `calculus_lesson`), and schedule future reviews.

**Swarm Agent**: This is the agent that benefits most from proper LangGraph usage. A proper swarm implementation would:
1. Compile a **supervisor graph** with a `Send` node that fans out to multiple worker subgraphs
2. Each worker subgraph independently invokes `claim-next` against the blackboard
3. A `merge` node collects results and produces a final summary
4. All workers share a checkpointer for state coordination

**Companion Agent**: Add a fact extraction node that analyzes the conversation and writes new graph nodes/edges, an importance scorer that curates the context window when facts exceed limits, and a personality modulator that adjusts tone based on ephemeral emotional state.

### Why the Focus on Infrastructure First

Building the memory engine first — before adding complex agent graph logic — follows the principle of separating **storage** from **computation**. The PostgreSQL schemas, indexes, and retrieval queries in this project are reusable regardless of the agent framework on top. You could replace LangGraph with a different orchestration layer (CrewAI, AutoGen, raw async Python) and the memory engine would still serve all six patterns.

---

## Going Further: Production Considerations

### Observability
Add LangSmith/LangFuse callbacks to trace retrieval-to-generation pipelines. Log every memory engine query with timing, result count, and error codes.

### Testing and Evaluation
The 50-test suite in `tests/` validates functional correctness. For production, add:
- **Retrieval quality evals**: Precision@K and NDCG for each pattern using labeled query→document pairs
- **Concurrency stress tests**: 100+ parallel swarm agents claiming tasks under load
- **Decay accuracy tests**: Pre-compute expected decayed scores and assert within tolerance
- **Role isolation penetration tests**: Verify that no combination of queries leaks cross-role documents

### Scaling
- For >1M vectors in the developer/task patterns, consider IVF instead of HNSW for faster index builds
- For the companion graph, consider adding recursive CTEs for deeper relationship traversal (friend-of-friend queries)
- For the swarm blackboard, add a `claimed_at` timestamp and a reaper process that releases tasks stalled in `IN_PROGRESS` beyond a timeout
