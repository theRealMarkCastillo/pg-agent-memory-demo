# Agent Memory: Storage, Context, and Control Flow

This guide explains the implementation in this repository and how to evaluate it. The baseline was checked on October 24, 2026: LangGraph 1.2.17, LangChain Core 1.7.0, OpenAI integration 1.7.2, Checkpoint 4.3.0, and PostgreSQL saver 3.2.0. Exact pins are in [requirements](demo_agents/requirements.txt).

The six examples cover memory **and coordination**. They are useful building blocks, not a complete taxonomy of cognition or a production certification. Current behavior below is grounded in source and regression tests. [Six additional workflows](USE_CASES.md) now implement approval, evidence, procedure promotion, timelines, research, and analytics in a separate extension host.

## Contents

1. [The memory problem in AI agents](#the-memory-problem-in-ai-agents)
2. [A taxonomy of agent memory](#a-taxonomy-of-agent-memory)
3. [Workshop path](#workshop-path)
4. [Pattern 1: Developer workspace memory](#pattern-1-developer-workspace-memory)
5. [Pattern 2: Autonomous task memory](#pattern-2-autonomous-task-memory)
6. [Pattern 3: Enterprise knowledge memory](#pattern-3-enterprise-knowledge-memory)
7. [Pattern 4: Adaptive tutor memory](#pattern-4-adaptive-tutor-memory)
8. [Pattern 5: Multi-agent swarm memory](#pattern-5-multi-agent-swarm-memory)
9. [Pattern 6: Relational companion memory](#pattern-6-relational-companion-memory)
10. [LangGraph integration analysis](#langgraph-integration-analysis)
11. [Evaluation layers](#evaluation-layers)
12. [Going further: Production considerations](#going-further-production-considerations)
13. [Primary references](#primary-references)

## The memory problem in AI agents

A model request operates on supplied context. Persistence, retrieval, and memory updates are application responsibilities even when an API offers conversation state. Reusing a recorded trajectory provides an example to the next request; it does not update model weights. Likewise, storing a generated assertion does not make it true.

A useful memory design answers five questions:

1. **What is the record?** A source document, observed event, user assertion, model extraction, or verified result?
2. **Who owns it?** Which user, role, project, branch, workflow, and application principal can access or change it?
3. **When is it valid?** Current, superseded, expired, or historical?
4. **How does it enter context?** Which eligibility filters, candidate retrieval, ranking, and prompt budget apply?
5. **How can it be corrected or removed?** Which source records, derived vectors, checkpoints, exports, and backups are affected?

The repository implements parts of this lifecycle. It also makes important gaps visible: extraction is fallible, prompt construction is character-bounded rather than token-aware, and external effects are not exactly-once.

## A taxonomy of agent memory

Separate these concerns before choosing a database or agent framework:

| Concern | Scope and lifetime | Implementation here | Key distinction |
|---------|--------------------|---------------------|-----------------|
| Run context | One invocation | Frozen `AgentContext` passed through LangGraph `Runtime` | Application-provided scope; not inferred from a model's arguments |
| Working state | Graph steps and a conversation thread | `MessagesState`, pattern-specific state, PostgreSQL checkpoints | Stored history can outlive the smaller prompt actually sent to a model |
| Semantic/domain memory | Across threads | Symbols, policy documents, graph assertions | Requires eligibility, freshness, and source interpretation |
| Episodic memory | Across interactions | Task trajectories and companion episodes/chunks | An account of an event; outcome quality depends on its evidence |
| Procedural memory | Reusable instructions | Trajectories can be retrieved as examples | The extension checker gates versioned promotion; the original trajectory agent still uses supplied scores |
| Temporary context | Until a deadline | Companion ephemerals with `expires_at` | Filtering expired records is different from physically deleting them |
| Coordination state | Task lifecycle | Blackboard rows and expiring leases | A queue claim is not a user memory or a guarantee of external execution |

The [LangGraph persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence) distinguishes thread checkpoints from cross-thread stores. This demo uses a domain API for long-term memory instead of a LangGraph Store: relational constraints, access checks, provenance, and specialized queries live behind that API.

The six patterns are composable. A support assistant might combine policy retrieval, customer assertions, case episodes, and a workflow checkpoint without introducing four separate agents.

## Workshop path

Read [operating contracts](REMEDIATION.md) and [launch instructions](RUNBOOK.md) first. Use the disposable runner for deterministic work:

```bash
make test PYTHON=.venv-tests/bin/python ENGINE_PYTHON=.venv-engine/bin/python
```

This starts a disposable PostgreSQL instance and engine, supplies fixture embeddings, and removes its services afterward. It does not load the project's `.env` or call the configured chat model. A standalone `pytest` command is not equivalent: many tests require the runner's database, engine, and generated credentials.

For the live demonstrations, configure a tool-capable chat model and an embedding endpoint matching `halfvec(1536)`, generate credentials, and run:

```bash
make build up
make demo
```

The seed uses an admin credential, resets the demo companion user, and explicitly resumes that user's memory. Each run gets fresh thread IDs. Demo 6d deletes the demo user's companion memory and disables future companion recording. Use a disposable demo dataset for the workshop.

| Lab | Inspect or run | Expected observation | Failure to discuss |
|-----|----------------|----------------------|--------------------|
| 1. Separate state from memory | Developer demos 1/1b; [runtime](demo_agents/agents/runtime.py), [developer graph](demo_agents/agents/developer_agent.py) | One thread retains conversation; retrieval refreshes the system message each turn | A new thread has no old dialogue but can still retrieve the same authorized domain records |
| 2. Compare retrieval arms | [Developer router](memory_engine/routers/developer.py), [lexical regression](tests/test_main_sync_regressions.py) | A natural-language identifier query can match through the lexical arm | Similarity and RRF scores are not calibrated confidence probabilities |
| 3. Trace authorization | [Enterprise graph](demo_agents/agents/enterprise_agent.py), `test_scope_bound_through_actual_tool_node` in [agent regressions](tests/test_agent_regressions.py) | A model's forged admin role is overwritten; the engine checks caller permissions | A system prompt is not an access-control boundary |
| 4. Observe contention | Demo 5; [swarm graph](demo_agents/agents/swarm_agent.py) and [router](memory_engine/routers/swarm.py) | Pull-mode workers claim separate queue rows and completion checks leases | A worker can crash after an external effect and before recording completion |
| 5. Correct and forget | Demos 6c/6d; [companion tests](tests/test_agent_regressions.py) | Moving cities closes the prior single-valued assertion; forgetting also purges companion checkpoint threads | Other domains, extension files, hook excerpts, exports, and backups have separate lifecycles |
| 6. Recover and bound work | [V1 tests](tests/test_langgraph_v1.py) | Saved parallel `Send` payloads resume after rebuilding the native graph; excessive cycles raise `GraphRecursionError` | The test is not a full process-crash or old-version checkpoint migration test |
| 7. Evaluate the result | [Evaluation layers](#evaluation-layers), [trace tooling](README.md#memory-testing-pipeline-tools) | Separate storage correctness, extraction agreement, retrieval metrics, and answer quality | Passing synthetic fixtures does not establish provider quality |

A useful learner deliverable is a short record of the input, authorized scope, retrieved record IDs, tool results, final answer, and what the checks actually established. Avoid grading success by how confident the answer sounds.

## Pattern 1: Developer workspace memory

**Outcome:** find relevant symbols within an authorized project and branch, then optionally inspect or write workspace files.

`dev_code_symbols` stores symbol identity, path, signature, content, and a `halfvec(1536)` embedding. Identity is project + branch + file + symbol name + symbol type. Reconciliation can replace a file's symbols and remove renamed/deleted entries; a symbol index is not automatically kept current by the graph.

The [retrieval query](memory_engine/routers/developer.py) combines semantic and lexical candidates with RRF (`k=60`). Each arm retrieves up to 50 candidates and the fused query returns up to 10. The lexical arm combines trigram, `word_similarity`, and prefix matching; a whole-string similarity alone performs poorly when a short identifier is compared with a full question. Project, branch, and optional symbol type constrain retrieval.

The [graph](demo_agents/agents/developer_agent.py) runs `search → agent → tools → agent`, ending when the model returns no tool calls. Available tools are `search_code_symbols`, `store_code_symbol`, `execute_shell_command`, `read_file`, and `write_file`. Shell execution remains disabled in the Compose runner.

**Boundary:** the demo is symbol retrieval and scoped file tooling, not an AST indexer, dependency-aware code navigator, or proof of correct code generation. Evaluate identifier recall, branch isolation, stale-symbol removal, and end-to-end task correctness separately. Choose between source chunks and symbols based on the query; neither representation universally dominates.

## Pattern 2: Autonomous task memory

**Outcome:** retrieve comparable task attempts and use them as context for a bounded execution attempt.

`task_trajectories` contains a goal embedding, JSON action sequence, result, and supplied score. The [query](memory_engine/routers/task_agent.py) filters by caller-authorized agent scope, a minimum score (default 0.7), and similarity, returning up to five trajectories. A threshold selects records with reported high scores; it does not independently verify them.

The [graph](demo_agents/agents/task_agent.py) runs `recall → agent → tools → agent`. Tools include trajectory search/store, allowlisted HTTP, workspace files, and opt-in sandboxed shell execution. Demo 2 writes a script and reads it back; this verifies the file contents were saved, not that the script executed successfully.

**Boundary:** the model can propose a trajectory score. No independent grader promotes a trajectory to a trusted procedure, and recall does not train the model. Failures can also be useful evidence; a production design may retain failure reasons and environment versions rather than simply excluding them. [Verified procedural memory](USE_CASES.md#3-verified-procedural-memory) implements that distinction in a separate, bounded CSV-to-JSON example.

## Pattern 3: Enterprise knowledge memory

**Outcome:** retrieve eligible current policies for an authenticated principal and abstain when there is no supporting policy evidence.

`enterprise_documents` stores document content, role eligibility, ACTIVE/ARCHIVED state, validity intervals, vectors, and a full-text index. The [router](memory_engine/routers/enterprise.py) checks the caller's authorized role and applies eligibility to both the vector and lexical arms before fusing ranks. RRF combines ranks from different score scales; its candidate limits and thresholds still need workload evaluation.

The [graph](demo_agents/agents/enterprise_agent.py) performs an initial search and permits additional search or storage through tools. A final response with no initial documents and no successful document-bearing tool artifact is replaced by an abstention. This is a programmatic evidence-presence check, not a citation verifier or a complete grounding guarantee.

**Boundary:** the database checks authorization; the model does not grant a role. Policy updates are versioned, but there is no immutable audit ledger, approval workflow, or claim-level citation system. For a learning extension, add source IDs and version-aware citations before expanding into multi-agent routing.

## Pattern 4: Adaptive tutor memory

**Outcome:** expose stored proficiency, a simple time-decay heuristic, and prerequisite readiness to a tutor.

`tutor_skills` supports one parent per skill; `tutor_user_progress` stores scores and review times. The [recursive query](memory_engine/routers/tutor.py) walks transitive prerequisites and computes a fixed exponential decay:

```sql
COALESCE(proficiency_score, 0.0) *
POWER(0.95, seconds_since_review / 86400.0) AS decayed_score
```

A missing progress row is `not_started`; a practiced skill below 0.5 is a `gap`; otherwise it is `mastered`. Readiness requires every transitive prerequisite to meet the threshold. These are application labels derived from the formula, not measured psychological states.

The [graph](demo_agents/agents/tutor_agent.py) runs `assess → agent → tools → agent`, with `get_skill_gaps` and `update_skill_progress`. The prompt asks for demonstrated mastery before recording improvement; no independent assessment engine enforces that request.

**Boundary:** fixed 0.95 retention is an educational simplification, not a calibrated learner model or a spaced-repetition scheduler. A meaningful next experiment would use scored exercises, preserve assessment evidence, and compare review scheduling against a baseline on held-out responses.

## Pattern 5: Multi-agent swarm memory

**Outcome:** coordinate concurrent workers through transactional claims and expiring leases.

The name “swarm” is historical. The [implementation](demo_agents/agents/swarm_agent.py) is a deterministic supervisor over existing tasks, parallel `Send` workers, and an aggregate node. The supervisor does not ask a model to decompose a goal. In assigned mode it chooses task IDs; in pull mode it only sizes the worker pool and workers race for claims. The CLI demonstrates pull mode.

The [claim transaction](memory_engine/routers/swarm.py) selects either a pending row or an expired in-progress row:

```sql
SELECT task_id FROM swarm_blackboard
WHERE ($1::text IS NULL OR workflow_id = $1)
  AND ($2::uuid IS NULL OR task_id = $2)
  AND (status = 'PENDING'
       OR (status = 'IN_PROGRESS' AND lease_until < clock_timestamp()))
ORDER BY created_at, task_id
LIMIT 1 FOR UPDATE SKIP LOCKED;
```

The same transaction sets the claimant, a random lease token, and a five-minute deadline. Completion requires the matching task, claimant, active token, and IN_PROGRESS state. Worker code owns the lease, validates it before each tool call, and overrides model-supplied completion credentials. Claims and tokens do not round-trip through the model.

Workers use a local tool loop capped at eight model rounds. Graph tasks default to at most five concurrent executions. `Send` schedules branches; the PostgreSQL queue coordinates claims. They solve different problems. This local Compose runner does not demonstrate a fleet of distributed graph workers.

**Boundary:** row locking prevents conflicting concurrent claims; an expired lease intentionally permits a later claimant. A pre-dispatch lease check cannot prevent an already-running remote action from completing after lease loss. Add destination idempotency, an outbox where appropriate, lease renewal, and failure/retry policy before relying on external effects. See the [PostgreSQL locking documentation](https://www.postgresql.org/docs/16/sql-select.html#SQL-FOR-UPDATE-SHARE); no throughput ceiling is established by this demo.

## Pattern 6: Relational companion memory

**Outcome:** retrieve user-scoped assertions and episodes, track updates and temporary context, and support explicit forgetting.

The [engine](memory_engine/routers/companion.py) uses graph nodes/edges, episodes/chunks, ephemerals, fact-source associations, and a memory-state record. The latter tracks whether recording is enabled, the deletion generation, and embedding-space identity.

| Domain | Records | Interpretation |
|--------|---------|----------------|
| Assertions | Graph nodes and versioned edges | Extracted or supplied claims with current/historical state |
| Episodes | Episode text, chunks, source associations | Retained interaction evidence; subject to deletion |
| Temporary context | Expiring ephemerals | Omitted from context after expiry; filtering does not physically purge the row |
| Recording policy | Memory-state row | Disabled after forgetting until explicitly resumed; stale generations cannot write |

Assertions have `user`, `self`, or `shared` scope. `self` records persona claims from assistant outputs, not real personal experience. `shared` records claims about interactions. Retrieving assistant-generated claims back into context can reinforce mistakes, so persona consistency must be evaluated separately from truth about the user.

The extractor emits a constrained UPPER_SNAKE vocabulary; the engine normalizes stored predicates to lowercase snake_case. Known aliases and normalized entity keys reduce duplicates, but do not provide general entity resolution. Single-valued predicates (`lives_in`, `works_at`, `married_to`, `has_name`, `has_age`, `birthday_on`) close prior active values in the same subject scope. That policy is deliberately narrow: a person can have several homes or jobs, so broader domains need an explicit cardinality/conflict policy.

`valence` and `intensity` are bounded extractor annotations, not validated emotion measurements. Negative emotion does not terminate a current fact. Provenance identifies a source episode; it supports inspection, not automatic verification of the source's claim. Observed validity intervals preserve history without arbitrary retrospective bitemporal correction.

The [graph](demo_agents/agents/companion_agent.py) runs `retrieve → agent ↔ tools → extract`. Retrieval uses the current query and limits; it does not fetch every fact. Tools can search episodes, store facts/episodes/ephemerals, terminate a scoped relationship, or forget memory. The extraction node uses a JSON prompt and parser, then writes episodes and assertions through the engine. Provider-native schema-constrained extraction is a possible improvement, not the current implementation.

Forgetting is exclusive within a tool batch: sibling writes are cancelled. The engine deletes that user's companion facts, provenance, episodes/chunks, ephemerals, and companion checkpoint threads, disables companion recording, and advances the generation. The custom saver rejects stale-generation checkpoint writes. The public wrapper returns only a deletion acknowledgement after forgetting. Other domains, extension evidence/checkpoints, workspace files, and project/branch hook excerpts are outside this endpoint's deletion scope.

**Boundary:** exported traces, backups, external provider retention, and prompt-injection resistance require separate treatment. Character-bounded prompts still contain retrieved text inside system-message content; treating that text as untrusted evidence and testing adversarial instructions is a high-priority follow-up. The implementation does not establish that retrieved instructions cannot redirect the model.

## LangGraph integration analysis

### Choice of agent API

Use a deterministic function or workflow when the process is known. Use LangChain's `create_agent` when a standard model/tool loop and middleware fit the task. Use custom LangGraph nodes when the application needs explicit retrieval, grounding, extraction, forgetting, or lease boundaries. More agents or deeper graphs are not inherently better.

The [v1 migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1) replaces deprecated `langgraph.prebuilt.create_react_agent` with `langchain.agents.create_agent`. This repository uses neither prebuilt; its custom `StateGraph` approach remains supported. Switching to `create_agent` would be a design change, not a prerequisite for currency.

| Primitive | Use in this repository | Important limit |
|-----------|------------------------|-----------------|
| `MessagesState`, `START`, `END` | Shared message reducer and explicit graph boundaries | State includes prompt-ready strings in several patterns; storing raw records and formatting on demand is a useful future refactor |
| `Runtime[AgentContext]` | Immutable scope snapshot constructed by `ScopedGraph` | Input scope must already be authenticated/authorized by the application; it is not a credential |
| `ToolNode` | Scoped dispatch in five agents | Swarm uses its own lease-aware loop; tool errors are explicitly returned as error messages |
| `Send` and reducers | Parallel worker fan-out and merged reports | Concurrency is not a distributed deployment guarantee |
| `AsyncPostgresSaver` | Checkpoints for CLI graphs | Native resume is tested; the scoped wrapper only accepts new-turn dictionaries |
| `durability="sync"` | Default when a saver is present | No exactly-once guarantee for remote effects; no durability setting is applied without a saver |
| `interrupt()` and `Command(resume=...)` | Authorized extension approval/resume service | Persisted owner/thread mapping supplies authority; a resume value cannot approve an action |
| Compiled subgraphs, middleware, streaming | Extension choices | Not exposed as application features in this demo |

### Context and execution budgets

`prompt_messages` selects complete turns within a 32,000-character budget and retains the latest system message. It does not split a tool-call/result pair. An oversized current turn raises an error. `refresh_messages` replaces the prior system message by ID and caps system content at 20,000 characters.

This bounds an individual prompt, not the size of the stored checkpoint history. It is not token counting, summarization, or long-term compaction. Model choice, tool schemas, and provider overhead affect the actual context limit. See [context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering) for the distinction between the model's context and persistent state.

The wrapper sets 32 graph supersteps and five concurrent graph tasks by default; callers can override top-level `recursion_limit` and `max_concurrency`. Exhaustion raises `GraphRecursionError`. Model requests use a 60-second HTTP timeout and at most two retries; these are not tool retries or an end-to-end deadline. Message `.text` handles provider content blocks for display and extraction.

### Upgrade and verification boundary

Rebuild the demo image after dependency changes. The saver runs `setup()` to apply upstream checkpoint migrations. Its subclass now implements the companion deletion-generation fence; the legacy `pending_sends` conversion and private loader override are removed. The CLI closes both pools in `finally`; embedding applications should call `await close_checkpointer()` during shutdown.

Use fresh demo thread IDs, which the runner generates automatically. Back up valued data before upgrading. Resuming historical in-flight 0.6 checkpoints, downgrading a migrated database, and full crash recovery across every pattern remain unverified. `make clean` deletes the volume and is not an upgrade step.

The fan-out regression interrupts after the supervisor, rebuilds a native graph, reads pending tasks from PostgreSQL, and resumes with `None`. That demonstrates checkpoint serialization/reconstruction in one process. The separate extension service now supplies scope-authorized resume, a persisted thread-to-owner mapping, revision-bound review, and local-file reconciliation. Its regression kills the actual host during review and exits after atomic file replacement before database acknowledgement. The resumed host reconciles the file digest without rewriting it. This additional evidence applies to that local destination; do not expose raw graph resume to untrusted callers.

### Lessons from the final design review

The [final review](DESIGN_REVIEW.md) found failures that successful graph execution alone did not expose. Approval requests held database connections while waiting for graph locks, starving nodes that needed the same pool. Separate API and node pools now break that circular wait. A bounded burst regression verifies this behavior; selecting production pool sizes still requires load measurements.

PostgreSQL cannot transact an ordinary filesystem replacement. Approval therefore commits `applying` intent before the write, holds a destination lock through acknowledgement, and freezes edits/cancellation after intent. A retry acknowledges matching content; conflicting later content preserves the attempt and returns 409. A later policy change cannot invalidate the observation that the file already contains the reviewed content. See the [recovery table](EXTENSIONS.md#recovery-and-limits) before extending this pattern to another destination.

Evidence completeness is separate from citation validity. The research workflow now checks every expected labelled statement, records coverage, and withholds a one-sided conclusion if extraction omitted evidence. Old reports without that protocol are excluded from trusted recall. Support checks expiry against the current clock after lock waits. These are application invariants; a larger model or another agent would not enforce them.

## Evaluation layers

Keep these results separate when reporting progress:

| Layer | Existing evidence/tooling | What to add or measure |
|-------|--------------------------|------------------------|
| Storage and API contracts | Disposable integration tests | Invariants, eligibility, provenance, expiry, deletion, migration repeatability |
| Graph execution | Scripted-model regressions and PostgreSQL resume test | Dispatch scope, loop limits, pending branches, real approval-host death and restart; other crash boundaries remain open |
| Extraction | `benchmark_extraction.py` with normalized one-to-one matching | Human-reviewed labels; precision/recall by subject, predicate, negation, and temporal update |
| Retrieval | Fixture ranking checks; labeled probes in `eval_recall.py` | Recall@K, Precision@K, stale/unauthorized retrieval rate on a representative corpus |
| Answer and task outcome | Live demo exercises the configured provider | Evidence-supported claims, abstention, verified file/API effects, cost and latency |
| Operations | No published load/service benchmark | p50/p95 latency, queue contention, checkpoint growth, deletion completion, recovery time |

A probe without expected labels is inspection, not a recall score. Labels extracted by another model measure agreement unless independently reviewed. A model-reported success score is a claim, not an objective task grade. Keep a held-out set and record model/provider, prompt version, embedding-space identity, dataset revision, limits, and the tested code revision with results.

For a retrieval experiment, compare lexical-only, vector-only, and fused results on the **same eligible records**. Include no-answer cases, near-duplicate facts, old/current facts, and queries that would match another user's data. Evaluate the answer separately: improved retrieval does not by itself establish fewer unsupported claims.

A minimal outcome comparison should hold the model, task set, and tool budget fixed and compare no domain memory with retrieved memory. A subsequent experiment can compare unrestricted writes with verified promotion. This tests whether memory improves the task rather than merely demonstrating that a database can return records.

Provider evaluations require explicit credentials and a disposable dataset; the deterministic runner deliberately strips live-model settings. See [verification commands and label semantics](REMEDIATION.md#verification). The October 24 final-review run observed **207 passed and one optional live-model test skipped**, including 33 extension cases and ten final-review regressions beyond the 164-pass modernization baseline. The [implementation evidence](IMPLEMENTATION_PLAN.md#progress-and-evidence) and [review report](DESIGN_REVIEW.md) record commands and boundaries; this is a dated local result, not a rolling claim about the current branch.

## Going further: Production considerations

Start from a concrete outcome, failure budget, and access model. An extra reflection loop, hierarchy, or framework feature is not a substitute for those requirements.

| Concern | Practical next step | Evidence required |
|---------|---------------------|-------------------|
| Memory integrity | Keep source IDs, proposed/accepted state, correction and promotion policy | Adversarial and contradictory records cannot silently become trusted instructions |
| Retrieval and context | Retain raw records, format evidence on demand, measure token budgets and quality | Representative recall/abstention and answer-grounding comparison |
| Recovery and effects | Separate review, commit, and external-action boundaries; introduce destination idempotency | Failure injection before/after effects does not duplicate the action |
| Observability | Correlate run/thread, tool call, source IDs, scope decisions, and outcomes; redact content deliberately | A failure can be reconstructed without logging credentials or unnecessary personal data |
| Retention | Define checkpoint, domain, trace/export, and backup lifetimes | Deletion and expiry assertions hold for every system in the declared boundary |
| Concurrency | Test contention, lease expiry/renewal, and slow workers under a fixed workload | Measured correctness and latency under the expected load |
| Deployment | Separate privileges, migrations, and provider compatibility checks | Verified startup/upgrade/recovery on the intended environment |

Do not pick a vector store or index by a universal row-count threshold. Measure filtered recall, write rate, memory, index build cost, and query plans. [pgvector](https://github.com/pgvector/pgvector) documents HNSW/IVFFlat tradeoffs and filtering behavior; choosing an index is workload-dependent. No latency, throughput, or production scale is established here.

The additional teaching modules begin with [approval and resume](USE_CASES.md#1-approval-and-resume), [evidence-backed support](USE_CASES.md#2-evidence-backed-support), and [verified procedural memory](USE_CASES.md#3-verified-procedural-memory). Continue with incidents, research and analytics. [Extension operations](EXTENSIONS.md) includes the offline command, API shapes, permission split, and explicit provider comparison.

## Primary references

- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents): choosing deterministic and model-directed control flow.
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph): node boundaries, raw state, and error handling.
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence): checkpoints versus long-term stores.
- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts): approval/resume and replay behavior.
- [Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering): selecting model, tool, and lifecycle context.
- [V1 migration](https://docs.langchain.com/oss/python/migrate/langgraph-v1): current agent API and supported graph primitives.
- [pgvector](https://github.com/pgvector/pgvector): vector types, indexing, filtered ANN, and recall measurement.
- [PostgreSQL 16 SELECT locking clauses](https://www.postgresql.org/docs/16/sql-select.html#SQL-FOR-UPDATE-SHARE): queue-style row selection with `SKIP LOCKED`.
