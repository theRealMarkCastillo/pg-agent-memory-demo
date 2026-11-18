# Review remediation and operating changes

The review findings are covered by API, graph, database, and pipeline regressions in
`tests/test_review_regressions.py`, `tests/test_agent_regressions.py`, and
`tests/test_pipeline_regressions.py`, `tests/test_followup_regressions.py`, and
`tests/test_main_sync_regressions.py`. The subsequent LangGraph modernization and
six-workflow review add `tests/test_langgraph_v1.py`, `tests/test_extension_*.py`,
and `tests/test_final_review_regressions.py`. The [final code/design report](DESIGN_REVIEW.md)
records five corrected defects and the October 24 result: **207 passed, 1 skipped**.
Existing integration tests now run against
fresh data per test, and assert real outcomes instead of accepting empty results.

## Start or upgrade

1. Keep the existing `.env` with model/database configuration. Run
   `python tools/configure_auth.py` once to generate `.env.auth`. It refuses to
   overwrite an existing file. Do not commit or share either credential file.
2. Run `make build` and `make up`. Startup applies versioned SQL migrations under
   an advisory lock and transaction. Schema changes also apply to existing volumes.
   All migrations run, including extension migrations `004`–`010` even if the
   extension host is not started. `009` creates a database role and requires
   role-creating migration privileges. The supplied local Compose database user
   has them; a production design must separate migration and runtime privileges.
3. For a new demo database, run `make seed`, then `make demo`. Seeding uses the
   separate admin credential; model-driven agents receive the scoped demo token.
   The companion demo deletes that user's companion records/checkpoints at its end
   and disables companion recording; it does not delete the user's other domains.
4. To use the API directly, send `Authorization: Bearer <token>`. `/health` alone
   is unauthenticated. Token configuration maps to permissions and allowed users,
   roles, projects/branches, workflows, and agents. A role request never grants a role.
5. Existing companion rows retain their data but need persisted vectors backfilled:
   POST `/companion/memory/{user_id}/reindex` with an authorized write credential.
   This atomically re-embeds nodes, complete assertions, and episode chunks. Migration
   `002_embedding_space.sql` marks existing vector spaces as unknown; companion vector
   reads/writes return 409 until reindexing succeeds. Plain unranked context reads,
   termination, and deletion remain available. Provider URL, model name, dimension,
   and optional `EMBEDDING_REVISION` identify the vector space. Bump the revision for
   provider weight changes behind an unchanged model name. Failed reindexing preserves
   the previous vectors and identity. Developer, enterprise, and trajectory vectors
   still require separate reingestion when changing the shared embedding configuration.

Stop old application/graph hosts and back up valued database and workspace data
before upgrading. LangGraph's saver applies its own checkpoint migrations. Use
fresh thread IDs for the original demo; resuming historical 0.6 checkpoints remains
unverified. See the [LangGraph upgrade boundary](GUIDE.md#upgrade-and-verification-boundary).
Migration `010` introduces `applying` and `effect_started_at`; older extension
hosts must not run against the upgraded approval lifecycle. Restart the updated
extension host and reconcile pending effects through its authorized `/run` API.

Research reports without `complete-labelled-evidence/2` remain stored but cannot
be accepted or returned as trusted assertions. Generate a new report ID and
review its current evidence. Do not retrofit protocol metadata onto old results.
See [extension recovery and limits](EXTENSIONS.md#recovery-and-limits) for both upgrades.

Existing duplicate symbols are retained in `migration_duplicate_symbols`, while
only the newest version remains in the current symbol index. Duplicate active
policies are archived. Migration refuses invalid cross-user provenance or out-of-range
scores rather than silently discarding data. Correct those records before retrying.
Previously lost Unicode names or overwritten history cannot be reconstructed by a migration.

Legacy companion checkpoints have arbitrary thread IDs without trustworthy user ownership.
They cannot safely be assigned to a user after the fact. Stop old runners before upgrading a populated checkpoint database. Export any
required operational history, then an administrator can POST
`/companion/legacy-checkpoints/purge` with
`{"confirmation":"delete unowned checkpoint history"}` to explicitly remove
unowned legacy checkpoint rows. Forgetting returns 409 while such rows remain,
rather than falsely claiming deletion. Retire exported history according to your
retention policy. New companion checkpoints use a user namespace and
are deleted with that user's memory. Do not describe legacy retained backups as forgotten.

## Memory and execution contracts

- Companion deletion removes that user's companion facts, provenance links,
  episodes/chunks, ephemerals, and companion PostgreSQL checkpoint threads.
  It also disables further companion recording. Other domains, extension evidence
  and checkpoints, workspace files, and project/branch hook excerpts are outside
  this endpoint's deletion boundary. Hook session removal has its own
  [procedure](AGENT_HOOKS.md#verify-and-remove-memory); extensions currently require
  disposable data because a general retention/deletion service is not implemented.
  POST `/companion/memory/{user_id}/resume` is an explicit application/user opt-in;
  the model has no resume tool. Generation checks reject delayed writes after deletion.
- Companion graphs use the deletion-aware PostgreSQL saver or no saver. Their public
  execution entry point is `ainvoke`; raw graph/checkpointer calls bypass application
  policy and must not be exposed to untrusted clients.
- Episodes use a caller-provided ingestion ID, or a content digest when omitted.
  Reusing an ID with different content is rejected. Embeddings complete before
  transactional publication; failed ingestion leaves no partial episode.
- Graph edges preserve closed versions and multiple source episodes. These are
  observed validity intervals and recording timestamps, not a complete bitemporal
  database with arbitrary retrospective correction. Evidence retains original text.
- Source symbol identity is project + branch + file + symbol name + symbol type.
  Upserts replace that symbol; DELETE `/developer/symbols` with project, branch,
  and file removes a deleted file's symbols. POST `/developer/symbols/reconcile` atomically replaces one file with the supplied
  `symbols` list, removing renamed/deleted entries; an empty list clears the file. Fully qualified symbol names distinguish nested definitions.
- Enterprise versions are archived when replaced; DELETE `/enterprise/documents`
  archives the active title/role version. Expiry requires a timezone-aware timestamp.
- Hybrid retrieval fuses independent lexical and vector ranks with RRF, k=60.
  Candidate limits and similarity thresholds are parameters requiring evaluation.
  HNSW iterative scans are configured on every pool checkout, including after reset.
- Fact vectors contain subject, predicate, and object. Retrieval limits candidates
  in SQL and caps salience influence. Target-only nodes do not displace assertions.
  Agent prompts select complete turns within a character budget; an oversized current
  turn produces an explicit budget error. This is a conservative character budget,
  not a model-specific token measurement.
- Swarm claims carry a random lease token and expire after five minutes. Completion
  requires the claimant, current lease, and IN_PROGRESS state. Expired tasks are
  reclaimable; old workers cannot complete a replacement worker's lease.
  Workers claim in application code before invoking a model, validate the lease before
  each tool dispatch, and stop on failed claims or lost leases. Completion receives
  application-owned task/agent/lease values. Remote effects still require idempotency
  or destination-side fencing: a pre-dispatch check cannot guarantee exactly-once execution.
- File tools reject traversal and symlinks. Shell tools are disabled unless a trusted
  local operator enables `ENABLE_SHELL_SANDBOX=1` with Docker available. Each command
  runs in a disposable container with only the workspace mounted, no network, no agent
  credentials, a read-only root filesystem, and resource limits. The Compose agent
  does not mount the Docker socket. Container isolation is not a VM security boundary.
- HTTP tools require an explicit HTTPS host allowlist and do not follow redirects.
  Allow only destinations authorized to receive model-supplied requests.
- Extension approval commits an immutable effect attempt before local file I/O.
  A retry acknowledges matching proposed content, or requires current evidence
  before a fresh write. Conflicting content returns 409 and preserves the attempt.
  Separate API and approval-node pools prevent waiting requests from starving
  graph execution. Support expiry is checked against the clock after lock waits;
  research requires complete labelled evidence for a one-sided conclusion.

## Verification

Use separate environments for the pinned engine and test/agent dependencies:

```sh
uv venv --python 3.12 .venv-engine
uv pip install --python .venv-engine/bin/python -r memory_engine/requirements.txt
uv venv --python 3.12 .venv-tests
uv pip install --python .venv-tests/bin/python -r tests/requirements.txt
.venv-tests/bin/python tests/run_isolated.py --engine-python .venv-engine/bin/python
```

The runner starts a disposable database and engine on loopback ports, generates test
credentials, resets test data between integration cases, and removes the containers
on exit. It also tests the real Docker shell boundary. Embeddings are deterministic
lexical fixtures: passing establishes application logic, not real semantic quality.
No production `.env` or remote provider credentials are loaded by this runner.

The separate extension demonstration runs twice in the same disposable database
and workspace and checks accepted record counts and applied-file modification times:

```bash
make extensions-demo PYTHON=.venv-tests/bin/python ENGINE_PYTHON=.venv-engine/bin/python
```

It also runs a scripted held-out comparison using authorized PostgreSQL procedure
recall. The [review evidence](DESIGN_REVIEW.md#verification) distinguishes these
checks from unrun live-provider, load, and remote-deployment validation.

Run provider evaluations separately against a disposable stack, with `LIVE_LLM_EVAL=1`
and explicitly supplied credentials. Extraction evaluation now matches canonical
predicates, exact normalized entities, and subject scope with one-to-one label matching.
When labels come from another model, results measure agreement, not verified truth.
`eval_recall.py` accepts labeled probes and reports precision/recall at the requested
limit; `--min-recall` makes misses fail the command. Unlabeled probes are inspection only.
Example probe: `[{"query":"Where do I live?","expected":[{"predicate":"LIVES_IN","object_value":"Tokyo","scope":"user"}]}]`.
Labels now match source entity as well as scope, predicate, and object. Implicit sources
are user (`user`, `me`, `i`, `myself`), companion (`companion`, `ai`, `self`), and shared
(`we`, `us`). Supply `source_entity` for another person or `source_aliases` for explicit
equivalent names, such as `"source_aliases":["Alice"]` on a user label. Names are never
assumed equivalent merely because they appear in the same memory scope.

Replay extracts turns concurrently and commits in source order, creating source episodes
before facts. Trace files are ordered by recorded start time when available, with filename
as a fallback for undated exports. Use chronologically named files if timestamps are absent.
Extracted terminations require an explicit `subject` (`user`, `self`, or `shared`);
ambiguous terminations are rejected. Tool termination filters are applied independently.
Trace downloads default to the preceding 30 days ending at current UTC time. Set
`--lookback-days`, or explicit timezone-aware `--end-time` and `--min-time` bounds,
for other windows. Cleanup attempts all requested users and exits nonzero if any deletion fails.

Normal `make down` and `make ci` preserve persistent volumes. `make clean` explicitly
removes them. Initialization/migration and deletion should be exercised on a disposable
copy before upgrading data that must be retained.
