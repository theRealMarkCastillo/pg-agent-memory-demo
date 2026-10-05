# Review remediation and operating changes

The review findings are covered by API, graph, database, and pipeline regressions in
`tests/test_review_regressions.py`, `tests/test_agent_regressions.py`, and
`tests/test_pipeline_regressions.py`. Existing integration tests now run against
fresh data per test, and assert real outcomes instead of accepting empty results.

## Start or upgrade

1. Keep the existing `.env` with model/database configuration. Run
   `python tools/configure_auth.py` once to generate `.env.auth`. It refuses to
   overwrite an existing file. Do not commit or share either credential file.
2. Run `make build` and `make up`. Startup applies versioned SQL migrations under
   an advisory lock and transaction. Schema changes also apply to existing volumes.
3. For a new demo database, run `make seed`, then `make demo`. Seeding uses the
   separate admin credential; model-driven agents receive the scoped demo token.
   The companion demo performs a real full-user deletion at its end.
4. To use the API directly, send `Authorization: Bearer <token>`. `/health` alone
   is unauthenticated. Token configuration maps to permissions and allowed users,
   roles, projects/branches, workflows, and agents. A role request never grants a role.
5. Existing companion rows retain their data but need persisted vectors backfilled:
   POST `/companion/memory/{user_id}/reindex` with an authorized write credential.
   This embeds complete assertions. Run it after changing the embedding model too;
   all stored vectors must use the same model and dimensionality.

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

- User deletion removes facts, all provenance links, episodes/chunks, ephemerals,
  and namespaced PostgreSQL checkpoints. It also disables further recording.
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
- File tools reject traversal and symlinks. Shell tools are disabled unless a trusted
  local operator enables `ENABLE_SHELL_SANDBOX=1` with Docker available. Each command
  runs in a disposable container with only the workspace mounted, no network, no agent
  credentials, a read-only root filesystem, and resource limits. The Compose agent
  does not mount the Docker socket. Container isolation is not a VM security boundary.
- HTTP tools require an explicit HTTPS host allowlist and do not follow redirects.
  Allow only destinations authorized to receive model-supplied requests.

## Verification

Use separate environments for the pinned engine and test/agent dependencies:

```sh
python3 -m venv .venv-engine
.venv-engine/bin/pip install -r memory_engine/requirements.txt
python3 -m venv .venv-tests
.venv-tests/bin/pip install -r tests/requirements.txt
.venv-tests/bin/python tests/run_isolated.py --engine-python .venv-engine/bin/python
```

The runner starts a disposable database and engine on loopback ports, generates test
credentials, resets test data between integration cases, and removes the containers
on exit. It also tests the real Docker shell boundary. Embeddings are deterministic
lexical fixtures: passing establishes application logic, not real semantic quality.
No production `.env` or remote provider credentials are loaded by this runner.

Run provider evaluations separately against a disposable stack, with `LIVE_LLM_EVAL=1`
and explicitly supplied credentials. Extraction evaluation now matches canonical
predicates, exact normalized entities, and subject scope with one-to-one label matching.
When labels come from another model, results measure agreement, not verified truth.
`eval_recall.py` accepts labeled probes and reports precision/recall at the requested
limit; `--min-recall` makes misses fail the command. Unlabeled probes are inspection only.
Example probe: `[{"query":"Where do I live?","expected":[{"predicate":"LIVES_IN","object_value":"Tokyo","scope":"user"}]}]`.

Replay extracts turns concurrently and commits in source order, creating source episodes
before facts. Trace files are ordered by recorded start time when available, with filename
as a fallback for undated exports. Use chronologically named files if timestamps are absent.

Normal `make down` and `make ci` preserve persistent volumes. `make clean` explicitly
removes them. Initialization/migration and deletion should be exercised on a disposable
copy before upgrading data that must be retained.
