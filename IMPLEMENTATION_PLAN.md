# Implementation Plan: Six Additional Agent Use Cases

Created October 24, 2026. Status: complete at the deterministic local acceptance boundary below, including the subsequent [final code and design review](DESIGN_REVIEW.md).

## Objective and scope

Implement all six modules proposed in [USE_CASES.md](USE_CASES.md) as small, runnable, documented teaching examples on the modernized LangGraph stack. Complete them in the order below. Each module must exercise its actual graph, API, and persistence boundaries rather than only simulate the intended behavior in prose.

The required completion boundary is a deterministic, locally verified implementation using PostgreSQL and fixture inputs/model responses. Include an explicit opt-in provider evaluation path for model-dependent modules; report live results separately when run. Provider quality, production readiness, remote deployment, and exactly-once behavior at untested destinations are outside this goal.

No additional vendors, hosted services, agent hierarchies, or framework migrations are required. Keep the existing six patterns and twelve-scenario demo working. Commit, push, deployment, and external messaging are not part of this plan.

## Starting state and ownership

- The working tree already contains the LangGraph modernization, revised material, and separate agent-hook work. Preserve those changes and record their status before implementation.
- The previous modernization suite observed 164 passed and one skipped. Refresh the baseline rather than treating that result as current for later revisions.
- The scoped wrapper accepts new-turn dictionaries, not authorized resume requests. PostgreSQL fan-out resume has a native graph regression; full process restart and file-effect recovery need new evidence.
- Use the existing migration system and choose the next available migration number at implementation time. Do not overwrite the agent-hook migration or edit historical applied migrations.
- Keep model requests out of deterministic checks. Run real database tests with the disposable test runner and separate engine/test environments.

## Delivery sequence

| Stage | Deliverable | Dependency | Exit condition |
|-------|-------------|------------|----------------|
| 0 | Baseline and fixture contracts | Existing working tree | Current checks recorded; scope, fixtures, and failure cases written |
| 1 | Shared application boundaries | Stage 0 | Authorized module scopes, schemas, migrations, and test isolation established |
| 2 | Approval and resume | Stage 1 | Reject/edit/approve and actual process restart verified; local effect reconciles across retries |
| 3 | Evidence-backed support | Stage 2 review mechanism | Versioned evidence, corrections, abstention, and reviewed drafts demonstrated |
| 4 | Verified procedural memory | Stage 1; reuse review where needed | Independent checker controls promotion; failed/retired attempts excluded from trusted reuse |
| 5 | Incident timeline | Stage 1 | Late events, corrections, duplicate ingestion, and source revocation demonstrated |
| 6 | Research evidence notebook | Stages 2–3 evidence/review conventions | Versioned sources and contradictions survive parallel extraction and aggregation |
| 7 | Read-only analytics | Stage 1 | Typed plan validation and actual restricted database access both enforce scope |
| 8 | Integrated delivery and review | All modules | Full regressions, repeatable demo, documentation, and final evidence report pass |

Stage numbers describe execution order; they do not require a generic framework spanning all modules. Share only stable contracts that remove actual duplication.

## Stage 0: Baseline and contracts

1. Record Git status, existing changes, dependency versions, and the current disposable-suite result. Diagnose failures before attributing them to new work.
2. Define a small fixture set for each module with explicit expected records, outputs, denials, and failure behavior. Keep expected answers/checkers independent from model output.
3. Establish stable fixture IDs and namespaces so existing data and demonstrations remain isolated.
4. Record the module contracts and intended commands in a progress/evidence section of this plan. Update statuses as stages pass, with commands and limitations.

## Stage 1: Shared application boundaries

1. Add module permissions and ownership checks through the existing authentication mechanism. Distinguish submitting a proposal/attempt from reviewing or promoting it.
2. Add typed request/response contracts and versioned migrations with uniqueness, ownership, and status-transition constraints. Store source/version IDs and evidence digests where they are needed.
3. Add an application service for authorized checkpoint resume. Resolve the owner and thread from persisted application records, reconstruct trusted runtime context, and validate the resume payload. Reject raw caller-selected checkpoint ownership.
4. Make deterministic model outputs injectable into the same graphs used by the demonstrations. Add live adapters only where generation is useful; keep validation and authorization in application code.
5. Extend disposable test setup/reset for new tables and roles. Add failure-injection hooks only at boundaries required to prove recovery.

Acceptance: migrations are repeatable; unauthorized identities cannot retrieve or resume another owner's records; the original suite remains passing. Do not expose unrestricted native graph invocation, database credentials, or model-controlled promotion tools.

## Stage 2: Approval and resume

Deliver a proposal/review API, a LangGraph interrupt/resume workflow, and a local fixture-file demonstration.

- Persist the owner, target, base digest, proposed revision/diff, proposal digest, reviewer, decision, thread, and effect key.
- Review binds to an exact revision. Editing produces a new reviewable revision; an old approval cannot authorize it. Authenticate the reviewer separately from the model.
- Keep proposal construction, review interruption, and effect application in separate nodes. Reject changed base content before application.
- Use atomic local-file replacement plus destination-state reconciliation. On retry, distinguish an already-applied revision from an unexpected conflicting file. Protect concurrent reviewers/applicators with an explicit locking or compare-and-set contract.
- Test rejection, edit/reapproval, forged owner/reviewer, stale revision/base, duplicate approval, concurrent approval, cancellation, and crash after file application but before result recording.
- Stop and restart the graph-host process while review is pending, then resume through the authorized service against the same PostgreSQL checkpoint and workspace.

Acceptance: rejection has zero file effects; only the approved revision applies; restart preserves review state; duplicate/retried application converges to one observed local update. Report this result as local destination reconciliation, not a guarantee for arbitrary remote effects.

## Stage 3: Evidence-backed support

Deliver a single support workflow that combines current policy evidence and user-scoped case assertions, generates a typed draft, and routes review through Stage 2 without sending messages externally.

- Preserve document/version IDs, case ownership, assertion origin, source IDs, correction links, and the draft's evidence snapshot.
- Verify that every cited source belongs to the authorized retrieval result. Separate citation membership from whether the cited text supports the claim.
- Use a small fixture-supported claim vocabulary or deterministic validators for the initial example; ambiguous or unsupported claims abstain or require review rather than receiving a false grounding guarantee.
- Corrections supersede the appropriate assertion while preserving history. Recheck evidence freshness before accepting a reviewed draft; a policy revision makes the stale draft require regeneration/review.
- Test current/expired/unauthorized policy, cross-user near matches, contradictory assertions, corrections, invented citations, no evidence, and instructions embedded in retrieved content.

Acceptance: the demo shows a corrected answer, resolved versioned citations, an abstention, and stale-evidence rejection. Document residual model entailment and prompt-injection risks.

## Stage 4: Verified procedural memory

Deliver a fixed CSV-to-JSON task family with independent output validation and versioned procedure promotion.

- Store attempts independently of the existing model-supplied trajectory score. Record specification/checker versions, environment identity, artifacts, and evidence digests.
- An application-owned checker validates actual artifacts. Only authorized promotion can move a verified procedure into trusted recall.
- Preserve rejected attempts for diagnostics; trusted retrieval excludes unverified, retired, incompatible, and obsolete versions.
- Test wrong output, missing evidence, forged high score, checker/spec changes, duplicate attempts, unauthorized promotion, retirement, and applicability mismatch.
- Provide a held-out comparison harness that holds task set, model, and tool budget constant with and without procedure retrieval. Scripted runs demonstrate mechanics; only observed provider runs can support claims about model improvement.

Acceptance: a plausible failed attempt cannot enter trusted recall, a valid attempt can be promoted, and retirement/version changes alter eligibility. Do not execute arbitrary instructions extracted from hook summaries.

## Stage 5: Incident timeline

Deliver append-only event ingestion and a source-linked timeline/summary graph using synthetic incidents.

- Store event identity, incident scope, occurrence/receipt timestamps, source revision, digest, and correction/revocation links.
- Define deterministic projection rules for ordering, corrections, clock conflicts, and revoked evidence. Preserve arrival history separately from the current projection.
- Summaries label hypotheses separately from observed events. Recompute or mark summaries stale when their evidence changes.
- Test out-of-order and duplicate events, conflicting clocks, explicit corrections, revoked sources, inaccessible incidents, and concurrent ingestion.

Acceptance: a late event changes the projected timeline without erasing receipt history; sources resolve correctly; similar incidents remain isolated. No remediation actions or causal-certainty claims are required.

## Stage 6: Research evidence notebook

Deliver a local-document workflow with versioned evidence, bounded parallel extraction, aggregation, and human-reviewed reusable assertions.

- Store document revisions/digests, claims, exact evidence references, and support/contradiction/insufficient-evidence relationships.
- Route eligible sources to isolated `Send` workers and aggregate structured results. Preserve contradictory evidence and unknowns.
- Keep generated reports separate from source material so reports cannot become self-confirming evidence on later runs.
- Test unsupported claims, contradiction, superseded/duplicate documents, adversarial source instructions, inaccessible documents, and deterministic merge behavior.
- Include a sequential baseline using the same fixtures; measure provider cost/latency only when a provider run is explicitly configured.

Acceptance: the report cites exact eligible source versions, preserves disagreement, rejects invented evidence, and requires authorized review before assertions enter reusable memory.

## Stage 7: Read-only analytics

Deliver a strict typed query-plan DSL over a small synthetic operational dataset; no free-form SQL endpoint.

- Permit named views, selected columns, fixed aggregates, bounded filters, row limits, and approved join shapes. Compile plans to parameterized application-authored SQL.
- Use a dedicated database role with no mutation privileges and tenant isolation enforced in the database. Keep its connection separate from privileged migration/application connections.
- Use statement timeouts and bounded result sizes. Restrict exposed functions and views as well as table privileges.
- Record the executed plan and observed result used by the answer. Treat an empty authorized result as scoped absence, not global absence.
- Test valid aggregates, forged tenants, forbidden columns, mutation-shaped plans, disallowed joins, excessive limits, expensive execution, empty results, and direct restricted-role access bypassing the application validator.

Acceptance: invalid plans are denied before execution, database isolation still holds when the validator is bypassed, resource limits are exercised, and summaries match observed rows.

## Stage 8: Integration, material, and final verification

1. Add a separate extension-demo command/entrypoint so the original twelve scenarios remain a stable teaching path. Support a deterministic mode with no provider requests and an explicit opt-in live mode where applicable.
2. Seed extension fixtures repeatably. Running extension demos twice must not duplicate accepted events, decisions, promoted procedures, or local effects. Document retained history and intentional resets.
3. Run the full disposable suite, including the actual restart/recovery case and restricted database-role checks. Verify scoped credentials, test isolation, migration repeatability, and resource cleanup.
4. Perform an implementation review against the written contracts. Fix actionable correctness issues before marking stages complete; rerun relevant checks after fixes and then the full integration gate.
5. Update README, GUIDE, RUNBOOK, and USE_CASES with actual commands, diagrams, examples, tested behavior, and remaining limits. Promote a module from proposed only after its acceptance checks pass.
6. Validate documentation links/examples and record the final dependency/code boundary, commands, outcomes, skips, and any provider evaluations separately. Preserve unrelated work and leave changes reviewable.

## Definition of done

The implementation goal is complete only when all six modules have runnable implementations, meaningful passing acceptance checks, migrations/authorization where required, a repeatable deterministic extension demo, and updated teaching material. The original deterministic suite must remain passing, with skips explained.

Live-provider quality can remain explicitly unverified if no provider evaluation is configured; the repository must supply the opt-in evaluation path and must not substitute scripted results for real-model evidence. A finished plan or a passing happy-path demo alone does not complete this goal.

## Progress and evidence

| Stage | Status | Evidence |
|-------|--------|----------|
| 0. Baseline/contracts | Passed | Refreshed disposable suite: 164 passed, 1 skipped (39.21s). Fixture contracts: `demo_agents/extensions/CONTRACTS.md` |
| 1. Shared boundaries | Passed | Separate authenticated host; strict request and published response schemas across 21 paths; ownership and submit/review/promote/run permissions; migrations `004`–`010` reused across host starts; isolated table reset and tenant logins |
| 2. Approval/resume | Passed | Four approval tests cover revision binding, denial, cancellation, concurrent retries, actual host kill during review and exit after file replacement, then restart/reconciliation |
| 3. Support | Passed | Six support cases cover correction, contradiction, expiry, missing/cross-user evidence, stale review/application, malformed answers and invented/incomplete citations |
| 4. Procedures | Passed | Eight procedure cases cover actual artifact verification, forged scores, incompatible versions, duplicate attempts, promotion permission and retirement; held-out scripted comparison uses authorized PostgreSQL recall |
| 5. Incidents | Passed | Two integration tests cover arrival/occurrence order, corrections, revocation, stale summaries, future clocks, scope and concurrent duplicate ingestion |
| 6. Research | Passed | Two integration/graph tests cover versioned evidence, contradiction/unknowns, sequential/parallel equivalence, invented quotes, scope, review and freshness; demo reads local fixture documents |
| 7. Analytics | Passed | Eleven cases cover strict plan rejection, aggregates/join/empty results, direct restricted-login RLS/column/mutation/role denials, and actual statement timeout through the graph |
| 8. Integration/material | Passed | Initial acceptance: 197 passed, 1 skipped (67.02s). Final review: **207 passed, 1 skipped** (79.19s), including ten new adverse cases; two-run offline demo preserves accepted counts and file mtimes; evidence below and in the final review |

### Final acceptance evidence — October 24, 2026

```bash
.venv-tests/bin/python tests/run_isolated.py --engine-python .venv-engine/bin/python
.venv-tests/bin/python tests/run_isolated.py --engine-python .venv-engine/bin/python --extensions-demo
.venv-tests/bin/python -m demo_agents.extensions.evaluate --mode scripted --output /tmp/pg-extension-evaluation.json
uv pip check --python .venv-tests/bin/python
uvx ruff check --select F demo_agents/extensions memory_engine/db.py tests/extension_host.py tests/test_extension_*.py tests/test_final_review_regressions.py
uvx ruff format --check demo_agents/extensions memory_engine/db.py tests/extension_host.py tests/test_extension_*.py tests/test_final_review_regressions.py tests/run_isolated.py
git diff --check
```

The final full suite adds 33 extension cases and ten final-review cases to the refreshed 164-pass baseline. The one skip is the optional real-model extraction test, gated on `LIVE_LLM_EVAL=1` and provider credentials; the deterministic runner intentionally clears live settings. One existing `python_multipart` deprecation warning remains. Separate engine and graph/test dependency environments were retained.

The demo's first and second runs both observed 3 proposals, 2 reviews, 2 attempts, 2 incident events, 2 research documents, 2 research reports and 1 analytics run. No accepted record count or applied-file modification time changed on the second run. Its held-out scripted evaluation used actual authorized PostgreSQL procedure recall and matched sequential/parallel research output. Standalone evaluation additionally records protocol/spec/checker/environment versions, dataset and implementation digests, Git revision/dirty state, model identity and provider usage/latency when available.

Implementation review found and fixed an integration issue: approval checkpoint names initially triggered the original companion legacy-checkpoint safeguard. Approval now uses the existing `domain:scope-hash:thread` convention; the safeguard remains intact and the original forget/seed regressions pass. Malformed support outputs fail closed to abstention. Analytics identity is bound to a distinct login via `session_user`, so caller-controlled tenant settings cannot change scope. Response contracts decode PostgreSQL JSON into API objects, and per-run diagnostic logs avoid collisions between disposable runners.

Initial documentation acceptance parsed/rendered seven Markdown files and parsed 23 shell examples and six SQL blocks. The final review parsed/rendered all ten repository Markdown documents, validated relative links/anchors/table columns, and parsed 30 shell examples and six SQL blocks. This includes the older remediation and hook guides; results are recorded in [DESIGN_REVIEW.md](DESIGN_REVIEW.md#verification). All 22 operations across 21 extension paths expose response schemas. [EXTENSIONS.md](EXTENSIONS.md) records actual commands, permissions, diagrams, recovery boundaries and provider evaluation. Disposable hosts, pools, containers and workspaces were closed/removed by the runners.

### Final review corrections — October 24, 2026

Eight new adverse cases first failed against the pre-review implementation, then
passed after correcting five defects: approval pool starvation; editable/cancellable
unacknowledged file effects; incomplete research extraction; policy expiry evaluated
at transaction start; and deeply nested artifact JSON escaping as an internal error.
Two additional cases cover a later approved write after a crash and exclusion of
legacy research reports. All ten pass in the 207-pass full suite.

Migration `010` adds durable effect intent. Approval nodes use a separate pool;
attempted effects cannot be edited/cancelled and recovery preserves conflicting
history. Research reports include extraction coverage and a protocol version;
trusted recall excludes old reports. Support checks the current clock after lock
waits. The artifact checker persists parse failures as diagnostics. The final
two-run demo again observed identical record counts and file modification times.

The design review retains explicit workflows, application-owned authority, domain
evidence separate from checkpoints, and independent validation. It does not add
more agents to solve concurrency or evidence-integrity problems. All documentation
now distinguishes companion deletion from extension and hook lifecycles.

No live provider quality or production-load claim is made. Support/research use intentionally bounded fixture semantics; procedure checking verifies observed artifacts before authorized promotion; local-file reconciliation does not prove arbitrary remote effects. Extension evidence/checkpoint retention and deployment privilege separation remain explicit operating limitations. No commit, push, deployment or external message was performed; prior modernization and separate hook changes remain in the working tree.
