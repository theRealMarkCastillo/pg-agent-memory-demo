# Extension fixture contracts

The six examples use a separate `/extensions` API and `ext_` database tables.
Authenticated `users` grants scope every record. Credentials have a server-owned
`subject`; `extensions:read`/`write` allow API access, and `submit`, `review`,
`promote`, and `run` separately authorize actions. The model never receives a
credential that can review or promote. Tests use Alice and Bob with deliberately
similar data, and a separate reviewer credential scoped to Alice.

| Module | Fixture and independent expected result | Required adverse cases |
|---|---|---|
| Approval | Empty local file → exact reviewed UTF-8 revision | Stale digest/revision, forged scope/reviewer/thread, rejection, cancellation, pool-saturating retry burst, process death before review and after replacement, edit/cancel after an attempted effect, intervening approved write |
| Support | Versioned refund policy plus corrected purchase assertion | Expiry including lock waits, contradiction, unsupported claim/citation, cross-user evidence, stale review, embedded instructions, policy change after an unacknowledged file effect |
| Procedure | CSV `name,quantity` → sorted JSON objects with integer quantities | Wrong output, forged score, missing or deeply nested artifact, incompatible spec/checker/environment, retirement, unauthorized promotion |
| Incident | Synthetic service outage with late event and correction | Duplicate/conflicting event IDs, receipt vs occurrence order, impossible clock, revocation, inaccessible incident, concurrent ingestion |
| Research | Local papers with explicit support/contradiction statements | Unknown claim, invented quotation, superseded source, duplicate/adversarial text, omitted contrary evidence, legacy extraction protocol, unauthorized source, deterministic parallel merge |
| Analytics | Synthetic tenant orders | Forged tenant, arbitrary SQL/column/join/function, resource limit, empty scope, direct-role database isolation |

Stable fixture UUIDs in the demonstration namespace allow two successive runs to
converge on the same accepted records. Tests use fresh UUIDs and a disposable
database, and compare outputs against independently written expectations.

The approval workspace is owned by the graph-host process. Each user gets a
hashed subdirectory and basename-only targets. Application writers serialize on
the destination. Committed `applying` intent freezes the revision before file I/O.
Retry acknowledges matching proposed content without replacing it. A conflicting
later write leaves the attempted effect visible and returns 409. Atomic replacement
and digest reconciliation cover this local destination; arbitrary concurrent
operating-system writers and remote effects are outside this contract. A
host-configured crash marker exists solely to test death between file replacement
and database acknowledgement. API waiters and approval nodes use separate pools.

Research completeness is defined only over the labelled fixture grammar, not
general natural-language claims. Every expected unique statement must appear in
validated extraction before a one-sided conclusion is eligible. Reports carry
the extraction protocol; trusted recall excludes older reports even when previously
accepted. Procedure inputs are bounded to 100 CSV rows, 20,000 CSV characters,
and 30,000 artifact JSON characters, with exact expected schema/value equality.

See the [final review](../../DESIGN_REVIEW.md) for reproduced defects, corrections,
and the observed validation boundary, and [operations](../../EXTENSIONS.md) for
recovery and retention limits.
