"""Published response contracts; decode PostgreSQL JSON instead of double encoding."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID
from pydantic import Json
from .common import StrictModel

Object = dict[str, Any] | Json[dict[str, Any]]


class ProposalRecord(StrictModel):
    proposal_id: UUID
    user_id: str
    target: str
    base_digest: str
    content: str
    digest: str
    revision: int
    status: Literal[
        "pending", "approved", "applying", "rejected", "cancelled", "applied"
    ]
    thread_id: str
    effect_key: UUID
    evidence: Object
    created_at: datetime
    applied_at: datetime | None
    effect_started_at: datetime | None
    diff: str | None = None


class RunResult(StrictModel):
    status: Literal[
        "pending", "approved", "applying", "rejected", "cancelled", "applied"
    ]
    revision: int | None = None


class PolicyRecord(StrictModel):
    document_id: UUID
    version: int
    user_id: str
    content: str
    digest: str
    refund_days: int
    valid_until: datetime


class AssertionRecord(StrictModel):
    assertion_id: UUID
    case_id: UUID
    user_id: str
    source: str
    origin: Literal["user_report", "operator_correction"]
    purchase_age_days: int
    digest: str
    supersedes: UUID | None
    created_at: datetime


class BoundedAnswer(StrictModel):
    claim: Literal["eligible", "ineligible", "abstain"]
    citations: list[str]


class SupportResult(StrictModel):
    answer: BoundedAnswer
    proposal: ProposalRecord | None
    sources: dict[str, str] | None = None


class AttemptRecord(StrictModel):
    attempt_id: UUID
    user_id: str
    source_csv: str
    artifact: str | None
    artifact_digest: str | None
    procedure: str
    spec_version: str
    checker_version: str
    environment: str
    verified: bool
    errors: list[str] | Json[list[str]]
    status: Literal["candidate", "promoted", "retired"]
    promoter: str | None
    created_at: datetime


class PromotionResult(StrictModel):
    status: Literal["promoted", "retired"]
    attempt_id: UUID | None = None


class ProcedureRecall(StrictModel):
    procedures: list[AttemptRecord]


class EventRecord(StrictModel):
    event_id: UUID
    incident_id: UUID
    user_id: str
    source_id: str
    source_revision: int
    occurred_at: datetime
    received_at: datetime
    arrival_order: int
    content: str
    digest: str
    supersedes: UUID | None
    revokes: UUID | None
    clock_conflict: bool | None = None


class TimelineResult(StrictModel):
    history: list[EventRecord]
    timeline: list[EventRecord]
    evidence_digest: str
    summary: Object


class SummaryResult(StrictModel):
    summary: Object
    stale: bool


class DocumentRecord(StrictModel):
    document_id: UUID
    version: int
    notebook_id: UUID
    user_id: str
    content: str
    digest: str


class ResearchReportRecord(StrictModel):
    report_id: UUID
    notebook_id: UUID
    user_id: str
    evidence_digest: str
    digest: str
    report: Object
    claims: list[str] | Json[list[str]]
    reviewer: str | None
    accepted: bool | None


class AcceptedResult(StrictModel):
    accepted: bool


class ResearchRecall(StrictModel):
    assertions: list[dict[str, Any]]


class AnalyticsResult(StrictModel):
    run_id: UUID
    plan: Object
    summary: Object
