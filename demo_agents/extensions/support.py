"""Bounded refund-policy vocabulary: membership and entailment are separate gates."""

from typing import Literal, TypedDict
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, AwareDatetime
from langgraph.graph import START, END, StateGraph
from .common import StrictModel, canonical, digest, locked, owned, permission
from .approval import Proposal, create_proposal

from .responses import PolicyRecord, AssertionRecord, SupportResult

router = APIRouter()


class Policy(StrictModel):
    document_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    content: str = Field(max_length=20000)
    refund_days: int = Field(ge=0, le=365)
    valid_until: AwareDatetime


class Assertion(StrictModel):
    assertion_id: UUID
    case_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    source: str = Field(max_length=20000)
    origin: Literal["user_report", "operator_correction"] = "user_report"
    purchase_age_days: int = Field(ge=0, le=10000)
    supersedes: UUID | None = None


class DraftRequest(StrictModel):
    proposal_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    document_id: UUID
    case_id: UUID


class Answer(StrictModel):
    claim: Literal["eligible", "ineligible", "abstain"]
    citations: list[str] = Field(max_length=8)


@router.post("/support/policies", response_model=PolicyRecord)
async def policy(body: Policy, request: Request):
    permission(request, "review")
    # The restricted example vocabulary has a deterministic semantics. Other
    # paragraphs are untrusted source text and cannot change the parsed rule.
    if body.content.splitlines()[:1] != [
        f"Refunds are available within {body.refund_days} days."
    ]:
        raise HTTPException(422, "Policy must begin with the supported refund rule")
    async with locked(request.app.state.pool, f"support:{body.user_id}") as conn:
        old = await conn.fetchrow(
            "SELECT * FROM ext_policies WHERE document_id=$1 ORDER BY version DESC LIMIT 1",
            body.document_id,
        )
        if old:
            owned(request, old)
            if old["user_id"] != body.user_id:
                raise HTTPException(409, "Document ownership cannot change")
        exact = await conn.fetchrow(
            "SELECT * FROM ext_policies WHERE document_id=$1 AND version=$2",
            body.document_id,
            body.version,
        )
        if exact:
            if (exact["digest"], exact["valid_until"]) != (
                digest(body.content),
                body.valid_until,
            ):
                raise HTTPException(409, "Version already contains different evidence")
            return dict(exact)
        if old and body.version != old["version"] + 1:
            raise HTTPException(409, "Policy revisions must be consecutive")
        row = await conn.fetchrow(
            "INSERT INTO ext_policies VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING *",
            body.document_id,
            body.version,
            body.user_id,
            body.content,
            digest(body.content),
            body.refund_days,
            body.valid_until,
        )
    return dict(row)


@router.post("/support/assertions", response_model=AssertionRecord)
async def assertion(body: Assertion, request: Request):
    permission(request, "submit")
    if body.origin == "operator_correction":
        permission(request, "review")
    if body.source.splitlines()[:1] != [
        f"Purchase age: {body.purchase_age_days} days."
    ]:
        raise HTTPException(
            422, "Assertion must begin with the supported age statement"
        )
    async with locked(request.app.state.pool, f"support:{body.user_id}") as conn:
        old = await conn.fetchrow(
            "SELECT * FROM ext_assertions WHERE assertion_id=$1", body.assertion_id
        )
        if old:
            owned(request, old)
            if any(str(old[k]) != str(v) for k, v in body.model_dump().items()):
                raise HTTPException(
                    409, "Assertion ID already contains different evidence"
                )
            return dict(old)
        if body.supersedes:
            prior = owned(
                request,
                await conn.fetchrow(
                    "SELECT * FROM ext_assertions WHERE assertion_id=$1",
                    body.supersedes,
                ),
            )
            if (prior["user_id"], prior["case_id"]) != (body.user_id, body.case_id):
                raise HTTPException(403, "Correction must remain in its original case")
            if await conn.fetchval(
                "SELECT 1 FROM ext_assertions WHERE supersedes=$1", body.supersedes
            ):
                raise HTTPException(409, "Correct the latest assertion instead")
        row = await conn.fetchrow(
            "INSERT INTO ext_assertions(assertion_id,case_id,user_id,source,origin,purchase_age_days,digest,supersedes) VALUES($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *",
            body.assertion_id,
            body.case_id,
            body.user_id,
            body.source,
            body.origin,
            body.purchase_age_days,
            digest(body.source),
            body.supersedes,
        )
    return dict(row)


async def retrieve(conn, user, document, case):
    policy = await conn.fetchrow(
        "SELECT * FROM ext_policies WHERE document_id=$1 AND user_id=$2 ORDER BY version DESC LIMIT 1",
        UUID(str(document)),
        user,
    )
    assertions = await conn.fetch(
        "SELECT a.* FROM ext_assertions a WHERE case_id=$1 AND user_id=$2 AND NOT EXISTS(SELECT 1 FROM ext_assertions b WHERE b.supersedes=a.assertion_id) ORDER BY assertion_id",
        UUID(str(case)),
        user,
    )
    # Transaction-start time can precede a long advisory-lock wait.
    now = await conn.fetchval("SELECT clock_timestamp()")
    if not policy or policy["valid_until"] <= now or not assertions:
        return {"claim": "abstain", "citations": [], "snapshot": {}}
    ages = {a["purchase_age_days"] for a in assertions}
    if len(ages) != 1:
        return {"claim": "abstain", "citations": [], "snapshot": {}}
    refs = [f"policy:{policy['document_id']}:{policy['version']}"] + [
        f"assertion:{a['assertion_id']}" for a in assertions
    ]
    return {
        "claim": "eligible"
        if next(iter(ages)) <= policy["refund_days"]
        else "ineligible",
        "citations": refs,
        "snapshot": {
            "kind": "support",
            "document_id": str(document),
            "case_id": str(case),
            "policy_version": policy["version"],
            "policy_digest": policy["digest"],
            "assertions": {str(a["assertion_id"]): a["digest"] for a in assertions},
        },
        "sources": {
            refs[0]: policy["content"],
            **{f"assertion:{a['assertion_id']}": a["source"] for a in assertions},
        },
    }


async def validate_snapshot(conn, user, evidence):
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,0))", f"support:{user}"
    )
    if evidence.get("kind") != "support":
        raise HTTPException(409, "Unknown evidence type")
    current = await retrieve(conn, user, evidence["document_id"], evidence["case_id"])
    if current["snapshot"] != evidence:
        raise HTTPException(409, "Evidence changed or expired; regenerate and review")


class SupportState(TypedDict, total=False):
    evidence: dict
    answer: dict
    accepted: bool


def build_support_graph(load_evidence, generate=None):
    async def retrieve_node(state):
        return {"evidence": await load_evidence()}

    async def draft(state):
        evidence = state["evidence"]
        result = (
            await generate(evidence)
            if generate
            else {k: evidence[k] for k in ("claim", "citations")}
        )
        try:
            answer = Answer.model_validate(result).model_dump()
        except ValueError:
            answer = {"claim": "abstain", "citations": []}
        return {"answer": answer}

    def verify(state):
        evidence, answer = state["evidence"], state["answer"]
        membership = set(answer["citations"]) <= set(evidence["citations"])
        support = answer["claim"] == evidence["claim"] and set(
            answer["citations"]
        ) == set(evidence["citations"])
        accepted = membership and support and answer["claim"] != "abstain"
        return {
            "accepted": accepted,
            "answer": answer if accepted else {"claim": "abstain", "citations": []},
        }

    graph = StateGraph(SupportState)
    for name, node in (
        ("retrieve", retrieve_node),
        ("draft", draft),
        ("verify", verify),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


@router.post("/support/drafts", response_model=SupportResult)
async def draft(body: DraftRequest, request: Request):
    permission(request, "submit")

    async def load():
        async with locked(request.app.state.pool, f"support:{body.user_id}") as conn:
            return await retrieve(conn, body.user_id, body.document_id, body.case_id)

    graph = build_support_graph(
        load, getattr(request.app.state, "support_generator", None)
    )
    result = await graph.ainvoke({}, {"recursion_limit": 8})
    if not result["accepted"]:
        return {"answer": result["answer"], "proposal": None}
    evidence = result["evidence"]
    content = canonical(result["answer"]) + "\n"
    proposal = await create_proposal(
        request,
        Proposal(
            proposal_id=body.proposal_id,
            user_id=body.user_id,
            target=f"support-{body.proposal_id}.json",
            base_digest=digest(""),
            content=content,
        ),
        evidence["snapshot"],
    )
    return {
        "answer": result["answer"],
        "sources": evidence["sources"],
        "proposal": proposal,
    }
