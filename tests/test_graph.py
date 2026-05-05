"""End-to-end graph routing tests, using the real agents but a mocked LLM + retriever.

Verifies the conditional edges in core.graph:
  • planner.route=='resolve_with_kb' → rag → resolver
  • planner.route=='resolve_direct'  → resolver (no rag)
  • planner.route=='escalate'        → escalation (no resolver/qa)
  • qa verdict 'revise' loops back to resolver
  • qa verdict 'revise' hitting QA_MAX_RETRIES escalates
  • qa verdict 'escalate' jumps straight to escalation
  • cx_analyst always runs at the end
"""
from __future__ import annotations

from core.graph import run_ticket
from core.state import (
    CXInsights,
    Classification,
    DraftReply,
    EscalationPacket,
    KBChunk,
    Plan,
    QAResult,
)


# Reusable canned responses --------------------------------------------------

CLS_BILLING = Classification(
    intent="billing",
    priority="medium",
    sentiment="neutral",
    language="en",
    summary="Refund question.",
    confidence=0.9,
)
CLS_FRUSTRATED = Classification(
    intent="complaint",
    priority="urgent",
    sentiment="frustrated",
    language="en",
    summary="Double-charge complaint.",
    confidence=0.95,
)
PLAN_KB = Plan(route="resolve_with_kb", needs_kb=True, rationale="Standard policy q.")
PLAN_DIRECT = Plan(route="resolve_direct", needs_kb=False, rationale="Generic praise.")
PLAN_ESCALATE = Plan(route="escalate", needs_kb=False, rationale="Frustrated + refund edge.")

DRAFT_OK = DraftReply(text="Hi! Per our refund policy you are eligible.", citations=["billing"])
QA_PASS = QAResult(verdict="pass", issues=[], feedback=None)
QA_REVISE = QAResult(verdict="revise", issues=["tone"], feedback="Be warmer.")
QA_ESCALATE = QAResult(verdict="escalate", issues=["legal"], feedback="Outside my scope.")

ESC_PACKET = EscalationPacket(
    summary="Frustrated customer reports double-charge.",
    suggested_next_steps=["Verify in Stripe", "Refund if duplicate"],
    customer_facing_acknowledgement="Hi — I'm routing this to our billing specialist now.",
    severity="urgent",
)
CX_DEFAULT = CXInsights(
    churn_risk="medium",
    satisfaction_signal="neutral",
    tags=["billing"],
    notes=None,
)


# ---------------------------------------------------------------------------

def test_graph_resolve_with_kb_path(fake_llm, fake_retrieve, sample_state):
    fake_retrieve.set([KBChunk(doc_id="billing", title="Refund", text="Refund policy: customers are eligible for refunds within the documented window.", score=0.8)])
    fake_llm.respond(Classification, CLS_BILLING)
    fake_llm.respond(Plan, PLAN_KB)
    fake_llm.respond(DraftReply, DRAFT_OK)
    fake_llm.respond(QAResult, QA_PASS)
    fake_llm.respond(CXInsights, CX_DEFAULT)

    out = run_ticket(sample_state)

    assert out.final_route == "auto_reply"
    assert out.final_response == DRAFT_OK.text
    assert out.kb_chunks, "RAG should have populated kb_chunks"
    assert out.escalation is None
    assert out.cx is not None


def test_graph_resolve_direct_path_skips_rag(fake_llm, fake_retrieve, sample_state):
    # If rag.run is called, fake_retrieve.last_query becomes non-None.
    fake_llm.respond(Classification, CLS_BILLING)
    fake_llm.respond(Plan, PLAN_DIRECT)
    fake_llm.respond(DraftReply, DraftReply(text="Thanks for reaching out!", citations=[]))
    fake_llm.respond(QAResult, QA_PASS)
    fake_llm.respond(CXInsights, CX_DEFAULT)

    out = run_ticket(sample_state)

    assert out.final_route == "auto_reply"
    assert out.kb_chunks == []
    assert fake_retrieve.last_query is None, "RAG should be skipped on resolve_direct"


def test_graph_planner_escalate_skips_resolver(fake_llm, fake_retrieve, sample_state):
    fake_llm.respond(Classification, CLS_FRUSTRATED)
    fake_llm.respond(Plan, PLAN_ESCALATE)
    fake_llm.respond(EscalationPacket, ESC_PACKET)
    fake_llm.respond(CXInsights, CX_DEFAULT)
    # DraftReply / QAResult intentionally NOT registered — would raise if called.

    out = run_ticket(sample_state)

    assert out.final_route == "escalated"
    assert out.escalation is not None
    assert out.escalation.severity == "urgent"
    assert out.draft is None
    assert out.qa is None
    assert fake_llm.calls_for(DraftReply) == []
    assert fake_llm.calls_for(QAResult) == []


def test_graph_qa_revise_then_pass_loops_resolver(fake_llm, fake_retrieve, sample_state):
    fake_retrieve.set([KBChunk(doc_id="billing", title="Refund", text="Refund policy: customers are eligible for refunds within the documented window.", score=0.8)])
    fake_llm.respond(Classification, CLS_BILLING)
    fake_llm.respond(Plan, PLAN_KB)
    fake_llm.respond(
        DraftReply,
        [
            DraftReply(text="first attempt", citations=["billing"]),
            DraftReply(text="revised attempt", citations=["billing"]),
        ],
    )
    fake_llm.respond(QAResult, [QA_REVISE, QA_PASS])
    fake_llm.respond(CXInsights, CX_DEFAULT)

    out = run_ticket(sample_state)

    assert out.final_route == "auto_reply"
    assert out.final_response == "revised attempt"
    assert out.qa_attempts == 2
    # Resolver was called twice, QA twice.
    assert len(fake_llm.calls_for(DraftReply)) == 2
    assert len(fake_llm.calls_for(QAResult)) == 2


def test_graph_qa_max_retries_escalates(fake_llm, fake_retrieve, sample_state):
    fake_retrieve.set([KBChunk(doc_id="billing", title="Refund", text="Refund policy: customers are eligible for refunds within the documented window.", score=0.8)])
    fake_llm.respond(Classification, CLS_BILLING)
    fake_llm.respond(Plan, PLAN_KB)
    fake_llm.respond(
        DraftReply,
        [
            DraftReply(text="attempt 1", citations=["billing"]),
            DraftReply(text="attempt 2", citations=["billing"]),
        ],
    )
    fake_llm.respond(QAResult, [QA_REVISE, QA_REVISE])
    fake_llm.respond(EscalationPacket, ESC_PACKET)
    fake_llm.respond(CXInsights, CX_DEFAULT)

    out = run_ticket(sample_state)

    assert out.final_route == "escalated"
    assert out.escalation is not None
    assert out.qa_attempts == 2


def test_graph_qa_escalate_verdict(fake_llm, fake_retrieve, sample_state):
    fake_retrieve.set([KBChunk(doc_id="policies", title="Limits", text="Refund policy: customers are eligible for refunds within the documented window.", score=0.8)])
    fake_llm.respond(Classification, CLS_BILLING)
    fake_llm.respond(Plan, PLAN_KB)
    fake_llm.respond(DraftReply, DRAFT_OK)
    fake_llm.respond(QAResult, QA_ESCALATE)
    fake_llm.respond(EscalationPacket, ESC_PACKET)
    fake_llm.respond(CXInsights, CX_DEFAULT)

    out = run_ticket(sample_state)

    assert out.final_route == "escalated"
    assert out.qa.verdict == "escalate"
    assert out.escalation is not None
