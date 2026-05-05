"""Tests for the programmatic grounding check used by the QA agent."""
from agents import qa
from core.grounding import check_grounding, find_factual_claims
from core.state import DraftReply, KBChunk, QAResult


def test_grounding_passes_when_numbers_appear_in_chunks():
    chunks = ["Refund window is 14 days for monthly plans and 30 days for annual."]
    draft = "We offer a refund within 30 days for annual plans."
    assert check_grounding(draft, chunks) == []


def test_grounding_flags_invented_number():
    chunks = ["Refund window is 14 days for monthly plans and 30 days for annual."]
    draft = "We offer a refund within 60 days for annual plans."
    issues = check_grounding(draft, chunks)
    assert len(issues) == 1
    assert "60" in issues[0]


def test_grounding_passes_for_pure_greeting():
    chunks = ["Refund window is 14 days for monthly plans."]
    draft = "Hello, thank you for reaching out."
    # No numbers, no policy keywords -> no claims to check
    assert find_factual_claims(draft) == []
    assert check_grounding(draft, chunks) == []


def test_grounding_flags_unsupported_policy_claim():
    chunks = ["Account access is via the Settings menu."]
    draft = "You can request a refund for any reason at any time."
    issues = check_grounding(draft, chunks)
    assert any("policy claim" in i for i in issues)


def test_qa_uses_grounding_gate(fake_llm, sample_state):
    """QA short-circuits with revise when grounding fails — no LLM call."""
    sample_state.kb_chunks = [
        KBChunk(doc_id="billing", title="Refund", text="Refund within 14 days.", score=0.9)
    ]
    sample_state.draft = DraftReply(
        text="You can refund within 60 days.", citations=["billing"]
    )

    out = qa.run(sample_state)

    assert out.qa is not None
    assert out.qa.verdict == "revise"
    # Should have produced issues mentioning the invented number.
    assert any("60" in i for i in out.qa.issues)
    # Crucially: no LLM call needed.
    assert fake_llm.calls_for(QAResult) == []


def test_qa_skips_grounding_when_no_kb(fake_llm, sample_state):
    """Without KB chunks (resolve_direct path), grounding is bypassed; LLM judges."""
    sample_state.kb_chunks = []
    sample_state.draft = DraftReply(text="Thanks for reaching out!", citations=[])
    fake_llm.respond(QAResult, QAResult(verdict="pass", issues=[], feedback=None))

    out = qa.run(sample_state)

    assert out.qa.verdict == "pass"
    assert len(fake_llm.calls_for(QAResult)) == 1
