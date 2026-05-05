import pytest

from agents import qa
from core.state import DraftReply, KBChunk, QAResult


def _with_draft(state):
    # Chunk text intentionally contains the policy keywords used in the draft
    # so the grounding gate passes through to the LLM verdict.
    state.kb_chunks = [
        KBChunk(
            doc_id="billing",
            title="Refund policy",
            text="Refund policy: every customer is eligible. We process refunds for monthly and annual plans.",
            score=0.8,
        )
    ]
    state.draft = DraftReply(
        text="Hello, here is our refund policy.", citations=["billing"]
    )
    return state


def test_qa_requires_draft(fake_llm, sample_state):
    with pytest.raises(AssertionError):
        qa.run(sample_state)


def test_qa_pass_increments_attempts(fake_llm, sample_state):
    _with_draft(sample_state)
    fake_llm.respond(QAResult, QAResult(verdict="pass", issues=[], feedback=None))

    out = qa.run(sample_state)

    assert out.qa.verdict == "pass"
    assert out.qa_attempts == 1


def test_qa_revise_with_feedback(fake_llm, sample_state):
    _with_draft(sample_state)
    fake_llm.respond(
        QAResult,
        QAResult(verdict="revise", issues=["tone"], feedback="Be more empathetic."),
    )

    out = qa.run(sample_state)

    assert out.qa.verdict == "revise"
    assert "Be more empathetic" in out.qa.feedback


def test_qa_attempts_count_across_calls(fake_llm, sample_state):
    _with_draft(sample_state)
    fake_llm.respond(
        QAResult,
        [
            QAResult(verdict="revise", issues=["x"], feedback="fix"),
            QAResult(verdict="pass", issues=[], feedback=None),
        ],
    )

    qa.run(sample_state)
    qa.run(sample_state)

    assert sample_state.qa_attempts == 2
    assert sample_state.qa.verdict == "pass"
