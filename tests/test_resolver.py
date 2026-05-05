from agents import resolver
from core.state import Classification, DraftReply, KBChunk, QAResult


def _setup(state):
    state.classification = Classification(
        intent="billing",
        priority="medium",
        sentiment="neutral",
        language="en",
        summary="refund",
        confidence=0.9,
    )
    state.kb_chunks = [
        KBChunk(doc_id="billing", title="Refund policy", text="14 days monthly, 30 days annual.", score=0.9)
    ]


def test_resolver_drafts_reply_with_citations(fake_llm, sample_state):
    _setup(sample_state)
    fake_llm.respond(
        DraftReply,
        DraftReply(
            text="Hi — totally understand. Per our policy you can request a refund within 30 days.",
            citations=["billing"],
        ),
    )

    out = resolver.run(sample_state)

    assert out.draft is not None
    assert out.draft.citations == ["billing"]
    assert "policy" in out.draft.text


def test_resolver_includes_qa_feedback_on_revise(fake_llm, sample_state):
    _setup(sample_state)
    sample_state.qa = QAResult(
        verdict="revise",
        issues=["wrong refund window"],
        feedback="Use 30 days for annual, not 14.",
    )
    captured = {}

    def _capture(system, user, schema):
        captured["user"] = user
        return DraftReply(text="revised", citations=["billing"])

    fake_llm.respond(DraftReply, _capture)

    resolver.run(sample_state)

    assert "30 days for annual" in captured["user"]
    assert "wrong refund window" in captured["user"]


def test_resolver_without_kb_signals_no_snippets(fake_llm, sample_state):
    sample_state.classification = Classification(
        intent="general_inquiry",
        priority="low",
        sentiment="positive",
        language="en",
        summary="praise",
        confidence=0.95,
    )
    sample_state.kb_chunks = []
    captured = {}

    def _capture(system, user, schema):
        captured["user"] = user
        return DraftReply(text="Thanks for the kind words!", citations=[])

    fake_llm.respond(DraftReply, _capture)

    resolver.run(sample_state)

    assert "no knowledge base snippets" in captured["user"]
