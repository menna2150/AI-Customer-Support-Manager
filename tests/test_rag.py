from agents import rag
from core.state import Classification, KBChunk


def test_rag_populates_kb_chunks(fake_retrieve, sample_state):
    sample_state.classification = Classification(
        intent="billing",
        priority="medium",
        sentiment="neutral",
        language="en",
        summary="Customer wants a refund.",
        confidence=0.9,
    )
    fake_retrieve.set(
        [
            KBChunk(doc_id="billing", title="Refund policy", text="14 days for monthly...", score=0.82),
            KBChunk(doc_id="billing", title="Failed payments", text="...", score=0.41),
        ]
    )

    out = rag.run(sample_state)

    assert len(out.kb_chunks) == 2
    assert out.kb_chunks[0].doc_id == "billing"
    assert any("rag" in line for line in out.trace)


def test_rag_query_contains_summary_and_body(fake_retrieve, make_state_factory):
    state = make_state_factory(body="UNIQUE-RAG-BODY refund please")
    state.classification = Classification(
        intent="billing",
        priority="low",
        sentiment="neutral",
        language="en",
        summary="REFUND-SUMMARY-MARKER",
        confidence=0.7,
    )
    fake_retrieve.set([])

    rag.run(state)

    q = fake_retrieve.last_query
    assert q is not None
    assert "REFUND-SUMMARY-MARKER" in q
    assert "UNIQUE-RAG-BODY" in q


def test_rag_passes_intent_filter(fake_retrieve, sample_state):
    sample_state.classification = Classification(
        intent="billing",
        priority="medium",
        sentiment="neutral",
        language="en",
        summary="refund",
        confidence=0.9,
    )
    fake_retrieve.set([])

    rag.run(sample_state)

    assert fake_retrieve.last_intent == "billing"


def test_rag_skips_intent_filter_for_unmapped_intents(fake_retrieve, sample_state):
    sample_state.classification = Classification(
        intent="other",
        priority="low",
        sentiment="neutral",
        language="en",
        summary="weird",
        confidence=0.5,
    )
    fake_retrieve.set([])

    rag.run(sample_state)

    assert fake_retrieve.last_intent is None
