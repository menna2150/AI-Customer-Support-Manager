"""RAG Agent — retrieve relevant KB chunks based on the ticket + classification.

Uses the predicted intent as a Chroma metadata filter when available, then falls
back to unfiltered search if the filter returns nothing.
"""
from __future__ import annotations

from config import RAG_TOP_K
from core.state import TicketState
from knowledge_base.retriever import retrieve


_KB_INTENTS = {
    "billing",
    "account_access",
    "technical_issue",
    "general_inquiry",
    "cancellation",
    "complaint",
}


def _build_query(state: TicketState) -> str:
    parts: list[str] = []
    if state.classification:
        parts.append(state.classification.summary)
        parts.append(f"intent: {state.classification.intent}")
    if state.ticket.subject:
        parts.append(state.ticket.subject)
    parts.append(state.ticket.body[:400])
    return " | ".join(parts)


def run(state: TicketState) -> TicketState:
    query = _build_query(state)

    intent_filter = None
    if state.classification and state.classification.intent in _KB_INTENTS:
        intent_filter = state.classification.intent

    chunks = retrieve(query, k=RAG_TOP_K, intent=intent_filter)
    state.kb_chunks = chunks
    titles = ", ".join(f"{c.doc_id}/{c.title}({c.score:.2f})" for c in chunks) or "(none)"
    state.log(
        "rag",
        f"retrieved {len(chunks)} chunks (intent_filter={intent_filter or 'none'}): {titles}",
    )
    return state
