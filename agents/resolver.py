"""Resolver Agent — draft the customer-facing reply, grounded in KB chunks if present."""
from __future__ import annotations

from core.llm import structured
from core.state import DraftReply, TicketState

SYSTEM = """You are a senior customer support specialist. Write the reply that will be sent to the customer.

Rules:
- Tone: empathetic, concise, concrete. Acknowledge their situation in one sentence first, THEN give the steps.
- If KB snippets are provided, ground every factual claim in them. Cite them by doc_id in the citations field.
  Do NOT invent policies, time windows, or features that are not in the snippets.
- If KB snippets are NOT provided, only answer if the question is general/conversational. Otherwise reply
  with a short acknowledgement and say a teammate will follow up.
- Never disclose internal systems, other customers, or data outside our policy.
- Match the customer's language (e.g. reply in Arabic if the ticket is in Arabic).
- Sign off with a single line, no name placeholder.
- Plain text, no markdown headers."""


def _format_chunks(state: TicketState) -> str:
    if not state.kb_chunks:
        return "(no knowledge base snippets — answer only if generic)"
    out = []
    for c in state.kb_chunks:
        out.append(f"[{c.doc_id}] {c.title}\n{c.text}")
    return "\n\n---\n\n".join(out)


def _qa_feedback(state: TicketState) -> str:
    if state.qa and state.qa.verdict == "revise" and state.qa.feedback:
        return (
            "\n\nIMPORTANT — a previous draft was rejected by QA. "
            f"Issues: {'; '.join(state.qa.issues)}. "
            f"Feedback: {state.qa.feedback}\n"
            "Address every point above in this revised draft."
        )
    return ""


def run(state: TicketState) -> TicketState:
    t = state.ticket
    cls = state.classification
    user = f"""Customer: {t.customer.name or t.customer.id} (plan: {t.customer.plan or "unknown"})
Language: {cls.language if cls else "en"}
Sentiment: {cls.sentiment if cls else "neutral"}
Subject: {t.subject or ""}
Body:
\"\"\"{t.body}\"\"\"

Knowledge base snippets:
{_format_chunks(state)}{_qa_feedback(state)}

Write the reply now."""

    draft = structured(SYSTEM, user, DraftReply, max_tokens=800, state=state, agent="resolver")
    state.draft = draft
    state.log(
        "resolver",
        f"drafted reply ({len(draft.text)} chars, citations={draft.citations})",
    )
    return state
