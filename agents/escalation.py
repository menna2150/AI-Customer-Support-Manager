"""Escalation Agent — produce a human-ready handoff packet."""
from __future__ import annotations

from core.llm import structured
from core.state import EscalationPacket, TicketState

SYSTEM = """You are preparing a handoff to a human support agent. Produce:
- summary                          : 3-5 bullet equivalents (as a paragraph) covering who, what, when, why escalated.
- suggested_next_steps             : 2-5 concrete actions the human should take next.
- customer_facing_acknowledgement  : a short message to send to the customer right now telling them a teammate will follow up.
                                     Match their language. Empathetic, no false promises about timelines.
- severity                         : urgent/high/medium/low based on classification + business impact.

Keep the summary factual. Do not invent customer details that weren't provided."""


def run(state: TicketState) -> TicketState:
    cls = state.classification
    plan = state.plan
    qa = state.qa
    t = state.ticket

    history = "\n".join(t.conversation_history) or "(none)"
    user = f"""Ticket id: {t.id}
Customer: {t.customer.name or t.customer.id} (plan: {t.customer.plan or "unknown"}, tenure_days: {t.customer.tenure_days})
Channel: {t.channel}
Language: {cls.language if cls else "en"}

Classification: intent={cls.intent if cls else "?"}, priority={cls.priority if cls else "?"}, sentiment={cls.sentiment if cls else "?"}
Planner rationale: {plan.rationale if plan else "(no plan)"}
QA notes: {qa.feedback if qa else "(none)"} ; issues: {qa.issues if qa else []}

Conversation history:
{history}

Latest message:
\"\"\"{t.body}\"\"\""""

    pkt = structured(SYSTEM, user, EscalationPacket, max_tokens=900, state=state, agent="escalation")
    state.escalation = pkt
    state.log("escalation", f"severity={pkt.severity} steps={len(pkt.suggested_next_steps)}")
    return state
