"""CX Analyst Agent — tag churn risk + satisfaction signals for analytics."""
from __future__ import annotations

from core.llm import structured
from core.state import CXInsights, TicketState

SYSTEM = """You are a CX analyst. Read the ticket + how it was resolved and emit insights for analytics.

Fields:
- churn_risk            : 'high' if the customer threatens to leave, mentions a competitor, has a high-priority unresolved issue,
                          or repeated complaints. 'medium' for negative sentiment without explicit churn signals.
                          'low' otherwise.
- satisfaction_signal   : the customer's expected satisfaction at the END of this interaction.
- tags                  : 2-5 short snake_case tags useful for grouping (e.g. 'refund_request', 'sso_setup', 'sync_error').
- notes                 : one optional sentence on a non-obvious insight (a pattern, a missing KB doc, etc.) — leave null if nothing notable."""


def run(state: TicketState) -> TicketState:
    cls = state.classification
    t = state.ticket
    outcome = state.final_route or "unknown"
    user = f"""Ticket id: {t.id}
Customer plan: {t.customer.plan or "unknown"}, tenure_days: {t.customer.tenure_days}
Intent: {cls.intent if cls else "?"}, priority: {cls.priority if cls else "?"}, sentiment: {cls.sentiment if cls else "?"}
Outcome: {outcome}
Body excerpt: {t.body[:500]}

Emit CX insights."""

    cx = structured(SYSTEM, user, CXInsights, state=state, agent="cx_analyst")
    state.cx = cx
    state.log("cx_analyst", f"churn={cx.churn_risk} tags={cx.tags}")
    return state
