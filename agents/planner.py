"""Planner Agent — decide the route: resolve_with_kb / resolve_direct / escalate."""
from __future__ import annotations

from core.llm import structured
from core.state import Plan, TicketState

SYSTEM = """You are the planner for a support workflow. Given a classified ticket, choose ONE route.

DEFAULT TO 'resolve_with_kb' for any policy or how-to question. Most tickets should be answered, not escalated. A calm, accurate reply often de-escalates a frustrated customer better than a handoff.

ONLY escalate when AT LEAST ONE of these is clearly true:
  1. The customer explicitly demands a human or threatens chargeback/legal action.
  2. The action requires backend/database access that an answer alone can't provide
     (account merges, manual data fixes, charge reversals).
  3. The request is clearly outside policy (refund 6 months in, contract exception).
  4. A legal, security, or compliance concern is raised.

Routes:
- 'resolve_with_kb' : the question is about billing, account access, technical how-tos, cancellation,
                      or any documented policy. needs_kb=true. THIS IS THE DEFAULT.
- 'resolve_direct'  : pure acknowledgement, praise, or generic chat that doesn't reference any policy.
                      needs_kb=false.
- 'escalate'        : only when one of the four conditions above is met.

Frustration alone is NOT a reason to escalate — answer the underlying question if it's answerable.

Provide a one-line rationale."""


def run(state: TicketState) -> TicketState:
    cls = state.classification
    assert cls is not None, "Planner requires Classification first"

    user = f"""Ticket: {state.ticket.subject or ""}
Body excerpt: {state.ticket.body[:400]}
Classification: intent={cls.intent}, priority={cls.priority}, sentiment={cls.sentiment}, confidence={cls.confidence}
Customer plan: {state.ticket.customer.plan or "unknown"}

Choose the route."""

    plan = structured(SYSTEM, user, Plan, state=state, agent="planner")
    state.plan = plan
    state.log("planner", f"route={plan.route} needs_kb={plan.needs_kb} :: {plan.rationale}")
    return state
