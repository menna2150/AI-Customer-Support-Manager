"""LangGraph workflow wiring the 7 agents.

Topology:
                 ┌──────────────┐
   start ─►──►  classifier  ──►  planner  ──►  router
                                              │
        ┌──────────────────────┬──────────────┘
        ▼                      ▼              ▼
       rag         (resolve_direct)      escalate ─►──┐
        │                      │                       │
        ▼                      ▼                       │
      resolver  ◄──────────────┘                       │
        │                                              │
        ▼                                              │
        qa ──► (pass)─► finalize_reply ──► cx ─► END   │
        │                                              │
        ├── (revise & attempts<MAX) ──► resolver       │
        └── (revise & attempts==MAX) | (escalate) ─────┤
                                                       ▼
                                              escalation ─► finalize_escalation ─► cx ─► END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents import classifier, planner, rag, resolver, qa, escalation, cx_analyst
from config import QA_MAX_RETRIES
from core.state import TicketState


def _route_after_planner(state: TicketState) -> str:
    assert state.plan is not None
    if state.plan.route == "escalate":
        return "escalation"
    if state.plan.needs_kb:
        return "rag"
    return "resolver"


def _route_after_qa(state: TicketState) -> str:
    assert state.qa is not None
    if state.qa.verdict == "pass":
        return "finalize_reply"
    if state.qa.verdict == "escalate":
        return "escalation"
    if state.qa_attempts >= QA_MAX_RETRIES:
        state.log("qa", f"max retries ({QA_MAX_RETRIES}) reached → escalating")
        return "escalation"
    return "resolver"


def _finalize_reply(state: TicketState) -> TicketState:
    assert state.draft is not None
    state.final_response = state.draft.text
    state.final_route = "auto_reply"
    state.log("finalize", "auto-reply ready")
    return state


def _finalize_escalation(state: TicketState) -> TicketState:
    assert state.escalation is not None
    state.final_response = state.escalation.customer_facing_acknowledgement
    state.final_route = "escalated"
    state.log("finalize", "escalation packet ready")
    return state


def build_graph():
    g = StateGraph(TicketState)

    g.add_node("classifier", classifier.run)
    g.add_node("planner", planner.run)
    g.add_node("rag", rag.run)
    g.add_node("resolver", resolver.run)
    g.add_node("qa", qa.run)
    g.add_node("escalation", escalation.run)
    g.add_node("finalize_reply", _finalize_reply)
    g.add_node("finalize_escalation", _finalize_escalation)
    g.add_node("cx_analyst", cx_analyst.run)

    g.set_entry_point("classifier")
    g.add_edge("classifier", "planner")

    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"rag": "rag", "resolver": "resolver", "escalation": "escalation"},
    )

    g.add_edge("rag", "resolver")
    g.add_edge("resolver", "qa")

    g.add_conditional_edges(
        "qa",
        _route_after_qa,
        {
            "finalize_reply": "finalize_reply",
            "resolver": "resolver",
            "escalation": "escalation",
        },
    )

    g.add_edge("escalation", "finalize_escalation")
    g.add_edge("finalize_reply", "cx_analyst")
    g.add_edge("finalize_escalation", "cx_analyst")
    g.add_edge("cx_analyst", END)

    return g.compile()


def run_ticket(state: TicketState) -> TicketState:
    graph = build_graph()
    raw = graph.invoke(state)
    # LangGraph may return a dict or the model — normalize.
    if isinstance(raw, TicketState):
        return raw
    return TicketState.model_validate(raw)
