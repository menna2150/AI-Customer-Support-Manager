"""End-to-end demo without an LLM.

Runs the parts of the pipeline that don't need Claude/Groq:
  • kNN intent classifier (offline, sentence-transformers)
  • Rule-based planner (routing decided from intent + confidence)
  • RAG retrieval (Chroma, intent-filtered)

For each ticket it prints classification + plan + the top KB chunks the
Resolver would have used. With a real LLM key you'd swap the kNN classifier
back for the LLM Classifier and let Resolver/QA/Escalation/CX run too.

Run:
    python demo_no_llm.py
    python demo_no_llm.py --ticket "I want to cancel my subscription"
"""
from __future__ import annotations
import argparse

from rich.console import Console
from rich.panel import Panel

from core.knn_classifier import predict_intent, warmup
from core.state import Customer, Ticket, TicketState
from data.bitext_loader import load_eval_tickets
from agents import rag

console = Console()


CONFIDENCE_ESCALATE = 0.50  # below this, hand off to a human


def rule_based_plan(intent: str, confidence: float) -> tuple[str, bool, str]:
    """Returns (route, needs_kb, rationale) — mimics agents/planner.py without an LLM."""
    if confidence < CONFIDENCE_ESCALATE:
        return "escalate", False, f"Low confidence ({confidence:.2f}) — route to a human."
    if intent in ("billing", "account_access", "technical_issue", "cancellation", "general_inquiry"):
        return "resolve_with_kb", True, f"On-topic intent ({intent}); KB likely has the answer."
    if intent == "complaint":
        return "escalate", False, "Complaint — needs a human."
    return "resolve_direct", False, "Generic — no KB needed."


def render(state: TicketState) -> None:
    t = state.ticket
    console.rule(f"[bold]{t.id}[/bold]  ·  {t.subject or ''}")
    console.print(Panel(t.body, title="Customer message", border_style="cyan"))

    cls = state.classification
    console.print(
        Panel(
            f"intent     = [bold]{cls.intent}[/bold]\n"
            f"confidence = {cls.confidence:.2f}\n"
            f"summary    = {cls.summary}",
            title="kNN classification",
            border_style="magenta",
        )
    )

    plan = state.plan
    console.print(
        Panel(
            f"route = [bold]{plan.route}[/bold]   needs_kb = {plan.needs_kb}\n{plan.rationale}",
            title="Rule-based plan",
            border_style="yellow",
        )
    )

    if state.kb_chunks:
        body = "\n\n".join(
            f"[{c.doc_id}] [bold]{c.title}[/bold]  (score {c.score:.2f})\n{c.text[:300]}"
            + ("…" if len(c.text) > 300 else "")
            for c in state.kb_chunks
        )
        console.print(Panel(body, title="Retrieved KB chunks (intent-filtered)", border_style="blue"))
    elif plan.needs_kb:
        console.print(Panel("(no chunks retrieved)", title="RAG", border_style="red"))


def run_one(state: TicketState) -> None:
    # 1. kNN classify
    intent, confidence = predict_intent(state.ticket.body)
    from core.state import Classification

    state.classification = Classification(
        intent=intent,  # type: ignore[arg-type]
        priority="medium",
        sentiment="neutral",
        language="en",
        summary=state.ticket.body[:80],
        confidence=confidence,
    )

    # 2. Rule-based plan
    from core.state import Plan

    route, needs_kb, rationale = rule_based_plan(intent, confidence)
    state.plan = Plan(route=route, needs_kb=needs_kb, rationale=rationale)  # type: ignore[arg-type]

    # 3. RAG (only if plan says so)
    if needs_kb:
        rag.run(state)

    render(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticket", help="custom ticket body (one-off)")
    ap.add_argument("--n", type=int, default=3, help="number of real Bitext tickets to run")
    args = ap.parse_args()

    console.print("[dim]warming up kNN index…[/dim]")
    warmup()

    if args.ticket:
        state = TicketState(
            ticket=Ticket(
                id="DEMO-CUSTOM",
                channel="web",
                subject="custom",
                body=args.ticket,
                customer=Customer(id="DEMO"),
            )
        )
        run_one(state)
        return

    for s in load_eval_tickets(n=args.n):
        run_one(s)


if __name__ == "__main__":
    main()
