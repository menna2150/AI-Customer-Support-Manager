"""CLI entry point.

Usage:
    python main.py                          # interactive: paste a ticket body
    python main.py --ticket-id T002         # run one of the bundled samples
    python main.py --all                    # run every sample ticket
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import ROOT
from core.graph import run_ticket
from core.state import Customer, Ticket, TicketState

console = Console()
SAMPLES_PATH = ROOT / "data" / "sample_tickets.json"


def load_samples() -> list[dict]:
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


def state_from_dict(d: dict) -> TicketState:
    customer = Customer(**d["customer"])
    ticket = Ticket(
        id=d["id"],
        channel=d.get("channel", "web"),
        subject=d.get("subject"),
        body=d["body"],
        customer=customer,
        conversation_history=d.get("conversation_history", []),
    )
    return TicketState(ticket=ticket)


def render(state: TicketState) -> None:
    t = state.ticket
    console.rule(f"[bold]Ticket {t.id}[/bold]  ·  {t.subject or ''}")
    console.print(f"[dim]from {t.customer.name or t.customer.id} on {t.channel}[/dim]")
    console.print(Panel(t.body, title="Customer message", border_style="cyan"))

    if state.classification:
        c = state.classification
        tbl = Table(show_header=False, box=None, pad_edge=False)
        tbl.add_row("intent", c.intent)
        tbl.add_row("priority", c.priority)
        tbl.add_row("sentiment", c.sentiment)
        tbl.add_row("language", c.language)
        tbl.add_row("confidence", f"{c.confidence:.2f}")
        tbl.add_row("summary", c.summary)
        console.print(Panel(tbl, title="Classification", border_style="magenta"))

    if state.plan:
        console.print(
            Panel(
                f"route = [bold]{state.plan.route}[/bold]   needs_kb = {state.plan.needs_kb}\n"
                f"{state.plan.rationale}",
                title="Plan",
                border_style="yellow",
            )
        )

    if state.kb_chunks:
        body = "\n".join(
            f"• [{c.doc_id}] {c.title}  (score {c.score:.2f})" for c in state.kb_chunks
        )
        console.print(Panel(body, title="RAG retrieved", border_style="blue"))

    if state.final_route == "auto_reply":
        console.print(
            Panel(
                state.final_response or "",
                title=f"Auto-reply  ·  citations={state.draft.citations if state.draft else []}",
                border_style="green",
            )
        )
    elif state.final_route == "escalated" and state.escalation:
        e = state.escalation
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(e.suggested_next_steps))
        console.print(
            Panel(
                f"[bold]Severity:[/bold] {e.severity}\n\n"
                f"[bold]Summary:[/bold]\n{e.summary}\n\n"
                f"[bold]Suggested next steps:[/bold]\n{steps}\n\n"
                f"[bold]Customer ack message:[/bold]\n{e.customer_facing_acknowledgement}",
                title="Escalation packet",
                border_style="red",
            )
        )

    if state.cx:
        cx = state.cx
        console.print(
            Panel(
                f"churn_risk = [bold]{cx.churn_risk}[/bold]   "
                f"satisfaction = {cx.satisfaction_signal}\n"
                f"tags = {cx.tags}\n"
                f"notes = {cx.notes or '—'}",
                title="CX insights",
                border_style="white",
            )
        )

    console.print(Panel("\n".join(state.trace), title="Trace", border_style="dim"))

    if state.llm_calls:
        cost = state.total_cost_usd
        console.print(
            Panel(
                f"calls={len(state.llm_calls)}   "
                f"tokens={state.total_tokens:,}   "
                f"latency={state.total_latency_ms/1000:.1f}s   "
                f"cost={'free' if cost == 0 else f'${cost:.5f}'}",
                title="Run cost",
                border_style="cyan",
            )
        )


def run_one(d: dict) -> None:
    state = state_from_dict(d)
    final = run_ticket(state)
    render(final)


def interactive() -> None:
    console.print("[bold]Paste a ticket body, then Ctrl-Z + Enter (Windows) to submit:[/bold]")
    body = sys.stdin.read().strip()
    if not body:
        console.print("[red]No input.[/red]")
        return
    d = {
        "id": "T-INTERACTIVE",
        "channel": "web",
        "subject": "Interactive ticket",
        "body": body,
        "customer": {"id": "U-INTERACTIVE", "plan": "unknown", "tenure_days": 0},
    }
    run_one(d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket-id")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for d in load_samples():
            run_one(d)
        return

    if args.ticket_id:
        for d in load_samples():
            if d["id"] == args.ticket_id:
                run_one(d)
                return
        console.print(f"[red]Ticket {args.ticket_id} not found.[/red]")
        return

    interactive()


if __name__ == "__main__":
    main()
