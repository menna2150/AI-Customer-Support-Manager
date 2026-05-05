"""Full-pipeline evaluation.

Runs the entire 7-agent graph on N held-out Bitext tickets and reports:
  • intent classification accuracy
  • escalation rate
  • avg / p50 / p95 latency per ticket
  • avg cost per ticket (USD)
  • avg LLM tokens per ticket
  • hallucination rate (drafts that QA flagged via grounding gate)

Usage:
    python -m eval.pipeline_eval --n 50
    python -m eval.pipeline_eval --n 100 --intents billing,account_access
"""
from __future__ import annotations
import argparse
import time
from collections import Counter
from statistics import mean, median

from rich.console import Console
from rich.table import Table

from core.graph import run_ticket
from core.state import Customer, Ticket, TicketState
from data.bitext_loader import train_test_split
from data.intent_map import to_intent

console = Console()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def to_state(row, idx: int) -> TicketState:
    return TicketState(
        ticket=Ticket(
            id=f"EVAL-{idx:04d}",
            channel="web",
            subject=row.bitext_intent.replace("_", " "),
            body=row.instruction,
            customer=Customer(id=f"BX-{idx:04d}", plan="unknown"),
        )
    )


def evaluate(n: int, intents: list[str] | None, seed: int = 7) -> None:
    import random

    _, test_rows = train_test_split()
    rows = test_rows
    if intents:
        rows = [r for r in rows if r.our_intent in intents]
    rng = random.Random(seed)
    sampled = rng.sample(rows, k=min(n, len(rows)))

    console.rule(f"[bold]Pipeline eval — {len(sampled)} tickets[/bold]")
    console.print("[dim]Each ticket runs the full 7-agent graph.[/dim]\n")

    correct_intent = 0
    escalated = 0
    grounding_revisions = 0
    total_errors = 0
    intent_total: Counter = Counter()
    intent_correct: Counter = Counter()

    latencies: list[float] = []
    costs: list[float] = []
    tokens: list[int] = []

    for i, row in enumerate(sampled):
        truth = to_intent(row.bitext_intent, row.category)
        state = to_state(row, i)

        ticket_start = time.perf_counter()
        try:
            final = run_ticket(state)
        except Exception as e:
            total_errors += 1
            console.print(f"  [{i+1:>3}/{len(sampled)}] ERROR  {type(e).__name__}: {str(e)[:80]}")
            continue
        elapsed = time.perf_counter() - ticket_start

        pred = final.classification.intent if final.classification else "other"
        intent_total[truth] += 1
        if pred == truth:
            correct_intent += 1
            intent_correct[truth] += 1

        if final.final_route == "escalated":
            escalated += 1

        # Grounding revision = at least one QA call with verdict=revise from the gate.
        # We can detect by inspecting trace lines.
        if any("verdict=revise (grounding gate)" in line for line in final.trace):
            grounding_revisions += 1

        latencies.append(elapsed)
        costs.append(final.total_cost_usd)
        tokens.append(final.total_tokens)

        marker = "OK  " if pred == truth else "MISS"
        outcome = final.final_route or "—"
        console.print(
            f"  [{i+1:>3}/{len(sampled)}] {marker}  "
            f"truth={truth:<18} pred={pred:<18} "
            f"out={outcome:<10} "
            f"t={elapsed:.1f}s  tok={final.total_tokens:>5}  ${final.total_cost_usd:.5f}"
        )

    # ----- Summary -----
    n_runs = len(latencies)
    if n_runs == 0:
        console.print("[red]No successful runs.[/red]")
        return

    acc = correct_intent / n_runs
    esc_rate = escalated / n_runs
    halluc_rate = grounding_revisions / n_runs

    console.rule("[bold]Summary[/bold]")
    summary = Table(show_header=False, box=None)
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Tickets run            ", f"{n_runs} ({total_errors} errors)")
    summary.add_row("Intent accuracy        ", f"{acc:.1%}  ({correct_intent}/{n_runs})")
    summary.add_row("Escalation rate        ", f"{esc_rate:.1%}  ({escalated}/{n_runs})")
    summary.add_row("Grounding-gate revisions", f"{halluc_rate:.1%}  ({grounding_revisions}/{n_runs})")
    summary.add_row("Avg latency / ticket   ", f"{mean(latencies):.1f}s")
    summary.add_row("p50 / p95 latency      ", f"{median(latencies):.1f}s / {percentile(latencies, 0.95):.1f}s")
    summary.add_row("Avg tokens / ticket    ", f"{int(mean(tokens)):,}")
    summary.add_row("Avg cost / ticket      ", "free" if mean(costs) == 0 else f"${mean(costs):.5f}")
    summary.add_row("Total cost             ", "free" if sum(costs) == 0 else f"${sum(costs):.4f}")
    console.print(summary)

    if intent_total:
        per_intent = Table(title="Per-intent classification accuracy")
        per_intent.add_column("intent")
        per_intent.add_column("n", justify="right")
        per_intent.add_column("correct", justify="right")
        per_intent.add_column("acc", justify="right")
        for intent in sorted(intent_total):
            n_i = intent_total[intent]
            c_i = intent_correct[intent]
            per_intent.add_row(intent, str(n_i), str(c_i), f"{c_i/max(1,n_i):.0%}")
        console.print(per_intent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--intents", default="", help="comma-separated subset")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    intents = [i.strip() for i in args.intents.split(",") if i.strip()] or None
    evaluate(args.n, intents, args.seed)


if __name__ == "__main__":
    main()
