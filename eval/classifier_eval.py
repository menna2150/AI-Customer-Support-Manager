"""Measure the Classifier agent's intent accuracy on real Bitext data.

Usage:
    python -m eval.classifier_eval --n 50
    python -m eval.classifier_eval --n 200 --intents billing,account_access

Prints overall accuracy, per-intent precision/recall, and a confusion table.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from agents import classifier
from core.state import Customer, Ticket, TicketState
from data.bitext_loader import load_rows
from data.intent_map import to_intent

console = Console()


def sample_eval_rows(n: int, intents: list[str] | None, seed: int = 7):
    import random

    rows = load_rows()
    if intents:
        rows = [r for r in rows if r.our_intent in intents]
    rng = random.Random(seed)
    return rng.sample(rows, k=min(n, len(rows)))


def to_state(row, idx: int) -> TicketState:
    customer = Customer(id=f"BX-{idx:04d}", plan="unknown")
    ticket = Ticket(
        id=f"EVAL-{idx:04d}",
        channel="web",
        subject=row.bitext_intent.replace("_", " "),
        body=row.instruction,
        customer=customer,
    )
    return TicketState(ticket=ticket)


def evaluate(n: int, intents: list[str] | None) -> None:
    rows = sample_eval_rows(n, intents)
    console.rule(f"[bold]Classifier evaluation: {len(rows)} tickets[/bold]")

    correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_intent_total: Counter = Counter()
    per_intent_correct: Counter = Counter()
    misses: list[tuple[str, str, str]] = []  # (true, pred, body)

    for i, row in enumerate(rows):
        truth = to_intent(row.bitext_intent, row.category)
        state = to_state(row, i)
        try:
            classifier.run(state)
        except Exception as e:
            console.print(f"[red]error on row {i}: {e}[/red]")
            continue
        pred = state.classification.intent if state.classification else "other"

        per_intent_total[truth] += 1
        confusion[truth][pred] += 1
        if pred == truth:
            correct += 1
            per_intent_correct[truth] += 1
        elif len(misses) < 8:
            misses.append((truth, pred, row.instruction[:120]))

        console.print(
            f"  [{i+1:>3}/{len(rows)}] truth={truth:<18} pred={pred:<18} "
            f"{'OK' if pred == truth else 'MISS'}"
        )

    # Summary
    overall = correct / max(1, len(rows))
    console.rule("[bold]Summary[/bold]")
    console.print(f"Overall accuracy: [bold]{overall:.1%}[/bold]  ({correct}/{len(rows)})\n")

    by_intent = Table(title="Per-intent accuracy")
    by_intent.add_column("intent")
    by_intent.add_column("n", justify="right")
    by_intent.add_column("correct", justify="right")
    by_intent.add_column("acc", justify="right")
    for intent in sorted(per_intent_total):
        n_i = per_intent_total[intent]
        c_i = per_intent_correct[intent]
        by_intent.add_row(intent, str(n_i), str(c_i), f"{c_i / max(1, n_i):.0%}")
    console.print(by_intent)

    cm = Table(title="Confusion (rows=truth, cols=pred)")
    all_labels = sorted({lab for k in confusion for lab in [k] + list(confusion[k])})
    cm.add_column("truth ↓ / pred →")
    for c in all_labels:
        cm.add_column(c, justify="right")
    for truth in all_labels:
        row = [truth]
        for pred in all_labels:
            v = confusion[truth].get(pred, 0)
            row.append(str(v) if v else "·")
        cm.add_row(*row)
    console.print(cm)

    if misses:
        console.print("\n[bold]Sample misclassifications:[/bold]")
        for truth, pred, body in misses:
            console.print(f"  truth={truth} pred={pred}  ::  {body}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--intents", default="", help="comma-separated subset, e.g. billing,account_access")
    args = ap.parse_args()
    intents = [i.strip() for i in args.intents.split(",") if i.strip()] or None
    evaluate(args.n, intents)


if __name__ == "__main__":
    main()
