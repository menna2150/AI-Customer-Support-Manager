"""Offline accuracy eval for the embedding-kNN classifier.

Trains kNN on the Bitext train split, evaluates on a held-out sample from the
test split. No API key required.

Usage:
    python -m eval.classifier_eval_knn --n 50
    python -m eval.classifier_eval_knn --n 200 --k 5
"""
from __future__ import annotations
import argparse
import random
from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from core.knn_classifier import predict_intent, warmup
from data.bitext_loader import train_test_split
from data.intent_map import to_intent

console = Console()


def evaluate(n: int, k: int, seed: int) -> None:
    console.print("[dim]warming up kNN index (first run embeds ~5k train rows)…[/dim]")
    warmup()

    _, test_rows = train_test_split()
    rng = random.Random(seed)
    sampled = rng.sample(test_rows, k=min(n, len(test_rows)))

    console.rule(f"[bold]Offline kNN eval — n={len(sampled)}, k={k}[/bold]")

    correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_intent_total: Counter = Counter()
    per_intent_correct: Counter = Counter()
    misses: list[tuple[str, str, float, str]] = []
    confidences: list[float] = []

    for i, row in enumerate(sampled):
        truth = to_intent(row.bitext_intent, row.category)
        pred, conf = predict_intent(row.instruction, k=k)
        confidences.append(conf)

        per_intent_total[truth] += 1
        confusion[truth][pred] += 1
        if pred == truth:
            correct += 1
            per_intent_correct[truth] += 1
        elif len(misses) < 8:
            misses.append((truth, pred, conf, row.instruction[:120]))

        marker = "OK  " if pred == truth else "MISS"
        console.print(
            f"  [{i+1:>3}/{len(sampled)}] {marker}  "
            f"truth={truth:<18} pred={pred:<18} conf={conf:.2f}"
        )

    overall = correct / max(1, len(sampled))
    console.rule("[bold]Summary[/bold]")
    console.print(f"Overall accuracy : [bold]{overall:.1%}[/bold]  ({correct}/{len(sampled)})")
    console.print(f"Mean confidence  : {sum(confidences)/len(confidences):.2f}")
    console.print()

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
    all_labels = sorted({lab for k_ in confusion for lab in [k_] + list(confusion[k_])})
    cm.add_column("truth / pred")
    for c in all_labels:
        cm.add_column(c, justify="right")
    for truth in all_labels:
        row = [truth]
        for pred in all_labels:
            v = confusion[truth].get(pred, 0)
            row.append(str(v) if v else ".")
        cm.add_row(*row)
    console.print(cm)

    if misses:
        console.print("\n[bold]Sample misclassifications:[/bold]")
        for truth, pred, conf, body in misses:
            console.print(
                f"  truth={truth:<18} pred={pred:<18} conf={conf:.2f}  ::  {body}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="held-out test rows to evaluate")
    ap.add_argument("--k", type=int, default=7, help="neighbours per prediction")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    evaluate(args.n, args.k, args.seed)


if __name__ == "__main__":
    main()
