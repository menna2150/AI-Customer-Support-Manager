"""Bitext customer-support dataset loader.

Source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
The CSV is ~5MB and is cached locally on first call. Network is only hit if the
cache file is missing — useful for offline runs and CI.
"""
from __future__ import annotations
import csv
import io
import random
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config import ROOT
from core.state import Customer, Intent, Ticket, TicketState
from data.intent_map import heuristic_priority, to_intent

CACHE_DIR = ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "bitext_customer_support.csv"

# Pinned to a specific revision is best for reproducibility, but `main` is fine
# for a demo project.
DATASET_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


@dataclass(frozen=True)
class BitextRow:
    instruction: str
    category: str
    bitext_intent: str
    response: str
    flags: str

    @property
    def our_intent(self) -> Intent:
        return to_intent(self.bitext_intent, self.category)


def _ensure_cached() -> Path:
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 0:
        return CACHE_FILE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[bitext_loader] downloading {DATASET_URL} -> {CACHE_FILE}")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as r:
        data = r.read()
    CACHE_FILE.write_bytes(data)
    return CACHE_FILE


@lru_cache(maxsize=1)
def load_rows() -> list[BitextRow]:
    """Parse the cached CSV (download if missing)."""
    path = _ensure_cached()
    rows: list[BitextRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                BitextRow(
                    instruction=(r.get("instruction") or "").strip(),
                    category=(r.get("category") or "").strip(),
                    bitext_intent=(r.get("intent") or "").strip(),
                    response=(r.get("response") or "").strip(),
                    flags=(r.get("flags") or "").strip(),
                )
            )
    return rows


def train_test_split(
    test_frac: float = 0.2,
    seed: int = 19,
) -> tuple[list[BitextRow], list[BitextRow]]:
    """Deterministic global split — kNN trains on `train`, eval samples from `test`.

    Always returns the same split for the same seed, so kNN cache is stable.
    """
    rows = load_rows()
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_at = int(len(rows) * (1.0 - test_frac))
    train = [rows[i] for i in indices[:split_at]]
    test = [rows[i] for i in indices[split_at:]]
    return train, test


def load_eval_tickets(
    n: int = 20,
    seed: int = 7,
    intents: Optional[list[Intent]] = None,
) -> list[TicketState]:
    """Sample N rows from the held-out test split and wrap each as a TicketState."""
    _, test_rows = train_test_split()
    rows = test_rows
    if intents:
        rows = [r for r in rows if r.our_intent in intents]
    rng = random.Random(seed)
    chosen = rng.sample(rows, k=min(n, len(rows)))
    out: list[TicketState] = []
    for i, r in enumerate(chosen):
        cust = Customer(
            id=f"BX-{i:04d}",
            name=None,
            plan="unknown",
            tenure_days=None,
            history_summary=None,
        )
        t = Ticket(
            id=f"BITEXT-{i:04d}",
            channel="web",
            subject=r.bitext_intent.replace("_", " "),
            body=r.instruction,
            customer=cust,
        )
        out.append(TicketState(ticket=t))
    return out


def load_few_shot_examples(per_intent: int = 1, seed: int = 13) -> list[dict]:
    """Pick a few real (instruction, intent, category) triples — one per Bitext intent.

    Returned shape:
        {"instruction": "...", "intent": <our Intent>, "bitext_intent": "...", "category": "..."}
    """
    rows = load_rows()
    rng = random.Random(seed)
    by_intent: dict[str, list[BitextRow]] = {}
    for r in rows:
        by_intent.setdefault(r.bitext_intent, []).append(r)

    examples = []
    for bitext_intent, group in by_intent.items():
        picks = rng.sample(group, k=min(per_intent, len(group)))
        for p in picks:
            examples.append(
                {
                    "instruction": p.instruction,
                    "intent": p.our_intent,
                    "bitext_intent": bitext_intent,
                    "category": p.category,
                }
            )
    return examples


def load_kb_pairs(max_per_intent: int = 50, seed: int = 21) -> list[dict]:
    """Real (query, response) pairs to ingest as KB examples — case-based retrieval.

    Each becomes a chunk in the vector store with metadata {doc_id, intent, ...}.
    """
    rows = load_rows()
    rng = random.Random(seed)
    by_intent: dict[str, list[BitextRow]] = {}
    for r in rows:
        if not r.response:
            continue
        by_intent.setdefault(r.our_intent, []).append(r)

    out: list[dict] = []
    for our_intent, group in by_intent.items():
        picks = rng.sample(group, k=min(max_per_intent, len(group)))
        for i, r in enumerate(picks):
            out.append(
                {
                    "doc_id": "bitext_cases",
                    "title": f"{r.bitext_intent} (real example)",
                    "intent": our_intent,
                    "text": (
                        f"Customer asked: {r.instruction}\n\n"
                        f"Reference answer:\n{r.response}"
                    ),
                    "meta_id": f"bx::{our_intent}::{i}",
                }
            )
    return out


if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} Bitext rows from {CACHE_FILE}")
    by_intent: dict[str, int] = {}
    for r in rows:
        by_intent[r.our_intent] = by_intent.get(r.our_intent, 0) + 1
    print("Distribution after mapping:")
    for k, v in sorted(by_intent.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>20} : {v}")
