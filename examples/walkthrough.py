"""End-to-end traced walkthrough of two contrasting tickets.

Demonstrates:
  • a clean RAG-resolved path (T004 — SSO question)
  • an escalation path (T003 — frustrated, double-charge complaint)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import run_one, load_samples  # noqa: E402

WANTED = {"T004", "T003"}


def main() -> None:
    samples = [t for t in load_samples() if t["id"] in WANTED]
    for t in samples:
        run_one(t)


if __name__ == "__main__":
    main()
