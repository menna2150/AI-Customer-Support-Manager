"""Offline k-NN intent classifier.

Embeds the Bitext train split with sentence-transformers, then classifies each
incoming ticket by majority vote of the k nearest neighbours (cosine sim).

Useful as:
  • a fully offline baseline (no API key required)
  • a cheap pre-filter in production — gate the expensive LLM behind a
    high-confidence kNN prediction
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import EMBEDDING_MODEL, ROOT
from core.state import Intent
from data.bitext_loader import train_test_split

CACHE_DIR = ROOT / "data" / "cache"
EMB_FILE = CACHE_DIR / "knn_train_embeddings.npy"
LABEL_FILE = CACHE_DIR / "knn_train_labels.npy"

TRAIN_CAP = 5000  # cap on training rows — keeps embedding step under a minute


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _index() -> tuple[np.ndarray, np.ndarray]:
    """Returns (embeddings: [N, D] float32 normalized, labels: [N] str)."""
    if EMB_FILE.exists() and LABEL_FILE.exists():
        embs = np.load(EMB_FILE)
        labels = np.load(LABEL_FILE, allow_pickle=True)
        return embs, labels

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_rows, _ = train_test_split()
    if len(train_rows) > TRAIN_CAP:
        # Stratify roughly by intent so all classes are represented.
        rng = np.random.default_rng(31)
        by_intent: dict[str, list] = {}
        for r in train_rows:
            by_intent.setdefault(r.our_intent, []).append(r)
        per_intent_cap = max(1, TRAIN_CAP // max(1, len(by_intent)))
        sampled = []
        for intent, group in by_intent.items():
            idx = rng.choice(len(group), size=min(per_intent_cap, len(group)), replace=False)
            sampled.extend(group[i] for i in idx)
        train_rows = sampled

    texts = [r.instruction for r in train_rows]
    labels = np.array([r.our_intent for r in train_rows], dtype=object)
    print(f"[knn] embedding {len(texts)} train rows…")
    embs = _model().encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(EMB_FILE, embs)
    np.save(LABEL_FILE, labels)
    print(f"[knn] cached embeddings to {EMB_FILE.name} ({embs.shape})")
    return embs, labels


def predict_intent(text: str, k: int = 7) -> tuple[Intent, float]:
    """Returns (predicted_intent, confidence_in_[0,1]).

    Confidence = fraction of the top-k neighbours that voted for the winner,
    weighted by their cosine similarity.
    """
    embs, labels = _index()
    q = _model().encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    sims = embs @ q  # cosine sim because both are L2-normalized
    top_idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    top_sims = sims[top_idx]
    top_labels = labels[top_idx]

    # Weighted vote: sum the cosine sim per label, pick the heaviest.
    scores: dict[str, float] = {}
    for lab, s in zip(top_labels, top_sims):
        scores[str(lab)] = scores.get(str(lab), 0.0) + float(s)
    winner = max(scores, key=scores.get)
    confidence = scores[winner] / max(1e-9, sum(scores.values()))
    return winner, confidence  # type: ignore[return-value]


def warmup() -> None:
    """Build the index now (useful for one-off setup so the eval doesn't pay for it)."""
    _index()
