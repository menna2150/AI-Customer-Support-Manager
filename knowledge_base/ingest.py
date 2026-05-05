"""Build the Chroma vector store.

Sources:
  1. Local markdown docs under knowledge_base/docs/    (always)
  2. Bitext (instruction, response) real cases         (--with-bitext)

Each chunk carries an `intent` metadata tag so the RAG agent can filter by the
classifier's predicted intent.

Run:
    python -m knowledge_base.ingest
    python -m knowledge_base.ingest --with-bitext --max-per-intent 80
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, EMBEDDING_MODEL, KB_COLLECTION, KB_DOCS_DIR


# Folder name → our intent label.
DOC_TO_INTENT = {
    "billing": "billing",
    "cancellation": "cancellation",
    "account_access": "account_access",
    "technical": "technical_issue",
    "policies": "general_inquiry",
}


def chunk_markdown(text: str, max_chars: int = 800) -> list[tuple[str, str]]:
    sections = re.split(r"(?m)^##\s+", text)
    out: list[tuple[str, str]] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        lines = sec.splitlines()
        heading = lines[0].strip().lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        if len(body) <= max_chars:
            out.append((heading, body))
        else:
            for i in range(0, len(body), max_chars):
                out.append((heading, body[i : i + max_chars]))
    return out


def collect_markdown_chunks() -> tuple[list[str], list[str], list[dict]]:
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    docs_dir = Path(KB_DOCS_DIR)
    for md_path in sorted(docs_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        doc_top = md_path.stem
        intent = DOC_TO_INTENT.get(doc_top, "general_inquiry")
        for i, (heading, chunk) in enumerate(chunk_markdown(text)):
            ids.append(f"{doc_top}::{i}")
            docs.append(f"{heading}\n\n{chunk}")
            metas.append(
                {
                    "doc_id": doc_top,
                    "title": heading,
                    "source": md_path.name,
                    "intent": intent,
                }
            )
    return ids, docs, metas


def collect_bitext_chunks(max_per_intent: int) -> tuple[list[str], list[str], list[dict]]:
    from data.bitext_loader import load_kb_pairs

    pairs = load_kb_pairs(max_per_intent=max_per_intent)
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for p in pairs:
        ids.append(p["meta_id"])
        docs.append(p["text"])
        metas.append(
            {
                "doc_id": p["doc_id"],
                "title": p["title"],
                "source": "bitext",
                "intent": p["intent"],
            }
        )
    return ids, docs, metas


def build(with_bitext: bool, max_per_intent: int) -> int:
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if KB_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(KB_COLLECTION)

    coll = client.create_collection(name=KB_COLLECTION, embedding_function=embed_fn)

    ids, docs, metas = collect_markdown_chunks()
    print(f"  markdown chunks: {len(docs)}")

    if with_bitext:
        bx_ids, bx_docs, bx_metas = collect_bitext_chunks(max_per_intent)
        print(f"  bitext chunks  : {len(bx_docs)}")
        ids += bx_ids
        docs += bx_docs
        metas += bx_metas

    if not docs:
        print("Nothing to ingest.", file=sys.stderr)
        return 0

    coll.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested {len(docs)} chunks into '{KB_COLLECTION}' at {CHROMA_DIR}")
    return len(docs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-bitext", action="store_true",
                    help="Also ingest real (instruction, response) pairs from Bitext")
    ap.add_argument("--max-per-intent", type=int, default=60,
                    help="Cap Bitext examples per intent (default 60)")
    args = ap.parse_args()
    build(with_bitext=args.with_bitext, max_per_intent=args.max_per_intent)
