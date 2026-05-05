"""Query helper used by the RAG agent.

Heavy deps (chromadb, sentence-transformers) are imported lazily so unit tests
that monkeypatch `retrieve` don't require them to be installed.

Supports:
  • dense semantic search over the Chroma collection
  • optional metadata filter (e.g. only chunks tagged with intent='billing')
  • graceful fallback that widens the filter if the intent-restricted query
    returns nothing
"""
from __future__ import annotations
from functools import lru_cache
from typing import Optional

from config import CHROMA_DIR, EMBEDDING_MODEL, KB_COLLECTION, RAG_TOP_K
from core.state import KBChunk


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=KB_COLLECTION, embedding_function=embed_fn)


def _to_chunks(res: dict) -> list[KBChunk]:
    chunks: list[KBChunk] = []
    if not res.get("ids") or not res["ids"][0]:
        return chunks
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[0.0] * len(ids)])[0]
    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
        score = max(0.0, 1.0 - float(dist))
        chunks.append(
            KBChunk(
                doc_id=meta.get("doc_id", _id),
                title=meta.get("title", ""),
                text=doc,
                score=score,
            )
        )
    return chunks


def retrieve(
    query: str,
    k: int = RAG_TOP_K,
    *,
    intent: Optional[str] = None,
) -> list[KBChunk]:
    """Hybrid retrieval. If `intent` is set, search both the intent bucket and
    the full corpus, merge, dedupe, and return top-k by score. This way a
    cross-bucket policy (e.g. "refund" living in `billing` while the classifier
    picks `cancellation`) still surfaces."""
    coll = _collection()

    if not intent:
        return _to_chunks(coll.query(query_texts=[query], n_results=k))

    filtered = _to_chunks(
        coll.query(query_texts=[query], n_results=k, where={"intent": intent})
    )
    unfiltered = _to_chunks(coll.query(query_texts=[query], n_results=k))

    # Dedupe by (doc_id, first 80 chars of text); keep the higher score.
    seen: dict[tuple, KBChunk] = {}
    for c in filtered + unfiltered:
        key = (c.doc_id, c.text[:80])
        if key not in seen or seen[key].score < c.score:
            seen[key] = c

    return sorted(seen.values(), key=lambda c: -c.score)[:k]
