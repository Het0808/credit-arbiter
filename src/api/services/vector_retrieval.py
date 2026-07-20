"""Semantic policy retrieval backed by Qdrant + local embeddings (RAG upgrade).

Drop-in alternative to the TF-IDF retriever: same return shape
({clauses:[{clause_id, source_id, scheme, title, text, rule, score,
corpus_version}], retrieval_failed, corpus_version, scheme}) so callers (policy
engine, assessment, UI citations) don't change. Enabled with RETRIEVER=vector;
retrieval.retrieve() falls back to TF-IDF automatically if this path errors.

Embeddings use fastembed (local ONNX, no API key). Qdrant runs either embedded
in-memory (QDRANT_URL unset or ":memory:") or against a server (QDRANT_URL).
Clauses of the active corpus are indexed lazily on first use and cached per
corpus version; ids are deterministic (uuid5) so re-indexing is idempotent.
"""

from __future__ import annotations

import os
import uuid

QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "halcyon_policies")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# Cosine floor below which retrieval is treated as failed (semantic scale differs
# from TF-IDF; tune per embedding model). Kept modest to avoid false failures.
VECTOR_SCORE_FLOOR = float(os.getenv("VECTOR_SCORE_FLOOR", "0.35"))

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "halcyon-credit-policy")

_client = None
_indexed_versions: set[str] = set()


def _get_client():
    """Return a shared QdrantClient (embedded in-memory or a server)."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        if QDRANT_URL in ("", ":memory:"):
            _client = QdrantClient(location=":memory:")
        else:
            _client = QdrantClient(url=QDRANT_URL)
        _client.set_model(EMBEDDING_MODEL)
    return _client


def index_clauses(clauses: list[dict], corpus_version: str, *, force: bool = False) -> int:
    """Upsert a corpus version's clauses into Qdrant. Idempotent (uuid5 ids)."""
    client = _get_client()
    if corpus_version in _indexed_versions and not force:
        return 0

    documents, metadata, ids = [], [], []
    for clause in clauses:
        documents.append(f"{clause['title']}. {clause['text']} {' '.join(clause.get('tags', []))}")
        metadata.append(
            {
                "clause_id": clause["clause_id"],
                "scheme": clause.get("scheme"),
                "title": clause["title"],
                "text": clause["text"],
                "rule": clause.get("rule"),
                "corpus_version": corpus_version,
            }
        )
        ids.append(str(uuid.uuid5(_NAMESPACE, f"{corpus_version}:{clause['clause_id']}")))

    client.add(collection_name=QDRANT_COLLECTION, documents=documents, metadata=metadata, ids=ids)
    _indexed_versions.add(corpus_version)
    return len(documents)


def _scheme_filter(scheme: str | None, corpus_version: str):
    from qdrant_client import models

    conditions = [models.FieldCondition(key="corpus_version", match=models.MatchValue(value=corpus_version))]
    if scheme is not None:
        conditions.append(models.FieldCondition(key="scheme", match=models.MatchValue(value=scheme)))
    return models.Filter(must=conditions)


def vector_retrieve(policy_index, query: str, scheme: str | None = None, top_k: int = 3) -> dict:
    """Semantic retrieval over ``policy_index``'s clauses. Same shape as TF-IDF retrieve()."""
    corpus_version = policy_index.corpus_version
    index_clauses(policy_index.clauses, corpus_version)

    base = {"corpus_version": corpus_version, "scheme": scheme}
    hits = _get_client().query(
        collection_name=QDRANT_COLLECTION,
        query_text=query,
        query_filter=_scheme_filter(scheme, corpus_version),
        limit=top_k,
    )
    if not hits or hits[0].score < VECTOR_SCORE_FLOOR:
        return {"clauses": [], "retrieval_failed": True, **base}

    clauses = []
    for h in hits:
        if h.score < VECTOR_SCORE_FLOOR:
            break
        m = h.metadata
        clauses.append(
            {
                "clause_id": m["clause_id"],
                "source_id": m["clause_id"],
                "scheme": m.get("scheme"),
                "title": m.get("title"),
                "text": m.get("text"),
                "rule": m.get("rule"),
                "score": round(float(h.score), 4),
                "corpus_version": corpus_version,
            }
        )
    return {"clauses": clauses, "retrieval_failed": False, **base}
