"""Multi-scheme policy retrieval engine (FR-4 / US-105, US-203, US-204).

Uses TF-IDF + cosine similarity over the 20-clause, 6-scheme policy corpus
(data/policy_corpus_v1.0.json) - a deliberate no-API-key, no-network-call
choice appropriate at this corpus scale. It is a real, standard sparse-
retrieval baseline, not a toy stub: explainable (term overlap drives the
match) and trivially inside the 600ms retrieval budget. The upgrade path
(embeddings / a vector DB) is the right move once the corpus grows well
beyond a few dozen clauses (per assumption A-2) - not needed at this scale.

Sprint 1's single-scheme v0.1 corpus (data/policy_corpus_personal_loan_v0.1.json)
is left in place, unmodified, so any decision record referencing it can still
be replayed against the exact clause text that was retrieved at the time
(US-207 policy version replay).
"""

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
POLICY_CORPUS_PATH = os.path.join(REPO_ROOT, "data", "policy_corpus_v1.0.json")

RETRIEVAL_FAILURE_FLOOR = 0.05


def _load_corpus(path: str = POLICY_CORPUS_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_query_from_profile(profile: dict) -> str:
    """Render an applicant's normalised profile into a short text query -
    the "application context" FR-4 refers to."""
    parts = [
        f"loan amount {profile.get('amt_credit')}",
        f"annual income {profile.get('amt_income_total')}",
        f"annuity {profile.get('amt_annuity')}",
        f"employment days {profile.get('days_employed')}",
        f"region rating {profile.get('region_rating_client')}",
    ]
    return " ".join(parts)


def retrieve(query: str, scheme: str = None, top_k: int = 3) -> dict:
    """Return the top-k matching policy clauses for a free-text query.

    Args:
        query: Free-text application context.
        scheme: If given, only clauses tagged with this scheme are eligible
            candidates (US-204: scheme-aware retrieval). If None, all schemes
            are searched (used by ad-hoc /policy/retrieve?query= calls).
        top_k: Max clauses to return.

    Response shape: {clauses: [{clause_id, source_id, title, text, score,
    scheme, version}], retrieval_failed}. retrieval_failed=True (empty
    clauses) when the best match falls below the similarity floor, or when
    the requested scheme has no candidate clauses at all - downstream this
    must be treated as a missing input (forces a Refer recommendation),
    never a fabricated match.
    """
    candidate_indices = (
        [i for i, clause in enumerate(_CLAUSES) if clause["scheme"] == scheme]
        if scheme
        else list(range(len(_CLAUSES)))
    )
    if not candidate_indices:
        return {"clauses": [], "retrieval_failed": True}

    query_vector = _VECTORIZER.transform([query])
    similarities = cosine_similarity(query_vector, _CLAUSE_MATRIX)[0]

    ranked = sorted(candidate_indices, key=lambda i: similarities[i], reverse=True)
    top_score = similarities[ranked[0]] if ranked else 0.0

    if top_score < RETRIEVAL_FAILURE_FLOOR:
        return {"clauses": [], "retrieval_failed": True}

    clauses = []
    for i in ranked[:top_k]:
        if similarities[i] < RETRIEVAL_FAILURE_FLOOR:
            break
        clause = _CLAUSES[i]
        clauses.append(
            {
                "clause_id": clause["clause_id"],
                "source_id": clause["clause_id"],
                "scheme": clause["scheme"],
                "version": _CORPUS["version"],
                "title": clause["title"],
                "text": clause["text"],
                "score": round(float(similarities[i]), 4),
            }
        )

    return {"clauses": clauses, "retrieval_failed": False}


def retrieve_for_profile(profile: dict, scheme: str = None, top_k: int = 3) -> dict:
    return retrieve(build_query_from_profile(profile), scheme=scheme, top_k=top_k)


def corpus_metadata() -> dict:
    return {
        "version": _CORPUS["version"],
        "effective_date": _CORPUS["effective_date"],
        "schemes": _CORPUS["schemes"],
        "clause_count": len(_CLAUSES),
    }


def reindex_corpus(path: str = POLICY_CORPUS_PATH) -> dict:
    """Manual re-index trigger (US-207): reload the policy corpus from
    `path` and make it live for every subsequent retrieve() call in this
    process, without a restart. Old decision records stay replayable
    regardless of what's currently loaded - each stores the clause text and
    version it actually saw (assessment.py's evidence_chain), not just a
    version pointer."""
    global _CORPUS, _CLAUSES, _CLAUSE_TEXTS, _VECTORIZER, _CLAUSE_MATRIX

    corpus = _load_corpus(path)
    clauses = corpus["clauses"]
    clause_texts = [f"{clause['title']}. {clause['text']}" for clause in clauses]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    clause_matrix = vectorizer.fit_transform(clause_texts)

    _CORPUS, _CLAUSES, _CLAUSE_TEXTS, _VECTORIZER, _CLAUSE_MATRIX = (
        corpus,
        clauses,
        clause_texts,
        vectorizer,
        clause_matrix,
    )
    return corpus_metadata()


reindex_corpus(POLICY_CORPUS_PATH)
