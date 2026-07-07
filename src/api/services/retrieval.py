"""Single-scheme policy retrieval engine (FR-4 / US-105).

Uses TF-IDF + cosine similarity over the 5-clause Personal Loan policy corpus
- a deliberate no-API-key, no-network-call choice appropriate for a 5-document
single-scheme POC. It is a real, standard sparse-retrieval baseline, not a
toy stub: explainable (term overlap drives the match) and trivially inside
the 600ms retrieval budget. The upgrade path (embeddings / a vector DB) is
appropriate once the corpus grows to 15-25 clauses across multiple loan
schemes in Sprint 2+ (per assumption A-2) - not needed at this scale.
"""

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
POLICY_CORPUS_PATH = os.path.join(REPO_ROOT, "data", "policy_corpus_personal_loan_v0.1.json")

RETRIEVAL_FAILURE_FLOOR = 0.05


def _load_corpus(path: str = POLICY_CORPUS_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_CORPUS = _load_corpus()
_CLAUSES = _CORPUS["clauses"]
_CLAUSE_TEXTS = [f"{clause['title']}. {clause['text']}" for clause in _CLAUSES]

_VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
_CLAUSE_MATRIX = _VECTORIZER.fit_transform(_CLAUSE_TEXTS)


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


def retrieve(query: str, top_k: int = 3) -> dict:
    """Return the top-k matching policy clauses for a free-text query.

    Response shape: {clauses: [{clause_id, title, text, score}], retrieval_failed}.
    retrieval_failed=True (empty clauses) when the best match falls below the
    similarity floor - downstream this must be treated as a missing input
    (forces a Refer recommendation), never a fabricated match.
    """
    query_vector = _VECTORIZER.transform([query])
    similarities = cosine_similarity(query_vector, _CLAUSE_MATRIX)[0]

    ranked = sorted(range(len(_CLAUSES)), key=lambda i: similarities[i], reverse=True)
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
                "title": clause["title"],
                "text": clause["text"],
                "score": round(float(similarities[i]), 4),
            }
        )

    return {"clauses": clauses, "retrieval_failed": False}


def retrieve_for_profile(profile: dict, top_k: int = 3) -> dict:
    return retrieve(build_query_from_profile(profile), top_k=top_k)


def corpus_metadata() -> dict:
    return {
        "scheme": _CORPUS["scheme"],
        "version": _CORPUS["version"],
        "effective_date": _CORPUS["effective_date"],
        "clause_count": len(_CLAUSES),
    }
