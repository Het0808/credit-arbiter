"""Scheme-aware policy retrieval engine (FR-4 / US-204) with version
management and replay (US-207).

Retrieval uses TF-IDF + cosine similarity over the policy corpus. Every
returned clause carries a stable ``source_id`` (its ``clause_id``) and the
``corpus_version`` it came from, so downstream explanations and audit records
are traceable to an exact clause in an exact corpus version (AC-3).

Multiple corpus versions can be loaded simultaneously: the *active* version
serves live traffic, while any older version can still be queried for replay
(US-207 AC: "old retrievals can be replayed against the old version"). TF-IDF
is a deliberate no-API-key, no-network sparse baseline appropriate for a
~20-clause corpus; the upgrade path to embeddings / a vector DB applies once
the corpus grows well beyond a few schemes (assumption A-2).
"""

import glob
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Default active corpus version. v1.0 is the multi-scheme rulebook (US-203);
# v0.1 (single-scheme Personal Loan) is retained purely for replay.
DEFAULT_ACTIVE_VERSION = "v1.0"

RETRIEVAL_FAILURE_FLOOR = 0.05


def _discover_corpora() -> dict:
    """Map corpus_version -> file path by scanning data/ for corpus JSON files."""
    registry = {}
    candidates = glob.glob(os.path.join(DATA_DIR, "policy_corpus*.json"))
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                corpus = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        version = corpus.get("corpus_version") or corpus.get("version")
        if version:
            registry[version] = path
    return registry


class PolicyIndex:
    """An immutable, in-memory TF-IDF index over one corpus version.

    Normalises both the v0.1 (top-level ``scheme``) and v1.0 (per-clause
    ``scheme``) corpus shapes into a single internal representation.
    """

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as fh:
            corpus = json.load(fh)

        self.path = path
        self.corpus_version = corpus.get("corpus_version") or corpus.get("version")
        self.effective_date = corpus.get("effective_date")
        default_scheme = corpus.get("scheme")  # v0.1 single-scheme fallback

        self.clauses = []
        for clause in corpus["clauses"]:
            self.clauses.append(
                {
                    "clause_id": clause["clause_id"],
                    "scheme": clause.get("scheme", default_scheme),
                    "title": clause["title"],
                    "text": clause["text"],
                    "tags": clause.get("tags", []),
                    "rule": clause.get("rule"),
                }
            )

        self.schemes = sorted({c["scheme"] for c in self.clauses if c["scheme"]})
        # Include tags in the indexed text so tag terms contribute to matching.
        self._clause_texts = [
            f"{c['title']}. {c['text']} {' '.join(c['tags'])}" for c in self.clauses
        ]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._clause_texts)

    def retrieve(self, query: str, scheme: str = None, top_k: int = 3) -> dict:
        """Return the top-k clauses for a query, optionally restricted to one scheme.

        When ``scheme`` is given, only that scheme's clauses are candidates
        (US-204 AC). Returns ``retrieval_failed=True`` with no clauses when the
        best candidate falls below the similarity floor, or when the requested
        scheme has no clauses - both must be treated downstream as a missing
        input that forces a Refer, never a fabricated match.
        """
        candidate_idxs = list(range(len(self.clauses)))
        if scheme is not None:
            candidate_idxs = [i for i in candidate_idxs if self.clauses[i]["scheme"] == scheme]

        base = {"corpus_version": self.corpus_version, "scheme": scheme}
        if not candidate_idxs:
            return {"clauses": [], "retrieval_failed": True, **base}

        similarities = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(candidate_idxs, key=lambda i: similarities[i], reverse=True)

        if similarities[ranked[0]] < RETRIEVAL_FAILURE_FLOOR:
            return {"clauses": [], "retrieval_failed": True, **base}

        clauses = []
        for i in ranked[:top_k]:
            if similarities[i] < RETRIEVAL_FAILURE_FLOOR:
                break
            clause = self.clauses[i]
            clauses.append(
                {
                    "clause_id": clause["clause_id"],
                    "source_id": clause["clause_id"],
                    "scheme": clause["scheme"],
                    "title": clause["title"],
                    "text": clause["text"],
                    "rule": clause["rule"],
                    "score": round(float(similarities[i]), 4),
                    "corpus_version": self.corpus_version,
                }
            )
        return {"clauses": clauses, "retrieval_failed": False, **base}

    def get_clause(self, clause_id: str) -> dict | None:
        for clause in self.clauses:
            if clause["clause_id"] == clause_id:
                return {
                    "clause_id": clause["clause_id"],
                    "source_id": clause["clause_id"],
                    "scheme": clause["scheme"],
                    "title": clause["title"],
                    "text": clause["text"],
                    "corpus_version": self.corpus_version,
                }
        return None

    def metadata(self) -> dict:
        return {
            "corpus_version": self.corpus_version,
            "effective_date": self.effective_date,
            "schemes": self.schemes,
            "clause_count": len(self.clauses),
            "source_file": os.path.basename(self.path),
        }


# --- Module-level version registry & active-index management (US-207) ---

_REGISTRY = _discover_corpora()
_INDEX_CACHE: dict = {}
_ACTIVE_VERSION = DEFAULT_ACTIVE_VERSION if DEFAULT_ACTIVE_VERSION in _REGISTRY else (
    next(iter(_REGISTRY)) if _REGISTRY else None
)


def get_index(version: str = None) -> PolicyIndex:
    """Return the (cached) index for a version, defaulting to the active one."""
    version = version or _ACTIVE_VERSION
    if version not in _REGISTRY:
        raise KeyError(f"Unknown policy corpus version: {version!r}. Known: {sorted(_REGISTRY)}")
    if version not in _INDEX_CACHE:
        _INDEX_CACHE[version] = PolicyIndex(_REGISTRY[version])
    return _INDEX_CACHE[version]


def list_versions() -> dict:
    return {
        "active_version": _ACTIVE_VERSION,
        "available_versions": sorted(_REGISTRY),
        "versions": {v: PolicyIndex(p).metadata() for v, p in sorted(_REGISTRY.items())},
    }


def reindex(version: str = None) -> dict:
    """Re-scan data/ for corpus files, rebuild the index for ``version`` (or the
    active version), make it active, and return its metadata. This is the
    manual re-index trigger from US-207 - call it after editing/adding a corpus.
    """
    global _REGISTRY, _ACTIVE_VERSION
    _REGISTRY = _discover_corpora()
    target = version or _ACTIVE_VERSION
    if target not in _REGISTRY:
        raise KeyError(f"Unknown policy corpus version: {target!r}. Known: {sorted(_REGISTRY)}")
    _INDEX_CACHE.pop(target, None)  # force a fresh build
    _ACTIVE_VERSION = target
    return get_index(target).metadata()


def active_version() -> str:
    return _ACTIVE_VERSION


# --- Public retrieval API (backward compatible with US-105 callers) ---


def build_query_from_profile(profile: dict) -> str:
    """Render an applicant's normalised profile into a short text query."""
    parts = [
        f"loan amount {profile.get('amt_credit')}",
        f"annual income {profile.get('amt_income_total')}",
        f"annuity {profile.get('amt_annuity')}",
        f"employment days {profile.get('days_employed')}",
        f"region rating {profile.get('region_rating_client')}",
    ]
    return " ".join(parts)


def retrieve(query: str, scheme: str = None, top_k: int = 3, corpus_version: str = None) -> dict:
    """Retrieve top-k clauses for a query against the active (or a specified) corpus.

    ``scheme`` restricts candidates to one loan scheme (US-204). ``corpus_version``
    enables replay against an older corpus (US-207).
    """
    return get_index(corpus_version).retrieve(query, scheme=scheme, top_k=top_k)


def retrieve_for_profile(profile: dict, scheme: str = None, top_k: int = 3, corpus_version: str = None) -> dict:
    return retrieve(build_query_from_profile(profile), scheme=scheme, top_k=top_k, corpus_version=corpus_version)


def get_clause(clause_id: str, corpus_version: str = None) -> dict | None:
    """Fetch a single clause's text + version by id (for citation display, US-309)."""
    return get_index(corpus_version).get_clause(clause_id)


def corpus_metadata() -> dict:
    return get_index().metadata()
