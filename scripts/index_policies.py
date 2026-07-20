"""Index the active policy corpus into Qdrant (vector RAG).

Run once after starting Qdrant (and whenever the corpus changes):

    RETRIEVER=vector QDRANT_URL=http://localhost:6333 python -m scripts.index_policies

With an embedded/in-memory Qdrant the app indexes lazily on first query, so this
script is only needed for a persistent Qdrant server.
"""

from src.api.services import retrieval as retrieval_service
from src.api.services.vector_retrieval import QDRANT_COLLECTION, QDRANT_URL, index_clauses


def main() -> None:
    for version in retrieval_service.list_versions()["available_versions"]:
        index = retrieval_service.get_index(version)
        n = index_clauses(index.clauses, index.corpus_version, force=True)
        print(f"indexed {n} clauses (corpus {version}) into Qdrant collection "
              f"'{QDRANT_COLLECTION}' at {QDRANT_URL}")


if __name__ == "__main__":
    main()
