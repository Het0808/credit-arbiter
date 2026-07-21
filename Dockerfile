# Backend image: FastAPI + trained model (ML) + vector RAG + LLM explanation.
FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required by lightgbm / onnxruntime (fastembed).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install all serving deps (base + ml + rag + llm). Copied first for layer caching.
COPY requirements.txt requirements-ml.txt requirements-rag.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt -r requirements-rag.txt -r requirements-llm.txt

# App code + assets the runtime needs.
COPY src ./src
COPY scripts ./scripts
COPY data/policy_corpus_v1.0.json data/policy_corpus_personal_loan_v0.1.json ./data/
COPY data/eval ./data/eval
# Trained model + metadata (gitignored file must exist in the build context;
# if absent, the app falls back to the rule-based scorer).
COPY models/production ./models/production

# Production defaults; secrets/URLs come from the environment (compose / .env).
ENV RISK_SCORER=ml \
    RETRIEVER=vector \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
