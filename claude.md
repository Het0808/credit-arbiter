# Halcyon Credit - Agentic Underwriting Copilot

This document provides context and progress for AI Developer Agents working on this project. Read this before diving into the code to understand the architecture and current state of development.

## Project Context
Halcyon Credit is a digital consumer lender building an **Agentic Underwriting Copilot**. The copilot uses ML prediction and a RAG (Retrieval-Augmented Generation) policy engine to generate evidence-backed lending recommendations for human underwriters.

## Architecture Decisions
1. **Backend (Python / FastAPI)**:
   - Located in `src/api/`, split into `routers/` (HTTP layer) and `services/` (pure, testable business logic).
   - Used for ML inference, RAG policy evaluation, and serving API endpoints.
   - **Database**: PostgreSQL (currently using a local SQLite `test.db` fallback for development) handled by `SQLAlchemy`.
   - **Authentication**: Custom JWT-based authentication using `PyJWT` and `passlib[bcrypt]`.

2. **Frontend (Vanilla HTML/JS + Vite)**:
   - Located in `ui/`.
   - Built with raw HTML, Vanilla CSS (glassmorphism dark theme), and Javascript without heavy frameworks like React.
   - Handles the Underwriter Web UI (Login, Queue, Application Detail, Evidence Panel, Accept/Override).

3. **Data & Notebooks**:
   - `notebooks/`: Jupyter notebooks for ML experimentation.
   - `data/`: Datasets, policy corpus, and retrieval eval sets. See `data/README.md`.

## Progress Summary (Sprint 1)
- [x] **Project Scaffolding**: Standard Python ML folder structure (`src`, `tests`, `data`, `notebooks`, `scripts`).
- [x] **Git Repository**: Initialized `main` and `dev` branches.
- [x] **US-101 Backend Authentication**: FastAPI endpoints for `/api/auth/register`, `/api/auth/login`, `/api/users/me`.
- [x] **US-101 Frontend Authentication**: Vite UI with login/registration forms storing JWTs in `localStorage`.
- [x] **US-102 Application Ingestion**: `src/api/services/ingestion.py` normalises a CSV row into an `Application` profile, flags `INCOMPLETE` on missing required fields without crashing. Seeded from `data/sample_applications.csv`.
- [x] **US-103 Baseline ML Risk Score**: `POST /api/score` via `src/api/services/scoring.py` (see placeholder note below).
- [x] **US-104 Minimal Policy Corpus**: `data/policy_corpus_personal_loan_v0.1.json`, 5 clauses, versioned.
- [x] **US-105 RAG Retrieval Engine**: `POST /api/policy/retrieve` via `src/api/services/retrieval.py` (TF-IDF, see placeholder note below).
- [x] **US-106 Recommendation Generation**: `POST /api/assess` via `src/api/services/assessment.py`, kill-switch + rule table.
- [x] **US-107 Underwriter UI**: Queue, application detail, Assess, evidence panel, Accept/Override controls in `ui/`.
- [x] **US-108 Audit Record Persistence**: `decision_record` table (`DecisionRecord` model), queryable via `GET /api/assessments/{id}`.
- [x] **US-109 Mock Regulatory Stub**: `POST /api/regulatory/verify` via `src/api/services/regulatory.py`.
- [x] **US-110 Demo Script**: `docs/demo_script.md`.
- [x] **US-111 Retrieval Accuracy Spike**: `tests/test_retrieval_accuracy.py`, >=85% pass bar met.

## Sprint 1 Placeholders & Known Simplifications
- **Risk scorer is rule-based, not ML-trained.** `src/api/services/scoring.py` computes a probability from debt-to-income, loan-to-income, and employment-tenure ratios. It is a deliberate stand-in for FR-2's trained LogisticRegression, pending the real Home Credit Default Risk 2018 dataset. It already excludes `CODE_GENDER` and `DAYS_BIRTH` from its inputs (assumption A-8b) - keep that exclusion when the real model replaces it.
- **`data/sample_applications.csv` is synthetic**, hand-authored for the POC (10 rows) - not the real ~307K-row Home Credit dataset. The ingestion pipeline code is real and accepts the real file unchanged once sourced; see `data/README.md`.
- **No LLM is used anywhere in Sprint 1.** This is correct per scope - Sprint 1 only requires FR-1/FR-2/FR-4/FR-6/FR-8/FR-10, not FR-9 (narrative explanation generation). An LLM provider/API key has not been chosen yet; that decision and FR-9 are deferred to a later sprint.
- **Retrieval uses TF-IDF + cosine similarity**, not an embedding model - appropriate for a 5-clause single-scheme POC. Upgrade to embeddings/a vector DB when the corpus grows across multiple schemes (Sprint 2+, per assumption A-2).
- **`decision_record` underwriter fields are updated in place** (not a separate insert-only audit_event table) when an underwriter accepts/overrides - matches the AC's literal "appended to the same record" wording. A stricter insert-only audit table is deferred to a later sprint.
- **Recommendation rule table is a Sprint-1 simplification** of FR-3's full multi-scheme policy engine (out of scope until Sprint 2).

## Environment Variables
- Create a `.env` file at the root.
- Required keys:
  - `DATABASE_URL` (e.g., `sqlite:///./test.db` or `postgresql://user:pass@host/db`)
  - `JWT_SECRET_KEY`
  - `JWT_ALGORITHM`
  - `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`

## How to Run
- **Seed the database** (idempotent, run once and any time `data/` changes): `python -m scripts.seed_db`
- **Backend**: `source .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000`
- **Frontend**: `cd ui && npm run dev`
- **Tests**: `pytest` from the repo root
- **Demo**: see `docs/demo_script.md` (includes the seeded demo underwriter login)
