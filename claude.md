# Halcyon Credit - Agentic Underwriting Copilot

This document provides context and progress for AI Developer Agents working on this project. Read this before diving into the code to understand the architecture and current state of development.

## Project Context
Halcyon Credit is a digital consumer lender building an **Agentic Underwriting Copilot**. The copilot uses ML prediction and a RAG (Retrieval-Augmented Generation) policy engine to generate evidence-backed lending recommendations for human underwriters.

## Architecture Decisions
1. **Backend (Python / FastAPI)**:
   - Located in `src/api/`.
   - Used for ML inference, RAG policy evaluation, and serving API endpoints.
   - **Database**: PostgreSQL (currently using a local SQLite `test.db` fallback for development) handled by `SQLAlchemy`.
   - **Authentication**: Custom JWT-based authentication using `PyJWT` and `passlib[bcrypt]`.

2. **Frontend (Vanilla HTML/JS + Vite)**:
   - Located in `ui/`.
   - Built with raw HTML, Vanilla CSS (glassmorphism dark theme), and Javascript without heavy frameworks like React.
   - Handles the Underwriter Web UI (Login, Dashboard, Evidence Panel).

3. **Data & Notebooks**:
   - `notebooks/`: Jupyter notebooks for ML experimentation.
   - `data/`: Datasets and mock regulatory structures.

## Progress Summary (Sprint 1)
- [x] **Project Scaffolding**: Standard Python ML folder structure (`src`, `tests`, `data`, `notebooks`, `scripts`).
- [x] **Git Repository**: Initialized `main` and `dev` branches.
- [x] **Backend Authentication**: Created FastAPI endpoints for `/api/auth/register`, `/api/auth/login`, and `/api/users/me`.
- [x] **Frontend Authentication**: Created a sleek Vite UI with login and registration forms that store JWTs in `localStorage`.
- [ ] **Data Ingestion Pipeline**: Needs implementation.
- [ ] **Baseline ML Model (Risk Prediction)**: Needs implementation.
- [ ] **Single-Scheme RAG**: Needs implementation.
- [ ] **Underwriter Evidence Panel (UI)**: Pending integration with ML/RAG backend.

## Environment Variables
- Create a `.env` file at the root.
- Required keys:
  - `DATABASE_URL` (e.g., `sqlite:///./test.db` or `postgresql://user:pass@host/db`)
  - `JWT_SECRET_KEY`
  - `JWT_ALGORITHM`
  - `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`

## How to Run
- **Backend**: `source .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000`
- **Frontend**: `cd ui && npm run dev`
