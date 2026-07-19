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

## Progress Summary (Sprint 2 — in progress)
- [x] **US-201 Production ML Hardening**: `src/risk_model/train_hardened.py` retrains the champion LightGBM on the real 307K-row Home Credit data with auxiliary-table aggregates (`src/risk_model/aux_features.py`: bureau, previous_application, installments, POS, credit-card → `data/home_credit_data/aux_features.parquet`). **Held-out ROC-AUC 0.7755** (up from 0.7654). See placeholder note on the 0.80 target below.
- [x] **US-201 Fairness fix (A-8b)**: `src/risk_model/config.py` now EXCLUDES `CODE_GENDER`, `DAYS_BIRTH`, and the `AGE_YEARS` proxy from the model feature set (previously leaked in). Enforced by `tests/test_risk_model.py::test_protected_features_are_excluded_from_model_inputs` and an assertion in the trainer.
- [x] **US-202 SHAP Top Contributors**: `src/risk_model/shap_explain.py` returns top-5 contributors each with a human-readable `label` and `direction` (increases/decreases risk).
- [x] **US-203 Multi-Scheme Corpus v1.0**: `data/policy_corpus_v1.0.json` — 6 schemes (Personal, Education, Vehicle, Business, First-Time, Low-Income), 21 clauses, each clause carries a machine-readable `rule`. v0.1 retained for replay.
- [x] **US-204 Scheme-Aware Retrieval**: `src/api/services/retrieval.py` — scheme-filtered TF-IDF, every clause returns `source_id` + `corpus_version`.
- [x] **US-205 Retrieval Quality Monitoring**: `src/api/services/retrieval_monitor.py` + `scripts/retrieval_quality_report.py` (context precision/recall + failure-rate alert; currently 100%/100%/0%).
- [x] **US-206 Policy Evaluation Engine**: `src/api/services/policy_engine.py` — deterministic rule evaluator; a failed rule can never yield Approve. Wired into `assessment.py`.
- [x] **US-207 Policy Version Management**: version registry + `POST /api/policy/reindex`, `GET /api/policy/versions`; `decision_record.policy_version` stamped on each decision.
- [x] **US-209 PII Redaction Layer**: `src/api/services/pii_redaction.py` — redacts SSN/DOB/account/card/email/phone and blocks any residual-PII prompt before an LLM call.
- [ ] **US-208 Explanation Generation (FR-9)**: DEFERRED — needs an LLM provider (not yet chosen). The PII-redaction gate it depends on is already built.

## Progress Summary (Sprint 4 — complete)
- [x] **US-401 Immutable Audit Log**: `src/api/services/audit_log.py` — append-only SHA-256 hash chain (`audit_event`); every decision + external call logged; `GET /api/ops/audit/verify` + `/audit/reconstruct/{id}`. Tamper-evident (test-enforced).
- [x] **US-402 Cost Metering + Hard Cutoff**: `src/api/services/cost_meter.py`; per-app cost persisted on the decision record; >$0.08 projected → human fallback.
- [x] **US-403 Load Test + P95**: `scripts/load_test.py` → `reports/ops/load_test.json`. 50-concurrent, P95 ≈ 0.34s (≤20s), 0 errors, per-stage attribution. (In-process; re-certify on deployed stack.)
- [x] **US-404 Hallucination Eval**: `src/api/services/explanation.py` (grounded generator, US-208 stand-in, no LLM) + `hallucination_eval.py` (faithfulness/hallucination harness, blocks release below thresholds). 0% hallucination by construction.
- [x] **US-405 Kill-Switch + Degraded Mode**: `src/api/services/kill_switch.py` + `POST /api/ops/kill-switch`; automatic degraded-mode routing to human review on any guardrail breach.
- [x] **US-406 Secrets + Least-Privilege**: `src/api/settings.py` — secrets from env only, per-tool scopes; secret-literal scan test; external calls audited.
- [x] **US-407 Ops Dashboard**: `src/api/services/ops_metrics.py` + `GET /api/ops/dashboard` (throughput, P95, cost/app, acceptance, override, fairness gap + thresholds/alerts) and a UI dashboard panel.
- [x] **US-408 Pilot (simulated)**: `scripts/run_pilot.py` → `docs/PILOT_RESULTS.md`. 3×20 simulation: 58% acceptance (below 75% target), 19 min median review (meets). Override remediation list produced. Real sign-off needs human underwriters.
- [x] **US-409 Acceptance Verification**: `docs/ACCEPTANCE_VERIFICATION.md` maps AC-1…AC-11 to evidence (8 PASS, 1 PARTIAL, 2 NOT MET, each with remediation).
- [x] **US-410 Runbook**: `docs/RUNBOOK.md` — deploy/rollback/kill-switch/on-call/failure-modes + degraded-mode drill.

## Sprint 4 Placeholders & Known Simplifications
- **AC-2 (AUC≥0.80, F1≥0.72) NOT MET**: AUC 0.7755; F1 target infeasible at ~8% prevalence — flagged to PO.
- **AC-7 hallucination**: harness + grounded generator in place; a real LLM judge (FR-9) is deferred pending provider choice.
- **AC-11 pilot acceptance 58% < 75% (simulated)**: driven by confidence-gate escalation; remediation documented; needs a real human pilot.
- **AC-8 threshold discrepancy**: PRD says <$0.05, US-402 story says <$0.08; actual ~$0.02 is under both — reconcile with PO.
- **Load test / cost are in-process** (no network/LLM); re-certify on the deployed stack.

## Progress Summary (Sprint 3 — complete)
- [x] **US-301 Document Upload & Storage**: `Document` model + `src/api/services/document_service.py`; `POST/GET /api/applications/{id}/documents` (type-validated; unsupported types 400).
- [x] **US-302 Document Verification**: completeness (missing required docs per scheme) + consistency (name/income agreement) via `GET /api/applications/{id}/documents/verify`.
- [x] **US-303 Regulatory Validation**: `src/api/services/regulatory.py` now runs identity/employment/tax/sanctions checks with exponential-backoff retry; escalates on exhaustion, never fabricates. Worst case ~0.28s (inside the 500ms budget).
- [x] **US-304 Fairness Monitor + Hard-Block**: `src/api/services/fairness_monitor.py` computes decision-history segment approval-rate deltas; a >5pp gap pauses the scheme (`SchemePause`), and `assessment.py` blocks auto-decisions for paused schemes. Endpoints under `/api/fairness`.
- [x] **US-305 Proxy Leakage**: `src/risk_model/proxy_leakage.py` — Cramér's V / correlation ratio for OCCUPATION_TYPE & REGION_RATING_CLIENT vs gender/age → `reports/ml/PROXY_LEAKAGE.md`. Finding: OCCUPATION_TYPE materially correlates with gender (V≈0.40).
- [x] **US-306 Escalation Workflow**: any escalation trigger routes to `GET /api/assessments/queue/human-review` with a reason code; ML confidence <0.60 forces Refer.
- [x] **US-307 Evidence Chain Completion**: assessment assembles all 6 components (risk, risk-factors, policy clauses, doc findings, regulatory, fairness); any missing component → Refer (kill-switch). `decision_record.evidence_complete` stored.
- [x] **US-308 Override Capture**: `POST /api/assessments/{id}/decision` requires a `reason_code` (enum) + note on override; `GET /api/assessments/metrics/override-rate`.
- [x] **US-309 Evidence Panel UI**: `ui/` renders the 5 evidence sections (Risk · Policy · Docs · Regulatory · Fairness) with citations + timestamps; clicking a clause citation opens its source text + version (`GET /api/policy/clause/{id}`).

## Sprint 3 Placeholders & Known Simplifications
- **Document OCR is out of scope**: `declared_name` / `declared_income` are supplied as structured metadata at upload; the completeness/consistency checks are identical to what they'd be over OCR-extracted fields.
- **Regulatory services are deterministic mocks** (hash-based verdicts + a `force_fail` transient-outage simulation) — no live KYC/bureau integration in v1.
- **The live assessment path uses the rule-based scorer** (`services/scoring.py`), not the trained ML model, because the seeded POC applications aren't rows in the 307K HC dataset the ML model expects. `explain_score()` supplies risk factors so the evidence chain stays complete. Wiring the ML model end-to-end needs an inference feature-assembly path for arbitrary applications (future work).
- **UI Vite build requires `npm install` in `ui/`** (node_modules not committed). JS is syntax-validated.

## Sprint 2 Placeholders & Known Simplifications
- **ML ROC-AUC is 0.7755, below the US-201 AC target of 0.80.** This is an honest improvement from the prior 0.7654 champion using all five auxiliary tables plus the fairness exclusions. Reaching 0.80 on Home Credit realistically requires Kaggle-scale feature engineering (hundreds of features across all tables). Treat 0.80 as a stretch target requiring a dedicated FE effort.
- **The US-201 AC also lists `F1 ≥ 0.72`, which is not attainable on this dataset.** At ~8% default prevalence, threshold-optimised F1 tops out near 0.30–0.33 (this model: best F1 0.33 @ threshold 0.66). Flag this AC to the PO as likely a spec error.
- **US-208 (LLM explanation generation) is deferred** pending an LLM-provider decision. Everything it depends on (SHAP factors, policy clause source_ids, PII redaction) is in place.

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
