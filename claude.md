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
- **`data/sample_applications.csv` is synthetic**, hand-authored for the POC (10 rows) - not the real ~307K-row Home Credit dataset. The ingestion pipeline code is real and accepts the real file unchanged once sourced; see `data/README.md`.
- **`decision_record` underwriter fields are updated in place** (not a separate insert-only audit_event table) when an underwriter accepts/overrides - matches the AC's literal "appended to the same record" wording. A stricter insert-only audit table is deferred to a later sprint; the SHA-256 hash chain added in Sprint 4 (`record_hash`) gives tamper-evidence without that redesign.

## Progress Summary (Sprints 2-4)

The trained ML models pulled in from `dev` at the start of Sprint 2 had several
real bugs (fixed as part of this work, not just wrapped): `CODE_GENDER` and
`DAYS_BIRTH` were being fed to the model as direct predictors, a **regression
against assumption A-8b**; `requirements.txt` was missing `pandas`/`lightgbm`/
`shap`/`matplotlib`/`seaborn`, so a fresh clone couldn't run `src/risk_model/`
at all; `select_model.py`'s deploy step and `preprocess.py`'s data-path
fallback both pointed at paths that didn't exist. All fixed; both models were
retrained clean and the numbers below are the honest post-fix result.

- [x] **US-201 Production ML model**: `src/risk_model/` retrained with
  `CODE_GENDER`/`DAYS_BIRTH`/`AGE_YEARS` excluded from features. **AUC 0.7617
  (LightGBM champion, tuned), 0.7415 (Logistic Regression baseline) - both
  above the 0.70 do-not-ship floor but short of the 0.80 target (AC-2). A
  hyperparameter bump (n_estimators/depth/lr) moved AUC +0.004 - the ceiling
  is the feature set, not tuning; not chasing further.** F1 ~0.27,
  also short of the 0.72 target. This is disclosed, not hidden - see
  `reports/ml/lightgbm_metrics.json` and `model_comparison_logreg_vs_lightgbm.md`.
  Wired into the live `/api/assess` and `/api/score` endpoints via
  `src/api/services/scoring.py` -> `src/risk_model/predict.py:predict_from_profile()`
  (scores a live applicant profile directly, no CSV/SK_ID_CURR lookup;
  model+SHAP explainer cached module-level for latency).
- [x] **US-202 SHAP explainability**: top-5 risk factors with human-readable
  labels, returned by `/api/assess` and shown in the UI evidence panel.
- [x] **US-203/204 Multi-scheme policy + RAG**: `data/policy_corpus_v1.0.json`,
  20 clauses across all 6 schemes. `Application.loan_scheme` is derived at
  ingestion (`src/api/services/ingestion.py:_derive_loan_scheme`) - HC2018 has
  no native loan-purpose field, so this is a documented simplification
  (income-floor/thin-file signals route to the matching specialty program,
  everything else is deterministically hash-distributed). `retrieval.py`
  filters candidates by scheme before TF-IDF ranking.
- [x] **US-206 Policy rule evaluator**: `src/api/services/policy_evaluation.py`,
  numeric DTI/income/LTI rules per scheme kept in lockstep with the authored
  clause text. Any failed rule blocks Approve.
- [x] **US-208/209 Explanation generation + PII redaction**: real LLM via
  **Groq** (`src/api/services/explanation.py`), grounded prompt built only
  from aggregated evidence, passed through `pii_redaction.py`'s linter first.
  Post-generation grounding check discards any response that cites a clause
  ID not actually retrieved (falls back to a deterministic template - never a
  kill-switch trigger). ~$0.000025/assessment actual cost.
- [x] **US-301/302 Document verification**: upload endpoint
  (`src/api/routers/documents.py`), completeness check against a
  required-docs-by-scheme table, consistency check mocked (no real OCR in
  v1, same hash-determinism style as the regulatory mock).
- [x] **US-303 Regulatory sub-checks**: 4 named checks (identity, employment,
  tax, sanctions), independently deterministic, aggregated to one status.
- [x] **US-304/305 Fairness hard-block + proxy leakage**:
  `reports/ml/fairness_thresholds.json` (regenerate via
  `python -m src.risk_model.fairness_thresholds` after every retrain) is
  consulted live per assessment (`src/api/services/fairness_check.py`). Real
  finding: age is a material proxy even with `DAYS_BIRTH` excluded from
  training (`EMPLOYMENT_YEARS` r=0.35, `EXT_SOURCE_MEAN` r=0.28 vs age - see
  `reports/ml/PROXY_LEAKAGE.md`) - the 18-25 age band approves 25pp below
  baseline and is correctly hard-blocked, not silently auto-decided.
- [x] **US-306 Full HITL escalation**: every PRD §11 trigger implemented -
  missing risk score, retrieval failure, thin-file, low ML confidence
  (recalibrated floor, see `assessment.py:ML_CONFIDENCE_FLOOR` - the literal
  "<0.60" spec escalated almost everything given this model's real
  probability distribution), policy violation, missing/inconsistent
  documents, fairness alert, regulatory escalation.
- [x] **US-307/308/309 Evidence chain + override reason codes + evidence
  panel UI**: full evidence chain (SHAP, all matched clauses, policy rules,
  doc findings, regulatory sub-checks, fairness, narrative) in every decision
  record; `reason_code` enum alongside free-text override reason; UI evidence
  panel extended with all of the above.
- [x] **US-401 Hash-chain audit**: `DecisionRecord.record_hash` =
  sha256(previous record's hash + this record's content) - a POC-scale
  tamper-evidence approximation, not true WORM storage.
- [x] **US-402 Cost guardrail**: real per-call cost tracked from Groq's
  returned token usage; a pre-call worst-case estimate skips the LLM call
  entirely if it would breach $0.08 (`explanation.py:COST_GUARDRAIL_USD`).
- [x] **US-405 Kill-switch**: `HALCYON_KILL_SWITCH=true` env var forces every
  new assessment straight to Refer, checked first in `run_assessment()`.
- [x] **US-407 subset - Ops dashboard**: `GET /api/metrics` + a minimal UI
  view (throughput, recommendation mix, escalation/acceptance/override
  rates, avg cost, avg/p95 latency).

- [x] **US-403 load test**: `scripts/load_test.py` (stdlib only), 50
  concurrent x 200 `/api/assess` requests, P95 12.0s (AC-9: <=20s, PASS).
  Surfaced and fixed a real bug along the way: the default SQLAlchemy pool
  (5+10=15 connections) couldn't serve 50 concurrent requests, and the
  tempting fix (`StaticPool`, one shared sqlite3 connection) is not actually
  thread-safe despite `check_same_thread=False` - caused `bad parameter or
  other API misuse` errors under load. Real fix: `pool_size=50` + sqlite
  `timeout=15` busy-wait in `src/api/database.py`.

## Progress Summary (Sprint 5, in progress)
- [x] **US-205 Retrieval quality monitoring**:
  `src/api/services/retrieval_quality.py` - a pragmatic stand-in for RAGAS
  context precision/recall (single-relevant-clause-per-query eval against
  the existing `data/eval/retrieval_eval_set.json`, not the full LLM-judge
  RAGAS metric - same documented-simplification style as US-404's grounding
  check). precision = top-1 hit rate (matches how `assessment.py` actually
  consumes retrieval - only `clauses[0]` drives the recommendation); recall
  = hit-in-top-k. Runnable as the "daily eval job" via
  `python -m scripts.retrieval_quality_report` (currently 91.7%/100%, both
  above the 0.85 floor). Live retrieval failure rate (not just the static
  eval set) is now in `GET /api/metrics` - `retrieval_failure_rate` +
  `retrieval_failure_alert` (fires above the 5% AC threshold), surfaced on
  the ops dashboard.
- [x] **US-207 Policy version management & re-index trigger**: source_id +
  version were already stamped on every retrieved clause and persisted
  verbatim in `DecisionRecord.evidence_chain_json` (US-204) - old decisions
  are already replayable against the exact clause text/version they saw,
  no redesign needed there. New: `retrieval.py:reindex_corpus(path)` hot-
  reloads the TF-IDF index from disk without a restart, wired to
  `POST /api/policy/reindex` (auth-gated like every other endpoint - no
  precedent for role-scoped admin gating elsewhere in this codebase, so
  none was added here either), which also upserts the `PolicyCorpusVersion`
  row (reusing `scripts/seed_db.py`'s upsert-by-version+source_file
  pattern).
- [x] **US-406 subset - Secrets scan + external-call audit log**:
  `scripts/scan_secrets.py` (stdlib regex, no new dependency) scans every
  git-tracked file for hardcoded secret-shaped literals - flags known key
  prefixes (`gsk_`, `sk-`, `AKIA...`, `ghp_`, `xox...`) and generic
  `key/secret/token/password = "literal"` assignments, ignores
  `os.environ.get(...)` lookups and placeholder values. Run via
  `python -m scripts.scan_secrets`; asserted clean by
  `tests/test_secrets_scan.py::test_repo_scan_is_clean`. Every issued (not
  skipped) Groq call now writes a JSON-line audit entry via
  `src/api/audit_log.py` to `logs/external_calls.log` (gitignored - runtime
  output, not source), success and failure both recorded. Least-privilege
  IAM/vault integration deferred - no cloud infra in this POC to scope it
  against.
- [x] **US-410 Runbook**: `docs/RUNBOOK.md` - deploy, rollback (code/model/
  fairness-thresholds, in that order), kill-switch, a known-failure-mode
  table, on-call escalation steps, retrain sequence, and a 4-step
  degraded-mode tabletop drill (LLM outage, kill-switch, DB unreachable,
  recovery) with an expected-behavior check for each step.

- [x] **US-407 full - Alert on threshold breach**: `GET /api/metrics` now
  computes `avg_cost_alert` (AC-8, >$0.05), `p95_latency_alert` (AC-9,
  >20s), and `fairness_gap_alert`/`fairness_gap_pp` (AC-5, reusing
  `fairness_check.py`'s existing thresholds-file loader via the new
  `current_max_fairness_gap_pp()`), alongside the existing
  `retrieval_failure_alert`. Any breach is logged as a WARNING
  (`metrics.py`) - a real, durable, greppable alert, not just a dashboard
  value nobody's looking at - same minimal log-based style as
  `audit_log.py`, no email/Slack integration added since none exists in
  this POC.
- [x] **US-409 Final Acceptance Verification**: `docs/FINAL_ACCEPTANCE_VERIFICATION.md`
  maps every PRD AC-1...AC-11 to its evidence artefact. 6/11 PASS
  (AC-1/4/6/8/9/10), 3 FAIL-with-disclosed-cause (AC-2/3/5, all pre-existing
  known limitations, not hidden), 2 not measurable without more
  infra/humans (AC-7 needs an LLM-judge harness, AC-11 needs the real
  pilot).
- Fixed along the way: `scripts/scan_secrets.py` was flagging its own test
  file's synthetic secret-shaped fixtures as real findings (pre-existing
  bug, unrelated to this pass's changes, caught because `pytest` was run
  full-suite rather than just the new tests) - excluded
  `tests/test_secrets_scan.py` from the scan.

### Explicitly not done (needs real humans, not more code)
- **US-408 underwriter pilot**: requires 3 real underwriters processing real
  files; not applicable to an automated coding pass.
- **US-404 full hallucination-eval harness**: the grounding/citation check in
  `explanation.py` is a pragmatic per-response check, not the full
  LLM-judge + spot-check pipeline the PRD describes for CI gating.

### Known limitations carried forward
- AUC (0.7617) and F1 (~0.27) remain short of AC-2's targets even after the
  A-8b fix and a hyperparameter tune - the ceiling is the feature set, not
  something more code fixes. Reaching AC-2 needs new features (relational
  data from `bureau.csv`/`previous_application.csv`) or resampling.

## Environment Variables
- Create a `.env` file at the root.
- Required keys:
  - `DATABASE_URL` (e.g., `sqlite:///./test.db` or `postgresql://user:pass@host/db`)
  - `JWT_SECRET_KEY`
  - `JWT_ALGORITHM`
  - `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`
  - `GROQ_API_KEY` (FR-9 explanation generation, US-208) - without it, `generate_explanation()` degrades to the deterministic template automatically, never errors.
  - `GROQ_MODEL` (default `llama-3.1-8b-instant`)
  - `HALCYON_KILL_SWITCH` (optional, US-405) - set to `true` to force every new assessment to Refer.

## How to Run
- **Seed the database** (idempotent, run once and any time `data/` changes): `python -m scripts.seed_db`
- **Backend**: `source .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000`
- **Frontend**: `cd ui && npm run dev`
- **Tests**: `pytest` from the repo root (hermetic - a conftest.py autouse fixture disables real Groq calls so the suite never depends on network/API cost)
- **Retrain the ML models**: `python -m src.risk_model.archive_baseline && python -m src.risk_model.train_lightgbm && python -m src.risk_model.tune_thresholds && python -m src.risk_model.select_model && python -m src.risk_model.fairness && python -m src.risk_model.fairness_thresholds` (run in this order; the last two regenerate the live fairness hard-block lookup)
- **Demo**: see `docs/demo_script.md` (includes the seeded demo underwriter login)
