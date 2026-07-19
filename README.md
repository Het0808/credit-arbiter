# Halcyon Credit — Agentic Underwriting Copilot

A human-in-the-loop copilot for personal-loan underwriting: it ingests an
application, scores default risk with a trained ML model (with SHAP
explanations), retrieves and evaluates the applicable loan-scheme policy
(RAG), runs mock document/regulatory/fairness checks, and produces an
Approve/Decline/Refer recommendation with a full evidence chain — a human
underwriter always makes the final call. Built for the Futurense AI Clinic
capstone; see `docs/Halcyon_Credit_PRD.txt` for the full spec.

## Stack

- **Backend**: FastAPI + SQLAlchemy (SQLite for dev, Postgres-ready), JWT auth
- **ML**: LightGBM + Logistic Regression baseline (`src/risk_model/`), SHAP
- **RAG**: TF-IDF policy retrieval over a 6-scheme, 20-clause corpus
- **LLM**: Groq (explanation generation only, degrades to a template if unset)
- **Frontend**: vanilla HTML/CSS/JS + Vite, no framework

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in JWT_SECRET_KEY; GROQ_API_KEY optional
python -m scripts.seed_db
uvicorn src.api.main:app --reload --port 8000   # terminal 1
cd ui && npm install && npm run dev              # terminal 2
```

Login: `underwriter@halcyon.com` / `halcyon-demo-1`. Full walkthrough with
suggested demo applications: `docs/demo_script.md`.

## What it does

1. **Ingest** a loan application (`src/api/services/ingestion.py`)
2. **Score** default risk with the trained model, top-5 SHAP risk factors
   (`src/risk_model/`, wired in via `src/api/services/scoring.py`)
3. **Retrieve** the applicable policy clauses for the applicant's loan scheme
   and **evaluate** numeric rules (DTI/income/LTI) against them
   (`src/api/services/retrieval.py`, `policy_evaluation.py`)
4. **Verify** documents (completeness + mocked consistency) and run 4 mock
   regulatory sub-checks (identity/employment/tax/sanctions)
5. **Audit fairness** live against a precomputed guardrail — hard-blocks any
   demographic segment with an approval-rate gap over 5pp
   (`src/api/services/fairness_check.py`)
6. **Explain**: a Groq-generated narrative grounded only in the evidence
   above, with a post-hoc check that discards any response citing a policy
   clause that wasn't actually retrieved
7. **Recommend** Approve/Decline/Refer, always escalating to a human on any
   missing input, policy violation, fairness alert, or low-confidence score
8. Underwriter **accepts or overrides** (with a reason code); every decision
   is stored as a hash-chained, fully reconstructable audit record

## Tests & retraining

```bash
pytest                                      # 75 tests, hermetic (no live LLM calls)
python -m scripts.load_test                 # 50-concurrent load test
python -m src.risk_model.archive_baseline && \
python -m src.risk_model.train_lightgbm && \
python -m src.risk_model.tune_thresholds && \
python -m src.risk_model.select_model && \
python -m src.risk_model.fairness && \
python -m src.risk_model.fairness_thresholds  # retrain + redeploy + refresh fairness guardrail
```

## Known limitations

- Champion model AUC 0.76 / F1 0.27 — above the 0.70 do-not-ship floor, short
  of the 0.80/0.72 target (`reports/ml/lightgbm_metrics.json`). The ceiling
  is the feature set (confirmed via hyperparameter tuning), not something
  fixable by more code.
- Document parsing, cross-document consistency, and regulatory checks are
  mocked — no real OCR/KYC/bureau integration (explicit v1 scope).
- No underwriter pilot (US-408) — needs real underwriters, not code.

Full progress log and every sprint's story-by-story status: `claude.md`.
