# Halcyon Credit Demo Script - Full Pipeline (Sprints 1-4)

End-to-end walkthrough: ingestion -> ML risk score (SHAP) -> multi-scheme
policy retrieval + rule evaluation -> document verification -> regulatory
sub-checks -> fairness audit -> Groq-grounded narrative -> recommendation ->
human decision (with reason code) -> audit trail -> ops dashboard.

## Setup (once)

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.seed_db          # idempotent: seeds 10 sample applications
                                    # (across all 6 loan schemes), the v1.0
                                    # policy corpus, placeholder documents for
                                    # 9 of the 10 applications, and a demo
                                    # underwriter login
uvicorn src.api.main:app --reload --port 8000   # backend
cd ui && npm run dev                             # frontend, separate terminal
```

Add `GROQ_API_KEY` to `.env` for real LLM narrative generation (otherwise the
system automatically falls back to a deterministic template - never errors).

## Walkthrough

1. **Login.** `underwriter@halcyon.com` / `halcyon-demo-1`.
2. **Queue.** 10 seeded applications spanning all 6 loan schemes (Personal,
   Education, Vehicle, Business, First-Time Borrower, Low-Income Assistance -
   scheme is derived at ingestion, shown as "Loan Scheme" after assessing).
   2 are flagged `INCOMPLETE`.
3. **Open an application, click Assess.** The evidence panel shows, in order:
   - Risk score/band + **top-5 SHAP risk factors** with human-readable labels
   - Loan scheme + **all retrieved policy clauses** (not just the top one),
     each with its source clause ID and confidence
   - **Policy rule evaluation** (DTI/income/LTI numeric checks against the
     scheme's authored thresholds - pass/fail per rule)
   - **Document verification** (completeness against the scheme's required
     doc list + a mocked consistency check)
   - **Regulatory status** with all 4 sub-checks (identity/employment/tax/
     sanctions) broken out
   - **Fairness audit** - flags if any demographic segment's approval-rate
     gap exceeds the 5pp hard-block guardrail (a real, live-computed check
     against `reports/ml/fairness_thresholds.json`, not decorative)
   - **Narrative explanation** - Groq-generated (or templated fallback),
     citing only clause IDs actually retrieved for this assessment, with its
     cost in USD
   - Final recommendation (Approve/Decline/Refer) + escalation banner if
     applicable
4. **Decide.** **Accept**, or **Override** with a reason code (dropdown) plus
   free-text reason.
5. **Audit.** `GET /api/assessments/{id}` reconstructs the entire decision
   from stored artefacts alone - inputs, SHAP factors, every clause
   considered, policy rules, document findings, regulatory sub-checks,
   fairness result, narrative, cost, latency, and a SHA-256 hash chaining
   this record to the previous one (tamper-evidence).
6. **Ops Dashboard.** Click "Ops Dashboard" from the queue view for
   throughput, recommendation mix, escalation/acceptance/override rates,
   average cost per assessment, and average/P95 latency across every
   assessment run so far.

## Suggested demo applications (by external ID)

| ID | Scheme | Outcome | What it demonstrates |
|---|---|---|---|
| 100001 | Education | Approve | Clean happy path, high ML confidence |
| 100002 | Business | Approve | Clean happy path, different scheme |
| 100003 | Personal | Refer (low_ml_confidence) | Fairness alert also present (age band) |
| 100005 | Personal | Decline | Regulatory FAIL (identity sub-check), high risk |
| 100006 | Education | Refer (policy_violation) | DTI and LTI both exceed the Education scheme's caps |
| 100008 | First-Time Borrower | Refer (thin_file) | Every First-Time Borrower application is thin-file by definition |
| 100009 / 100010 | - | Refer (missing_risk_score) | INCOMPLETE ingestion (missing required field) |

To demo document verification live: upload a document via
`POST /api/applications/{id}/documents` (multipart, fields `doc_type` +
`file`) for application 100004, which is seeded with zero documents - its
`missing_documents` finding and recommendation change once you upload its
required doc types.

## Known, disclosed limitations (read before demoing)

- **Model quality**: the champion LightGBM model's AUC (0.7575) and F1
  (~0.27) are short of AC-2's targets (0.80 / 0.72) - both above the 0.70
  do-not-ship floor for AUC, not a hidden gap. See
  `reports/ml/lightgbm_metrics.json`.
- **Age is a material proxy** for the excluded `DAYS_BIRTH` feature, even
  though `DAYS_BIRTH` itself is never a model input - see
  `reports/ml/PROXY_LEAKAGE.md`. The fairness hard-block catches this live,
  it isn't swept under the rug.
- **Document consistency, cross-document OCR, and regulatory checks are all
  mocked** (deterministic, never fabricated) - no real KYC/bureau/document
  parsing integration exists, per the PRD's explicit v1 scope.
- **Load-test certification (US-403) and the underwriter pilot (US-408)**
  were not attempted - both require real infrastructure/humans, not code.

## Feedback log

| Date | Reviewer | Comment | Action |
|------|----------|---------|--------|
|      |          |         |        |
|      |          |         |        |
