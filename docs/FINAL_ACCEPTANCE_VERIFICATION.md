# Final Acceptance Verification (US-409, AC-1...AC-11)

Maps every PRD v2.0 acceptance criterion to the artefact that proves it, generated 2026-07-12. Each row is PASS (evidence meets the literal target) or FAIL (evidence exists, target not met — not silently hidden).

| AC | Criterion | Status | Evidence |
|----|-----------|--------|----------|
| AC-1 | End-to-end workflow completes without error on test cases | **PASS** | `tests/test_assessment.py` exercises ingestion → score → policy → fairness → recommendation → audit for the normal path, kill-switch paths, and record persistence. Full suite: `pytest` → 94 passed, 0 failed (run 2026-07-12). |
| AC-2 | ML AUC ≥ 0.80, F1 ≥ 0.72 | **FAIL** | `reports/ml/lightgbm_metrics.json`: AUC 0.7617, F1 0.2717. Above the 0.70 do-not-ship floor, short of target. `claude.md` documents the ceiling is the feature set (no relational `bureau.csv`/`previous_application.csv` features yet), not tuning — a hyperparameter sweep moved AUC only +0.004. |
| AC-3 | Policy retrieval accuracy ≥ 95% across all schemes | **FAIL** | `scripts/retrieval_quality_report` (2026-07-12): precision (top-1 hit rate) 91.67%, recall 100%, n=12 queries. Above the internal 85% regression floor in `tests/test_retrieval_accuracy.py`, below the PRD's 95% target. Eval set is the POC's 12-query set, not the PRD's full 30-scenario multi-scheme spike. |
| AC-4 | Policy adherence 100% — no recommendation violates a stated rule | **PASS** | `src/api/services/policy_evaluation.py` blocks Approve on any failed numeric rule; `tests/test_policy_evaluation.py` (6 tests) covers DTI/income/LTI/thin-file/unknown-scheme paths, all passing. |
| AC-5 | Fairness gap ≤ 5pp across segments (hard block if exceeded) | **FAIL (correctly hard-blocked)** | `reports/ml/fairness_thresholds.json`: max real gap is 24.19pp (AGE_BAND 18-25). Raw target not met, but every segment above 5pp is flagged `hard_block: true` and `fairness_check.py` forces human review live — see `reports/ml/PROXY_LEAKAGE.md`. Live dashboard now surfaces this as `fairness_gap_pp` / `fairness_gap_alert` in `GET /api/metrics` (added this pass, US-407). |
| AC-6 | Complete evidence chain on every recommendation | **PASS** | `DecisionRecord.evidence_chain_json` includes SHAP factors, matched clauses, policy rule results, doc findings, regulatory sub-checks, fairness result, and narrative — asserted in `tests/test_assessment.py` and rendered in the UI evidence panel. |
| AC-7 | Hallucination rate < 1% via LLM judge | **NOT MEASURED** | `explanation.py`'s grounding check (discards ungrounded citations) is a per-response guard, not the LLM-judge held-out measurement the AC specifies. Documented as deferred in `claude.md` (US-404). No held-out hallucination rate exists to report. |
| AC-8 | Cost per application < $0.05 | **PASS** | Real Groq token-usage cost ~$0.000025/assessment (claude.md), tracked live via `cost_usd` on every `DecisionRecord`; `avg_cost_alert` in `GET /api/metrics` would fire if this regressed above $0.05 (added this pass). |
| AC-9 | P95 latency ≤ 20s at 50 concurrent | **PASS** | `scripts/load_test.py`: P95 12.0s at 50×200 requests (documented result in claude.md). Live `p95_latency_alert` in `GET /api/metrics` now watches this in production traffic too (added this pass). |
| AC-10 | Underwriter can audit every recommendation from stored artefacts alone | **PASS** | Evidence chain (AC-6) + `record_hash` sha256 chain (`US-401`) persisted per record; nothing needed beyond the DB row to reconstruct or verify a decision. |
| AC-11 | Acceptance rate ≥ 75% in underwriter pilot (3×20 files) | **NOT RUN** | US-408 requires 3 real underwriters processing real files — not applicable to an automated pass. No pilot data exists yet. |

## Summary

- **PASS**: AC-1, AC-4, AC-6, AC-8, AC-9, AC-10 (6/11)
- **FAIL, evidence exists**: AC-2, AC-3, AC-5 — all disclosed with root cause, none silently passed
- **Not measurable without more infra/data**: AC-7 (needs LLM-judge harness), AC-11 (needs a human pilot)

## Remediation plan for the open ACs

- **AC-2**: needs relational features from `bureau.csv`/`previous_application.csv` (both already present under `data/home_credit_data/`) or resampling — a real feature-engineering pass, not a tuning pass.
- **AC-3**: expand `data/eval/retrieval_eval_set.json` to the PRD's full 30-scenario, multi-scheme spike; current 12-query set is POC-sized.
- **AC-5**: no code fix — the gap is a genuine property of the training data (age is a real risk-correlated proxy). Current mitigation (hard-block escalation) is the correct control per PRD's own kill-switch language; closing the raw gap would need bias-mitigation techniques (reweighing, threshold-per-group) which the PRD doesn't scope for this sprint.
- **AC-7**: build the LLM-judge hallucination harness against a held-out explanation set (out of scope for this pass — needs a judge-model integration).
- **AC-11**: schedule the underwriter pilot.
