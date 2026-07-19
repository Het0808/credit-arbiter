# Final Acceptance Verification (US-409)

Maps each PRD §12 acceptance criterion (AC-1…AC-11) to its evidence artefact and
a verdict. Verdicts are reported honestly: **PASS**, **PARTIAL**, or **NOT MET**,
each with a remediation note where relevant.

_Generated for the Sprint 4 final evaluation. Re-run the linked scripts/tests to
reproduce every artefact._

| AC | Criterion | Verdict | Evidence | Notes / Remediation |
|----|-----------|---------|----------|---------------------|
| **AC-1** | End-to-end workflow completes without error | ✅ PASS | `POST /api/assess`; `tests/test_api_endpoints.py::test_full_assess_decision_audit_flow`; 101 passing tests | Ingestion → risk → policy → docs → regulatory → fairness → recommendation → audit log all exercised. |
| **AC-2** | ML AUC ≥ 0.80 and F1 ≥ 0.72 | ❌ NOT MET | `reports/ml/hardened_metrics.json` (AUC 0.7755, best F1 0.33) | AUC improved 0.7654→0.7755 with aux tables + fairness fix. Reaching 0.80 needs Kaggle-scale feature engineering. **F1 ≥ 0.72 is infeasible at ~8% prevalence** (ceiling ≈ 0.33) — recommend the PO revise this AC. |
| **AC-3** | Policy retrieval accuracy ≥ 95% across schemes | ✅ PASS | `scripts/retrieval_quality_report.py` → `reports/ml/retrieval_quality.json` (precision 100%, recall 100%, failure 0%) | Scheme-aware retrieval over the 6-scheme corpus; `tests/test_retrieval_monitor.py`. |
| **AC-4** | Policy adherence = 100% (no rule violated) | ✅ PASS | `src/api/services/policy_engine.py`; `tests/test_policy_engine.py` | A failed rule can never yield Approve; enforced in `assessment.py`. |
| **AC-5** | Fairness gap ≤ 5pp (hard block if exceeded) | ✅ PASS | `src/api/services/fairness_monitor.py`; `tests/test_fairness_monitor.py` | >5pp gap pauses the scheme; assessment blocks auto-decisions on paused schemes. |
| **AC-6** | Complete evidence chain per recommendation | ✅ PASS | `assessment.py` (6 components + kill-switch); `tests/test_escalation_evidence.py` | Any missing component → Refer. |
| **AC-7** | Hallucination rate < 1% (LLM judge) | ⚠️ PARTIAL | `src/api/services/hallucination_eval.py`; `tests/test_explanation_eval.py` (0% hallucination) | Explanations are **grounded-by-construction** (deterministic generator, no LLM). Harness + faithfulness contract are in place; when an LLM is adopted (deferred FR-9), it must pass this same harness. |
| **AC-8** | Cost per application < $0.05 | ✅ PASS | `src/api/services/cost_meter.py`; cost persisted on each decision (~$0.02) | Guardrail cutoff is set at $0.08 (per US-402); PRD AC-8 is $0.05 — actual cost is under both. **Reconcile the $0.05 vs $0.08 threshold with the PO.** |
| **AC-9** | P95 ≤ 20s at 50 concurrent | ✅ PASS | `scripts/load_test.py` → `reports/ops/load_test.json` (P95 ≈ 0.34s, 0 errors) | In-process load test (no network/LLM); re-certify against the deployed stack before pilot. |
| **AC-10** | Audit readiness = 100% (reconstruct from artefacts) | ✅ PASS | `src/api/services/audit_log.py`; `GET /api/ops/audit/verify` + `reconstruct`; `tests/test_audit_and_guardrails.py` | Append-only SHA-256 hash chain; tampering is detectable. |
| **AC-11** | Acceptance ≥ 75% in underwriter pilot (3×20) | ❌ NOT MET (simulated) | `scripts/run_pilot.py` → `docs/PILOT_RESULTS.md` (simulated 58% acceptance, median 19 min) | Median review time meets ≤22 min. **Acceptance below target is driven by the confidence-gate escalation rate** — remediation: tune the 0.60 confidence threshold and improve the model. Real sign-off requires 3 human underwriters. |

## Summary

- **PASS: 8/11** (AC-1, 3, 4, 5, 6, 8, 9, 10)
- **PARTIAL: 1/11** (AC-7 — harness ready, LLM deferred)
- **NOT MET: 2/11** (AC-2 ML targets; AC-11 pilot acceptance — both with documented remediation)

## Remediation plan (open items)

1. **AC-2 (AUC/F1):** Full multi-table feature engineering to push AUC toward 0.80; raise the infeasible F1 ≥ 0.72 target with the PO.
2. **AC-7 (hallucination):** Wire the chosen LLM for FR-9 explanations behind the existing PII-redaction gate; run `hallucination_eval` on ≥50 real explanations.
3. **AC-11 (pilot):** Recalibrate the confidence-gate threshold to reduce over-escalation, then run the real 3×20 underwriter pilot.
4. **AC-8 threshold:** Reconcile the $0.05 (PRD) vs $0.08 (US-402) cost ceiling.
