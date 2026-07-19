# Halcyon Credit — Operations Runbook (US-410)

Operational runbook for the Agentic Underwriting Copilot: deploy, rollback,
kill-switch, on-call escalation, and known failure modes. Pair this with
`docs/ACCEPTANCE_VERIFICATION.md` and the demo in `docs/demo_script.md`.

---

## 1. Services & entry points

| Component | How to run | Notes |
|-----------|-----------|-------|
| Backend API | `uvicorn src.api.main:app --port 8000` | FastAPI; tables auto-created on boot |
| Frontend | `cd ui && npm install && npm run dev` | Vite; talks to `API_BASE` |
| DB seed | `python -m scripts.seed_db` | Idempotent; registers policy corpus versions |
| Retrieval quality job | `python -m scripts.retrieval_quality_report` | Exit ≠ 0 on quality breach |
| Load test | `python -m scripts.load_test 50 100` | Writes `reports/ops/load_test.json` |
| Pilot simulation | `python -m scripts.run_pilot` | Writes `docs/PILOT_RESULTS.md` |
| ML retrain | `python -m src.risk_model.train_hardened` | Rebuilds `models/production/risk_model_v1.pkl` |

## 2. Deploy

1. `pip install -r requirements.txt` (and `requirements-ingestion.txt` if ingesting).
2. Set environment secrets (never commit): `DATABASE_URL`, `JWT_SECRET_KEY`,
   `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`. Loaded via `src/api/settings.py`.
3. `python -m scripts.seed_db` to register policy corpus versions + demo user.
4. Start the API; verify `GET /api/health` → `{"status":"ok"}`.
5. Smoke test: `pytest -q` (must be green) and `python -m scripts.load_test`.

## 3. Rollback

- **Policy corpus:** activate the previous version — `POST /api/policy/reindex {"version":"v0.1"}`.
  Old decisions remain replayable against their stamped `policy_version`.
- **ML model:** restore the prior pickle from `models/` (e.g. `models/lightgbm/…`) into
  `models/production/risk_model_v1.pkl` and its `model_metadata.json`; no schema change needed.
- **Code:** redeploy the previous build tag. The audit log (`audit_event`) is append-only and
  survives rollbacks; verify integrity with `GET /api/ops/audit/verify`.

## 4. Kill-switch & degraded mode (US-405)

- **Activate:** `POST /api/ops/kill-switch {"active": true}` (or the UI "Global kill-switch" toggle).
  Every new assessment then returns **Refer** with `escalation_reason_code = kill_switch_active`,
  taking effect on the next request (< 60 s).
- **Deactivate:** `POST /api/ops/kill-switch {"active": false}`.
- **Automatic degraded mode** (no operator action needed) routes an application to human review when:
  incomplete evidence, retrieval failure, cost-guardrail breach, low model confidence (<0.60),
  unresolved regulatory checks, or a fairness scheme-pause.

## 5. On-call escalation

1. **P1 (auto-decisions on broken state):** activate the kill-switch immediately, then page the Tech Lead.
2. **Fairness hard-block fired:** a scheme is paused (`GET /api/fairness/paused-schemes`). Compliance
   reviews; release with `POST /api/fairness/release/{scheme}` only after sign-off.
3. **Quality/latency/cost alert on the ops dashboard:** triage the failing KPI (`GET /api/ops/dashboard`);
   the load-test report names the slowest stage.

## 6. Known failure modes → response

| Symptom | Likely cause | Response |
|---------|-------------|----------|
| Many decisions escalated | Model confidence gate (<0.60) firing on mid-risk cases | Expected; tune threshold (see AC-11 remediation) |
| `retrieval_failed` on assessments | Corpus not indexed / bad scheme | `POST /api/policy/reindex`; check `GET /api/policy/versions` |
| Scheme auto-Refers everything | Fairness hard-block pause active | Review + `POST /api/fairness/release/{scheme}` |
| Regulatory `escalate_for_review` | Mock service transient failure after retries | Retry the assessment; never fabricated |
| Audit `intact: false` | Tampering / corruption | Freeze writes, page Compliance + Tech Lead, restore from backup |
| Cost guardrail breaches | Oversized explanation projection | Inspect `cost_breakdown`; cap explanation tokens |

## 7. Degraded-mode tabletop drill (AC review)

1. Activate the kill-switch; confirm the next `/api/assess` returns Refer + `kill_switch_active`.
2. Confirm the escalated item appears in `GET /api/assessments/queue/human-review`.
3. Deactivate; confirm normal flow resumes.
4. Verify audit integrity before and after: `GET /api/ops/audit/verify` → `intact: true`.
