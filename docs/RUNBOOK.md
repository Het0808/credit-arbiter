# Halcyon Credit Runbook (US-410)

Operational reference for running, degrading, and recovering the copilot.
POC-scale: today's "deploy" is a single backend + frontend process pair, not
a cluster - this runbook reflects that honestly rather than describing
infrastructure that doesn't exist yet.

## Deploy

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in JWT_SECRET_KEY; GROQ_API_KEY optional
python -m scripts.seed_db         # idempotent
uvicorn src.api.main:app --port 8000          # backend, terminal 1
cd ui && npm install && npm run dev            # frontend, terminal 2
```

Pre-deploy checklist:
- `pytest` - 83 tests, all hermetic (no live LLM calls; a conftest.py autouse
  fixture disables the real Groq client).
- `python -m scripts.scan_secrets` - fails (exit 1) if any secret-shaped
  literal is committed (US-406).
- `reports/ml/fairness_thresholds.json` is present and matches the currently
  deployed model - if you retrained without regenerating it, the fairness
  hard-block will judge against a stale baseline (see Retrain below).

## Rollback

There is no blue/green deploy here - rollback means restoring the previous
known-good state of three independent artefacts, in this order:

1. **Code**: `git checkout <previous-tag-or-commit>` (or revert the bad
   commit), then re-run `pip install -r requirements.txt` in case
   dependencies changed.
2. **Model**: `src/risk_model/archive_baseline.py` archives the prior
   champion before every retrain (see `models/`) - restore the archived
   `.pkl` if the new model regresses AUC/fairness. Never roll back code
   without also checking whether the model file it expects still matches.
3. **Fairness thresholds**: `reports/ml/fairness_thresholds.json` must stay
   in lockstep with whichever model is live - if you roll back the model,
   regenerate thresholds against it
   (`python -m src.risk_model.fairness_thresholds`), don't reuse the newer
   model's thresholds against an older model.

Database rollback: `DecisionRecord` rows are hash-chained (US-401) - deleting
or editing a row breaks every subsequent hash. Don't repair audit data by
hand; if a bad row must be voided, add a new record referencing it rather
than mutating history.

## Kill-switch

`HALCYON_KILL_SWITCH=true` (env var, checked first in
`src/api/services/assessment.py:run_assessment()`) forces every new
assessment straight to `Refer` - use this to freeze automated recommendations
without stopping the service. Set it, restart the backend process, confirm
via one `/api/assess` call that the response's `recommendation` is `Refer`
with `escalation_flag=true`. Unset and restart to resume normal operation.

This is the single fastest lever in an incident - reach for it before
anything more surgical (patching a service, rolling back a model) when the
failure mode is "recommendations look wrong" and you need to stop the
bleeding immediately.

## Known failure modes and fallbacks

| Condition | System behavior | Action needed |
|---|---|---|
| `GROQ_API_KEY` unset, invalid, or Groq API down/timeout (>8s) | `explanation.py` degrades to a deterministic template narrative automatically. Never blocks a decision. | None urgent; check `logs/external_calls.log` for `"success": false` entries if it's persistent - may indicate a Groq outage worth tracking. |
| Per-call cost estimate would exceed `COST_GUARDRAIL_USD` ($0.08) | Call is skipped before being sent; template fallback used. | Investigate prompt size growth (e.g. unusually large evidence chain) if this starts firing often. |
| Any policy rule fails, fairness hard-block trips, regulatory sub-check fails, or documents are missing/inconsistent | Recommendation is escalated to `Refer`, never silently auto-approved (US-306). | Human underwriter reviews; no on-call action required, this is the system working as designed. |
| SQLite `database is locked` under concurrent load | `src/api/database.py` sets `pool_size=50` + `timeout=15`; under heavier load than the P95 12.0s / 50-concurrent baseline (`scripts/load_test.py`), requests will queue rather than fail outright, up to the timeout. | If timeouts start appearing, that's the signal to migrate `DATABASE_URL` to Postgres (already supported, no code change) rather than tuning SQLite further. |
| Fairness thresholds file missing or stale relative to the deployed model | Fairness check has nothing valid to compare against. | Regenerate immediately: `python -m src.risk_model.fairness_thresholds`. Do not run assessments against a model with no matching thresholds file - treat as equivalent to the kill-switch being needed. |

## On-call escalation

1. **Recommendations look systematically wrong** (wrong scheme, wrong risk
   band, fairness alerts spiking) -> set `HALCYON_KILL_SWITCH=true`
   immediately, then investigate. This buys time without an outage.
2. **Backend is down / erroring on every request** -> restart the uvicorn
   process; check `DATABASE_URL` connectivity first if Postgres is in use.
3. **Suspected secret leak** -> run `python -m scripts.scan_secrets` against
   the current `HEAD` and recent history; rotate the exposed credential
   (`GROQ_API_KEY` / `JWT_SECRET_KEY`) and redeploy with a fresh `.env`
   immediately, don't wait for root-cause.
4. **Anything unresolved after step 1-3** -> escalate to the model/policy
   owner with the specific `DecisionRecord.id`(s) in question - the full
   evidence chain (`GET /api/assessments/{id}`) is enough to reconstruct the
   decision without re-running anything live.

## Retrain (only when model quality or fairness needs it)

```bash
python -m src.risk_model.archive_baseline
python -m src.risk_model.train_lightgbm
python -m src.risk_model.tune_thresholds
python -m src.risk_model.select_model
python -m src.risk_model.fairness
python -m src.risk_model.fairness_thresholds
```

Order matters: the last two steps regenerate the live fairness guardrail
against whichever model `select_model` just deployed. Skipping them leaves
`/api/assess` fairness-checking against a stale model's thresholds.

## Degraded-mode tabletop drill

Run this after any change to the assessment pipeline, and periodically as a
standing drill:

1. Unset `GROQ_API_KEY` (or point it at an invalid key) and restart the
   backend. Run one `/api/assess` call. **Expected**: `source: "template"`,
   `cost_usd: 0.0`, recommendation unaffected.
2. Set `HALCYON_KILL_SWITCH=true`, restart, run one `/api/assess` call.
   **Expected**: `recommendation: "Refer"`, `escalation_flag: true`,
   regardless of input.
3. Point `DATABASE_URL` at an unreachable host, restart. **Expected**:
   backend fails to start (fail-fast, not silent data loss) - confirms there
   is no path where decisions are made without being persisted.
4. Unset `HALCYON_KILL_SWITCH` and restore `GROQ_API_KEY`/`DATABASE_URL`,
   restart, confirm one normal `/api/assess` call returns to
   `source: "llm"` (if the key is valid) and a non-`Refer`-only
   recommendation set.

A successful drill is all four steps behaving exactly as expected above -
any deviation is a regression in the fallback logic, not an acceptable
surprise, and blocks the next deploy until fixed.
