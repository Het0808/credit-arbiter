# Sprint 1 POC Demo Script (US-110)

End-to-end walkthrough of the Sprint 1 thin-slice: ingestion -> risk score ->
single-scheme policy retrieval -> recommendation -> human decision -> audit.

## Setup (once)

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.seed_db          # idempotent: seeds 10 sample applications,
                                    # the v0.1 policy corpus record, and a demo
                                    # underwriter login
uvicorn src.api.main:app --reload --port 8000   # backend
cd ui && npm run dev                             # frontend, separate terminal
```

## Walkthrough

1. **Login.** Open the UI and sign in as the seeded demo underwriter:
   `underwriter@halcyon.com` / `halcyon-demo-1`.
2. **Queue.** The dashboard shows the Application Queue - 10 seeded
   applications, including 2 flagged `INCOMPLETE` (missing a required field).
3. **Open an application.** Click any row to see the normalised applicant
   profile (income, credit, annuity, employment, education, family status,
   region rating, occupation). An `INCOMPLETE` application shows a banner
   naming the missing field(s).
4. **Assess.** Click **Assess**. Within a couple of seconds the Evidence &
   Recommendation panel shows:
   - Risk score (probability) and band (Low / Medium / High)
   - The top retrieved Personal Loan policy clause, its source clause ID, and
     retrieval confidence
   - The mock regulatory verification status
   - The recommendation (Approve / Decline / Refer) and, if applicable, an
     "Escalated for human review" banner
5. **Decide.** Click **Accept** to record the underwriter's agreement, or
   **Override** to supply a mandatory reason and record a disagreement.
6. **Audit.** Re-opening the application (or calling
   `GET /api/assessments/{id}`) shows the full decision record: inputs,
   score, retrieved clause, regulatory result, recommendation, evidence
   chain, and the underwriter's action/reason/timestamp - reconstructable
   from stored data alone.

## Stakeholder caveats (read before demoing)

- **Risk scorer is a placeholder.** `src/api/services/scoring.py` is a
  deterministic rule-based formula (debt-to-income, loan-to-income,
  employment tenure), not a trained model. It stands in for FR-2's
  LogisticRegression until the real Home Credit Default Risk 2018 dataset is
  sourced (see `data/README.md`). It already excludes `CODE_GENDER` and
  `DAYS_BIRTH` per assumption A-8b, so the real model can drop in without
  changing that guarantee.
- **Retrieval is TF-IDF, not embeddings.** Appropriate for a 5-clause
  single-scheme POC; an embedding/vector-DB upgrade is the right move once
  the corpus grows to 15-25 clauses across multiple schemes in Sprint 2+.
- **Regulatory check is a mock.** Deterministic PASS/FAIL, not a live bureau
  or KYC integration (explicitly out of scope for v1 per the PRD).
- **No LLM is used anywhere in this sprint.** Sprint 1 only requires FR-1
  (basic), FR-2 (baseline), FR-4 (single scheme), FR-6 (stub), FR-8 (POC),
  and FR-10 (basic) - not FR-9 (narrative explanation), so this is not a gap,
  it's in scope for a later sprint once an LLM provider is chosen.

## Feedback log

| Date | Reviewer | Comment | Action |
|------|----------|---------|--------|
|      |          |         |        |
|      |          |         |        |
