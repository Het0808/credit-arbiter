# Data Directory

## sample_applications.csv

This is **synthetic, hand-authored demo data** (10 rows) used to build and
demo the Sprint 1 proof-of-concept end-to-end flow (ingestion -> scoring ->
retrieval -> recommendation -> audit). It is **not** the real Home Credit
Default Risk 2018 dataset (~307,511 rows across `application_train.csv`,
`bureau.csv`, `bureau_balance.csv`) referenced in the PRD (assumption A-1,
A-4, A-5).

The column schema is a realistic subset of the real HC2018 `application_train`
columns, so the ingestion pipeline in `src/api/services/ingestion.py` is
written to accept the real file unchanged once it is sourced - swapping in
the real dataset is a data-only change, not a code change. One row
reproduces the real HC2018 `DAYS_EMPLOYED = 365243` sentinel value (used for
unemployed/pensioner applicants) to exercise that known data quirk. Two rows
are missing a required field to exercise the `INCOMPLETE` ingestion path.

## policy_corpus_personal_loan_v0.1.json

Synthetic Personal Loan policy rulebook (5 clauses) authored for the Sprint 1
single-scheme RAG POC (assumption A-2, user story US-104). Retrieved via the
TF-IDF engine in `src/api/services/retrieval.py`.

## eval/retrieval_eval_set.json

Labelled query -> expected-clause-id pairs used by the US-111 retrieval
accuracy spike (`tests/test_retrieval_accuracy.py`).
