# Proxy Leakage Assessment (US-305)

Assumption A-8b excludes `CODE_GENDER` and `DAYS_BIRTH` from the model's
direct inputs, but retained features can still act as **proxies** if they
correlate strongly with a protected attribute. This report quantifies that
risk for the champion LightGBM model's actual retained features (not the
PRD's illustrative `OCCUPATION_TYPE`/`REGION_RATING_CLIENT` example - neither
of those two fields is a model input in this implementation; see
`src/risk_model/config.py`).

Pearson correlation, computed on the full engineered training frame
(246,008 rows) between each retained numeric feature and (a) male gender
as a 0/1 indicator, (b) applicant age in years:

| Feature | r vs. male gender | r vs. age |
|---|---|---|
| `EXT_SOURCE_MEAN` | -0.089 | **0.280** |
| `CREDIT_INCOME_RATIO` | -0.126 | 0.122 |
| `EMPLOYMENT_YEARS` | -0.093 | **0.352** |
| `ANNUITY_INCOME_RATIO` | -0.118 | 0.082 |
| `INCOME_PER_PERSON` | 0.052 | 0.034 |

## Findings

1. **Gender proxy risk is low.** Every retained feature correlates with
   gender at `|r| < 0.13` - materially weaker than `CODE_GENDER`'s own
   direct r=0.054 with the TARGET reported in the PRD. This is consistent
   with the live fairness hard-block (`reports/ml/fairness_thresholds.json`)
   finding the `CODE_GENDER` approval-rate gap (F +1.97pp, M -3.83pp) within
   the 5pp guardrail after the A-8b fix (see `reports/ml/FAIRNESS_SUMMARY.md`).

2. **Age proxy risk is real and material.** `EMPLOYMENT_YEARS` (r=0.352) and
   `EXT_SOURCE_MEAN` (r=0.280) both correlate moderately with age - younger
   applicants mechanically have less employment tenure and (empirically in
   this dataset) lower external bureau scores, both legitimate underwriting
   signals that are *also* age-correlated. This is the direct mechanism
   behind the large age-band fairness gaps already observed live: the
   18-25 segment approves 25pp below the population baseline, the 65+
   segment 12pp above (`reports/ml/fairness_thresholds.json`,
   `hard_block: true` for both).

## Mitigation status

- **Detection (done):** the live fairness hard-block
  (`src/api/services/fairness_check.py`) already catches every application
  in an affected age band and forces escalation (`fairness_alert`) rather
  than auto-deciding - the PRD's guardrail is enforced today, not deferred.
- **Feature-level mitigation (not done, tracked for a future sprint):**
  removing or orthogonalizing `EMPLOYMENT_YEARS`/`EXT_SOURCE_MEAN` would
  reduce AUC (they are the top two SHAP drivers) and requires a fairness-vs-
  accuracy tradeoff decision the PRD reserves for the credit risk committee,
  not an engineering default. Candidate approaches for a follow-up: reweighing
  by age band during training, or an age-conditional post-processing
  threshold adjustment (distinct from the existing conservative/balanced/
  revenue-friendly global thresholds in `THRESHOLD_POLICY.md`).
