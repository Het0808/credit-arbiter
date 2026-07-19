"""Auxiliary-table feature engineering for ML hardening (US-201).

Aggregates the Home Credit *bureau* and *previous_application* tables to one
row per applicant (SK_ID_CURR) and caches the result as a parquet file so both
training and single-applicant inference use identical features. These external
credit-history aggregates are the highest-ROI additions for lifting AUC beyond
what the application table alone supports, and contain no protected attributes.

Only the columns needed for each aggregate are read from the (large) source
CSVs to keep memory bounded.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HC_DIR = PROJECT_ROOT / "data" / "home_credit_data" / "home-credit-default-risk"
BUREAU_CSV = HC_DIR / "bureau.csv"
PREV_APP_CSV = HC_DIR / "previous_application.csv"
INSTALLMENTS_CSV = HC_DIR / "installments_payments.csv"
POS_CASH_CSV = HC_DIR / "POS_CASH_balance.csv"
CREDIT_CARD_CSV = HC_DIR / "credit_card_balance.csv"
AUX_CACHE = PROJECT_ROOT / "data" / "home_credit_data" / "aux_features.parquet"

ID = "SK_ID_CURR"


def _aggregate_bureau() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "CREDIT_ACTIVE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
            "CREDIT_DAY_OVERDUE", "DAYS_CREDIT"]
    b = pd.read_csv(BUREAU_CSV, usecols=cols)
    b["_active"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    grouped = b.groupby(ID)
    out = pd.DataFrame({
        "BUREAU_LOAN_COUNT": grouped.size(),
        "BUREAU_ACTIVE_COUNT": grouped["_active"].sum(),
        "BUREAU_AMT_CREDIT_SUM_MEAN": grouped["AMT_CREDIT_SUM"].mean(),
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": grouped["AMT_CREDIT_SUM_DEBT"].sum(),
        "BUREAU_CREDIT_DAY_OVERDUE_MEAN": grouped["CREDIT_DAY_OVERDUE"].mean(),
        "BUREAU_CREDIT_DAY_OVERDUE_MAX": grouped["CREDIT_DAY_OVERDUE"].max(),
        "BUREAU_DAYS_CREDIT_MEAN": grouped["DAYS_CREDIT"].mean(),
    })
    return out


def _aggregate_previous_application() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION"]
    p = pd.read_csv(PREV_APP_CSV, usecols=cols)
    p["_approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["_refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    grouped = p.groupby(ID)
    out = pd.DataFrame({
        "PREV_APP_COUNT": grouped.size(),
        "PREV_APP_APPROVED_RATIO": grouped["_approved"].mean(),
        "PREV_APP_REFUSED_COUNT": grouped["_refused"].sum(),
        "PREV_AMT_APPLICATION_MEAN": grouped["AMT_APPLICATION"].mean(),
        "PREV_AMT_CREDIT_MEAN": grouped["AMT_CREDIT"].mean(),
        "PREV_DAYS_DECISION_MAX": grouped["DAYS_DECISION"].max(),
    })
    return out


def _aggregate_installments() -> pd.DataFrame:
    """Payment-behaviour aggregates - the strongest external delinquency signal."""
    cols = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    i = pd.read_csv(INSTALLMENTS_CSV, usecols=cols)
    # Days past due on each instalment (payment date - due date; positive => late).
    i["_dpd"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["_late"] = (i["_dpd"] > 0).astype(int)
    i["_pay_ratio"] = i["AMT_PAYMENT"] / i["AMT_INSTALMENT"].replace(0, np.nan)
    grouped = i.groupby(ID)
    out = pd.DataFrame({
        "INST_COUNT": grouped.size(),
        "INST_DPD_MEAN": grouped["_dpd"].mean(),
        "INST_DPD_MAX": grouped["_dpd"].max(),
        "INST_LATE_COUNT": grouped["_late"].sum(),
        "INST_PAYMENT_RATIO_MEAN": grouped["_pay_ratio"].mean(),
    })
    return out


def _aggregate_pos_cash() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF"]
    p = pd.read_csv(POS_CASH_CSV, usecols=cols)
    grouped = p.groupby(ID)
    out = pd.DataFrame({
        "POS_COUNT": grouped.size(),
        "POS_SK_DPD_MEAN": grouped["SK_DPD"].mean(),
        "POS_SK_DPD_MAX": grouped["SK_DPD"].max(),
    })
    return out


def _aggregate_credit_card() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"]
    c = pd.read_csv(CREDIT_CARD_CSV, usecols=cols)
    c["_util"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    grouped = c.groupby(ID)
    out = pd.DataFrame({
        "CC_COUNT": grouped.size(),
        "CC_AMT_BALANCE_MEAN": grouped["AMT_BALANCE"].mean(),
        "CC_UTILIZATION_MEAN": grouped["_util"].mean(),
        "CC_SK_DPD_MAX": grouped["SK_DPD"].max(),
    })
    return out


def build_aux_features(force: bool = False) -> pd.DataFrame:
    """Build (or load cached) per-applicant auxiliary aggregate features."""
    if AUX_CACHE.exists() and not force:
        return pd.read_parquet(AUX_CACHE)

    aux = _aggregate_bureau()
    for part in (_aggregate_previous_application(), _aggregate_installments(),
                 _aggregate_pos_cash(), _aggregate_credit_card()):
        aux = aux.join(part, how="outer")
    aux.index.name = ID
    aux = aux.reset_index()

    AUX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    aux.to_parquet(AUX_CACHE, index=False)
    return aux


def merge_aux_features(df: pd.DataFrame, aux: pd.DataFrame = None) -> pd.DataFrame:
    """Left-join auxiliary aggregates onto an application frame by SK_ID_CURR.

    Applicants with no bureau/previous history get 0-filled aggregates, which is
    the correct signal ("no external credit history") rather than a missing value.
    """
    if aux is None:
        aux = build_aux_features()
    aux_cols = [c for c in aux.columns if c != ID]
    # Idempotent: if aux columns are already merged, do nothing.
    if all(c in df.columns for c in aux_cols):
        return df
    if ID not in df.columns:
        return df
    merged = df.merge(aux, on=ID, how="left")
    merged[aux_cols] = merged[aux_cols].fillna(0.0)
    return merged


if __name__ == "__main__":
    aux = build_aux_features(force=True)
    print(f"Built aux features: {aux.shape[0]} applicants x {aux.shape[1] - 1} features")
    print(f"Cached to: {AUX_CACHE}")
    print(aux.describe().T[["mean", "min", "max"]].round(2))
