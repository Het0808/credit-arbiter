"""Compiles reports/ml/fairness_report.csv (from fairness.py) into a small
per-segment lookup table (reports/ml/fairness_thresholds.json) that the live
API can consult in constant time at assessment time (FR-7 / US-304), without
re-running the full batch fairness analysis on every request.

Re-run this after every fairness.py run (i.e. after every retrain):

    python -m src.risk_model.fairness_thresholds
"""

import json
from pathlib import Path

import pandas as pd


def compute_fairness_thresholds() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "reports" / "ml"
    csv_path = reports_dir / "fairness_report.csv"
    out_path = reports_dir / "fairness_thresholds.json"

    df = pd.read_csv(csv_path)

    segments: dict = {}
    overall_by_attribute: dict = {}
    for attribute, group in df.groupby("attribute"):
        total_size = group["sample_size"].sum()
        overall_rate = float((group["approval_rate"] * group["sample_size"]).sum() / total_size)
        overall_by_attribute[attribute] = overall_rate

        attribute_segments = {}
        for _, row in group.iterrows():
            delta_pp = round((float(row["approval_rate"]) - overall_rate) * 100, 2)
            attribute_segments[row["subgroup"]] = {
                "approval_rate": float(row["approval_rate"]),
                "sample_size": int(row["sample_size"]),
                "delta_pp": delta_pp,
                "hard_block": abs(delta_pp) > 5.0,
            }
        segments[attribute] = attribute_segments

    payload = {
        "hard_block_threshold_pp": 5.0,
        "overall_approval_rate_by_attribute": overall_by_attribute,
        "segments": segments,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Fairness thresholds written to: {out_path}")
    return out_path


if __name__ == "__main__":
    compute_fairness_thresholds()
