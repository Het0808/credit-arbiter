"""
Extract hyperparameter tuning results from the GridSearchCV CV CSV artifacts.
Compiles all evaluated hyperparameter combinations and their ROC-AUC scores,
prints a table, and saves the output to reports/ml/grid_search_all_results.csv.
"""

import glob
from pathlib import Path
import pandas as pd


def extract_results() -> None:
    """
    Search for cv_results.csv in the mlruns folder, load, clean, and export.
    """
    print("Searching for cv_results.csv files in local MLflow artifacts...")
    project_root = Path(__file__).resolve().parents[2]
    mlruns_dir = project_root / "mlruns"
    reports_dir = project_root / "reports" / "ml"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate all cv_results.csv paths under mlruns/
    search_pattern = str(mlruns_dir / "**" / "cv_results.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print("No cv_results.csv files found. Please make sure GridSearchCV ran successfully.")
        return
        
    print(f"Found {len(csv_files)} cv_results.csv file(s). Loading the latest one...")
    
    # Sort files by modification time to get the latest run
    csv_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    latest_csv = csv_files[0]
    print(f"Loading results from: {latest_csv}")
    
    # Load into pandas
    df = pd.read_csv(latest_csv)
    
    # Target columns
    param_cols = [c for c in df.columns if c.startswith("param_")]
    score_col = "mean_test_score"
    
    if score_col not in df.columns:
        print(f"Error: Could not locate score column '{score_col}' in the CSV.")
        return
        
    # Get columns we want
    lr_col = [c for c in param_cols if "learning_rate" in c]
    depth_col = [c for c in param_cols if "max_depth" in c]
    est_col = [c for c in param_cols if "n_estimators" in c]
    
    lr_name = lr_col[0] if lr_col else None
    depth_name = depth_col[0] if depth_col else None
    est_name = est_col[0] if est_col else None
    
    if not (lr_name and depth_name and est_name):
        print("Could not locate parameter columns. Available param columns:")
        print(param_cols)
        return
        
    df_filtered = df[[lr_name, depth_name, est_name, score_col]].copy()
    
    # Rename for readability
    df_filtered = df_filtered.rename(columns={
        lr_name: "learning_rate",
        depth_name: "max_depth",
        est_name: "n_estimators",
        score_col: "cv_roc_auc"
    })
    
    # Convert data types safely
    df_filtered["learning_rate"] = df_filtered["learning_rate"].astype(float)
    df_filtered["max_depth"] = df_filtered["max_depth"].astype(int)
    df_filtered["n_estimators"] = df_filtered["n_estimators"].astype(int)
    df_filtered["cv_roc_auc"] = df_filtered["cv_roc_auc"].astype(float)
    
    # Sort by ROC-AUC descending
    df_filtered = df_filtered.sort_values("cv_roc_auc", ascending=False).drop_duplicates()
    
    # Save to CSV
    csv_path = reports_dir / "grid_search_all_results.csv"
    df_filtered.to_csv(csv_path, index=False)
    print(f"Exported all {len(df_filtered)} tuning iterations to: {csv_path}")
    
    # Print the top 30 combinations
    print("\nTop 30 Hyperparameter Combinations by CV ROC-AUC:")
    print(df_filtered.head(30).to_string(index=False))


if __name__ == "__main__":
    extract_results()
