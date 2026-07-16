# XGBoost Model Performance Report

This report presents a detailed evaluation of the XGBoost challenger model (v1) trained on the Home Credit Default Risk dataset.

## Model Description
The model uses `XGBClassifier` integrated into a standard scikit-learn preprocessing pipeline. The model includes standard median imputation and scaling for numerical features, and most frequent imputation followed by one-hot encoding for categorical features. 

To address the severe class imbalance in the training set (~8% default rate), `scale_pos_weight` is set dynamically based on the ratio of negative to positive labels in the training split. Early stopping is applied using a 10% stratified validation split from the training dataset.

- **Algorithm**: `XGBClassifier`
- **Objective**: `binary:logistic`
- **Evaluation Metric**: `auc` (Area Under ROC Curve)
- **Early Stopping Rounds**: 50
- **Scale Pos Weight**: 12.6090 (calculated dynamically)
- **Training Timestamp**: 2026-07-12 19:48:43
- **Model Version**: v1

## Hyperparameters
The following hyperparameters were configured for this training run:
- `n_estimators`: 500 (with early stopping)
- `learning_rate`: 0.05
- `max_depth`: 6
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 5
- `gamma`: 1
- `random_state`: 42
- `tree_method`: "hist"

## Evaluation Metrics

The model was evaluated on the unseen test split (20% partition of the dataset, 61,503 records):

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.7655** |
| **Accuracy** | **0.7267** |
| **Precision** | **0.1783** |
| **Recall** | **0.6614** |
| **F1-Score** | **0.2809** |
| **Average Precision (PR-AUC)** | **0.2534** |

### Confusion Matrix
- **True Negatives (TN)**: 41,408 (Creditworthy applications approved)
- **False Positives (FP)**: 15,130 (Creditworthy applications blocked)
- **False Negatives (FN)**: 1,681 (Defaulting applications approved - toxic leakage)
- **True Positives (TP)**: 3,284 (Defaulting applications blocked)

## Visualizations
- **ROC Curve**: ![ROC Curve](plots/xgboost_roc_curve.png)
- **Precision-Recall Curve**: ![Precision-Recall Curve](plots/xgboost_precision_recall_curve.png)
- **Confusion Matrix Heatmap**: ![Confusion Matrix Heatmap](plots/xgboost_confusion_matrix.png)

## Operational Metrics
- **Feature Count**: 32 features
- **Training Time**: 2.61 seconds
- **Best Iteration**: 302 estimators (early stopped before 500 estimators)

## Advantages of XGBoost v1
1. **Regularization against Overfitting**: XGBoost incorporates L1 and L2 regularization directly into its objective function, preventing the deep tree overfitting observed in other ensemble architectures.
2. **Early Stopping Protection**: Training halts dynamically when validation AUC plateaus, ensuring optimal training epoch selection and saving compute budget.
3. **Imbalance Optimization**: Using dynamic `scale_pos_weight` ensures high sensitivity (Recall) to default events.
4. **Hist Tree Method Speed**: Utilizing the histogram-based split finder significantly speeds up training on large tabular datasets (over 240,000 training rows).

## Limitations of XGBoost v1
1. **Low Precision**: As with all models on this dataset, high sensitivity (Recall) to defaults leads to a high rate of False Positives (~27% false alarm rate), causing good borrowers to be rejected.
2. **Memory Footprint**: Fits the entire training set in RAM, which can grow significantly with larger feature vectors.
3. **Inference Latency**: Evaluating hundreds of decision trees is slower than a simple Logistic Regression, though still well within web API requirements (~ms latency).
