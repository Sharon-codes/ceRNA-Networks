# ESM-MambaTCR Advanced Validation & Clinical Metrics Report

This report documents the advanced classification performance of the ESM-MambaTCR architecture under the optimal decision threshold optimized via Youden's J statistic.

## Optimal Decision Threshold
- **Threshold**: `0.0766` (maximizing $\text{TPR} - \text{FPR}$)

## Metrics Summary Table

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Matthews Correlation Coefficient (MCC)** | `0.2977` | Balanced statistical rate for binary classification (critical for bioinformatics) |
| **F1-Score** | `0.6734` | Harmonic mean of precision and recall |
| **Precision (PPV)** | `0.6263` | Positive Predictive Value |
| **Recall (Sensitivity)** | `0.7282` | True Positive Rate / sensitivity to binding anchors |
| **Specificity (TNR)** | `0.5656` | True Negative Rate / rejection of decoy pairings |

## Confusion Matrix Details
- **True Negatives (TN)**: `1363`
- **False Positives (FP)**: `1047`
- **False Negatives (FN)**: `655`
- **True Positives (TP)**: `1755`
