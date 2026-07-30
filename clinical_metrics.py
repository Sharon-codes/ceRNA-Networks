import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix
)

from dataset import TCRDataset, TCRCollate, build_global_pool
from model import MambaTCR

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_predictions(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            cdr3_beta = batch["cdr3_beta"].to(device)
            cdr3_alpha = batch["cdr3_alpha"].to(device)
            peptide_plus_hla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            probs = torch.sigmoid(logits)
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds)

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    os.makedirs("./Evaluation", exist_ok=True)
    
    # Load model
    print("Loading model and SWA checkpoint...")
    model = MambaTCR(d_model=64, nhead=8, num_layers=2).to(device)
    checkpoint_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Production SWA checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Load test set
    test_csv = "./Processed/test.csv"
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split not found at {test_csv}")
    test_df = pd.read_csv(test_csv)
    
    # Set seed and prepare loader
    set_seed(42)
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)
    test_dataset = TCRDataset(test_csv)
    test_collate = TCRCollate(test_triplets, test_pool)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_collate)
    
    print("Running evaluation to gather predictions...")
    set_seed(42)
    with SuppressStdout():
        labels, probs = get_predictions(model, test_loader, device)
        
    print(f"Evaluating clinical metrics for {len(labels)} samples...")
    
    # Threshold Optimization via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Clip threshold to valid probability range
    optimal_threshold = np.clip(optimal_threshold, 0.0, 1.0)
    
    print(f"Optimal Decision Threshold (Youden's J): {optimal_threshold:.4f}")
    
    # Binarize predictions
    preds = (probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    
    # Format markdown report
    report_content = f"""# ESM-MambaTCR Advanced Validation & Clinical Metrics Report

This report documents the advanced classification performance of the ESM-MambaTCR architecture under the optimal decision threshold optimized via Youden's J statistic.

## Optimal Decision Threshold
- **Threshold**: `{optimal_threshold:.4f}` (maximizing $\\text{{TPR}} - \\text{{FPR}}$)

## Metrics Summary Table

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Matthews Correlation Coefficient (MCC)** | `{mcc:.4f}` | Balanced statistical rate for binary classification (critical for bioinformatics) |
| **F1-Score** | `{f1:.4f}` | Harmonic mean of precision and recall |
| **Precision (PPV)** | `{precision:.4f}` | Positive Predictive Value |
| **Recall (Sensitivity)** | `{recall:.4f}` | True Positive Rate / sensitivity to binding anchors |
| **Specificity (TNR)** | `{specificity:.4f}` | True Negative Rate / rejection of decoy pairings |

## Confusion Matrix Details
- **True Negatives (TN)**: `{tn}`
- **False Positives (FP)**: `{fp}`
- **False Negatives (FN)**: `{fn}`
- **True Positives (TP)**: `{tp}`
"""

    # Save to file
    out_path = "./Evaluation/clinical_metrics_report.md"
    with open(out_path, "w") as f:
        f.write(report_content)
        
    print(f"Clinical metrics report successfully saved to {out_path}")
    print("\nMetrics summary:")
    print(f"MCC:         {mcc:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")

if __name__ == "__main__":
    main()
