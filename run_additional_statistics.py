import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar

# Add workspace to path
sys.path.append(os.getcwd())
from dataset import TCRDataset, TCRCollate, build_global_pool
from model import MambaTCR
from run_zeroshot_comparators import EPIC_TRACE, train_model, evaluate_model, set_seed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Additional Statistics running on: {device}")
    
    # 1. Prepare LODO dataset splits
    print("Preparing LODO dataset...")
    train_df = pd.read_csv("./Processed/train.csv")
    val_df = pd.read_csv("./Processed/val.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
    
    # Write temp files
    train_temp_path = "train_temp_stats.csv"
    test_temp_path = "test_temp_stats.csv"
    train_df.to_csv(train_temp_path, index=False)
    test_df.to_csv(test_temp_path, index=False)
    
    train_dataset = TCRDataset(train_temp_path)
    test_dataset = TCRDataset(test_temp_path)
    
    # Correct triplet sets and global pools
    train_triplets = set(zip(train_df["cdr3_beta"], train_df["peptide"], train_df["hla_allele"]))
    train_pool = build_global_pool(train_df)
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)
    
    train_collate = TCRCollate(train_triplets, train_pool)
    test_collate = TCRCollate(test_triplets, test_pool)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=train_collate)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_collate)
    
    # 2. Evaluate our pre-trained Direct Concatenation model
    print("\nEvaluating Direct Concatenation (Baseline)...")
    baseline_model = MambaTCR(d_model=64).to(device)
    baseline_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        print("Successfully loaded baseline model checkpoint.")
    else:
        raise FileNotFoundError(f"Missing baseline checkpoint at {baseline_path}")
        
    base_auc, base_mcc, base_pred, y_true = evaluate_model(baseline_model, test_loader)
    print(f"Baseline AUC: {base_auc:.5f} | Youden MCC: {base_mcc:.5f}")
    
    # 3. Calculate 95% Confidence Interval for Precision@top-100
    n_bootstrap = 1000
    n_samples = len(y_true)
    np.random.seed(42)
    precisions_100 = []
    
    print("\nBootstrapping Precision@top-100 threshold...")
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_true = y_true[indices]
        sample_pred = base_pred[indices]
        
        # Sort and select top 100
        top100_idx = np.argsort(sample_pred)[::-1][:100]
        precisions_100.append(np.mean(sample_true[top100_idx]))
        
    p100_lower = np.percentile(precisions_100, 2.5)
    p100_upper = np.percentile(precisions_100, 97.5)
    p100_point = np.mean(y_true[np.argsort(base_pred)[::-1][:100]])
    print(f"Precision@top-100: {p100_point:.5f} (95% CI: [{p100_lower:.5f}, {p100_upper:.5f}])")
    
    # 4. Train and evaluate EPIC-TRACE
    print("\n--- Training EPIC-TRACE ---")
    set_seed(42)
    epic_model = EPIC_TRACE().to(device)
    train_model(epic_model, train_loader, epochs=15)
    epic_auc, epic_mcc, epic_pred, _ = evaluate_model(epic_model, test_loader)
    print(f"EPIC-TRACE AUC: {epic_auc:.5f} | Youden MCC: {epic_mcc:.5f}")
    
    # Find EPIC-TRACE's Youden-optimal threshold
    best_mcc = -1.0
    best_epic_thresh = 0.5
    for t in np.arange(0.01, 1.00, 0.01):
        preds = (epic_pred >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_epic_thresh = t
            
    print(f"EPIC-TRACE Optimal Threshold: {best_epic_thresh:.3f}")
    
    # 5. Run McNemar's Test
    base_binary = (base_pred >= 0.076).astype(int)
    epic_binary = (epic_pred >= best_epic_thresh).astype(int)
    
    base_correct = (base_binary == y_true)
    epic_correct = (epic_binary == y_true)
    
    # Contingency Table
    a = np.sum(base_correct & epic_correct)
    b = np.sum(base_correct & ~epic_correct)
    c = np.sum(~base_correct & epic_correct)
    d = np.sum(~base_correct & ~epic_correct)
    
    table = [[a, b], [c, d]]
    print("\nMcNemar Contingency Table:")
    print(f"                     EPIC Correct   EPIC Incorrect")
    print(f"Baseline Correct        {a:<14d} {b:<14d}")
    print(f"Baseline Incorrect      {c:<14d} {d:<14d}")
    
    mcnemar_res = mcnemar(table, exact=True)
    p_value = mcnemar_res.pvalue
    print(f"McNemar's test exact p-value: {p_value:.4e}")
    
    # Clean up temp files
    for path in [train_temp_path, test_temp_path]:
        if os.path.exists(path):
            os.remove(path)
            
    # Save to report file
    os.makedirs("Evaluation", exist_ok=True)
    report_path = "./Evaluation/additional_statistics_report.txt"
    with open(report_path, "w") as f:
        f.write("Additional Statistics & Baselines (Concerns 2 & 5)\n")
        f.write("==================================================\n")
        f.write(f"Precision@top-100 on LODO Test Set: {p100_point:.5f}\n")
        f.write(f"Precision@top-100 95% CI Bounds:    [{p100_lower:.5f}, {p100_upper:.5f}]\n\n")
        f.write(f"McNemar's Test comparing Baseline (tau=0.076) vs. EPIC-TRACE (tau={best_epic_thresh:.3f}):\n")
        f.write(f"  Contingency Table:\n")
        f.write(f"    Both Correct:      {a}\n")
        f.write(f"    Baseline only:     {b}\n")
        f.write(f"    EPIC-TRACE only:   {c}\n")
        f.write(f"    Both Incorrect:    {d}\n")
        f.write(f"  McNemar p-value:     {p_value:.5e}\n")
        
    print(f"\nAdditional statistics logged to {report_path}")

if __name__ == "__main__":
    main()
