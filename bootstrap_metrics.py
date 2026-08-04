import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

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
    
    print("Running initial evaluation to gather predictions...")
    set_seed(42)
    with SuppressStdout():
        labels, probs = get_predictions(model, test_loader, device)
        
    n_samples = len(labels)
    print(f"Gathered predictions for {n_samples} test samples.")
    
    # Empirical bootstrapping
    n_bootstrap = 1000
    np.random.seed(42)
    
    bootstrapped_roc_aucs = []
    bootstrapped_pr_aucs = []
    
    print(f"Running {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_labels = labels[indices]
        sample_probs = probs[indices]
        
        # Calculate ROC-AUC
        roc_auc = roc_auc_score(sample_labels, sample_probs)
        bootstrapped_roc_aucs.append(roc_auc)
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(sample_labels, sample_probs)
        pr_auc = auc(recall, precision)
        bootstrapped_pr_aucs.append(pr_auc)
        
    # Calculate 95% Confidence Intervals
    roc_lower = np.percentile(bootstrapped_roc_aucs, 2.5)
    roc_upper = np.percentile(bootstrapped_roc_aucs, 97.5)
    
    pr_lower = np.percentile(bootstrapped_pr_aucs, 2.5)
    pr_upper = np.percentile(bootstrapped_pr_aucs, 97.5)
    
    # Calculate mean metrics
    mean_roc = np.mean(bootstrapped_roc_aucs)
    mean_pr = np.mean(bootstrapped_pr_aucs)
    
    # Format results
    roc_ci_str = f"ROC-AUC: {mean_roc:.3f} [95% CI: {roc_lower:.3f} - {roc_upper:.3f}]"
    pr_ci_str = f"PR-AUC:  {mean_pr:.3f} [95% CI: {pr_lower:.3f} - {pr_upper:.3f}]"
    
    print("\nBootstrapped Results:")
    print("=====================")
    print(roc_ci_str)
    print(pr_ci_str)
    
    # Save to file
    out_path = "./Evaluation/bootstrap_metrics_results.txt"
    with open(out_path, "w") as f:
        f.write("ESM-MambaTCR Empirical Bootstrapping & 95% Confidence Intervals\n")
        f.write("===============================================================\n")
        f.write(roc_ci_str + "\n")
        f.write(pr_ci_str + "\n")
        
    print(f"\nBootstrap results successfully logged to {out_path}")

if __name__ == "__main__":
    main()
