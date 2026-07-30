import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

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

def evaluate_ablation(model, loader, device, mask_mode=None):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            cdr3_beta = batch["cdr3_beta"].to(device)
            cdr3_alpha = batch["cdr3_alpha"].to(device)
            peptide_plus_hla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            if mask_mode == "tcr":
                # Mask TCR (both beta and alpha)
                cdr3_beta = cdr3_beta * 0.0
                cdr3_alpha = cdr3_alpha * 0.0
            elif mask_mode == "peptide":
                # Mask Peptide-HLA
                peptide_plus_hla = peptide_plus_hla * 0.0
                
            logits = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            probs = torch.sigmoid(logits)
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())
            
    return roc_auc_score(all_targets, all_preds)

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
    
    # Run Baseline
    print("Running Baseline evaluation...")
    set_seed(42)
    with SuppressStdout():
        baseline_auc = evaluate_ablation(model, test_loader, device, mask_mode=None)
    print(f"Baseline Test ROC-AUC:      {baseline_auc:.4f}")
    
    # Run TCR-Only (Mask Peptide)
    print("Running TCR-Only (Mask Peptide) evaluation...")
    set_seed(42)
    with SuppressStdout():
        tcr_only_auc = evaluate_ablation(model, test_loader, device, mask_mode="peptide")
    print(f"TCR-Only Test ROC-AUC:      {tcr_only_auc:.4f}")
    
    # Run Peptide-Only (Mask TCR)
    print("Running Peptide-Only (Mask TCR) evaluation...")
    set_seed(42)
    with SuppressStdout():
        pep_only_auc = evaluate_ablation(model, test_loader, device, mask_mode="tcr")
    print(f"Peptide-Only Test ROC-AUC:  {pep_only_auc:.4f}")
    
    # Write to log
    out_path = "./Evaluation/unimodal_ablation_results.txt"
    with open(out_path, "w") as f:
        f.write("ESM-MambaTCR Unimodal Bias Ablation Results\n")
        f.write("============================================\n")
        f.write(f"Baseline Test ROC-AUC:      {baseline_auc:.4f}\n")
        f.write(f"TCR-Only (Masked Peptide):  {tcr_only_auc:.4f}\n")
        f.write(f"Peptide-Only (Masked TCR):  {pep_only_auc:.4f}\n")
        
    print(f"Ablation results successfully logged to {out_path}")

if __name__ == "__main__":
    main()
