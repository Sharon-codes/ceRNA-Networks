import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Limit PyTorch threads
torch.set_num_threads(1)

# Import dataset definitions
sys.path.append(os.getcwd())
from dataset import TCRDataset, build_global_pool, load_embedding, get_hla_pseudo
from model import MambaTCR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utility Metrics running on: {device}")

# ----------------------------------------------------
# 1. Custom Collators for negative sampling regimes
# ----------------------------------------------------

# (a) Synthetic Mutational Permutations Collator
class SyntheticMutateCollate:
    def __init__(self):
        pass
        
    def __call__(self, batch):
        batch_size = len(batch)
        beta_tensors = []
        alpha_tensors = []
        pephla_tensors = []
        pephla_neg_tensors = []
        
        for item in batch:
            raw_beta = item["raw_cdr3_beta"]
            raw_alpha = item["raw_cdr3_alpha"]
            raw_peptide = item["raw_peptide"]
            raw_hla = item["raw_hla_allele"]
            
            hla_pseudo = get_hla_pseudo(raw_hla)
            pephla_string = raw_peptide + hla_pseudo
            
            emb_beta = load_embedding(raw_beta)
            emb_alpha = load_embedding(raw_alpha)
            emb_pephla = load_embedding(pephla_string)
            
            beta_tensors.append(emb_beta)
            alpha_tensors.append(emb_alpha)
            pephla_tensors.append(emb_pephla)
            
            # Simulate mutation by adding noise to peptide residues
            pep_len = len(raw_peptide)
            emb_neg = emb_pephla.clone()
            num_mutations = max(1, int(pep_len * 0.2))
            positions = random.sample(range(pep_len), min(num_mutations, pep_len))
            for pos in positions:
                emb_neg[pos] = emb_neg[pos] + torch.randn(320) * 0.15
                
            pephla_neg_tensors.append(emb_neg)
            
        beta_batch = torch.stack(beta_tensors)
        alpha_batch = torch.stack(alpha_tensors)
        pephla_batch = torch.stack(pephla_tensors)
        pephla_neg = torch.stack(pephla_neg_tensors)
        
        beta_all = torch.cat([beta_batch, beta_batch], dim=0)
        alpha_all = torch.cat([alpha_batch, alpha_batch], dim=0)
        pephla_all = torch.cat([pephla_batch, pephla_neg], dim=0)
        labels_all = [1.0] * batch_size + [0.0] * batch_size
        
        return {
            "cdr3_beta": beta_all,
            "cdr3_alpha": alpha_all,
            "peptide_plus_hla": pephla_all,
            "label": torch.tensor(labels_all, dtype=torch.float)
        }

# (b) Uniform Random Pool Swapping Collator
class UniformRandomPoolCollate:
    def __init__(self, global_pool):
        self.global_pool = global_pool
        
    def __call__(self, batch):
        batch_size = len(batch)
        beta_tensors = []
        alpha_tensors = []
        pephla_tensors = []
        pephla_neg_tensors = []
        
        for item in batch:
            raw_beta = item["raw_cdr3_beta"]
            raw_alpha = item["raw_cdr3_alpha"]
            raw_peptide = item["raw_peptide"]
            raw_hla = item["raw_hla_allele"]
            
            hla_pseudo = get_hla_pseudo(raw_hla)
            pephla_string = raw_peptide + hla_pseudo
            
            beta_tensors.append(load_embedding(raw_beta))
            alpha_tensors.append(load_embedding(raw_alpha))
            pephla_tensors.append(load_embedding(pephla_string))
            
            # Select a negative peptide uniformly at random from global pool
            found = False
            for _ in range(20):
                g_pep, g_hla = random.choice(self.global_pool)
                if g_pep != raw_peptide:
                    neg_pep = g_pep
                    neg_hla = g_hla
                    found = True
                    break
            if not found:
                neg_pep = "AAAAA"
                neg_hla = "A*02:01"
                
            neg_pephla_string = neg_pep + get_hla_pseudo(neg_hla)
            pephla_neg_tensors.append(load_embedding(neg_pephla_string))
            
        beta_batch = torch.stack(beta_tensors)
        alpha_batch = torch.stack(alpha_tensors)
        pephla_batch = torch.stack(pephla_tensors)
        pephla_neg = torch.stack(pephla_neg_tensors)
        
        beta_all = torch.cat([beta_batch, beta_batch], dim=0)
        alpha_all = torch.cat([alpha_batch, alpha_batch], dim=0)
        pephla_all = torch.cat([pephla_batch, pephla_neg], dim=0)
        labels_all = [1.0] * batch_size + [0.0] * batch_size
        
        return {
            "cdr3_beta": beta_all,
            "cdr3_alpha": alpha_all,
            "peptide_plus_hla": pephla_all,
            "label": torch.tensor(labels_all, dtype=torch.float)
        }

# (c) Natural Decoy Derangement (Algorithm 2) Collator
# Imported from dataset.py (TCRCollate)

# ----------------------------------------------------
# Helper training function
# ----------------------------------------------------
def train_and_eval(collate_fn, train_dataset, test_loader, epochs=5):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    set_seed(42)
    model = MambaTCR(d_model=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            beta = batch["cdr3_beta"].to(device)
            alpha = batch["cdr3_alpha"].to(device)
            pephla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(beta, alpha, pephla)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            beta = batch["cdr3_beta"].to(device)
            alpha = batch["cdr3_alpha"].to(device)
            pephla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(beta, alpha, pephla)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    return roc_auc_score(all_targets, all_probs)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
def main():
    print("Preparing datasets...")
    train_df = pd.read_csv("./Processed/train.csv")
    val_df = pd.read_csv("./Processed/val.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
    
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
    
    train_temp_path = "./Processed/utility_train_temp.csv"
    test_temp_path = "./Processed/utility_test_temp.csv"
    train_df.to_csv(train_temp_path, index=False)
    test_df.to_csv(test_temp_path, index=False)
    
    # Setup test loader using standard collator
    from dataset import TCRCollate
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)
    test_dataset = TCRDataset(test_temp_path)
    test_collate = TCRCollate(test_triplets, test_pool)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_collate)
    
    # 1. Virtual Screening Enrichment Metrics
    print("\n--- Calculating Clinical Enrichment ---")
    baseline_model = MambaTCR(d_model=64).to(device)
    baseline_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        print("Successfully loaded baseline model checkpoint.")
    else:
        print("WARNING: baseline model checkpoint not found.")
        
    baseline_model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            beta = batch["cdr3_beta"].to(device)
            alpha = batch["cdr3_alpha"].to(device)
            pephla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            logits = baseline_model(beta, alpha, pephla)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_probs)
    
    # Sort predictions descending
    sort_idx = np.argsort(-y_pred)
    y_true_sorted = y_true[sort_idx]
    
    # Calculate Precision@top-100
    precision_100 = np.sum(y_true_sorted[:100]) / 100.0
    
    # Base prevalence (fraction of true positives in test set)
    base_prevalence = np.sum(y_true) / len(y_true)
    
    # Enrichment Factor
    ef_100 = precision_100 / base_prevalence
    
    print(f"Precision@top-100: {precision_100:.4f}")
    print(f"Base Prevalence: {base_prevalence:.4f}")
    print(f"Enrichment Factor (EF_100): {ef_100:.4f}")
    
    # 2. Decoy Ablation Study
    print("\n--- Running Decoy Ablation Study ---")
    train_dataset = TCRDataset(train_temp_path)
    train_pool = build_global_pool(train_df)
    train_triplets = set(zip(train_df["cdr3_beta"], train_df["peptide"], train_df["hla_allele"]))
    
    # (a) Synthetic Mutational Permutations
    print("Training on Synthetic Mutational Permutations...")
    collate_a = SyntheticMutateCollate()
    auc_a = train_and_eval(collate_a, train_dataset, test_loader, epochs=5)
    print(f"  Test ROC-AUC: {auc_a:.4f}")
    
    # (b) Uniform Random Pool Swapping
    print("Training on Uniform Random Pool Swapping...")
    collate_b = UniformRandomPoolCollate(train_pool)
    auc_b = train_and_eval(collate_b, train_dataset, test_loader, epochs=5)
    print(f"  Test ROC-AUC: {auc_b:.4f}")
    
    # (c) Natural Decoy Derangement
    print("Training on Natural Decoy Derangement (Algorithm 2)...")
    collate_c = TCRCollate(train_triplets, train_pool)
    auc_c = train_and_eval(collate_c, train_dataset, test_loader, epochs=5)
    print(f"  Test ROC-AUC: {auc_c:.4f}")
    
    # Clean up temp files
    for path in [train_temp_path, test_temp_path]:
        if os.path.exists(path):
            os.remove(path)
            
    # Write report to clinical_utility_and_decoys.txt
    out_path = "clinical_utility_and_decoys.txt"
    with open(out_path, "w") as f:
        f.write("=== Clinical Screening Enrichment & Decoy Ablation Report ===\n\n")
        f.write("1. Virtual Screening Enrichment (Baseline):\n")
        f.write(f"  Precision@top-100: {precision_100:.5f}\n")
        f.write(f"  Base Prevalence: {base_prevalence:.5f}\n")
        f.write(f"  Enrichment Factor (EF_100): {ef_100:.5f}\n\n")
        f.write("2. Negative Decoy Sampling Ablation Matrix:\n")
        f.write("| Negative Sampling Regime | Test ROC-AUC |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| (a) Synthetic Mutational Permutations | {auc_a:.5f} |\n")
        f.write(f"| (b) Uniform Random Pool Swapping | {auc_b:.5f} |\n")
        f.write(f"| (c) Natural Decoy Derangement (Algorithm 2) | {auc_c:.5f} |\n")
    print(f"\nReport successfully logged to {out_path}")

if __name__ == "__main__":
    main()
