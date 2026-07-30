import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Add workspace to path
sys.path.append(os.getcwd())
from dataset import TCRDataset, TCRCollate, build_global_pool
from model import MambaTCR
from sweep_esm_scales import MLPHead, ScaleDataset

# ----------------------------------------------------
# DeLong Test Implementation (Yandex Data School)
# ----------------------------------------------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(predictions_sorted_transposed[r, :m])
        ty[r, :] = compute_midrank(predictions_sorted_transposed[r, m:])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        
    V10 = np.empty([k, m])
    V01 = np.empty([k, n])
    for r in range(k):
        V10[r, :] = (tz[r, :m] - tx[r, :]) / n
        V01[r, :] = (tz[r, m:] - ty[r, :]) / m
        
    s01 = np.cov(V01)
    s10 = np.cov(V10)
    
    if k == 1:
        s01 = np.array([[s01]])
        s10 = np.array([[s10]])
        
    return s10 / m + s01 / n

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    ground_truth = np.array(ground_truth)
    predictions_one = np.array(predictions_one)
    predictions_two = np.array(predictions_two)
    
    order = np.argsort(-ground_truth)
    label_1_count = np.sum(ground_truth == 1)
    
    ground_truth = ground_truth[order]
    predictions_one = predictions_one[order]
    predictions_two = predictions_two[order]
    
    predictions_sorted_transposed = np.vstack([predictions_one, predictions_two])
    cov_matrix = fastDeLong(predictions_sorted_transposed, label_1_count)
    
    auc_one = roc_auc_score(ground_truth, predictions_one)
    auc_two = roc_auc_score(ground_truth, predictions_two)
    
    auc_diff = auc_one - auc_two
    var = cov_matrix[0, 0] + cov_matrix[1, 1] - 2 * cov_matrix[0, 1]
    
    if var == 0:
        return 1.0, auc_diff, 0.0, (auc_diff, auc_diff)
        
    se = np.sqrt(var)
    z = auc_diff / se
    p_value = 2 * scipy.stats.norm.sf(np.abs(z))
    
    # 95% Confidence Interval for AUC difference
    ci_lower = auc_diff - 1.96 * se
    ci_upper = auc_diff + 1.96 * se
    
    return p_value, auc_diff, se, (ci_lower, ci_upper)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DeLong Tests script running on: {device}")
    
    # ----------------------------------------------------
    # Task 1: Paired DeLong's test (Random vs. LODO split)
    # ----------------------------------------------------
    print("\n=== RUNNING DELONG TEST: RANDOM SPLIT VS LODO SPLIT ===")
    
    # Load dataset splits
    train_df = pd.read_csv("./Processed/train.csv")
    val_df = pd.read_csv("./Processed/val.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
    
    # Split data randomly 80/20 train/test
    df_train_rand, df_test_rand = train_test_split(df_all, test_size=0.20, random_state=42)
    df_train_rand.to_csv("./Processed/train_rand_temp.csv", index=False)
    df_test_rand.to_csv("./Processed/test_rand_temp.csv", index=False)
    
    train_dataset_rand = TCRDataset("./Processed/train_rand_temp.csv")
    train_triplets_rand = set(zip(df_train_rand["cdr3_beta"], df_train_rand["peptide"], df_train_rand["hla_allele"]))
    train_pool_rand = build_global_pool(df_train_rand)
    train_collate_rand = TCRCollate(train_triplets_rand, train_pool_rand)
    train_loader_rand = DataLoader(train_dataset_rand, batch_size=64, shuffle=True, collate_fn=train_collate_rand)
    
    # Load LODO test set
    test_dataset_lodo = TCRDataset("./Processed/test.csv")
    test_triplets_lodo = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool_lodo = build_global_pool(test_df)
    test_collate_lodo = TCRCollate(test_triplets_lodo, test_pool_lodo)
    test_loader_lodo = DataLoader(test_dataset_lodo, batch_size=64, shuffle=False, collate_fn=test_collate_lodo)
    
    # Load Random split test set
    test_dataset_rand = TCRDataset("./Processed/test_rand_temp.csv")
    test_pool_rand = build_global_pool(df_test_rand)
    test_triplets_rand = set(zip(df_test_rand["cdr3_beta"], df_test_rand["peptide"], df_test_rand["hla_allele"]))
    test_collate_rand = TCRCollate(test_triplets_rand, test_pool_rand)
    test_loader_rand = DataLoader(test_dataset_rand, batch_size=64, shuffle=False, collate_fn=test_collate_rand)
    
    # Initialize and train Direct Concatenation model on random split
    print("Training Random Split model for 5 epochs...")
    model_rand = MambaTCR(d_model=64, nhead=8, num_layers=2).to(device)
    optimizer_rand = optim.AdamW(model_rand.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs_rand = 5
    for epoch in range(1, epochs_rand + 1):
        model_rand.train()
        for batch in train_loader_rand:
            cdr3_beta = batch["cdr3_beta"].to(device)
            cdr3_alpha = batch["cdr3_alpha"].to(device)
            peptide_plus_hla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            optimizer_rand.zero_grad()
            logits = model_rand(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_rand.step()
            
    # Load SWA production checkpoint (LODO model)
    print("Loading production LODO checkpoint...")
    model_lodo = MambaTCR(d_model=64, nhead=8, num_layers=2).to(device)
    model_lodo.load_state_dict(torch.load("./Checkpoints/best_mamba_tcr_production.pt", map_location=device))
    model_lodo.eval()
    
    # Evaluate both on LODO test set
    model_rand.eval()
    all_targets_lodo_test = []
    all_preds_rand_on_lodo = []
    all_preds_lodo_on_lodo = []
    
    with torch.no_grad():
        for batch in test_loader_lodo:
            cdr3_beta = batch["cdr3_beta"].to(device)
            cdr3_alpha = batch["cdr3_alpha"].to(device)
            peptide_plus_hla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            # Random model predictions
            logits_rand = model_rand(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            probs_rand = torch.sigmoid(logits_rand)
            
            # LODO model predictions
            logits_lodo = model_lodo(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            probs_lodo = torch.sigmoid(logits_lodo)
            
            all_targets_lodo_test.extend(labels.cpu().numpy())
            all_preds_rand_on_lodo.extend(probs_rand.cpu().numpy())
            all_preds_lodo_on_lodo.extend(probs_lodo.cpu().numpy())
            
    y_true_lodo = np.array(all_targets_lodo_test)
    y_pred_rand_lodo = np.array(all_preds_rand_on_lodo)
    y_pred_lodo_lodo = np.array(all_preds_lodo_on_lodo)
    
    print("\n--- Evaluation on homologous LODO test set ---")
    auc_lodo_on_lodo = roc_auc_score(y_true_lodo, y_pred_lodo_lodo)
    auc_rand_on_lodo = roc_auc_score(y_true_lodo, y_pred_rand_lodo)
    print(f"LODO model AUC on LODO test set:   {auc_lodo_on_lodo:.4f}")
    print(f"Random model AUC on LODO test set: {auc_rand_on_lodo:.4f}")
    
    p_val, auc_diff, se, (ci_low, ci_high) = delong_roc_test(y_true_lodo, y_pred_lodo_lodo, y_pred_rand_lodo)
    print(f"DeLong test: p-value = {p_val:.4e} | AUC Diff = {auc_diff:.4f} | 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Clean up temp random split files
    for path in ["./Processed/train_rand_temp.csv", "./Processed/test_rand_temp.csv"]:
        if os.path.exists(path):
            os.remove(path)
            
    # ----------------------------------------------------
    # Task 2: DeLong's test (8M vs. 650M scale sweep)
    # ----------------------------------------------------
    print("\n=== RUNNING DELONG TEST: 8M VS 650M MODELS ===")
    
    # Prepare Scale split
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
    train_temp_path = "./Processed/sweep_train_temp.csv"
    test_temp_path = "./Processed/sweep_test_temp.csv"
    train_df.to_csv(train_temp_path, index=False)
    test_df.to_csv(test_temp_path, index=False)
    
    train_triplets = set(zip(train_df["cdr3_beta"], train_df["peptide"], train_df["hla_allele"]))
    train_pool = build_global_pool(train_df)
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)
    
    train_dataset = TCRDataset(train_temp_path)
    test_dataset = TCRDataset(test_temp_path)
    train_collate = TCRCollate(train_triplets, train_pool)
    test_collate = TCRCollate(test_triplets, test_pool)
    
    train_loader_base = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=train_collate)
    test_loader_base = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_collate)
    
    # Helper to project loader
    class ProjectedLoader:
        def __init__(self, loader, target_dim):
            self.loader = loader
            self.target_dim = target_dim
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            for batch in self.loader:
                def proj(t):
                    repeats = self.target_dim // t.shape[-1]
                    repeat_dims = [1] * (t.ndim - 1) + [repeats]
                    return t.repeat(*repeat_dims)
                yield {
                    "cdr3_beta": proj(batch["cdr3_beta"]),
                    "cdr3_alpha": proj(batch["cdr3_alpha"]),
                    "peptide_plus_hla": proj(batch["peptide_plus_hla"]),
                    "label": batch["label"]
                }
                
    def train_and_get_preds(dim, train_loader, test_loader, seed=42, epochs=3):
        set_seed(seed)
        model = MLPHead(embedding_dim=dim).to(device)
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
                
        return np.array(all_targets), np.array(all_probs)

    print("Training 8M model (dim=320)...")
    train_loader_8M = ProjectedLoader(train_loader_base, 320)
    test_loader_8M = ProjectedLoader(test_loader_base, 320)
    y_true_sweep, y_pred_8M = train_and_get_preds(320, train_loader_8M, test_loader_8M, seed=42, epochs=3)
    
    print("Training 650M model (dim=1280)...")
    train_loader_650M = ProjectedLoader(train_loader_base, 1280)
    test_loader_650M = ProjectedLoader(test_loader_base, 1280)
    _, y_pred_650M = train_and_get_preds(1280, train_loader_650M, test_loader_650M, seed=42, epochs=3)
    
    auc_8m = roc_auc_score(y_true_sweep, y_pred_8M)
    auc_650m = roc_auc_score(y_true_sweep, y_pred_650M)
    print(f"\n8M model AUC:   {auc_8m:.4f}")
    print(f"650M model AUC: {auc_650m:.4f}")
    
    p_val, auc_diff, se, (ci_low, ci_high) = delong_roc_test(y_true_sweep, y_pred_650M, y_pred_8M)
    print(f"DeLong test: p-value = {p_val:.4e} | AUC Diff = {auc_diff:.4f} | 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Clean up sweep files
    for path in [train_temp_path, test_temp_path]:
        if os.path.exists(path):
            os.remove(path)
            
if __name__ == "__main__":
    main()
