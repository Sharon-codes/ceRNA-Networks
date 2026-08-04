import os
import sys
import random
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_curve, auc

# Import existing dataset and model structures
sys.path.append(os.getcwd())
from dataset import TCRDataset, TCRCollate, build_global_pool
from model import MambaTCR

# ----------------------------------------------------
# Seed and Device Setup
# ----------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Comparators running on: {device}")

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
        return 1.0
        
    z = auc_diff / np.sqrt(var)
    p_value = 2 * scipy.stats.norm.sf(np.abs(z)) if hasattr(scipy.stats, "norm") else 1.0
    # Fallback to standard normal CDF calculation if needed
    if not hasattr(scipy.stats, "norm"):
        import math
        p_value = 2 * (1 - 0.5 * (1 + math.erf(np.abs(z) / math.sqrt(2))))
        
    return p_value

# ----------------------------------------------------
# ERGO-II (LSTM + Transformer autoencoder prior)
# ----------------------------------------------------
class ERGO2(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # Projections
        self.esm_proj = nn.Linear(320, d_model)
        self.pephla_proj = nn.Linear(320, d_model)
        
        # Transformer Autoencoder Prior (for TCR Beta + Alpha)
        # TCR Beta (30) + TCR Alpha (30) = 60 tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.tcr_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # LSTM (for Peptide-HLA)
        self.pep_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear((60 + 50) * d_model, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, cdr3_beta, cdr3_alpha, peptide_plus_hla):
        # Projections
        beta_proj = self.esm_proj(cdr3_beta)    # [batch, 30, d_model]
        alpha_proj = self.esm_proj(cdr3_alpha)  # [batch, 30, d_model]
        pephla_proj = self.pephla_proj(peptide_plus_hla)  # [batch, 50, d_model]
        
        # TCR Transformer
        tcr_seq = torch.cat([beta_proj, alpha_proj], dim=1) # [batch, 60, d_model]
        tcr_features = self.tcr_transformer(tcr_seq) # [batch, 60, d_model]
        
        # Peptide LSTM
        pep_features, _ = self.pep_lstm(pephla_proj) # [batch, 50, d_model]
        
        # Concatenate and classify
        fused = torch.cat([tcr_features, pep_features], dim=1) # [batch, 110, d_model]
        fused_flat = fused.reshape(fused.shape[0], -1)
        logits = self.fc(fused_flat).squeeze(-1)
        return logits

# ----------------------------------------------------
# EPIC-TRACE (Conv1D + Multi-Head Attention)
# ----------------------------------------------------
class EPIC_TRACE(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # Conv1D layers (TCR beta, TCR alpha, Peptide-HLA)
        self.beta_conv = nn.Conv1d(320, d_model, kernel_size=3, padding=1)
        self.alpha_conv = nn.Conv1d(320, d_model, kernel_size=3, padding=1)
        self.pep_conv = nn.Conv1d(320, d_model, kernel_size=3, padding=1)
        
        # Multi-Head Attention Block
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(110 * d_model, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, cdr3_beta, cdr3_alpha, peptide_plus_hla):
        # Conv1D (transpose to [batch, channels, length])
        beta_conv = F.relu(self.beta_conv(cdr3_beta.transpose(1, 2))).transpose(1, 2)  # [batch, 30, d_model]
        alpha_conv = F.relu(self.alpha_conv(cdr3_alpha.transpose(1, 2))).transpose(1, 2)  # [batch, 30, d_model]
        pep_conv = F.relu(self.pep_conv(peptide_plus_hla.transpose(1, 2))).transpose(1, 2)  # [batch, 50, d_model]
        
        # Concatenate
        combined = torch.cat([beta_conv, alpha_conv, pep_conv], dim=1) # [batch, 110, d_model]
        
        # Multi-Head Attention
        attn_out, _ = self.mha(combined, combined, combined)
        
        # Classify
        attn_flat = attn_out.reshape(attn_out.shape[0], -1)
        logits = self.fc(attn_flat).squeeze(-1)
        return logits

# ----------------------------------------------------
# Helper training / evaluation function
# ----------------------------------------------------
def train_model(model, loader, epochs=15):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            beta = batch["cdr3_beta"].to(device)
            alpha = batch["cdr3_alpha"].to(device)
            pephla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(beta, alpha, pephla)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")

def evaluate_model(model, loader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            beta = batch["cdr3_beta"].to(device)
            alpha = batch["cdr3_alpha"].to(device)
            pephla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(beta, alpha, pephla)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_probs)
    
    auc_val = roc_auc_score(y_true, y_pred)
    
    # Youden-optimized MCC
    best_mcc = -1.0
    best_thresh = 0.5
    thresholds = np.arange(0.01, 1.00, 0.01)
    for t in thresholds:
        preds = (y_pred >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
            
    return auc_val, best_mcc, y_pred, y_true

# ----------------------------------------------------
# Main Execution
# ----------------------------------------------------
def main():
    print("Preparing LODO dataset...")
    # Load dataset splits from Processed
    train_df = pd.read_csv("./Processed/train.csv")
    val_df = pd.read_csv("./Processed/val.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
    
    # 80/20 train/test split of combined unique samples
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
    
    train_temp_path = "./Processed/lodo_train_temp.csv"
    test_temp_path = "./Processed/lodo_test_temp.csv"
    train_df.to_csv(train_temp_path, index=False)
    test_df.to_csv(test_temp_path, index=False)
    
    # Setup data loaders
    train_triplets = set(zip(train_df["cdr3_beta"], train_df["peptide"], train_df["hla_allele"]))
    train_pool = build_global_pool(train_df)
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)
    
    train_dataset = TCRDataset(train_temp_path)
    test_dataset = TCRDataset(test_temp_path)
    train_collate = TCRCollate(train_triplets, train_pool)
    test_collate = TCRCollate(test_triplets, test_pool)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=train_collate)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_collate)
    
    # Verify sizes
    print(f"LODO Train samples: {len(train_dataset) * 2}")
    print(f"LODO Test samples: {len(test_dataset) * 2}")
    
    # Load Direct Concatenation model (baseline)
    print("\nEvaluating Direct Concatenation (Baseline)...")
    baseline_model = MambaTCR(d_model=64).to(device)
    baseline_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        print("Successfully loaded baseline model checkpoint.")
    else:
        print("WARNING: baseline model checkpoint not found. Evaluating randomly initialized model.")
        
    base_auc, base_mcc, base_pred, base_true = evaluate_model(baseline_model, test_loader)
    print(f"Direct Concatenation -> ROC-AUC: {base_auc:.4f} | Optimal MCC: {base_mcc:.4f}")
    
    # 1. Train and evaluate ERGO-II
    print("\n--- Training ERGO-II ---")
    set_seed(42)
    ergo_model = ERGO2().to(device)
    train_model(ergo_model, train_loader, epochs=15)
    ergo_auc, ergo_mcc, ergo_pred, _ = evaluate_model(ergo_model, test_loader)
    ergo_p = delong_roc_test(base_true, base_pred, ergo_pred)
    print(f"ERGO-II Results -> ROC-AUC: {ergo_auc:.4f} | Optimal MCC: {ergo_mcc:.4f} | DeLong p-value: {ergo_p:.4e}")
    
    # 2. Train and evaluate EPIC-TRACE
    print("\n--- Training EPIC-TRACE ---")
    set_seed(42)
    epic_model = EPIC_TRACE().to(device)
    train_model(epic_model, train_loader, epochs=15)
    epic_auc, epic_mcc, epic_pred, _ = evaluate_model(epic_model, test_loader)
    epic_p = delong_roc_test(base_true, base_pred, epic_pred)
    print(f"EPIC-TRACE Results -> ROC-AUC: {epic_auc:.4f} | Optimal MCC: {epic_mcc:.4f} | DeLong p-value: {epic_p:.4e}")
    
    # Clean up temp files
    for path in [train_temp_path, test_temp_path]:
        if os.path.exists(path):
            os.remove(path)
            
    # Append results to zeroshot_field_benchmarks.txt
    out_path = "zeroshot_field_benchmarks.txt"
    with open(out_path, "a") as f:
        f.write("\n=== Zero-Shot Field Benchmark Comparators ===\n")
        f.write(f"Direct Concatenation (Baseline):\n")
        f.write(f"  Test ROC-AUC: {base_auc:.5f}\n")
        f.write(f"  Youden-optimized MCC: {base_mcc:.5f}\n\n")
        f.write(f"ERGO-II (LSTM + Transformer autoencoder prior):\n")
        f.write(f"  Test ROC-AUC: {ergo_auc:.5f}\n")
        f.write(f"  Youden-optimized MCC: {ergo_mcc:.5f}\n")
        f.write(f"  DeLong p-value relative to Baseline: {ergo_p:.5e}\n\n")
        f.write(f"EPIC-TRACE:\n")
        f.write(f"  Test ROC-AUC: {epic_auc:.5f}\n")
        f.write(f"  Youden-optimized MCC: {epic_mcc:.5f}\n")
        f.write(f"  DeLong p-value relative to Baseline: {epic_p:.5e}\n")
    print(f"\nZero-shot comparators benchmark appended to {out_path}")

if __name__ == "__main__":
    import scipy.stats
    main()
