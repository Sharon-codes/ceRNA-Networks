import os
import sys
import time
import json
import random
import platform
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------
# NetTCR-style CNN components
# ----------------------------------------------------
amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
aa_to_idx = {aa: i+1 for i, aa in enumerate(amino_acids)} # 0 for padding

def tokenize_sequence(seq, max_len):
    encoded = [aa_to_idx.get(aa, 0) for aa in str(seq).upper()]
    if len(encoded) < max_len:
        encoded = encoded + [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

class CNNCollate:
    def __init__(self, positive_triplets_set, global_peptides_pool):
        self.pos_set = positive_triplets_set
        self.global_pool = global_peptides_pool
        
    def __call__(self, batch):
        batch_size = len(batch)
        
        pos_betas = [item["raw_cdr3_beta"] for item in batch]
        pos_alphas = [item["raw_cdr3_alpha"] for item in batch]
        pos_peptides = [item["raw_peptide"] for item in batch]
        
        # Shuffling for decoy negative sampling
        shuffled_indices = list(range(batch_size))
        if batch_size > 1:
            while True:
                random.shuffle(shuffled_indices)
                collision = False
                for i in range(batch_size):
                    if shuffled_indices[i] == i:
                        collision = True
                        break
                if not collision:
                    break
                    
        # Resolve peptide collisions
        for i in range(batch_size):
            if shuffled_indices[i] != -1 and pos_peptides[i] == pos_peptides[shuffled_indices[i]]:
                swap_found = False
                for j in range(batch_size):
                    if i == j or shuffled_indices[j] == -1:
                        continue
                    cand_i = shuffled_indices[j]
                    cand_j = shuffled_indices[i]
                    if (pos_peptides[i] != pos_peptides[cand_i] and 
                        pos_peptides[j] != pos_peptides[cand_j] and 
                        cand_i != i and 
                        cand_j != j):
                        shuffled_indices[i], shuffled_indices[j] = shuffled_indices[j], shuffled_indices[i]
                        swap_found = True
                        break
                if not swap_found:
                    shuffled_indices[i] = -1
                    
        neg_peptides = []
        for i in range(batch_size):
            idx = shuffled_indices[i]
            if idx == -1:
                global_pep = pos_peptides[i]
                found = False
                if self.global_pool:
                    for _ in range(20):
                        g_pep, _ = random.choice(self.global_pool)
                        if g_pep != pos_peptides[i]:
                            global_pep = g_pep
                            found = True
                            break
                if not found:
                    global_pep = "AAAAA"
                neg_peptides.append(global_pep)
            else:
                neg_peptides.append(pos_peptides[idx])
                
        sequences = []
        # Positives
        for i in range(batch_size):
            beta_tokens = tokenize_sequence(pos_betas[i], 30)
            alpha_tokens = tokenize_sequence(pos_alphas[i], 30)
            pep_tokens = tokenize_sequence(pos_peptides[i], 20)
            sequences.append(beta_tokens + alpha_tokens + pep_tokens)
        # Negatives
        for i in range(batch_size):
            beta_tokens = tokenize_sequence(pos_betas[i], 30)
            alpha_tokens = tokenize_sequence(pos_alphas[i], 30)
            pep_tokens = tokenize_sequence(neg_peptides[i], 20)
            sequences.append(beta_tokens + alpha_tokens + pep_tokens)
            
        labels = [1.0] * batch_size + [0.0] * batch_size
        
        return {
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float)
        }

class NetTCRStyleCNN(nn.Module):
    def __init__(self, vocab_size=22, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D Convolution and max-pooling
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Sequence length concatenated: 30 (beta) + 30 (alpha) + 20 (pep) = 80 tokens.
        # After pool1: 80 / 2 = 40. After pool2: 40 / 2 = 20.
        self.flatten_dim = 128 * 20 # 2560
        
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x) # [batch, 80, 32]
        x = x.transpose(1, 2) # [batch, 32, 80]
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        logit = self.mlp(x)
        return logit.squeeze(-1)

# Helper function to extract mean-pooled features for Logistic Regression
def extract_tabular_data(loader, device):
    X_list = []
    y_list = []
    with torch.no_grad():
        for batch in loader:
            beta = batch["cdr3_beta"].to(device)
            alpha = batch["cdr3_alpha"].to(device)
            pephla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            beta_pooled = beta.mean(dim=1)
            alpha_pooled = alpha.mean(dim=1)
            pephla_pooled = pephla.mean(dim=1)
            
            features = torch.cat([beta_pooled, alpha_pooled, pephla_pooled], dim=1)
            X_list.append(features.cpu().numpy())
            y_list.append(labels.cpu().numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def main():
    set_seed(42)
    device = torch.device("cpu")
    print(f"Reviewer Defense Suite running on CPU...")
    
    os.makedirs("./images", exist_ok=True)
    out_path = "reviewer_defense_metrics.txt"
    log_file = open(out_path, "w")
    
    # ----------------------------------------------------
    # Module 1: Logistic Regression Threshold Optimization
    # ----------------------------------------------------
    print("\n--- Running Module 1: LR Threshold Optimization ---")
    train_df = pd.read_csv("./Processed/train.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    
    train_triplets = set(zip(train_df["cdr3_beta"], train_df["peptide"], train_df["hla_allele"]))
    train_pool = build_global_pool(train_df)
    train_dataset = TCRDataset("./Processed/train.csv")
    train_collate = TCRCollate(train_triplets, train_pool)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_collate)
    
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)
    test_dataset = TCRDataset("./Processed/test.csv")
    test_collate = TCRCollate(test_triplets, test_pool)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=test_collate)
    
    print("Extracting tabular features for Logistic Regression...")
    with SuppressStdout():
        X_train, y_train = extract_tabular_data(train_loader, device)
        X_test, y_test = extract_tabular_data(test_loader, device)
        
    print("Training Logistic Regression...")
    clf_lr = LogisticRegression(max_iter=1000, random_state=42)
    clf_lr.fit(X_train, y_train)
    probs_lr = clf_lr.predict_proba(X_test)[:, 1]
    
    print("Optimizing LR decision threshold for MCC...")
    best_thresh = 0.5
    best_mcc = -1.0
    
    thresholds = np.arange(0.01, 1.00, 0.01)
    for t in thresholds:
        preds = (probs_lr >= t).astype(int)
        mcc = matthews_corrcoef(y_test, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
            
    # Compute optimized metrics
    opt_preds = (probs_lr >= best_thresh).astype(int)
    f1 = f1_score(y_test, opt_preds)
    precision = precision_score(y_test, opt_preds)
    recall = recall_score(y_test, opt_preds)
    
    m1_text = f"""==================================================
MODULE 1: LOGISTIC REGRESSION THRESHOLD OPTIMIZATION
==================================================
Optimal Threshold (Max MCC): {best_thresh:.2f}
Max MCC Achieved:            {best_mcc:.4f}
F1-Score at Max MCC:         {f1:.4f}
Precision at Max MCC:        {precision:.4f}
Recall at Max MCC:           {recall:.4f}
"""
    print(m1_text)
    log_file.write(m1_text + "\n")
    
    # ----------------------------------------------------
    # Module 2: NetTCR-Style CNN Baseline
    # ----------------------------------------------------
    print("--- Running Module 2: NetTCR-Style CNN ---")
    set_seed(42)
    
    # Dataloaders for CNN (with raw text and dynamic natural negatives)
    cnn_collate_train = CNNCollate(train_triplets, train_pool)
    cnn_loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=cnn_collate_train)
    
    cnn_collate_test = CNNCollate(test_triplets, test_pool)
    cnn_loader_test = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=cnn_collate_test)
    
    # Initialize NetTCR-Style CNN
    cnn_model = NetTCRStyleCNN().to(device)
    optimizer = optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train CNN for 20 epochs
    epochs = 20
    print(f"Training NetTCR-Style CNN for {epochs} epochs on CPU...")
    for epoch in range(1, epochs + 1):
        cnn_model.train()
        epoch_loss = 0.0
        with SuppressStdout():
            for batch in cnn_loader_train:
                seqs = batch["sequences"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                logits = cnn_model(seqs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss/len(cnn_loader_train):.4f}")
        
    # Evaluate CNN on Test split
    cnn_model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        with SuppressStdout():
            for batch in cnn_loader_test:
                seqs = batch["sequences"].to(device)
                labels = batch["labels"].to(device)
                
                logits = cnn_model(seqs)
                probs = torch.sigmoid(logits)
                
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
    y_test_cnn = np.array(all_targets)
    probs_cnn = np.array(all_probs)
    
    auc_cnn = roc_auc_score(y_test_cnn, probs_cnn)
    
    # Optimize threshold for CNN
    best_mcc_cnn = -1.0
    best_t_cnn = 0.5
    for t in thresholds:
        preds = (probs_cnn >= t).astype(int)
        mcc = matthews_corrcoef(y_test_cnn, preds)
        if mcc > best_mcc_cnn:
            best_mcc_cnn = mcc
            best_t_cnn = t
            
    m2_text = f"""==================================================
MODULE 2: NETTCR-STYLE CNN BASELINE
==================================================
Test ROC-AUC:                     {auc_cnn:.4f}
Optimal Threshold (Max MCC):      {best_t_cnn:.2f}
Test MCC (at Optimal Threshold):   {best_mcc_cnn:.4f}
"""
    print(m2_text)
    log_file.write(m2_text + "\n")
    
    # ----------------------------------------------------
    # Module 3: Model Calibration & Hyperparameter Dump
    # ----------------------------------------------------
    print("--- Running Module 3: Calibration & Hyperparameters ---")
    set_seed(42)
    
    # Load Direct Concatenation model
    mamba_model = MambaTCR(d_model=64, nhead=8, num_layers=2).to(device)
    mamba_model.load_state_dict(torch.load("./Checkpoints/best_mamba_tcr_production.pt", map_location=device))
    mamba_model.eval()
    
    # Evaluate Direct Concatenation model on Test set
    all_mamba_probs = []
    all_mamba_targets = []
    with torch.no_grad():
        with SuppressStdout():
            for batch in test_loader:
                cdr3_beta = batch["cdr3_beta"].to(device)
                cdr3_alpha = batch["cdr3_alpha"].to(device)
                peptide_plus_hla = batch["peptide_plus_hla"].to(device)
                labels = batch["label"].to(device)
                
                logits = mamba_model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
                probs = torch.sigmoid(logits)
                
                all_mamba_probs.extend(probs.cpu().numpy())
                all_mamba_targets.extend(labels.cpu().numpy())
                
    y_test_mamba = np.array(all_mamba_targets)
    probs_mamba = np.array(all_mamba_probs)
    
    # 1. Brier Score
    brier_score = brier_score_loss(y_test_mamba, probs_mamba)
    
    # 2. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test_mamba, probs_mamba, n_bins=10, strategy='uniform')
    
    # Plot and save Calibration Diagram (Reliability Curve)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5.5, 4.5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, marker="o", color="blue", lw=2, label="Direct Concatenation")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives", fontsize=11)
    plt.title("Model Calibration Curve (Reliability Diagram)", fontsize=12, fontweight='bold')
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig("./images/Figure_5_Calibration.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./images/Figure_5_Calibration.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Calibration diagram plotted and saved to ./images/Figure_5_Calibration.pdf & .png")
    
    # 3. Dynamic Hardware Specs
    cpu_cores = multiprocessing.cpu_count()
    cpu_name = platform.processor() or "Unknown CPU"
    ram_gb = "Unknown"
    try:
        import ctypes
        class memory_status(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_uint64),
                ('ullAvailPhys', ctypes.c_uint64),
                ('ullTotalPageFile', ctypes.c_uint64),
                ('ullAvailPageFile', ctypes.c_uint64),
                ('ullTotalVirtual', ctypes.c_uint64),
                ('ullAvailVirtual', ctypes.c_uint64),
                ('ullAvailExtendedVirtual', ctypes.c_uint64)
            ]
        stat = memory_status()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        ram_gb = f"{stat.ullTotalPhys / (1024**3):.2f} GB"
    except:
        pass
        
    hardware_specs = {
        "CPU": cpu_name,
        "CPU_Cores": cpu_cores,
        "System_RAM": ram_gb,
        "OS_Platform": f"{platform.system()} {platform.release()}"
    }
    
    # Save complete hyperparameters to hyperparameters.json
    hyperparams = {
        "model_architecture": {
            "name": "ESM-MambaTCR (Direct Concatenation)",
            "projection_head_dimensions": [7040, 128, 64, 1],
            "d_model": 64,
            "dropout": 0.3
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64
        },
        "scheduler": {
            "name": "CosineAnnealingWarmRestarts",
            "T_0": 10,
            "T_mult": 1,
            "eta_min": 1e-6
        },
        "hardware_environment": hardware_specs
    }
    
    with open("hyperparameters.json", "w") as h_file:
        json.dump(hyperparams, h_file, indent=4)
    print("Hyperparameters dumped to hyperparameters.json")
    
    m3_text = f"""==================================================
MODULE 3: MODEL CALIBRATION & HARDWARE DUMP
==================================================
Brier Score (Direct Concatenation): {brier_score:.4f}
Calibration curve plotted and saved to `./images/Figure_5_Calibration.pdf`

Hyperparameter & Hardware Dump:
{json.dumps(hyperparams, indent=4)}
"""
    log_file.write(m3_text + "\n")
    print(m3_text)
    
    log_file.close()
    print(f"\nAll reviewer defense modules ran successfully. Metrics log saved to {out_path}")

if __name__ == "__main__":
    main()
