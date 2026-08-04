import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from Bio.Align import substitution_matrices

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
print(f"SOTA Replications running on: {device}")

# Import dataset definitions
sys.path.append(os.getcwd())
from dataset import TCRDataset, build_global_pool

# ----------------------------------------------------
# Tokenization and BLOSUM50 Encoding Setup
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

# Load BLOSUM50 using Biopython
blosum50 = substitution_matrices.load("BLOSUM50")
aa_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
blosum50_aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

def get_blosum50_vector(aa):
    if aa in blosum50_aa_to_idx:
        return [blosum50[aa, col] for col in aa_list]
    else:
        return [0.0] * 20

def encode_sequence_blosum50(seq, max_len):
    encoded = []
    for char in str(seq).upper():
        encoded.append(get_blosum50_vector(char))
    if len(encoded) < max_len:
        encoded = encoded + [[0.0] * 20] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return np.array(encoded, dtype=np.float32)

BLOSUM_CACHE = {}

def get_blosum50_encoded(seq, max_len):
    key = (seq, max_len)
    if key in BLOSUM_CACHE:
        return BLOSUM_CACHE[key]
    encoded = encode_sequence_blosum50(seq, max_len)
    BLOSUM_CACHE[key] = encoded
    return encoded


# ----------------------------------------------------
# Custom SOTA Dataset Collator
# ----------------------------------------------------
class SOTACollate:
    def __init__(self, positive_triplets_set, global_peptides_pool):
        self.pos_set = positive_triplets_set
        self.global_pool = global_peptides_pool
        
    def __call__(self, batch):
        batch_size = len(batch)
        
        pos_betas = [item["raw_cdr3_beta"] for item in batch]
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
                
        # Construct tensors
        tcr_beta_tokens_list = []
        tcr_beta_blosum_list = []
        
        pep_tokens_pos_list = []
        pep_blosum_pos_list = []
        
        pep_tokens_neg_list = []
        pep_blosum_neg_list = []
        
        for i in range(batch_size):
            tcr_beta_tokens_list.append(tokenize_sequence(pos_betas[i], 30))
            tcr_beta_blosum_list.append(get_blosum50_encoded(pos_betas[i], 30))
            
            pep_tokens_pos_list.append(tokenize_sequence(pos_peptides[i], 20))
            pep_blosum_pos_list.append(get_blosum50_encoded(pos_peptides[i], 20))
            
            pep_tokens_neg_list.append(tokenize_sequence(neg_peptides[i], 20))
            pep_blosum_neg_list.append(get_blosum50_encoded(neg_peptides[i], 20))
            
        tcr_beta_tokens = torch.tensor(tcr_beta_tokens_list + tcr_beta_tokens_list, dtype=torch.long)
        tcr_beta_blosum = torch.tensor(np.array(tcr_beta_blosum_list + tcr_beta_blosum_list), dtype=torch.float)
        
        pep_tokens = torch.tensor(pep_tokens_pos_list + pep_tokens_neg_list, dtype=torch.long)
        pep_blosum = torch.tensor(np.array(pep_blosum_pos_list + pep_blosum_neg_list), dtype=torch.float)
        
        labels = torch.tensor([1.0] * batch_size + [0.0] * batch_size, dtype=torch.float)
        
        return {
            "tcr_beta_tokens": tcr_beta_tokens,
            "tcr_beta_blosum": tcr_beta_blosum,
            "pep_tokens": pep_tokens,
            "pep_blosum": pep_blosum,
            "labels": labels
        }

# ----------------------------------------------------
# Model 1: NetTCR-2.0 (Montemurro et al., 2021)
# ----------------------------------------------------
class NetTCR2(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernels = [1, 3, 5, 7, 9]
        
        # Parallel 1D-Convolutions for TCR-beta
        self.tcr_convs = nn.ModuleList([
            nn.Conv1d(in_channels=20, out_channels=16, kernel_size=k, padding=(k - 1) // 2)
            for k in self.kernels
        ])
        
        # Parallel 1D-Convolutions for Peptide
        self.pep_convs = nn.ModuleList([
            nn.Conv1d(in_channels=20, out_channels=16, kernel_size=k, padding=(k - 1) // 2)
            for k in self.kernels
        ])
        
        # Fully connected layer:
        # Concatenated features size: (16 * 5) for TCR-beta + (16 * 5) for Peptide = 160
        self.fc = nn.Sequential(
            nn.Linear(160, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1)
        )
        
    def forward(self, tcr_beta, peptide):
        # Transpose to [batch, channels, length]
        tcr_in = tcr_beta.transpose(1, 2)
        pep_in = peptide.transpose(1, 2)
        
        tcr_features = []
        for conv in self.tcr_convs:
            c = F.relu(conv(tcr_in))
            pooled = torch.max(c, dim=2)[0]
            tcr_features.append(pooled)
            
        pep_features = []
        for conv in self.pep_convs:
            c = F.relu(conv(pep_in))
            pooled = torch.max(c, dim=2)[0]
            pep_features.append(pooled)
            
        tcr_concat = torch.cat(tcr_features, dim=1) # [batch, 80]
        pep_concat = torch.cat(pep_features, dim=1) # [batch, 80]
        
        fused = torch.cat([tcr_concat, pep_concat], dim=1) # [batch, 160]
        logits = self.fc(fused).squeeze(-1) # [batch]
        return logits

# ----------------------------------------------------
# Model 2: TITAN (Gao et al., 2023)
# ----------------------------------------------------
class TITAN(nn.Module):
    def __init__(self, vocab_size=22, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D-CNN sequence token encoders
        self.tcr_cnn = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.pep_cnn = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Cross-Attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1)
        )
        
    def forward(self, tcr_tokens, pep_tokens):
        # Embed and encode TCR
        tcr_emb = self.embedding(tcr_tokens).transpose(1, 2) # [batch, D, L_tcr]
        H_tcr = F.relu(self.tcr_cnn(tcr_emb)).transpose(1, 2) # [batch, L_tcr, D]
        
        # Embed and encode Peptide
        pep_emb = self.embedding(pep_tokens).transpose(1, 2) # [batch, D, L_pep]
        H_pep = F.relu(self.pep_cnn(pep_emb)).transpose(1, 2) # [batch, L_pep, D]
        
        # Bidirectional Attention
        Q = self.q_proj(H_tcr)
        K = self.k_proj(H_pep)
        
        # Similarity matrix: [batch, L_tcr, L_pep]
        S = torch.matmul(Q, K.transpose(1, 2))
        
        # Attended representations
        attn_tcr_pep = F.softmax(S, dim=-1)
        C_tcr = torch.matmul(attn_tcr_pep, H_pep)
        
        attn_pep_tcr = F.softmax(S.transpose(1, 2), dim=-1)
        C_pep = torch.matmul(attn_pep_tcr, H_tcr)
        
        # Global max-pooling
        pooled_tcr = torch.max(C_tcr, dim=1)[0]
        pooled_pep = torch.max(C_pep, dim=1)[0]
        
        # Concatenate and classify
        fused = torch.cat([pooled_tcr, pooled_pep], dim=1)
        logits = self.fc(fused).squeeze(-1)
        return logits

# ----------------------------------------------------
# Train & Evaluate Functions
# ----------------------------------------------------
def train_model(model, loader, optimizer, criterion, epochs=15):
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            
            if isinstance(model, NetTCR2):
                tcr = batch["tcr_beta_blosum"].to(device)
                pep = batch["pep_blosum"].to(device)
                logits = model(tcr, pep)
            else:
                tcr = batch["tcr_beta_tokens"].to(device)
                pep = batch["pep_tokens"].to(device)
                logits = model(tcr, pep)
                
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
            labels = batch["labels"].to(device)
            
            if isinstance(model, NetTCR2):
                tcr = batch["tcr_beta_blosum"].to(device)
                pep = batch["pep_blosum"].to(device)
                logits = model(tcr, pep)
            else:
                tcr = batch["tcr_beta_tokens"].to(device)
                pep = batch["pep_tokens"].to(device)
                logits = model(tcr, pep)
                
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_probs)
    
    auc = roc_auc_score(y_true, y_pred)
    
    # Threshold optimization
    best_mcc = -1.0
    best_thresh = 0.5
    thresholds = np.arange(0.01, 1.00, 0.01)
    for t in thresholds:
        preds = (y_pred >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
            
    return auc, best_mcc, best_thresh

# ----------------------------------------------------
# Main Execution
# ----------------------------------------------------
def main():
    print("Loading datasets...")
    train_df = pd.read_csv("./Processed/train.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    
    train_triplets = set(zip(train_df["cdr3_beta"], train_df["peptide"], train_df["hla_allele"]))
    train_pool = build_global_pool(train_df)
    train_dataset = TCRDataset("./Processed/train.csv")
    train_collate = SOTACollate(train_triplets, train_pool)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_collate)
    
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)
    test_dataset = TCRDataset("./Processed/test.csv")
    test_collate = SOTACollate(test_triplets, test_pool)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_collate)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 1. Train NetTCR-2.0
    print("\n--- Training NetTCR-2.0 ---")
    set_seed(42)
    nettcr_model = NetTCR2().to(device)
    optimizer_nettcr = optim.AdamW(nettcr_model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_model(nettcr_model, train_loader, optimizer_nettcr, criterion, epochs=15)
    auc_nettcr, mcc_nettcr, thresh_nettcr = evaluate_model(nettcr_model, test_loader)
    print(f"NetTCR-2.0 Results -> ROC-AUC: {auc_nettcr:.4f} | Optimal MCC: {mcc_nettcr:.4f} at threshold {thresh_nettcr:.2f}")
    
    # 2. Train TITAN
    print("\n--- Training TITAN ---")
    set_seed(42)
    titan_model = TITAN().to(device)
    optimizer_titan = optim.AdamW(titan_model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_model(titan_model, train_loader, optimizer_titan, criterion, epochs=15)
    auc_titan, mcc_titan, thresh_titan = evaluate_model(titan_model, test_loader)
    print(f"TITAN Results -> ROC-AUC: {auc_titan:.4f} | Optimal MCC: {mcc_titan:.4f} at threshold {thresh_titan:.2f}")
    
    # Log results
    out_path = "sota_replication_results.txt"
    with open(out_path, "w") as f:
        f.write("=== SOTA Model Replication Results ===\n")
        f.write(f"NetTCR-2.0:\n")
        f.write(f"  Test ROC-AUC: {auc_nettcr:.5f}\n")
        f.write(f"  Optimal Threshold MCC: {mcc_nettcr:.5f} (threshold: {thresh_nettcr:.2f})\n\n")
        f.write(f"TITAN:\n")
        f.write(f"  Test ROC-AUC: {auc_titan:.5f}\n")
        f.write(f"  Optimal Threshold MCC: {mcc_titan:.5f} (threshold: {thresh_titan:.2f})\n")
    print(f"\nResults successfully logged to {out_path}")

if __name__ == "__main__":
    main()
