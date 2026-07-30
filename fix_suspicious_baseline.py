import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Add workspace to path
sys.path.append(os.getcwd())
from dataset import TCRDataset, TCRCollate, build_global_pool
from modules import PositionalEncoding, CrossAttentionFusion, ProjectionHead

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CrossAttentionMambaTCR(nn.Module):
    def __init__(self, d_model=64, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.esm_proj = nn.Linear(320, d_model)
        self.pephla_proj = nn.Linear(320, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=50)
        self.cross_attention = CrossAttentionFusion(d_model, nhead=nhead)
        self.projection_head = ProjectionHead(d_model, seq_len=61)
        
    def forward(self, cdr3_beta, cdr3_alpha, peptide_plus_hla):
        beta_proj = self.esm_proj(cdr3_beta)
        alpha_proj = self.esm_proj(cdr3_alpha)
        pephla_proj = self.pephla_proj(peptide_plus_hla)
        
        beta_proj = self.pos_encoder(beta_proj)
        alpha_proj = self.pos_encoder(alpha_proj)
        pephla_proj = self.pos_encoder(pephla_proj)
        
        batch_size = beta_proj.size(0)
        sep_tensor = torch.zeros(batch_size, 1, self.d_model, device=beta_proj.device)
        tcr_seq = torch.cat([beta_proj, sep_tensor, alpha_proj], dim=1) # [batch, 61, d_model]
        
        fused = self.cross_attention(tcr_seq, pephla_proj)
        logit = self.projection_head(fused)
        return logit.squeeze(-1)

def evaluate_model(model, loader, device):
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
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Baseline correction running on: {device}")
    
    # Load dataset
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
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_collate)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # ----------------------------------------------------
    # Step 1: Train original baseline
    # ----------------------------------------------------
    print("\n--- Training Original Cross-Attention Baseline ---")
    set_seed(42)
    model = CrossAttentionMambaTCR().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, 6):
        model.train()
        for batch in train_loader:
            cdr3_beta = batch["cdr3_beta"].to(device)
            cdr3_alpha = batch["cdr3_alpha"].to(device)
            peptide_plus_hla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    y_true, y_pred = evaluate_model(model, test_loader, device)
    raw_auc = roc_auc_score(y_true, y_pred)
    print(f"Original baseline test AUC: {raw_auc:.5f}")
    
    # ----------------------------------------------------
    # Step 2: Correct if mathematically forced to 0.50000
    # ----------------------------------------------------
    # Checking for degeneracy (e.g. constant predictions or NaN/extreme flatline)
    is_degenerate = (len(np.unique(y_pred.round(5))) <= 1) or np.isnan(y_pred).any() or (abs(raw_auc - 0.5) < 1e-5)
    
    if is_degenerate:
        print("\n[WARNING] Degenerate predictions/AUC of 0.50000 detected. Re-training with baseline fix...")
        set_seed(42)
        model_fixed = CrossAttentionMambaTCR().to(device)
        optimizer_fixed = optim.AdamW(model_fixed.parameters(), lr=5e-5, weight_decay=1e-4)
        
        # Train for 3 epochs with gradient clipping and lowered learning rate
        epochs_fixed = 3
        for epoch in range(1, epochs_fixed + 1):
            model_fixed.train()
            epoch_loss = 0.0
            for batch in train_loader:
                cdr3_beta = batch["cdr3_beta"].to(device)
                cdr3_alpha = batch["cdr3_alpha"].to(device)
                peptide_plus_hla = batch["peptide_plus_hla"].to(device)
                labels = batch["label"].to(device)
                
                optimizer_fixed.zero_grad()
                logits = model_fixed(cdr3_beta, cdr3_alpha, peptide_plus_hla)
                loss = criterion(logits, labels)
                loss.backward()
                
                # Apply gradient clipping
                nn.utils.clip_grad_norm_(model_fixed.parameters(), max_norm=1.0)
                optimizer_fixed.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch}/{epochs_fixed} | Loss: {epoch_loss/len(train_loader):.4f}")
            
        y_true_fixed, y_pred_fixed = evaluate_model(model_fixed, test_loader, device)
        fixed_auc = roc_auc_score(y_true_fixed, y_pred_fixed)
        print(f"Fixed baseline test AUC (unrounded): {fixed_auc:.5f}")
        final_auc = fixed_auc
    else:
        print("Predictions are not degenerate. Logging raw unrounded AUC.")
        final_auc = raw_auc
        
    # Save the exact float to 5 decimal places or more
    out_path = "fixed_baseline_metric.txt"
    with open(out_path, "w") as f:
        f.write(f"{final_auc:.5f}")
    print(f"\nSaved fixed baseline metric ({final_auc:.5f}) to {out_path}")

if __name__ == "__main__":
    main()
