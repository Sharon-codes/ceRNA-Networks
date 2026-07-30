import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Add workspace to path
sys.path.append(os.getcwd())
from dataset import TCRDataset, TCRCollate, build_global_pool
from modules import ProjectionHead
from run_zeroshot_comparators import set_seed

class AblationModel(nn.Module):
    def __init__(self, mode="mean", d_model=64):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        
        # ESM-2 8M hidden dimension is 320. Project to d_model.
        self.esm_proj = nn.Linear(320, d_model)
        self.pephla_proj = nn.Linear(320, d_model)
        
        if mode in ["mean", "max", "cls"]:
            # Concatenating the three pooled representations gives a vector of size 3 * d_model
            self.projection_head = nn.Sequential(
                nn.Linear(d_model * 3, d_model * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(d_model, 1)
            )
        else:
            # Direct Concatenation (Baseline flat structure)
            self.projection_head = ProjectionHead(d_model, seq_len=110)
            
    def forward(self, cdr3_beta, cdr3_alpha, peptide_plus_hla):
        # Project representations to d_model
        beta_proj = self.esm_proj(cdr3_beta)    # [batch, 30, d_model]
        alpha_proj = self.esm_proj(cdr3_alpha)  # [batch, 30, d_model]
        pephla_proj = self.pephla_proj(peptide_plus_hla)  # [batch, 50, d_model]
        
        if self.mode == "mean":
            beta_pool = beta_proj.mean(dim=1)
            alpha_pool = alpha_proj.mean(dim=1)
            pephla_pool = pephla_proj.mean(dim=1)
            fused = torch.cat([beta_pool, alpha_pool, pephla_pool], dim=1) # [batch, 3 * d_model]
            logit = self.projection_head(fused)
        elif self.mode == "max":
            beta_pool = beta_proj.max(dim=1)[0]
            alpha_pool = alpha_proj.max(dim=1)[0]
            pephla_pool = pephla_proj.max(dim=1)[0]
            fused = torch.cat([beta_pool, alpha_pool, pephla_pool], dim=1) # [batch, 3 * d_model]
            logit = self.projection_head(fused)
        elif self.mode == "cls":
            # Extract CLS token (index 0)
            beta_pool = beta_proj[:, 0, :]
            alpha_pool = alpha_proj[:, 0, :]
            pephla_pool = pephla_proj[:, 0, :]
            fused = torch.cat([beta_pool, alpha_pool, pephla_pool], dim=1) # [batch, 3 * d_model]
            logit = self.projection_head(fused)
        else:
            # Direct Concatenation (Modification C)
            fused = torch.cat([beta_proj, alpha_proj, pephla_proj], dim=1) # [batch, 110, d_model]
            logit = self.projection_head(fused)
            
        return logit.squeeze(-1)

def train_and_evaluate_ablation(mode, d_model, train_loader, test_loader, device, epochs=5):
    set_seed(42)
    model = AblationModel(mode=mode, d_model=d_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train
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
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_probs)
    return roc_auc_score(y_true, y_pred)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Architectural Ablations running on: {device}")
    
    # Prepare datasets
    train_df = pd.read_csv("./Processed/train.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    
    train_temp_path = "train_temp_ablation.csv"
    test_temp_path = "test_temp_ablation.csv"
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
    
    # Run ablations
    print("\n--- Training Modification A (Max-Pooling, d_model=64) ---")
    auc_a = train_and_evaluate_ablation("max", 64, train_loader, test_loader, device)
    print(f"Modification A ROC-AUC: {auc_a:.5f}")
    
    print("\n--- Training Modification B (CLS-Extraction, d_model=64) ---")
    auc_b = train_and_evaluate_ablation("cls", 64, train_loader, test_loader, device)
    print(f"Modification B ROC-AUC: {auc_b:.5f}")
    
    print("\n--- Training Modification C (Direct Concatenation, d_model=128) ---")
    auc_c = train_and_evaluate_ablation("direct_128", 128, train_loader, test_loader, device)
    print(f"Modification C ROC-AUC: {auc_c:.5f}")
    
    # Clean up temp files
    for path in [train_temp_path, test_temp_path]:
        if os.path.exists(path):
            os.remove(path)
            
    # Save to report file
    os.makedirs("Evaluation", exist_ok=True)
    report_path = "./Evaluation/architectural_ablations_report.txt"
    with open(report_path, "w") as f:
        f.write("Architectural Ablations Report (Stronger Ablations)\n")
        f.write("==================================================\n")
        f.write(f"Modification A (Max-Pooling, d_model=64):              {auc_a:.5f}\n")
        f.write(f"Modification B ([CLS]-Token Extraction, d_model=64):   {auc_b:.5f}\n")
        f.write(f"Modification C (Direct Concatenation, d_model=128):     {auc_c:.5f}\n")
        
    print(f"\nArchitectural ablations results logged to {report_path}")

if __name__ == "__main__":
    main()
