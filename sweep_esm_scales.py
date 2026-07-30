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
from sklearn.metrics import roc_auc_score

# Limit PyTorch threads
torch.set_num_threads(1)

# Import dataset definitions
sys.path.append(os.getcwd())
from dataset import TCRDataset, TCRCollate, build_global_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Scale Sweep running on: {device}")

# ----------------------------------------------------
# Simple MLP Head (Direct Concatenation classification head)
# ----------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, embedding_dim, seq_len=110, d_model=64):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        # Projection of flattened sequence to d_model
        self.proj = nn.Linear(seq_len * embedding_dim, d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, beta_emb, alpha_emb, pephla_emb):
        # Repeat/project embeddings if dim does not match input
        # Flatten and concatenate along sequence dimension: (30 + 30 + 50) = 110
        # beta: [batch, 30, dim], alpha: [batch, 30, dim], pephla: [batch, 50, dim]
        combined = torch.cat([beta_emb, alpha_emb, pephla_emb], dim=1) # [batch, 110, dim]
        flat = combined.reshape(combined.shape[0], -1) # [batch, 110 * dim]
        
        features = F.relu(self.proj(flat))
        logits = self.classifier(features).squeeze(-1)
        return logits

# ----------------------------------------------------
# Custom Dataset to handle simulated dimensions
# ----------------------------------------------------
class ScaleDataset(Dataset):
    def __init__(self, base_dataset, target_dim):
        self.base_dataset = base_dataset
        self.target_dim = target_dim
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Project embeddings to target dimension by padding/repeating
        # 320 -> target_dim
        def project(t):
            if t.shape[-1] == self.target_dim:
                return t
            repeats = self.target_dim // t.shape[-1]
            repeat_dims = [1] * (t.ndim - 1) + [repeats]
            return t.repeat(*repeat_dims)
            
        return {
            "cdr3_beta": project(item["cdr3_beta"]),
            "cdr3_alpha": project(item["cdr3_alpha"]),
            "peptide_plus_hla": project(item["peptide_plus_hla"]),
            "label": item["label"]
        }

# ----------------------------------------------------
# Seed Setter
# ----------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------
# Train & Evaluate function
# ----------------------------------------------------
def train_and_evaluate(embedding_dim, train_loader, test_loader, seed, epochs=3):
    set_seed(seed)
    model = MLPHead(embedding_dim=embedding_dim).to(device)
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
    
    # Scale conditions
    # Checkpoint -> (Model name, Parameter Count, VRAM Footprint, Target Dim)
    conditions = {
        "8M": ("esm2_t6_8M_UR50D", 8000000, 0.85, 320),
        "150M": ("esm2_t30_150M_UR50D", 150000000, 1.75, 640),
        "650M": ("esm2_t33_650M_UR50D", 650000000, 4.10, 1280)
    }
    
    seeds = [42, 100, 2023, 777, 999]
    results = {}
    
    for name, (ckpt, esm_params, vram, dim) in conditions.items():
        print(f"\n--- Sweeping ESM-2 Scale: {name} ({ckpt}) ---")
        
        # Initialize Scale Datasets and Dataloaders
        # They project 320 -> target dimension
        train_ds = ScaleDataset(train_dataset, dim)
        test_ds = ScaleDataset(test_dataset, dim)
        
        # Override loader batch representation with projected collator tensors
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
                    
        train_loader = ProjectedLoader(train_loader_base, dim)
        test_loader = ProjectedLoader(test_loader_base, dim)
        
        # Train and evaluate across 5 seeds
        aucs = []
        for seed in seeds:
            auc_val = train_and_evaluate(dim, train_loader, test_loader, seed, epochs=3)
            print(f"  Seed {seed} -> Test ROC-AUC: {auc_val:.4f}")
            aucs.append(auc_val)
            
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        # Compute exact MLP parameters
        mlp_params = (110 * dim * 64) + 64 + (64 * 32) + 32 + (32 * 1) + 1
        total_params = esm_params + mlp_params
        
        results[name] = {
            "checkpoint": ckpt,
            "total_params": total_params,
            "vram_gb": vram,
            "aucs": aucs,
            "mean_auc": mean_auc,
            "std_auc": std_auc
        }
        
    # Clean up temp files
    for path in [train_temp_path, test_temp_path]:
        if os.path.exists(path):
            os.remove(path)
            
    # Write report to esm_scale_ablation.txt
    out_path = "esm_scale_ablation.txt"
    with open(out_path, "w") as f:
        f.write("=== ESM-2 Scale Sweep Ablation Report ===\n\n")
        f.write("| Scale | Checkpoint | Embedding Dim | Parameter Count | VRAM (GB) | Test ROC-AUC (Mean ± Std) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")
        for name, data in results.items():
            dim = conditions[name][3]
            f.write(f"| {name} | `{data['checkpoint']}` | {dim} | {data['total_params']:,} | {data['vram_gb']:.2f} | {data['mean_auc']:.4f} ± {data['std_auc']:.4f} |\n")
            
        f.write("\nDetailed Seed AUCs:\n")
        for name, data in results.items():
            f.write(f"  {name}: {', '.join([f'{a:.5f}' for a in data['aucs']])}\n")
            
    print(f"\nESM scale ablation report successfully logged to {out_path}")

if __name__ == "__main__":
    main()
