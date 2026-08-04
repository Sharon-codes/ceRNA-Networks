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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())
from dataset import TCRDataset, TCRCollate, build_global_pool
from modules import PositionalEncoding, ProjectionHead

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------
# Modified Cross-Attention models to extract intermediate activations
# ----------------------------------------------------
class TrackingCrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tcr_encoded, pep_hla_encoded):
        B, L_q, D = tcr_encoded.shape
        _, L_k, _ = pep_hla_encoded.shape
        
        q = self.q_proj(self.q_norm(tcr_encoded))
        k = self.k_proj(self.k_norm(pep_hla_encoded))
        v = self.v_proj(self.k_norm(pep_hla_encoded))
        
        q = q.view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, D)
        attn_out = self.out_proj(attn_out)
        
        fused = self.norm(tcr_encoded + attn_out)
        return fused, attn_weights

class TrackingCrossAttentionMambaTCR(nn.Module):
    def __init__(self, d_model=64, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.esm_proj = nn.Linear(320, d_model)
        self.pephla_proj = nn.Linear(320, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=50)
        self.cross_attention = TrackingCrossAttentionFusion(d_model, nhead=nhead)
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
        
        fused, attn_weights = self.cross_attention(tcr_seq, pephla_proj)
        logit = self.projection_head(fused)
        return logit.squeeze(-1), tcr_seq, fused, attn_weights

def compute_pairwise_cosine_similarities(tensor):
    # tensor shape: [B, L, D]
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    tensor_normalized = tensor / norm
    
    # Batch matrix multiplication: [B, L, D] x [B, D, L] -> [B, L, L]
    sim_matrix = torch.bmm(tensor_normalized, tensor_normalized.transpose(1, 2))
    
    B, L, _ = sim_matrix.shape
    # Mask to select off-diagonal elements
    mask = ~torch.eye(L, dtype=torch.bool, device=tensor.device)
    mask = mask.unsqueeze(0).expand(B, L, L)
    
    return sim_matrix[mask].cpu().numpy()

def main():
    set_seed(42)
    device = torch.device("cpu")
    print("Loading data...")
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
    # Using larger batch size to extract representative activation statistics
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_collate)
    
    # ----------------------------------------------------
    # Step 1: Train the Baseline model dynamically
    # ----------------------------------------------------
    print("\n--- Training Baseline Cross-Attention Model (5 epochs) ---")
    model = TrackingCrossAttentionMambaTCR(d_model=64, nhead=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            cdr3_beta = batch["cdr3_beta"].to(device)
            cdr3_alpha = batch["cdr3_alpha"].to(device)
            peptide_plus_hla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits, _, _, _ = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
        
    # ----------------------------------------------------
    # Step 2: Extract Activations from Test Set Batch
    # ----------------------------------------------------
    print("\n--- Extracting Activations from Test Batch ---")
    model.eval()
    
    # Retrieve a single batch of size 128
    batch = next(iter(test_loader))
    with torch.no_grad():
        cdr3_beta = batch["cdr3_beta"].to(device)
        cdr3_alpha = batch["cdr3_alpha"].to(device)
        peptide_plus_hla = batch["peptide_plus_hla"].to(device)
        
        _, tensor_a, tensor_b, tensor_c = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
        
    print(f"Tensor A (Pre-Attention) shape:  {tensor_a.shape}")
    print(f"Tensor B (Post-Attention) shape: {tensor_b.shape}")
    print(f"Tensor C (Attention Weights) shape: {tensor_c.shape}")
    
    # ----------------------------------------------------
    # Step 3: Metric Calculations
    # ----------------------------------------------------
    print("\n--- Computing Metrics ---")
    sims_a = compute_pairwise_cosine_similarities(tensor_a)
    sims_b = compute_pairwise_cosine_similarities(tensor_b)
    
    mean_sim_a = sims_a.mean()
    mean_sim_b = sims_b.mean()
    std_sim_a = sims_a.std()
    std_sim_b = sims_b.std()
    
    # Variance of attention weights along the key dimension (dim=-1)
    attn_var = torch.var(tensor_c, dim=-1).mean().item()
    N_pairs = len(sims_a)
    
    print(f"Total Token Pairs Compared (N):           {N_pairs}")
    print(f"Pre-Attention Pairwise Cosine Similarity:  {mean_sim_a:.4f} \u00b1 {std_sim_a:.4f}")
    print(f"Post-Attention Pairwise Cosine Similarity: {mean_sim_b:.4f} \u00b1 {std_sim_b:.4f}")
    print(f"Mean Attention Variance:                  {attn_var:.6f}")
    
    # Log these exact statistics to a text file
    with open("oversmoothing_metrics.log", "w") as log_f:
        log_f.write("=== REPRESENTATION OVERSMOOTHING QUANTITATIVE ANALYSIS ===\n")
        log_f.write(f"Total Token Pairs Compared (N):           {N_pairs}\n")
        log_f.write(f"Pre-Attention Mean Cosine Similarity:  {mean_sim_a:.6f} \u00b1 {std_sim_a:.6f}\n")
        log_f.write(f"Post-Attention Mean Cosine Similarity: {mean_sim_b:.6f} \u00b1 {std_sim_b:.6f}\n")
        log_f.write(f"Mean Attention Variance:                  {attn_var:.8f}\n")
        log_f.write(f"Sequence Length (Query L_q):            {tensor_a.size(1)}\n")
        log_f.write(f"Sequence Length (Key L_k):              {tensor_c.size(-1)}\n")
        
    # ----------------------------------------------------
    # Step 4: Figure Generation
    # ----------------------------------------------------
    print("\n--- Plotting Figure 4 (Oversmoothing Quantification) ---")
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Subplot A: Token Homogenization (Pairwise Cosine Similarities)
    sns.kdeplot(sims_a, fill=True, color="#4c72b0", label="Pre-Attention", lw=2.5, ax=ax1)
    sns.kdeplot(sims_b, fill=True, color="#c44e52", label="Post-Attention", lw=2.5, ax=ax1)
    ax1.set_xlim([-1.05, 1.05])
    ax1.set_xlabel("Pairwise Cosine Similarity", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
    ax1.set_title("A. Token Homogenization Distribution", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper left", frameon=True)
    
    # Subplot B: Softmax Collapse (Attention Weights Histogram)
    attn_flat = tensor_c.cpu().numpy().flatten()
    # Subsample to keep plotting fast and responsive
    if len(attn_flat) > 100000:
        attn_flat = np.random.choice(attn_flat, size=100000, replace=False)
        
    # Sequence length L
    L_k = tensor_c.size(-1)
    uniform_point = 1.0 / L_k
    
    ax2.hist(attn_flat, bins=100, color="#55a868", alpha=0.85, edgecolor='none', density=True)
    ax2.axvline(uniform_point, color="#c44e52", linestyle="--", lw=2.5, label=f"Uniform Collapse (1/L = {uniform_point:.2f})")
    ax2.set_xlabel("Attention Weights", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
    ax2.set_title("B. Cross-Attention Softmax Collapse", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    os.makedirs("./images", exist_ok=True)
    plt.savefig("./images/Figure_4_Oversmoothing_Quant.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./images/Figure_4_Oversmoothing_Quant.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure successfully saved to ./images/Figure_4_Oversmoothing_Quant.pdf and .png!")

if __name__ == "__main__":
    main()
