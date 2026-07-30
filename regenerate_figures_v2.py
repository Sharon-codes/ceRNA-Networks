import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Add workspace to path
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
# Models for Figure 4 Activation Extraction
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
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    tensor_normalized = tensor / norm
    sim_matrix = torch.bmm(tensor_normalized, tensor_normalized.transpose(1, 2))
    B, L, _ = sim_matrix.shape
    mask = ~torch.eye(L, dtype=torch.bool, device=tensor.device)
    mask = mask.unsqueeze(0).expand(B, L, L)
    return sim_matrix[mask].cpu().numpy()

# ----------------------------------------------------
# Main Plotting script
# ----------------------------------------------------
def main():
    os.makedirs("./images", exist_ok=True)
    
    # ----------------------------------------------------
    # Plot Figure 1: Architecture Block Diagram
    # ----------------------------------------------------
    print("Plotting Figure 1 (Architecture Block Diagram)...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    color_grey = '#F2F2F2'
    color_red = '#FFE5E5'
    color_green = '#E5FFE5'
    border_grey = '#CCCCCC'
    border_red = '#FF9999'
    border_green = '#99FF99'
    
    title_font = {'weight': 'bold', 'size': 13}
    text_font = {'size': 11, 'ha': 'center', 'va': 'center'}
    
    # Column A: Flawed Standard
    ax.text(2.5, 9.6, "A. Flawed Standard (Representation Collapse)", title_font, ha='center', color='#800000')
    
    ax.text(2.5, 8.2, "Frozen ESM-2 Tensors\nTCR & Peptide-HLA\n(320-dim Embeddings)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_grey, ec=border_grey, lw=1.5))
            
    ax.text(2.5, 5.2, "Transformer/Mamba Encoders\n(Feature Homogenization /\nReduced Token Diversity)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_red, ec=border_red, lw=1.5))
            
    ax.text(2.5, 2.2, "Cross Attention\n(Softmax Flattening &\nGradient Attenuation)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_red, ec=border_red, lw=1.5))
            
    # Arrows A
    ax.annotate('', xy=(2.5, 6.0), xytext=(2.5, 7.3),
                arrowprops=dict(arrowstyle="-|>", color='#CC0000', lw=2, mutation_scale=15))
    ax.annotate('', xy=(2.5, 3.0), xytext=(2.5, 4.4),
                arrowprops=dict(arrowstyle="-|>", color='#CC0000', lw=2, mutation_scale=15))
                
    # Column B: Our Framework
    ax.text(7.5, 9.6, "B. Our Framework (Direct Concatenation)", title_font, ha='center', color='#006600')
    
    ax.text(7.5, 8.6, "Frozen ESM-2 Tensors\nTCR & Peptide-HLA\n(320-dim Embeddings)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_grey, ec=border_grey, lw=1.5))
            
    ax.text(7.5, 6.8, "Linear Projection\n(Dimension reduction to d_model)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    ax.text(7.5, 5.0, "Direct Concatenation\n(Preserve spatial sequence structures)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    ax.text(7.5, 3.2, "MLP Position-Weight Matrix\n(Implicit spatial positioning & prediction)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    ax.text(7.5, 1.4, "Binding Probability\n(No sigmoid/LN bottleneck)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    # Arrows B
    ax.annotate('', xy=(7.5, 7.5), xytext=(7.5, 8.0),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
    ax.annotate('', xy=(7.5, 5.7), xytext=(7.5, 6.2),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
    ax.annotate('', xy=(7.5, 3.9), xytext=(7.5, 4.4),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
    ax.annotate('', xy=(7.5, 2.1), xytext=(7.5, 2.6),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
                
    # Divider line
    ax.plot([5.0, 5.0], [0.5, 9.8], color='#999999', linestyle='--', lw=1.5)
    
    plt.tight_layout()
    plt.savefig("./images/Figure1_Architecture_v2.png", dpi=300, bbox_inches='tight')
    plt.savefig("./images/Figure1_Architecture_v2.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Successfully saved Figure 1 v2.")

    # ----------------------------------------------------
    # Plot Figure 4: Quantitative Oversmoothing Analysis
    # ----------------------------------------------------
    print("\nPlotting Figure 4 (Oversmoothing Quantification)...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset & dataloaders
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
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_collate)
    
    # Train tracking model dynamically
    model = TrackingCrossAttentionMambaTCR().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
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
            
    # Extract test batch activations
    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        cdr3_beta = batch["cdr3_beta"].to(device)
        cdr3_alpha = batch["cdr3_alpha"].to(device)
        peptide_plus_hla = batch["peptide_plus_hla"].to(device)
        _, tensor_a, tensor_b, tensor_c = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
        
    sims_a = compute_pairwise_cosine_similarities(tensor_a)
    sims_b = compute_pairwise_cosine_similarities(tensor_b)
    
    # Plotting Figure 4
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
    
    # Subplot B: Softmax Weight Distribution
    attn_flat = tensor_c.cpu().numpy().flatten()
    if len(attn_flat) > 100000:
        attn_flat = np.random.choice(attn_flat, size=100000, replace=False)
        
    L_k = tensor_c.size(-1)
    uniform_point = 1.0 / L_k
    
    ax2.hist(attn_flat, bins=100, color="#55a868", alpha=0.85, edgecolor='none', density=True)
    ax2.axvline(uniform_point, color="#c44e52", linestyle="--", lw=2.5, label=f"Uniform Collapse (1/L = {uniform_point:.2f})")
    ax2.set_xlabel("Attention Weights", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
    ax2.set_title("B. Cross-Attention Softmax Weight Distribution", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig("./images/Figure_4_Oversmoothing_Quant_v2.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("./images/Figure_4_Oversmoothing_Quant_v2.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully saved Figure 4 v2.")

if __name__ == "__main__":
    main()
