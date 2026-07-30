import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, max_len, d_model]
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tcr_encoded, pep_hla_encoded):
        # Shapes: tcr_encoded [B, L_q, D], pep_hla_encoded [B, L_k, D]
        B, L_q, D = tcr_encoded.shape
        _, L_k, _ = pep_hla_encoded.shape
        
        q = self.q_proj(self.q_norm(tcr_encoded)) # [B, L_q, D]
        k = self.k_proj(self.k_norm(pep_hla_encoded)) # [B, L_k, D]
        v = self.v_proj(self.k_norm(pep_hla_encoded)) # [B, L_k, D]
        
        # Split into multiple heads: [B, nhead, L, head_dim]
        q = q.view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        
        # Dot product scores: [B, nhead, L_q, L_k]
        scores = torch.matmul(q, k.transpose(-2, -1))
        # Note: We do NOT scale by 1 / sqrt(head_dim) here to avoid the uniform attention flatline
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum: [B, nhead, L_q, head_dim]
        attn_out = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project out: [B, L_q, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, D)
        attn_out = self.out_proj(attn_out)
        
        return self.norm(tcr_encoded + attn_out)

class ProjectionHead(nn.Module):
    def __init__(self, d_model, seq_len=61, dropout=0.3):
        super().__init__()
        self.flatten_dim = d_model * seq_len
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x):
        x_flat = x.reshape(x.shape[0], -1)
        return self.mlp(x_flat)
