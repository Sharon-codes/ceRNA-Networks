import torch
import torch.nn as nn
from modules import PositionalEncoding, CrossAttentionFusion, ProjectionHead

class MambaTCR(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=2):
        """
        ESM-MambaTCR Direct Concatenation Model:
        - ESM-2 Embeddings: Extract rich contextual representations using facebook/esm2_t6_8M_UR50D.
        - TCR and Peptide-HLA projections: Map raw 320-dimensional embeddings to d_model.
        - Direct Concatenation: Concat TCR beta, TCR alpha, and Peptide-HLA sequence embeddings directly.
        - Projection Head: Flattened MLP logit output.
        """
        super().__init__()
        self.d_model = d_model
        
        # ESM-2 8M hidden dimension is 320. Project to d_model.
        self.esm_proj = nn.Linear(320, d_model)
        self.pephla_proj = nn.Linear(320, d_model)
        
        # Flattened sequence length: 30 (beta) + 30 (alpha) + 50 (pep) = 110 tokens
        self.projection_head = ProjectionHead(d_model, seq_len=110)
        
    def forward(self, cdr3_beta, cdr3_alpha, peptide_plus_hla, hla_allele=None):
        # Shape verification checks
        batch_size = cdr3_beta.shape[0]
        assert cdr3_beta.shape[1] == 30, f"Expected cdr3_beta length 30, got {cdr3_beta.shape[1]}"
        assert cdr3_alpha.shape[1] == 30, f"Expected cdr3_alpha length 30, got {cdr3_alpha.shape[1]}"
        assert peptide_plus_hla.shape[1] == 50, f"Expected peptide_plus_hla length 50, got {peptide_plus_hla.shape[1]}"
        
        # Project representations to d_model
        beta_proj = self.esm_proj(cdr3_beta)    # [batch, 30, d_model]
        alpha_proj = self.esm_proj(cdr3_alpha)  # [batch, 30, d_model]
        pephla_proj = self.pephla_proj(peptide_plus_hla)  # [batch, 50, d_model]
        
        # Direct Concatenation: [beta; alpha; pep] (resulting sequence length: 110)
        fused = torch.cat([beta_proj, alpha_proj, pephla_proj], dim=1) # [batch, 110, d_model]
        
        # Projection to Logit
        logit = self.projection_head(fused) # [batch, 1]
        return logit.squeeze(-1) # [batch]
