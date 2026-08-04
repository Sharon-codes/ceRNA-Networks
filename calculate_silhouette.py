import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

# Add workspace to path
sys.path.append(os.getcwd())
from dataset import get_hla_pseudo, load_embedding
from model import MambaTCR

# Standard HLA Class I Supertype mapping logic (Sette & Sidney, 1999; Sidney et al., 2008)
def get_supertype(allele):
    allele = allele.upper().replace("HLA-", "").strip()
    if allele.startswith("A*") or allele.startswith("A"):
        num_part = allele.replace("A*", "").replace("A", "")
        digits = "".join([c for c in num_part if c.isdigit()])[:2]
        if not digits:
            return None
        if digits in ["02", "68", "69"]:
            if num_part.startswith("68:01"):
                return "A03"
            return "A02"
        elif digits in ["01", "25", "26", "29", "30", "32", "34", "36", "43", "80"]:
            return "A01"
        elif digits in ["03", "11", "31", "33", "66", "68"]:
            return "A03"
        elif digits in ["23", "24", "30"]:
            return "A24"
    elif allele.startswith("B*") or allele.startswith("B"):
        num_part = allele.replace("B*", "").replace("B", "")
        digits = "".join([c for c in num_part if c.isdigit()])[:2]
        if not digits:
            return None
        if digits in ["07", "35", "51", "53", "54", "55", "56", "67", "78", "42"]:
            return "B07"
        elif digits in ["08"]:
            return "B08"
        elif digits in ["15", "27", "38", "39", "48", "73"]:
            if num_part.startswith("15:02"):
                return "B62"
            return "B27"
        elif digits in ["18", "37", "40", "41", "44", "45", "47", "49", "50"]:
            return "B44"
        elif digits in ["57", "58"]:
            return "B58"
        elif digits in ["15", "46", "52"]:
            return "B62"
    return None

def main():
    device = torch.device("cpu")
    print("Loading model and projection weights...")
    model = MambaTCR(d_model=64, nhead=8, num_layers=2).to(device)
    checkpoint_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load all datasets to find unique HLA alleles and their pairings
    print("Loading datasets...")
    train_df = pd.read_csv("./Processed/train.csv")
    val_df = pd.read_csv("./Processed/val.csv")
    test_df = pd.read_csv("./Processed/test.csv")
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
    
    # Group by hla_allele to get a representative pairing for each unique allele
    unique_hla_rows = df_all.groupby("hla_allele").first().reset_index()
    print(f"Found {len(unique_hla_rows)} unique HLA alleles in the dataset.")
    
    raw_embeddings = []
    latent_embeddings = []
    supertypes = []
    valid_alleles = []
    
    # Iterate over unique HLA alleles to extract embeddings
    for _, row in unique_hla_rows.iterrows():
        allele = str(row["hla_allele"])
        peptide = str(row["peptide"])
        
        supertype = get_supertype(allele)
        if supertype is None:
            continue
            
        try:
            # Reconstruct the peptide-HLA string used in collation
            hla_pseudo = get_hla_pseudo(allele)
            pephla_str = peptide + hla_pseudo
            
            # Load pre-computed ESM-2 embedding
            emb = load_embedding(pephla_str)  # [50, 320]
            
            # Slice HLA portion (from L_peptide to L_peptide + 34)
            L_pep = len(peptide)
            hla_emb = emb[L_pep : L_pep + 34, :]  # [34, 320]
            mean_pooled = hla_emb.mean(dim=0)     # [320]
            
            # Project using model's linear layer
            with torch.no_grad():
                projected = model.pephla_proj(mean_pooled.to(device))  # [64]
                
            raw_embeddings.append(mean_pooled.numpy())
            latent_embeddings.append(projected.numpy())
            supertypes.append(supertype)
            valid_alleles.append(allele)
        except Exception as e:
            # Print warning if any fails
            print(f"Warning: Failed to process allele {allele}: {e}")
            
    raw_embeddings = np.array(raw_embeddings)
    latent_embeddings = np.array(latent_embeddings)
    supertypes = np.array(supertypes)
    
    print(f"\nSuccessfully extracted and mapped {len(supertypes)} HLA alleles to Class I supertypes.")
    print("Supertypes distribution:")
    unique_supertypes, counts = np.unique(supertypes, return_counts=True)
    for s, c in zip(unique_supertypes, counts):
        print(f"  {s}: {c}")
        
    # Calculate Silhouette Scores
    if len(unique_supertypes) > 1:
        sil_raw = silhouette_score(raw_embeddings, supertypes)
        sil_latent = silhouette_score(latent_embeddings, supertypes)
        
        print("\n=== SILHOUETTE SCORE CLUSTERING PROOF ===")
        print(f"Raw ESM-2 Mean-Pooled (320-dim) Silhouette Score:   {sil_raw:.4f}")
        print(f"Projected Latent Space (64-dim) Silhouette Score:   {sil_latent:.4f}")
        
        # Log to file
        out_path = "./Evaluation/silhouette_scores.txt"
        with open(out_path, "w") as f:
            f.write("HLA Supertype Silhouette Score Analysis\n")
            f.write("=======================================\n")
            f.write(f"Number of mapped alleles: {len(supertypes)}\n")
            f.write(f"Raw ESM-2 (320-dim) Silhouette Score:   {sil_raw:.5f}\n")
            f.write(f"Projected Latent (64-dim) Silhouette Score:   {sil_latent:.5f}\n")
            f.write("\nSupertypes distribution:\n")
            for s, c in zip(unique_supertypes, counts):
                f.write(f"  {s}: {c}\n")
                
        print(f"\nSilhouette scores logged successfully to {out_path}")
    else:
        print("Error: Not enough classes to compute silhouette score.")

if __name__ == "__main__":
    main()
