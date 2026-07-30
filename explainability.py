import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from model import MambaTCR
from dataset import load_embedding, get_hla_pseudo

def main():
    device = torch.device("cpu")
    print(f"Using device for XAI: {device}")

    # Create directories
    os.makedirs("./Evaluation/XAI", exist_ok=True)

    # 1. Load trained model
    print("Loading model...")
    model = MambaTCR(d_model=64, nhead=8, num_layers=2).to(device)
    
    # Try production checkpoint first, then fallback to standard checkpoint
    checkpoint_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if not os.path.exists(checkpoint_path):
        fallback_path = "./Checkpoints/best_mamba_tcr.pt"
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(f"Checkpoint not found at either {checkpoint_path} or {fallback_path}")
        print(f"Production checkpoint not found. Falling back to standard checkpoint: {fallback_path}")
        checkpoint_path = fallback_path
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Load test set
    print("Loading test set...")
    test_csv = "./Processed/test.csv"
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split not found at {test_csv}")
    
    df_test = pd.read_csv(test_csv)
    
    # To identify the top 50 highest-confidence true positive pairings, we first evaluate the model
    # on all test pairings (all of which are positive in the dataset; the collator is what creates negatives).
    print("Scoring the entire test set to select top 50 highest-confidence true positive pairings...")
    all_probs = []
    batch_size = 64
    num_samples = len(df_test)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Scoring Test Set"):
            batch_df = df_test.iloc[i : i + batch_size]
            
            beta_list = []
            alpha_list = []
            pephla_list = []
            
            for _, row in batch_df.iterrows():
                raw_beta = str(row["cdr3_beta"])
                raw_alpha = str(row["cdr3_alpha"]) if pd.notna(row["cdr3_alpha"]) and row["cdr3_alpha"] is not None else ""
                raw_peptide = str(row["peptide"])
                raw_hla = str(row["hla_allele"])
                hla_pseudo = get_hla_pseudo(raw_hla)
                pephla_str = raw_peptide + hla_pseudo
                
                beta_list.append(load_embedding(raw_beta))
                alpha_list.append(load_embedding(raw_alpha))
                pephla_list.append(load_embedding(pephla_str))
                
            beta_batch = torch.stack(beta_list).to(device)
            alpha_batch = torch.stack(alpha_list).to(device)
            pephla_batch = torch.stack(pephla_list).to(device)
            
            logits = model(beta_batch, alpha_batch, pephla_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            
    df_test["predicted_prob"] = all_probs
    
    # Select top 50
    df_top50 = df_test.sort_values(by="predicted_prob", ascending=False).head(50).copy()
    df_top50 = df_top50.reset_index(drop=True)
    print(f"Selected top 50 true positives. Confidence range: {df_top50['predicted_prob'].min():.4f} - {df_top50['predicted_prob'].max():.4f}")
    
    # 3. Perform Sliding Window Occlusion (W=1 and W=3)
    results = []
    
    with torch.no_grad():
        for idx, row in tqdm(df_top50.iterrows(), total=len(df_top50), desc="Occlusion sensitivity mapping"):
            raw_beta = str(row["cdr3_beta"])
            raw_alpha = str(row["cdr3_alpha"]) if pd.notna(row["cdr3_alpha"]) and row["cdr3_alpha"] is not None else ""
            raw_peptide = str(row["peptide"])
            raw_hla = str(row["hla_allele"])
            disease = str(row["disease"]) if "disease" in row else "Unknown"
            baseline_prob = row["predicted_prob"]
            
            hla_pseudo = get_hla_pseudo(raw_hla)
            pephla_str = raw_peptide + hla_pseudo
            
            # Load original embeddings (without batch dimension)
            beta_emb = load_embedding(raw_beta).to(device)        # [30, 320]
            alpha_emb = load_embedding(raw_alpha).to(device)      # [30, 320]
            pephla_emb = load_embedding(pephla_str).to(device)    # [50, 320]
            
            # String lengths for pad skipping (clipped to maximum tensor dimensions)
            L_beta = min(len(raw_beta), 30)
            L_alpha = min(len(raw_alpha), 30)
            L_pephla = min(len(pephla_str), 50)
            
            # Keep track of scores for each window size
            scores = {
                "beta_scores_w1": [0.0] * L_beta,
                "beta_scores_w3": [0.0] * L_beta,
                "alpha_scores_w1": [0.0] * L_alpha,
                "alpha_scores_w3": [0.0] * L_alpha,
                "pephla_scores_w1": [0.0] * L_pephla,
                "pephla_scores_w3": [0.0] * L_pephla,
            }
            
            for W in [1, 3]:
                # A. Occlude TCR Beta branch
                for i in range(L_beta):
                    indices = [j for j in range(i - W // 2, i + W // 2 + 1) if 0 <= j < L_beta]
                    beta_emb_occ = beta_emb.clone()
                    beta_emb_occ[indices, :] = 0.0
                    
                    # Run inference (add batch dimension)
                    logits = model(beta_emb_occ.unsqueeze(0), alpha_emb.unsqueeze(0), pephla_emb.unsqueeze(0))
                    prob_occ = torch.sigmoid(logits).item()
                    scores[f"beta_scores_w{W}"][i] = baseline_prob - prob_occ
                    
                # B. Occlude TCR Alpha branch
                for i in range(L_alpha):
                    indices = [j for j in range(i - W // 2, i + W // 2 + 1) if 0 <= j < L_alpha]
                    alpha_emb_occ = alpha_emb.clone()
                    alpha_emb_occ[indices, :] = 0.0
                    
                    logits = model(beta_emb.unsqueeze(0), alpha_emb_occ.unsqueeze(0), pephla_emb.unsqueeze(0))
                    prob_occ = torch.sigmoid(logits).item()
                    scores[f"alpha_scores_w{W}"][i] = baseline_prob - prob_occ
                    
                # C. Occlude Peptide-HLA branch
                for i in range(L_pephla):
                    indices = [j for j in range(i - W // 2, i + W // 2 + 1) if 0 <= j < L_pephla]
                    pephla_emb_occ = pephla_emb.clone()
                    pephla_emb_occ[indices, :] = 0.0
                    
                    logits = model(beta_emb.unsqueeze(0), alpha_emb.unsqueeze(0), pephla_emb_occ.unsqueeze(0))
                    prob_occ = torch.sigmoid(logits).item()
                    scores[f"pephla_scores_w{W}"][i] = baseline_prob - prob_occ
            
            entry = {
                "rank": idx + 1,
                "cdr3_beta": raw_beta,
                "cdr3_alpha": raw_alpha,
                "peptide": raw_peptide,
                "hla_allele": raw_hla,
                "disease": disease,
                "baseline_prob": baseline_prob,
                
                # Valid residues/characters
                "beta_chars": list(raw_beta)[:L_beta],
                "alpha_chars": list(raw_alpha)[:L_alpha],
                "pephla_chars": list(pephla_str)[:L_pephla],
                
                # Scores
                **scores
            }
            results.append(entry)
            
    # Save scores
    out_path = "./Evaluation/XAI/occlusion_scores.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Successfully saved occlusion scores for 50 samples to {out_path}")

if __name__ == "__main__":
    main()
