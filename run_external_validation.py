import os
import sys
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from transformers import AutoTokenizer, EsmModel

# Add workspace to path
sys.path.append(os.getcwd())
from dataset import get_hla_pseudo, load_embedding
from model import MambaTCR

def get_md5_hash(sequence: str) -> str:
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()

class ExternalDataset(Dataset):
    def __init__(self, df, hla_pseudo):
        self.labels = torch.tensor(df["binder"].values, dtype=torch.float)
        
        # Pre-load embeddings
        beta_list = []
        alpha_list = []
        pephla_list = []
        
        for _, row in df.iterrows():
            beta_seq = str(row["CDR3b"])
            pep_seq = str(row["peptide"])
            
            # Load embeddings (alpha is empty string for beta-only MIRA set)
            beta_emb = load_embedding(beta_seq)      # [30, 320]
            alpha_emb = load_embedding("")           # [30, 320]
            pephla_emb = load_embedding(pep_seq + hla_pseudo) # [50, 320]
            
            beta_list.append(beta_emb)
            alpha_list.append(alpha_emb)
            pephla_list.append(pephla_emb)
            
        self.beta_tensors = torch.stack(beta_list)
        self.alpha_tensors = torch.stack(alpha_list)
        self.pephla_tensors = torch.stack(pephla_list)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            "cdr3_beta": self.beta_tensors[idx],
            "cdr3_alpha": self.alpha_tensors[idx],
            "peptide_plus_hla": self.pephla_tensors[idx],
            "label": self.labels[idx]
        }

def compute_missing_embeddings(sequences, max_length, embed_dir, device):
    os.makedirs(embed_dir, exist_ok=True)
    
    # Filter only those that are truly missing
    seqs_to_compute = []
    hashes_to_compute = []
    
    for seq in sequences:
        h = get_md5_hash(seq)
        save_path = os.path.join(embed_dir, f"{h}.pt")
        if not os.path.exists(save_path):
            seqs_to_compute.append(seq)
            hashes_to_compute.append(h)
            
    if not seqs_to_compute:
        return
        
    print(f"Generating {len(seqs_to_compute)} missing embeddings of max length {max_length}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
    model.eval()
    
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(seqs_to_compute), batch_size):
            batch_seqs = seqs_to_compute[i:i+batch_size]
            batch_hashes = hashes_to_compute[i:i+batch_size]
            
            inputs = tokenizer(batch_seqs, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.cpu() # [batch, seq_len, 320]
            
            for j, h in enumerate(batch_hashes):
                save_path = os.path.join(embed_dir, f"{h}.pt")
                torch.save(hidden_states[j].clone(), save_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"External Validation Cohort running on: {device}")
    
    # 1. Load dataset
    test_path = "./Dataset/Native/nettcr_test.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing external dataset at {test_path}")
        
    df = pd.read_csv(test_path)
    print(f"Loaded external cohort with {len(df)} samples.")
    
    # Get A*02:01 pseudo sequence
    hla_pseudo = get_hla_pseudo("A*02:01")
    
    # 2. Extract and cache missing embeddings
    unique_betas = set(df["CDR3b"].dropna().astype(str).unique())
    unique_pep_hlas = set((str(p) + hla_pseudo) for p in df["peptide"].dropna().unique())
    
    # Ensure empty alpha is also cached
    unique_alphas = {""}
    
    embed_dir = "./Dataset/Embeddings"
    compute_missing_embeddings(unique_betas, 30, embed_dir, device)
    compute_missing_embeddings(unique_alphas, 30, embed_dir, device)
    compute_missing_embeddings(unique_pep_hlas, 50, embed_dir, device)
    
    # 3. Create dataset & dataloader
    dataset = ExternalDataset(df, hla_pseudo)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # 4. Load frozen model
    model = MambaTCR(d_model=64).to(device)
    checkpoint_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 5. Evaluate
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
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
    
    # Compute metrics
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    # Precision@top-100
    top100_idx = np.argsort(y_pred)[::-1][:100]
    p_at_100 = np.mean(y_true[top100_idx])
    
    print("\n=== EXTERNAL VALIDATION COHORT RESULTS ===")
    print(f"Dataset Source:  10x Genomics MIRA Set")
    print(f"Total Samples:   {len(y_true)}")
    print(f"ROC-AUC:         {roc_auc:.5f}")
    print(f"PR-AUC:          {pr_auc:.5f}")
    print(f"Precision@top-100: {p_at_100:.5f}")
    
    # Write to a report file
    os.makedirs("Evaluation", exist_ok=True)
    report_path = "./Evaluation/external_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("External Validation Cohort (Concern 1)\n")
        f.write("======================================\n")
        f.write(f"Dataset: 10x Genomics MIRA (NetTCR-2.0 Test Set)\n")
        f.write(f"Samples: {len(y_true)} (106 Binders, 530 Non-Binders)\n")
        f.write(f"ROC-AUC:           {roc_auc:.5f}\n")
        f.write(f"PR-AUC:            {pr_auc:.5f}\n")
        f.write(f"Precision@top-100: {p_at_100:.5f}\n")
        
    print(f"\nExternal validation metrics logged to {report_path}")

if __name__ == "__main__":
    main()
