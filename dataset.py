import os
import random
import logging
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import hashlib

logger = logging.getLogger(__name__)

# Global HLA Pseudo-sequence map
HLA_PSEUDO_MAP = {}
pseudo_map_path = os.path.join(os.path.dirname(__file__), "Processed", "hla_pseudo_map.json")
if os.path.exists(pseudo_map_path):
    try:
        with open(pseudo_map_path, "r") as f:
            HLA_PSEUDO_MAP = json.load(f)
        logger.info(f"Loaded unified HLA pseudo-sequence map with {len(HLA_PSEUDO_MAP)} entries.")
    except Exception as e:
        logger.error(f"Error loading hla_pseudo_map.json: {e}")
else:
    logger.warning("hla_pseudo_map.json not found. Locus fallbacks will be used.")

# Hard negatives mapping removed

def get_md5_hash(sequence: str) -> str:
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()

embed_dir = os.path.join(os.path.dirname(__file__), "Dataset", "Embeddings")

EMBEDDING_CACHE = {}
archive_path = os.path.join(os.path.dirname(__file__), "Dataset", "all_embeddings.pt")
if os.path.exists(archive_path):
    try:
        print(f"Loading pre-computed embedding archive from {archive_path}...")
        EMBEDDING_CACHE = torch.load(archive_path, weights_only=True)
        print(f"SUCCESS: Loaded {len(EMBEDDING_CACHE)} embeddings from archive.")
    except Exception as e:
        print(f"ERROR: Failed to load embedding archive: {e}")
else:
    print(f"WARNING: Embedding archive not found at {archive_path}!")

def load_embedding(sequence: str):
    h = get_md5_hash(sequence)
    if h in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[h]
        
    path = os.path.join(embed_dir, f"{h}.pt")
    if os.path.exists(path):
        # We explicitly use weights_only=True for safety where supported, but backwards compatible otherwise
        try:
            tensor = torch.load(path, weights_only=True)
        except TypeError:
            tensor = torch.load(path)
        EMBEDDING_CACHE[h] = tensor
        return tensor
    else:
        raise FileNotFoundError(f"Missing pre-computed embedding for sequence (hash: {h}): {path}")

def get_hla_pseudo(allele):
    """
    Translates an HLA allele name into its 34-amino-acid pseudo-sequence.
    Applies direct matches, cleaned keys, or locus-specific consensus sequences.
    """
    if not isinstance(allele, str) or pd.isna(allele):
        allele = ""
    allele = allele.strip()
    
    # 1. Try direct matches
    if allele in HLA_PSEUDO_MAP:
        return HLA_PSEUDO_MAP[allele]
        
    # Clean representation: A*02:01 -> A0201
    allele_clean = allele.replace('HLA-', '').replace('*', '').replace(':', '').replace('_', '').replace('-', '').upper().strip()
    if allele_clean in HLA_PSEUDO_MAP:
        return HLA_PSEUDO_MAP[allele_clean]
        
    # 2. Locus-specific consensus fallbacks
    # Class II (DQA1, DQB1, DRB1, DPB1, DPA1, Eb, Aa)
    if any(k in allele_clean for k in ["DQ", "DR", "DP", "AA", "EB"]):
        return "QEFFIASGAAVDAIMWLFLECYDLQRATYHVGFT"  # HLA-DRB1*01:01 consensus
    # C-locus
    if allele_clean.startswith("C"):
        return "YDSGYREKYRQADVSNLYLRSDSYTLAALAYTWY"  # HLA-C*07:02 consensus
    # Default Class I A/B-locus
    return "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"  # HLA-A*02:01 consensus

# Peptide mutation logic removed

class TCRDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (str): Path to the processed CSV file.
        """
        self.df = pd.read_csv(csv_path)
        
        # Verify columns exist
        required_cols = ["cdr3_beta", "peptide", "hla_allele"]
        for col in required_cols:
            assert col in self.df.columns, f"Required column '{col}' missing from {csv_path}"
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        raw_beta = str(row["cdr3_beta"])
        raw_alpha = str(row["cdr3_alpha"]) if pd.notna(row["cdr3_alpha"]) and row["cdr3_alpha"] is not None else ""
        raw_peptide = str(row["peptide"])
        raw_hla = str(row["hla_allele"])
        disease = str(row["disease"]) if "disease" in row else "Unknown"
        
        return {
            "raw_cdr3_beta": raw_beta,
            "raw_cdr3_alpha": raw_alpha,
            "raw_peptide": raw_peptide,
            "raw_hla_allele": raw_hla,
            "disease": disease
        }

class TCRCollate:
    def __init__(self, positive_triplets_set, global_peptides_pool):
        """
        Args:
            positive_triplets_set (set): A set of positive tuples: (raw_cdr3_beta, raw_peptide, raw_hla_allele)
            global_peptides_pool (list): A list of tuples (raw_peptide, raw_hla_allele) for global fallback.
        """
        self.pos_set = positive_triplets_set
        self.global_pool = global_peptides_pool
        
    def __call__(self, batch):
        batch_size = len(batch)
        
        # 1. Load pre-computed true embeddings for the batch
        beta_tensors = []
        alpha_tensors = []
        pephla_tensors = []
        
        pep_pos = []
        hla_pos = []
        for item in batch:
            raw_beta = item["raw_cdr3_beta"]
            raw_alpha = item["raw_cdr3_alpha"]
            raw_peptide = item["raw_peptide"]
            raw_hla = item["raw_hla_allele"]
            
            pep_pos.append(raw_peptide)
            hla_pos.append(raw_hla)
            
            # Translate HLA to pseudo-sequence and combine
            hla_pseudo = get_hla_pseudo(raw_hla)
            pephla_string = raw_peptide + hla_pseudo
            
            beta_tensors.append(load_embedding(raw_beta))
            alpha_tensors.append(load_embedding(raw_alpha))
            pephla_tensors.append(load_embedding(pephla_string))
            
        beta_batch = torch.stack(beta_tensors)
        alpha_batch = torch.stack(alpha_tensors)
        pephla_batch = torch.stack(pephla_tensors)
        
        # 2. Generate natural decoy negatives by shuffling the Peptide-HLA tensors
        shuffled_indices = list(range(batch_size))
        if batch_size > 1:
            while True:
                random.shuffle(shuffled_indices)
                collision = False
                for i in range(batch_size):
                    if shuffled_indices[i] == i:
                        collision = True
                        break
                if not collision:
                    break
        
        # Strict Value-Level Collision Check
        for i in range(batch_size):
            if shuffled_indices[i] != -1 and pep_pos[i] == pep_pos[shuffled_indices[i]]:
                # Collision detected! Swap with another index j that resolves it
                swap_found = False
                for j in range(batch_size):
                    if i == j or shuffled_indices[j] == -1:
                        continue
                    cand_i = shuffled_indices[j]
                    cand_j = shuffled_indices[i]
                    if (pep_pos[i] != pep_pos[cand_i] and 
                        pep_pos[j] != pep_pos[cand_j] and 
                        cand_i != i and 
                        cand_j != j):
                        shuffled_indices[i], shuffled_indices[j] = shuffled_indices[j], shuffled_indices[i]
                        swap_found = True
                        break
                
                # Global Fallback if no valid swap resolves the collision (highly dominated peptide)
                if not swap_found:
                    shuffled_indices[i] = -1
        
        # Load natural decoy peptide embeddings (with global pool fallback)
        pephla_neg_tensors = []
        pep_neg = []
        for i in range(batch_size):
            idx = shuffled_indices[i]
            if idx == -1:
                # Load from global pool fallback
                global_pep = pep_pos[i]
                global_hla = hla_pos[i]
                found = False
                if self.global_pool:
                    for _ in range(20):
                        g_pep, g_hla = random.choice(self.global_pool)
                        if g_pep != pep_pos[i]:
                            global_pep = g_pep
                            global_hla = g_hla
                            found = True
                            break
                if not found:
                    global_pep = "AAAAA"
                    global_hla = "A*02:01"
                
                pep_neg.append(global_pep)
                hla_pseudo = get_hla_pseudo(global_hla)
                pephla_neg_tensors.append(load_embedding(global_pep + hla_pseudo))
            else:
                pep_neg.append(pep_pos[idx])
                pephla_neg_tensors.append(pephla_batch[idx])
                
        pephla_neg = torch.stack(pephla_neg_tensors)
        
        # 3. Concatenate true and negative pairs
        beta_all = torch.cat([beta_batch, beta_batch], dim=0)
        alpha_all = torch.cat([alpha_batch, alpha_batch], dim=0)
        pephla_all = torch.cat([pephla_batch, pephla_neg], dim=0)
        
        labels_all = [1.0] * batch_size + [0.0] * batch_size
        pep_all = pep_pos + pep_neg
        
        # Temporary audit print disabled to optimize performance
        # print("\n--- COLLATOR AUDIT ---")
        # for idx in range(min(4, len(labels_all))):
        #     is_pos = "TRUE_POSITIVE" if labels_all[idx] == 1.0 else "DECOY/NEGATIVE"
        #     print(f"Index {idx}: label={labels_all[idx]} | type={is_pos} | peptide={pep_all[idx]}")
        # neg_start = batch_size
        # for idx in range(neg_start, min(neg_start + 2, len(labels_all))):
        #     is_pos = "TRUE_POSITIVE" if labels_all[idx] == 1.0 else "DECOY/NEGATIVE"
        #     print(f"Index {idx}: label={labels_all[idx]} | type={is_pos} | peptide={pep_all[idx]}")
        # print("----------------------\n")
        
        return {
            "cdr3_beta": beta_all,
            "cdr3_alpha": alpha_all,
            "peptide_plus_hla": pephla_all,
            "label": torch.tensor(labels_all, dtype=torch.float)
        }

def build_global_pool(df):
    """
    Builds a global pool of unique peptide-HLA pairs for negative sampling fallback.
    """
    pool = []
    unique_pairs = df.drop_duplicates(subset=["peptide", "hla_allele"])
    for _, row in unique_pairs.iterrows():
        pep_str = str(row["peptide"])
        hla_str = str(row["hla_allele"])
        pool.append((pep_str, hla_str))
    return pool
