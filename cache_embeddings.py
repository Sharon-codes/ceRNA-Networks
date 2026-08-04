import os
import hashlib
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
from dataset import get_hla_pseudo

def get_md5_hash(sequence: str) -> str:
    """Returns the MD5 hash of a given sequence string."""
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()

def main():
    # Setup directories
    data_dir = "./Processed"
    embed_dir = "./Dataset/Embeddings"
    os.makedirs(embed_dir, exist_ok=True)
    
    # Load dataset splits
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    dfs = []
    for path in [train_path, val_path, test_path]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    
    if not dfs:
        print("No CSV files found in ./Processed")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Extract unique sequences
    unique_beta = set(df_all["cdr3_beta"].dropna().astype(str).unique())
    unique_alpha = set(df_all["cdr3_alpha"].dropna().astype(str).unique())
    
    unique_peptides = set(df_all["peptide"].dropna().astype(str).unique())
    unique_hlas = set(df_all["hla_allele"].dropna().astype(str).unique())
    
    # Generate peptide-HLA combinations actually present
    pep_hla_pairs = set()
    for _, row in df_all.iterrows():
        pep = str(row["peptide"])
        hla = str(row["hla_allele"])
        pep_hla_pairs.add((pep, hla))
        
    print(f"Unique CDR3 Beta: {len(unique_beta)}")
    print(f"Unique CDR3 Alpha: {len(unique_alpha)}")
    print(f"Unique Peptides: {len(unique_peptides)}")
    print(f"Unique HLA Alleles: {len(unique_hlas)}")
    print(f"Unique Pep-HLA pairs: {len(pep_hla_pairs)}")
    
    # Use only true Peptide-HLA combinations
    all_pep_hla_pairs = pep_hla_pairs
    print(f"Total Pep-HLA pairs to embed: {len(all_pep_hla_pairs)}")
    
    # Pre-compute Pep-HLA strings
    unique_pep_hla_strings = set()
    for pep, hla in all_pep_hla_pairs:
        hla_pseudo = get_hla_pseudo(hla)
        unique_pep_hla_strings.add(pep + hla_pseudo)
    print(f"Unique Pep-HLA combined strings: {len(unique_pep_hla_strings)}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for feature caching.")
    
    print("Loading ESM-2 Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
    model.eval()
    
    def process_and_save(sequences, max_length, desc):
        sequences = list(sequences)
        batch_size = 256
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc=desc):
                batch_seqs = sequences[i:i+batch_size]
                
                # Check which ones need computing
                seqs_to_compute = []
                hashes_to_compute = []
                
                for seq in batch_seqs:
                    h = get_md5_hash(seq)
                    save_path = os.path.join(embed_dir, f"{h}.pt")
                    if not os.path.exists(save_path):
                        seqs_to_compute.append(seq)
                        hashes_to_compute.append(h)
                
                if not seqs_to_compute:
                    continue
                    
                inputs = tokenizer(seqs_to_compute, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state.cpu() # [batch, seq_len, 320]
                
                for j, h in enumerate(hashes_to_compute):
                    save_path = os.path.join(embed_dir, f"{h}.pt")
                    # Save individual tensor [seq_len, 320]
                    torch.save(hidden_states[j].clone(), save_path)
                    
    # Process all sequence types
    # Empty string might happen for alpha
    if "" in unique_alpha:
        process_and_save([""], max_length=30, desc="Embedding empty alpha")
        unique_alpha.remove("")
        
    process_and_save(unique_beta, max_length=30, desc="Embedding CDR3 Beta")
    process_and_save(unique_alpha, max_length=30, desc="Embedding CDR3 Alpha")
    process_and_save(unique_pep_hla_strings, max_length=50, desc="Embedding Peptide+HLA")

if __name__ == "__main__":
    main()
