import os
import argparse
import logging
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import TCRDataset, TCRCollate, build_global_pool
from model import MambaTCR

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def levenshtein_distance(s1, s2):
    """
    Computes the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def cluster_peptides(peptides, threshold=2):
    """
    Clusters peptides based on sequence similarity.
    """
    adj = defaultdict(list)
    for i in range(len(peptides)):
        for j in range(i + 1, len(peptides)):
            if levenshtein_distance(peptides[i], peptides[j]) <= threshold:
                adj[peptides[i]].append(peptides[j])
                adj[peptides[j]].append(peptides[i])
                
    visited = set()
    clusters = []
    for p in peptides:
        if p not in visited:
            component = []
            queue = [p]
            visited.add(p)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(component)
    return clusters

def split_by_peptide_clusters(df_split, train_ratio=0.88, val_ratio=0.12):
    """
    Splits the remainder dataset into train and val splits ensuring entire
    homologous peptide clusters stay within the same split to prevent leakage.
    """
    unique_peps = df_split["peptide"].unique().tolist()
    clusters = cluster_peptides(unique_peps, threshold=2)
    np.random.seed(42)
    np.random.shuffle(clusters)
    
    train_peptides = []
    val_peptides = []
    train_count = 0
    val_count = 0
    
    for cluster in clusters:
        cluster_count = df_split[df_split["peptide"].isin(cluster)].shape[0]
        assigned = train_count + val_count
        if assigned == 0:
            train_peptides.extend(cluster)
            train_count += cluster_count
        else:
            train_diff = train_ratio - (train_count / assigned)
            val_diff = val_ratio - (val_count / assigned)
            if train_diff >= val_diff:
                train_peptides.extend(cluster)
                train_count += cluster_count
            else:
                val_peptides.extend(cluster)
                val_count += cluster_count
                
    train_df = df_split[df_split["peptide"].isin(train_peptides)].copy()
    val_df = df_split[df_split["peptide"].isin(val_peptides)].copy()
    return train_df, val_df

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        # Move tokenized inputs to device
        cdr3_beta = batch["cdr3_beta"].to(device)
        cdr3_alpha = batch["cdr3_alpha"].to(device)
        peptide_plus_hla = batch["peptide_plus_hla"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        labels = labels.to(torch.float32).unsqueeze(1)
        loss = criterion(logits, labels)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        all_targets.extend(labels.squeeze(-1).cpu().numpy())
        all_preds.extend(torch.sigmoid(logits).squeeze(-1).detach().cpu().float().numpy())
        
    epoch_loss = total_loss / len(all_targets)
    epoch_roc_auc = roc_auc_score(all_targets, all_preds)
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    epoch_pr_auc = auc(recall, precision)
    
    return epoch_loss, epoch_roc_auc, epoch_pr_auc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            cdr3_beta = batch["cdr3_beta"].to(device)
            cdr3_alpha = batch["cdr3_alpha"].to(device)
            peptide_plus_hla = batch["peptide_plus_hla"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(cdr3_beta, cdr3_alpha, peptide_plus_hla)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            labels = labels.to(torch.float32).unsqueeze(1)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            all_targets.extend(labels.squeeze(-1).cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).squeeze(-1).cpu().float().numpy())
            
    eval_loss = total_loss / len(all_targets)
    eval_roc_auc = roc_auc_score(all_targets, all_preds)
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    eval_pr_auc = auc(recall, precision)
    
    return eval_loss, eval_roc_auc, eval_pr_auc

def main():
    parser = argparse.ArgumentParser(description="Upgraded MambaTCR Training and LODO/LOAO CV Pipeline")
    parser.add_argument("--data_dir", type=str, default="./Processed", help="Directory containing processed splits")
    parser.add_argument("--checkpoint_dir", type=str, default="./Checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--d_model", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick dry-run with a small subset of data")
    
    # Strict Biological Partitioning Arguments
    parser.add_argument("--split_strategy", type=str, default="homology", 
                        choices=["homology", "leave_one_allele", "leave_one_disease"],
                        help="Biological partitioning cross-validation strategy")
    parser.add_argument("--held_out_disease", type=str, default="Breast Cancer",
                        help="Disease to leave out for leave_one_disease strategy")
    parser.add_argument("--held_out_allele", type=str, default="",
                        help="HLA allele family (e.g. A*11) to leave out for leave_one_allele (defaults to dynamic ~10 percent choice)")
    
    # Stochastic Weight Averaging (SWA)
    parser.add_argument("--swa_start", type=int, default=50, help="Epoch to start SWA weight averaging")
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device("cpu")
    logger.info(f"Using device: {device} (forced for CPU production)")
    
    # Load raw preprocessed partitions
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Combine into a unified set to partition dynamically
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["cdr3_beta", "peptide", "hla_allele"]).copy()
    
    # Track held-out metadata
    held_out_info = ""
    
    # 2. Partition data according to selected strategy
    if args.split_strategy == "homology":
        logger.info("Applying standard homology-based partitioning.")
        # Retain original splits from disk
        pass
    elif args.split_strategy == "leave_one_allele":
        # Dynamic Allele Selection representing ~10% of the dataset
        df_all["hla_prefix"] = df_all["hla_allele"].apply(lambda x: str(x).split(":")[0] if pd.notna(x) else "")
        prefix_counts = df_all["hla_prefix"].value_counts()
        total_size = len(df_all)
        target_size = 0.10 * total_size
        
        best_prefix = args.held_out_allele
        if not best_prefix:
            min_diff = float("inf")
            for prefix, count in prefix_counts.items():
                if prefix in ["", "Unknown", "NONE", "C*"]:
                    continue
                diff = abs(count - target_size)
                if diff < min_diff:
                    min_diff = diff
                    best_prefix = prefix
                    
        # Filter all matching rows for the test set
        mask_allele = df_all["hla_allele"].astype(str).str.startswith(best_prefix) | df_all["hla_prefix"].astype(str).str.startswith(best_prefix)
        test_df = df_all[mask_allele].copy()
        df_remainder = df_all[~mask_allele].copy()
        
        # Split remainder into train/val splits
        train_df, val_df = split_by_peptide_clusters(df_remainder)
        held_out_info = f"HLA allele cluster: {best_prefix}"
        logger.info(f"Isolating held-out allele cluster: {held_out_info} ({len(test_df)} records, {len(test_df)/total_size*100:.2f}%)")
        assert len(test_df) > 0, f"No records found for held-out allele prefix '{best_prefix}'!"
        
    elif args.split_strategy == "leave_one_disease":
        disease_name = args.held_out_disease
        mask_disease = df_all["disease"].astype(str).str.contains(disease_name, case=False, na=False)
        test_df = df_all[mask_disease].copy()
        df_remainder = df_all[~mask_disease].copy()
        
        train_df, val_df = split_by_peptide_clusters(df_remainder)
        held_out_info = f"Disease: {disease_name}"
        logger.info(f"Isolating held-out disease: {held_out_info} ({len(test_df)} records, {len(test_df)/len(df_all)*100:.2f}%)")
        assert len(test_df) > 0, f"No records found for held-out disease '{disease_name}'!"

    # 3. Handle Dry-Run Subsampling
    if args.dry_run:
        logger.info("Running in DRY-RUN mode. Subsampling datasets for quick verification...")
        train_df = train_df.head(40)
        val_df = val_df.head(10)
        test_df = test_df.head(10)
        
    # Write temporary files for dataset loader references
    train_temp = os.path.join(args.data_dir, "train_temp.csv")
    val_temp = os.path.join(args.data_dir, "val_temp.csv")
    test_temp = os.path.join(args.data_dir, "test_temp.csv")
    
    train_df.to_csv(train_temp, index=False)
    val_df.to_csv(val_temp, index=False)
    test_df.to_csv(test_temp, index=False)

    # 4. Construct Leakage-free negative sampling pools and triplets
    train_triplets = set(zip(train_df["cdr3_beta"], train_df["peptide"], train_df["hla_allele"]))
    train_pool = build_global_pool(train_df)
    
    val_triplets = set(zip(val_df["cdr3_beta"], val_df["peptide"], val_df["hla_allele"]))
    val_pool = build_global_pool(val_df)
    
    test_triplets = set(zip(test_df["cdr3_beta"], test_df["peptide"], test_df["hla_allele"]))
    test_pool = build_global_pool(test_df)

    # 5. Initialize ESM-2 Tokenizer
    logger.info("Skipping tokenizer initialization (using pre-computed features)...")

    # PyTorch Datasets
    train_dataset = TCRDataset(train_temp)
    val_dataset = TCRDataset(val_temp)
    test_dataset = TCRDataset(test_temp)

    # Collate loaders with dynamic Easy/Hard negative sampling
    train_collate = TCRCollate(train_triplets, train_pool)
    val_collate = TCRCollate(val_triplets, val_pool)
    test_collate = TCRCollate(test_triplets, test_pool)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_collate)

    # 6. Initialize upgraded MambaTCR model
    model = MambaTCR(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    logger.info("Upgraded ESM-MambaTCR Model initialized successfully.")

    # Set up SWA Model wrappers
    swa_model = AveragedModel(model)

    # Criteria and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine Annealing with Warm Restarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    # SWA learning rate scheduler
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

    # Training epochs setup
    best_val_auc = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_mamba_tcr.pt")
    production_path = os.path.join(args.checkpoint_dir, "best_mamba_tcr_production.pt")
    
    # Epoch thresholds (dry-run scales epochs down)
    unfreeze_epoch = 1 if args.dry_run else 30
    swa_start_epoch = 1 if args.dry_run else args.swa_start
    total_epochs = 2 if args.dry_run else args.epochs
    use_swa = total_epochs >= 15

    logger.info(f"Starting training loop for {total_epochs} epochs...")
    last_train_auc = 0.0
    for epoch in range(1, total_epochs + 1):
        
        # Train & Evaluate
        train_loss, train_auc, train_pr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        last_train_auc = train_auc
        val_loss, val_auc, val_pr = evaluate(model, val_loader, criterion, device)
        
        # B. Step scheduler depending on SWA start threshold
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        logger.info(
            f"Epoch {epoch:02d}/{total_epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, PR-AUC: {train_pr:.4f} | "
            f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, PR-AUC: {val_pr:.4f}"
        )
        
        # Check validation AUC for early stopping and checkpoint saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save standard best weights
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"--> Saved new best model checkpoint to {checkpoint_path} (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and (not use_swa or epoch < swa_start_epoch):
                logger.warning(f"Early stopping triggered! No improvement in Val ROC-AUC for {args.patience} epochs.")
                break
                
    # 7. Final Model Selection
    if use_swa and total_epochs >= swa_start_epoch:
        logger.info(f"Saving final SWA-averaged weights as the production model to {production_path}...")
        torch.save(swa_model.module.state_dict(), production_path)
    else:
        logger.info(f"Using best validation checkpoint weights from {checkpoint_path} as the production model...")
        if os.path.exists(checkpoint_path):
            import shutil
            shutil.copy(checkpoint_path, production_path)
        
    logger.info(f"Loading final weights from {production_path} for testing...")
    model.load_state_dict(torch.load(production_path))
    
    # 8. Test evaluation & metadata printing
    test_loss, test_auc, test_pr = evaluate(model, test_loader, criterion, device)
    
    logger.info("==========================================================")
    logger.info("                  FINAL TEST EVALUATION                  ")
    logger.info("==========================================================")
    if args.split_strategy == "leave_one_allele":
        logger.info(f"Evaluating zero-shot generalization on held-out allele: {held_out_info}")
    elif args.split_strategy == "leave_one_disease":
        logger.info(f"Evaluating zero-shot generalization on held-out disease: {held_out_info}")
    else:
        logger.info("Evaluating standard homology-based split")
    logger.info("==========================================================")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test ROC-AUC: {test_auc:.4f}")
    logger.info(f"Test PR-AUC: {test_pr:.4f}")
    logger.info("==========================================================")
    
    # Explicit terminal print output as required by summary instructions
    print(f"Final Train ROC-AUC: {last_train_auc:.4f}")
    print(f"Final Test ROC-AUC: {test_auc:.4f}")
    print(f"Final Test PR-AUC: {test_pr:.4f}")

    # Clean up temporary CSV files
    for path in [train_temp, val_temp, test_temp]:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main()
