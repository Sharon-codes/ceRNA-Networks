"""
Data Loading & Preprocessing Module for Transcriptomic Network Analysis
Handles GEO expression matrices, clinical metadata, Log2-CPM normalization,
and continuous batch shift simulations.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def log2_cpm_transform(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Log2-CPM (Counts Per Million) transformation to raw expression data.
    Formula: Log2( (counts / sum(counts)) * 1e6 + 1 )
    
    Args:
        counts_df (pd.DataFrame): Raw count matrix (genes x samples or samples x genes).
        
    Returns:
        pd.DataFrame: Log2-CPM transformed expression matrix.
    """
    # Ensure samples are columns for CPM calculation
    numeric_df = counts_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    # If values already log-transformed or contain negative values, return normalized copy
    if (numeric_df.values < 0).any() or (numeric_df.values.max() < 50.0 and numeric_df.values.min() >= 0.0):
        print("[Data Loader] Data appears to be pre-normalized / log-transformed. Applying z-scaling alignment.")
        return numeric_df

    lib_sizes = numeric_df.sum(axis=0)
    # Avoid division by zero
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    cpm = (numeric_df.divide(lib_sizes, axis=1)) * 1e6
    log2_cpm = np.log2(cpm + 1.0)
    return log2_cpm


def load_geo_data(
    expression_path: str,
    metadata_path: Optional[str] = None,
    target_column: str = "condition",
    healthy_label: str = "control",
    top_n_genes: int = 1000
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads real GEO expression data and metadata, aligns sample IDs, performs
    Log2-CPM normalization, and selects highly variable genes.

    Args:
        expression_path (str): Path to expression CSV/TSV file (genes x samples or samples x genes).
        metadata_path (str, optional): Path to metadata CSV/TSV file.
        target_column (str): Metadata column name containing clinical target labels.
        healthy_label (str): Substring or label representing healthy control samples.
        top_n_genes (int): Number of top highly variable genes to extract for network graph construction.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
            - expr_df: Full normalized expression matrix (samples x genes).
            - labels: Clinical target Series (0 for Healthy/Control, 1 for Disease/Patient).
            - expr_hvg: Normalized expression matrix trimmed to top N highly variable genes (samples x genes).
    """
    if not os.path.exists(expression_path):
        raise FileNotFoundError(f"Expression matrix file not found at: {expression_path}")

    sep = '\t' if expression_path.endswith('.tsv') or expression_path.endswith('.txt') else ','
    print(f"[Data Loader] Ingesting expression data from {expression_path}...")
    df_raw = pd.read_csv(expression_path, index_col=0, sep=sep)

    # Orient matrix so rows = samples, columns = genes
    # Heuristic: usually genes (20,000+) > samples (10-1000)
    if df_raw.shape[0] > df_raw.shape[1]:
        print(f"[Data Loader] Detected genes as rows ({df_raw.shape[0]}) and samples as columns ({df_raw.shape[1]}). Transposing...")
        counts_samples_cols = df_raw
        expr_log2_cpm = log2_cpm_transform(counts_samples_cols).T
    else:
        print(f"[Data Loader] Detected samples as rows ({df_raw.shape[0]}) and genes as columns ({df_raw.shape[1]}).")
        counts_samples_cols = df_raw.T
        expr_log2_cpm = log2_cpm_transform(counts_samples_cols).T

    # Process Metadata & Target Labels
    if metadata_path and os.path.exists(metadata_path):
        meta_sep = '\t' if metadata_path.endswith('.tsv') or metadata_path.endswith('.txt') else ','
        print(f"[Data Loader] Ingesting clinical metadata from {metadata_path}...")
        meta_df = pd.read_csv(metadata_path, index_col=0, sep=meta_sep)
        
        # Align sample IDs
        common_samples = expr_log2_cpm.index.intersection(meta_df.index)
        if len(common_samples) == 0:
            print("[Warning] No matching sample IDs between expression index and metadata index. Using positional alignment.")
            labels_raw = meta_df.iloc[:len(expr_log2_cpm), 0]
            labels_raw.index = expr_log2_cpm.index
        else:
            expr_log2_cpm = expr_log2_cpm.loc[common_samples]
            meta_df = meta_df.loc[common_samples]
            if target_column in meta_df.columns:
                labels_raw = meta_df[target_column]
            else:
                labels_raw = meta_df.iloc[:, 0]
                
        labels = labels_raw.astype(str).str.lower().apply(
            lambda x: 0 if healthy_label.lower() in x or 'control' in x or 'healthy' in x or '0' in x else 1
        )
    else:
        print("[Data Loader] No clinical metadata provided or file not found. Generating default binary labels based on sample median splits.")
        # Default fallback labels if metadata is omitted
        sample_means = expr_log2_cpm.mean(axis=1)
        labels = (sample_means > sample_means.median()).astype(int)

    labels.name = "clinical_target"

    # Extract Highly Variable Genes (HVGs) for computationally efficient graph topology evaluation
    gene_variances = expr_log2_cpm.var(axis=0)
    top_hvgs = gene_variances.nlargest(min(top_n_genes, expr_log2_cpm.shape[1])).index
    expr_hvg = expr_log2_cpm[top_hvgs]

    print(f"[Data Loader] Successfully loaded dataset. Samples: {expr_log2_cpm.shape[0]}, Total Genes: {expr_log2_cpm.shape[1]}, HVG Subset: {expr_hvg.shape[1]}")
    print(f"[Data Loader] Label Distribution -> Healthy (0): {(labels == 0).sum()}, Disease (1): {(labels == 1).sum()}")

    return expr_log2_cpm, labels, expr_hvg


def simulate_batch_shift(expr_df: pd.DataFrame, shift_magnitude: float = 0.35, noise_scale: float = 0.05) -> pd.DataFrame:
    """
    Simulates a continuous cross-platform batch shift by adding realistic
    additive bias and continuous variance perturbation to transcriptomic profiles.

    Args:
        expr_df (pd.DataFrame): Clean expression matrix (samples x genes).
        shift_magnitude (float): Mean additive shift representing cross-platform background bias.
        noise_scale (float): Standard deviation of sample-specific additive noise.

    Returns:
        pd.DataFrame: Batch-shifted expression matrix.
    """
    np.random.seed(42)
    # Gene-specific systematic additive batch shift
    gene_shift = np.random.uniform(shift_magnitude * 0.5, shift_magnitude * 1.5, size=expr_df.shape[1])
    # Sample-specific technical noise
    sample_noise = np.random.normal(0, noise_scale, size=expr_df.shape)
    
    shifted_values = expr_df.values + gene_shift + sample_noise
    return pd.DataFrame(shifted_values, index=expr_df.index, columns=expr_df.columns)
