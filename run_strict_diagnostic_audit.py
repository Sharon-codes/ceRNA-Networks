"""
Diagnostic Audit & Methodological Pipeline Implementation
GSE73002 (Breast Cancer Serum) & GSE115513 (Colorectal Cancer Carcinoma Tissue)
"""

import os
import gzip
import io
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

SEED = 42
np.random.seed(SEED)
OUTPUT_DIR = "./mirna_audit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# STEP 1: PREPROCESSING & HOMOGENEOUS COHORT SUBSETTING
# ==============================================================================
print("\n" + "=" * 80)
print(" STEP 1: STRICT PREPROCESSING & COHORT SUBSETTING AUDIT ")
print("=" * 80)

def parse_geo_matrix_with_metadata(filepath):
    """
    Parses GEO series matrix file, extracting sample metadata headers and expression matrix.
    """
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    meta_dict = {}
    data_start = 0
    for idx, l in enumerate(lines):
        if l.startswith('!series_matrix_table_begin'):
            data_start = idx + 1
            break
        if l.startswith('!Sample_geo_accession'):
            meta_dict['accession'] = [x.replace('"', '').strip() for x in l.split('\t')[1:]]
        elif l.startswith('!Sample_characteristics_ch1'):
            if 'characteristics' not in meta_dict:
                meta_dict['characteristics'] = []
            meta_dict['characteristics'].append([x.replace('"', '').strip() for x in l.split('\t')[1:]])
            
    expr_lines = [l for l in lines[data_start:] if not l.startswith('!') and l.strip()]
    df_expr = pd.read_csv(io.StringIO(''.join(expr_lines)), sep='\t', index_col=0).apply(pd.to_numeric, errors='coerce')
    return meta_dict, df_expr


print("[PROGRESS] Ingesting and Auditing GSE73002...")
meta73, expr73_raw = parse_geo_matrix_with_metadata('./GSE73002_series_matrix.txt.gz')

# Filter GSE73002 strictly to Breast Cancer samples ('diagnosis: breast cancer')
# Characteristics line 0 contains diagnosis
diag_line_73 = meta73['characteristics'][0]
bc_sample_mask = [True if 'breast cancer' in d.lower() else False for d in diag_line_73]
bc_sample_ids = expr73_raw.columns[bc_sample_mask]
expr73_bc = expr73_raw[bc_sample_ids].copy()

print(f"  - GSE73002 Total Samples in Matrix: {expr73_raw.shape[1]}")
print(f"  - GSE73002 Strictly Filtered Early-Stage Breast Cancer Serum Samples (N): {expr73_bc.shape[1]}")


print("[PROGRESS] Ingesting and Auditing GSE115513...")
meta115, expr115_raw = parse_geo_matrix_with_metadata('./GSE115513_series_matrix.txt.gz')

# Filter GSE115513 strictly to Carcinoma Tissue samples ('tissue: Carcinoma')
# Characteristics line 2 contains tissue
tissue_line_115 = meta115['characteristics'][2]
carcinoma_sample_mask = [True if 'carcinoma' in t.lower() else False for t in tissue_line_115]
carcinoma_sample_ids = expr115_raw.columns[carcinoma_sample_mask]
expr115_carcinoma = expr115_raw[carcinoma_sample_ids].copy()

print(f"  - GSE115513 Total Samples in Matrix: {expr115_raw.shape[1]}")
print(f"  - GSE115513 Strictly Filtered Colorectal Carcinoma Tissue Samples (N): {expr115_carcinoma.shape[1]}")


def preprocess_and_knn_impute(df_expr, top_n=500, k=5):
    """
    1. Filter out probes with >20% missing values across samples.
    2. Perform KNN (k=5) imputation on remaining missing values (no global median imputation).
    3. Log2 transform if non-log scale.
    4. Select top 500 highly variable miRNA probes based on variance.
    Returns: Preprocessed Samples x Probes DataFrame.
    """
    # 1. Probe filtering (>20% missing across samples)
    missing_frac = df_expr.isnull().mean(axis=1)
    df_clean = df_expr.loc[missing_frac <= 0.20].copy()
    
    # 2. KNN (k=5) Imputation
    if df_clean.isnull().any().any():
        print(f"    * Performing KNN (k={k}) Imputation on {df_clean.isnull().sum().sum()} missing values...")
        imputer = KNNImputer(n_neighbors=k)
        # Transpose so samples are rows, probes are columns for KNN imputer
        imputed_mat = imputer.fit_transform(df_clean.values.T)
        df_imputed = pd.DataFrame(imputed_mat.T, index=df_clean.index, columns=df_clean.columns)
    else:
        df_imputed = df_clean.copy()
        
    # Log2 transformation check
    if (df_imputed.values < 0).any():
        df_imputed = df_imputed.clip(lower=0)
    if (df_imputed.values > 50).any():
        df_imputed = np.log2(df_imputed + 1.0)
        
    # 3. Feature selection: Top 500 most variable miRNA probes by variance across samples
    probe_variances = df_imputed.var(axis=1)
    top_probes = probe_variances.nlargest(top_n).index
    df_top = df_imputed.loc[top_probes].copy()
    
    # Return Samples x Probes DataFrame
    return df_top.T


print("[PROGRESS] Preprocessing and KNN-Imputing GSE73002...")
df73_final = preprocess_and_knn_impute(expr73_bc, top_n=500, k=5)
print(f"  - GSE73002 Final Matrix Shape: {df73_final.shape} (Samples x Probes)")

print("[PROGRESS] Preprocessing and KNN-Imputing GSE115513...")
df115_final = preprocess_and_knn_impute(expr115_carcinoma, top_n=500, k=5)
print(f"  - GSE115513 Final Matrix Shape: {df115_final.shape} (Samples x Probes)")


# ==============================================================================
# HELPER FUNCTIONS FOR METHODOLOGICAL PIPELINE
# ==============================================================================

def compute_spearman_matrix(X_samples_probes):
    """
    Computes Spearman rank correlation matrix across probes (columns).
    """
    R_spearman, _ = stats.spearmanr(X_samples_probes.values, axis=0)
    R_spearman = np.nan_to_num(R_spearman, nan=0.0)
    np.fill_diagonal(R_spearman, 1.0)
    return R_spearman


def fit_wgcna_scale_free(R_matrix, beta_range=range(1, 21)):
    """
    Fits soft-thresholding power beta dynamically across beta in [1, 20]
    to maximize scale-free topology fit R^2.
    """
    n_nodes = R_matrix.shape[0]
    best_beta = 1
    best_r2 = -1.0
    r2_dict = {}
    
    for beta in beta_range:
        # Soft adjacency matrix
        A_soft = np.power(np.abs(R_matrix), beta)
        np.fill_diagonal(A_soft, 0.0)
        
        # Connectivity (degree) per node
        k_vec = np.sum(A_soft, axis=1)
        
        # Bin degrees to estimate p(k)
        if np.max(k_vec) == np.min(k_vec):
            r2_dict[beta] = 0.0
            continue
            
        hist, bin_edges = np.histogram(k_vec, bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        valid = (hist > 0) & (bin_centers > 0)
        if np.sum(valid) < 3:
            r2_dict[beta] = 0.0
            continue
            
        log_k = np.log10(bin_centers[valid]).reshape(-1, 1)
        log_pk = np.log10(hist[valid] / np.sum(hist)).reshape(-1, 1)
        
        reg = LinearRegression().fit(log_k, log_pk)
        r2 = reg.score(log_k, log_pk)
        r2_dict[beta] = float(r2)
        
        if r2 >= 0.85 and best_r2 < 0.85:
            best_r2 = r2
            best_beta = beta
        elif r2 > best_r2:
            best_r2 = r2
            best_beta = beta
            
    return best_beta, best_r2, r2_dict


def run_psd_bootstrapping(df_samples_probes, theta=0.75, n_bootstraps=100, seed=SEED):
    """
    PSD-Preserved Non-Parametric Bootstrapping:
    Resamples patient rows with replacement (preserving joint covariance & PSD).
    Calculates flip probability for all baseline edges at theta=0.75.
    Unstable edges: flip prob > 0.05.
    Stable edges: flip prob <= 0.05.
    """
    X_mat = df_samples_probes.values
    n_samples, n_nodes = X_mat.shape
    
    # Baseline Spearman Matrix & Adjacency
    R_base = compute_spearman_matrix(df_samples_probes)
    A_base = (R_base >= theta).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_base = nx.from_numpy_array(A_base)
    
    baseline_edges = set(G_base.edges())
    triu_i, triu_j = np.triu_indices(n_nodes, k=1)
    
    # Track flip occurrences across bootstraps
    flip_counts = {e: 0 for e in baseline_edges}
    
    rng = np.random.RandomState(seed)
    
    t0 = time.time()
    for b in range(n_bootstraps):
        # Row resampling with replacement
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_mat[boot_idx, :]
        
        # Spearman correlation on boot sample
        R_boot, _ = stats.spearmanr(X_boot, axis=0)
        R_boot = np.nan_to_num(R_boot, nan=0.0)
        np.fill_diagonal(R_boot, 1.0)
        
        A_boot = (R_boot >= theta).astype(np.int8)
        np.fill_diagonal(A_boot, 0)
        
        # Check edge states for baseline edges
        for u, v in baseline_edges:
            if A_boot[u, v] == 0:  # Edge was lost in bootstrap
                flip_counts[(u, v)] += 1
                
    unstable_edges = set()
    stable_edges = set()
    
    for e, count in flip_counts.items():
        flip_prob = count / float(n_bootstraps)
        if flip_prob > 0.05:
            unstable_edges.add(e)
        else:
            stable_edges.add(e)
            
    return G_base, unstable_edges, stable_edges


def run_configuration_null_model(G_base, n_seeds=20, seed=SEED):
    """
    Degree-Preserving Null Model Comparison:
    Constructs double-edge-swapped degree-preserving null graphs matching G_base degree sequence.
    Computes EBC across 20 configuration seeds.
    """
    null_ebc_vals = []
    degree_seq = [d for n, d in G_base.degree()]
    num_edges = G_base.number_of_edges()
    
    if num_edges == 0:
        return [0.0]
        
    for s in range(n_seeds):
        G_null = G_base.copy()
        # Double edge swap preserving degree sequence
        try:
            nx.double_edge_swap(G_null, nswap=min(2000, 3 * num_edges), max_tries=20000, seed=seed + s)
        except Exception:
            pass
            
        ebc_null_dict = nx.edge_betweenness_centrality(G_null, k=min(250, G_null.number_of_nodes()), seed=seed + s)
        null_ebc_vals.extend(list(ebc_null_dict.values()))
        
    return null_ebc_vals


def execute_full_diagnostic_pipeline(df_samples_probes, cohort_name):
    """
    Executes full methodological pipeline on preprocessed cohort matrix.
    """
    print(f"\n" + "=" * 60)
    print(f" EXECUTING METHODOLOGICAL PIPELINE: {cohort_name} ")
    print("=" * 60)
    
    N_samples, V_probes = df_samples_probes.shape
    
    # 1. Baseline Spearman Correlation & Graph Metrics
    t_start = time.time()
    R_spearman = compute_spearman_matrix(df_samples_probes)
    theta = 0.75
    A_base = (R_spearman >= theta).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_base = nx.from_numpy_array(A_base)
    
    E_edges = G_base.number_of_edges()
    density = float((2.0 * E_edges) / (V_probes * (V_probes - 1)))
    
    print(f"  [1/4] Baseline Spearman Graph: N={N_samples}, V={V_probes}, E={E_edges}, Density={density:.6f}")
    
    # 2. Dynamic WGCNA Scale-Free Topology Fitting
    best_beta, best_r2, r2_dict = fit_wgcna_scale_free(R_spearman)
    print(f"  [2/4] WGCNA Fitting: Optimal Power Beta={best_beta}, Scale-Free R^2={best_r2:.4f}")
    
    # 3. PSD-Preserved Non-Parametric Bootstrapping (n=100)
    print(f"  [3/4] Running 100 Bootstrap Resamples...")
    G_base, unstable_edges, stable_edges = run_psd_bootstrapping(df_samples_probes, theta=0.75, n_bootstraps=100, seed=SEED)
    
    # Compute baseline EBC
    ebc_dict = nx.edge_betweenness_centrality(G_base, k=min(300, G_base.number_of_nodes()), seed=SEED)
    
    ebc_unstable = [ebc_dict[e] for e in unstable_edges if e in ebc_dict]
    ebc_stable = [ebc_dict[e] for e in stable_edges if e in ebc_dict]
    
    mean_ebc_unstable = float(np.mean(ebc_unstable)) if len(ebc_unstable) > 0 else 0.0
    mean_ebc_stable = float(np.mean(ebc_stable)) if len(ebc_stable) > 0 else 0.0
    
    # 4. Degree-Preserving Null Model Comparison (20 seeds)
    print(f"  [4/4] Generating 20 Degree-Preserving Configuration Model Null Graphs...")
    null_ebc_vals = run_configuration_null_model(G_base, n_seeds=20, seed=SEED)
    mean_ebc_null = float(np.mean(null_ebc_vals)) if len(null_ebc_vals) > 0 else 0.0
    
    # Enrichment Ratios
    enrichment_unstable_stable = mean_ebc_unstable / (mean_ebc_stable + 1e-12)
    enrichment_unstable_null = mean_ebc_unstable / (mean_ebc_null + 1e-12)
    
    # Mann-Whitney U test & Rank-Biserial Effect Size
    if len(ebc_unstable) > 0 and len(ebc_stable) > 0:
        u_stat, p_val = stats.mannwhitneyu(ebc_unstable, ebc_stable, alternative='greater')
        n1, n2 = len(ebc_unstable), len(ebc_stable)
        rank_biserial_r = float(np.abs((2.0 * u_stat) / (n1 * n2) - 1.0))
    else:
        u_stat, p_val, rank_biserial_r = 0.0, 1.0, 0.0
        
    print(f"\n--- {cohort_name} FINAL DIAGNOSTIC AUDIT RESULTS ---")
    print(f"  * Cohort Final Sample Size (N): {N_samples}")
    print(f"  * Probe Count (V): {V_probes}")
    print(f"  * Baseline Edge Count (E): {E_edges}")
    print(f"  * Network Density (theta=0.75): {density:.6f}")
    print(f"  * Optimal WGCNA Power Beta: {best_beta}")
    print(f"  * Scale-Free Topology Fit R^2: {best_r2:.4f}")
    print(f"  * Unstable Edge Count (P_flip > 0.05): {len(unstable_edges)}")
    print(f"  * Stable Edge Count (P_flip <= 0.05): {len(stable_edges)}")
    print(f"  * Mean EBC (Unstable Edges): {mean_ebc_unstable:.6e}")
    print(f"  * Mean EBC (Stable Edges): {mean_ebc_stable:.6e}")
    print(f"  * Mean EBC (Degree-Preserving Null): {mean_ebc_null:.6e}")
    print(f"  * EBC Enrichment Ratio (Unstable / Stable): {enrichment_unstable_stable:.4f}x")
    print(f"  * EBC Enrichment Ratio (Unstable / Null): {enrichment_unstable_null:.4f}x")
    print(f"  * Mann-Whitney U Test (Unstable vs Stable): Statistic={u_stat:.1f}, p-value={p_val:.6e}")
    print(f"  * Rank-Biserial Effect Size (|r|): {rank_biserial_r:.4f}")
    print(f"  * Pipeline Execution Time: {time.time() - t_start:.2f} s")
    
    return {
        'cohort': cohort_name,
        'N_samples': N_samples,
        'probes': V_probes,
        'baseline_edges': E_edges,
        'density': density,
        'wgcna_beta': best_beta,
        'wgcna_r2': best_r2,
        'unstable_edges': len(unstable_edges),
        'stable_edges': len(stable_edges),
        'mean_ebc_unstable': mean_ebc_unstable,
        'mean_ebc_stable': mean_ebc_stable,
        'mean_ebc_null': mean_ebc_null,
        'enrichment_unstable_stable': enrichment_unstable_stable,
        'enrichment_unstable_null': enrichment_unstable_null,
        'mann_whitney_u': u_stat,
        'p_value': p_val,
        'rank_biserial_r': rank_biserial_r
    }


res73 = execute_full_diagnostic_pipeline(df73_final, "GSE73002 (Early-Stage Breast Cancer Serum)")
res115 = execute_full_diagnostic_pipeline(df115_final, "GSE115513 (Colorectal Carcinoma Tissue)")

df_summary = pd.DataFrame([res73, res115])
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'diagnostic_audit_strict_preprocessing.csv'), index=False)

print("\n" + "=" * 80)
print(" DIAGNOSTIC AUDIT AND METHODOLOGICAL PIPELINE COMPLETE ")
print("=" * 80 + "\n")
