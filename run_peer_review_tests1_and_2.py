"""
Peer-Review Revision Tests 1 & 2
Test 1: Intra-Cohort Split-Half Technical Noise Audit (GSE73002)
Test 2: GSE115513 Density-Matched Threshold Sweep
"""

import os
import gzip
import io
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats

# Set global random seed for exact reproducibility
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "./mirna_audit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# DATA INGESTION & PREPROCESSING
# ==============================================================================
print("\n" + "=" * 80)
print(" INGESTING EMPIRICAL GEO DATASETS FOR TESTS 1 & 2 ")
print("=" * 80)

def load_clean_expression_matrix(filepath, n_samples=907, top_n_genes=500):
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    data_start = next(i for i, l in enumerate(lines) if l.startswith('!series_matrix_table_begin')) + 1
    expr_lines = [l for l in lines[data_start:] if not l.startswith('!') and l.strip()]
    df_raw = pd.read_csv(io.StringIO(''.join(expr_lines)), sep='\t', index_col=0).apply(pd.to_numeric, errors='coerce')
    
    # Subset N samples
    df_sub = df_raw.iloc[:, :min(n_samples, df_raw.shape[1])].copy()
    
    # Remove probes with >20% missing values
    missing_pct = df_sub.isnull().mean(axis=1)
    df_clean = df_sub.loc[missing_pct <= 0.20].copy()
    
    # Row-wise median imputation
    df_clean = df_clean.apply(lambda row: row.fillna(row.median()), axis=1)
    
    # Log2 transform if necessary
    if (df_clean.values <= 0).any():
        df_clean = np.log2(df_clean.clip(lower=0) + 1.0)
    else:
        df_clean = np.log2(df_clean + 1.0)
        
    # Top N most variable probes by standard deviation
    top_genes = df_clean.std(axis=1).nlargest(top_n_genes).index
    df_top = df_clean.loc[top_genes]
    
    # Transpose to Samples x Probes for correlation calculation
    return df_top.T

df_gse73002 = load_clean_expression_matrix('./GSE73002_series_matrix.txt.gz', n_samples=907, top_n_genes=500)
df_gse115513 = load_clean_expression_matrix('./GSE115513_series_matrix.txt.gz', n_samples=606, top_n_genes=500)

print(f"[+] GSE73002 Matrix Shape: {df_gse73002.shape} (Samples x Probes)")
print(f"[+] GSE115513 Matrix Shape: {df_gse115513.shape} (Samples x Probes)")


# ==============================================================================
# TEST 1: INTRA-COHORT SPLIT-HALF TECHNICAL NOISE AUDIT (GSE73002)
# ==============================================================================
print("\n" + "=" * 80)
print(" TEST 1: INTRA-COHORT SPLIT-HALF TECHNICAL NOISE AUDIT (GSE73002) ")
print("=" * 80)

# Step 1: Randomly split 907 samples into two equal halves (453 vs 454)
n_samples_total = df_gse73002.shape[0]
shuffled_indices = np.random.RandomState(SEED).permutation(n_samples_total)

half_a_idx = shuffled_indices[:n_samples_total // 2]
half_b_idx = shuffled_indices[n_samples_total // 2:]

half_a_expr = df_gse73002.iloc[half_a_idx]
half_b_expr = df_gse73002.iloc[half_b_idx]

print(f"[*] Split GSE73002 into Half A ({half_a_expr.shape[0]} samples) and Half B ({half_b_expr.shape[0]} samples)")

# Step 2: Compute Pearson correlation matrix independently for Half A and Half B
R_half_a = np.corrcoef(half_a_expr.values.T)
R_half_a = np.nan_to_num(R_half_a, nan=0.0)
np.fill_diagonal(R_half_a, 1.0)

R_half_b = np.corrcoef(half_b_expr.values.T)
R_half_b = np.nan_to_num(R_half_b, nan=0.0)
np.fill_diagonal(R_half_b, 1.0)

# Step 3: Compute absolute difference |Delta r| between Half A and Half B
delta_r_tech_matrix = np.abs(R_half_a - R_half_b)
triu_idx = np.triu_indices(500, k=1)
abs_delta_r_tech_vals = delta_r_tech_matrix[triu_idx]

# 95th percentile of technical noise distribution
delta_r_tech_95 = float(np.percentile(abs_delta_r_tech_vals, 95))
print(f"[*] 95th Percentile Intra-Cohort Technical Noise (Delta_r_tech): {delta_r_tech_95:.6f}")

# Step 4: Full GSE73002 correlation matrix
R_full_73 = np.corrcoef(df_gse73002.values.T)
R_full_73 = np.nan_to_num(R_full_73, nan=0.0)
np.fill_diagonal(R_full_73, 1.0)

# Baseline graph at theta = 0.75
theta_1 = 0.75
A_base_73 = (R_full_73 >= theta_1).astype(np.int8)
np.fill_diagonal(A_base_73, 0)
G_base_73 = nx.from_numpy_array(A_base_73)

# Apply Delta_r_tech as heteroscedastic noise
rng_t1 = np.random.RandomState(SEED)
n_genes = 500
raw_noise_std = (1.0 - np.abs(R_full_73))
noise_t1 = rng_t1.normal(0.0, raw_noise_std, size=(n_genes, n_genes))
noise_t1 = (noise_t1 + noise_t1.T) / 2.0
max_noise_abs = np.max(np.abs(noise_t1))
if max_noise_abs > 0:
    noise_t1 = noise_t1 * (delta_r_tech_95 / max_noise_abs)

R_pert_t1 = np.clip(R_full_73 + noise_t1, -1.0, 1.0)
np.fill_diagonal(R_pert_t1, 1.0)

A_pert_73 = (R_pert_t1 >= theta_1).astype(np.int8)
np.fill_diagonal(A_pert_73, 0)
G_pert_73 = nx.from_numpy_array(A_pert_73)

edges_base_set_73 = set(G_base_73.edges())
edges_pert_set_73 = set(G_pert_73.edges())

flipped_lost_73 = edges_base_set_73 - edges_pert_set_73
flipped_gained_73 = edges_pert_set_73 - edges_base_set_73
all_flipped_73 = flipped_lost_73 | flipped_gained_73
preserved_edges_73 = edges_base_set_73 & edges_pert_set_73

# Compute EBC using sampling for high performance
ebc_dict_73 = nx.edge_betweenness_centrality(G_base_73, k=min(300, G_base_73.number_of_nodes()), seed=SEED)

ebc_flipped_73 = [ebc_dict_73[e] for e in flipped_lost_73 if e in ebc_dict_73]
ebc_preserved_73 = [ebc_dict_73[e] for e in preserved_edges_73 if e in ebc_dict_73]

mean_ebc_flip_73 = float(np.mean(ebc_flipped_73)) if len(ebc_flipped_73) > 0 else 0.0
mean_ebc_pres_73 = float(np.mean(ebc_preserved_73)) if len(ebc_preserved_73) > 0 else 0.0
enrichment_73 = mean_ebc_flip_73 / (mean_ebc_pres_73 + 1e-12)

if len(ebc_flipped_73) > 0 and len(ebc_preserved_73) > 0:
    u_stat_73, p_val_73 = stats.mannwhitneyu(ebc_flipped_73, ebc_preserved_73, alternative='greater')
else:
    u_stat_73, p_val_73 = 0.0, 1.0

df_test1 = pd.DataFrame([{
    'dataset': 'GSE73002',
    'delta_r_tech_95th_percentile': delta_r_tech_95,
    'total_baseline_edges': len(edges_base_set_73),
    'total_flipped_edges': len(all_flipped_73),
    'mean_ebc_flipped': mean_ebc_flip_73,
    'mean_ebc_preserved': mean_ebc_pres_73,
    'ebc_enrichment_ratio': enrichment_73,
    'mann_whitney_u': u_stat_73,
    'p_value': p_val_73
}])
df_test1.to_csv(os.path.join(OUTPUT_DIR, 'test1_split_half_technical_noise.csv'), index=False)

print("\n--- TEST 1 RESULTS ---")
print(df_test1.to_string(index=False))


# ==============================================================================
# TEST 2: GSE115513 DENSITY-MATCHED THRESHOLD SWEEP
# ==============================================================================
print("\n" + "=" * 80)
print(" TEST 2: GSE115513 DENSITY-MATCHED THRESHOLD SWEEP ")
print("=" * 80)

# Step 1: Compute GSE115513 Pearson correlation matrix
R_full_115 = np.corrcoef(df_gse115513.values.T)
R_full_115 = np.nan_to_num(R_full_115, nan=0.0)
np.fill_diagonal(R_full_115, 1.0)

# Fixed empirical max Delta R = 0.958
DELTA_R_EMP_MAX = 0.958
theta_sweep_115 = [0.55, 0.60, 0.65, 0.70]
sweep_rows_115 = []

for th in theta_sweep_115:
    A_base_115 = (R_full_115 >= th).astype(np.int8)
    np.fill_diagonal(A_base_115, 0)
    G_base_115 = nx.from_numpy_array(A_base_115)
    
    # Calibrated heteroscedastic noise at DELTA_R_EMP_MAX = 0.958
    rng_t2 = np.random.RandomState(SEED)
    raw_noise_std = (1.0 - np.abs(R_full_115))
    noise_t2 = rng_t2.normal(0.0, raw_noise_std, size=(n_genes, n_genes))
    noise_t2 = (noise_t2 + noise_t2.T) / 2.0
    max_noise_abs = np.max(np.abs(noise_t2))
    if max_noise_abs > 0:
        noise_t2 = noise_t2 * (DELTA_R_EMP_MAX / max_noise_abs)
        
    R_pert_t2 = np.clip(R_full_115 + noise_t2, -1.0, 1.0)
    np.fill_diagonal(R_pert_t2, 1.0)
    
    A_pert_115 = (R_pert_t2 >= th).astype(np.int8)
    np.fill_diagonal(A_pert_115, 0)
    G_pert_115 = nx.from_numpy_array(A_pert_115)
    
    edges_base_set_115 = set(G_base_115.edges())
    edges_pert_set_115 = set(G_pert_115.edges())
    
    flipped_lost_115 = edges_base_set_115 - edges_pert_set_115
    flipped_gained_115 = edges_pert_set_115 - edges_base_set_115
    all_flipped_115 = flipped_lost_115 | flipped_gained_115
    preserved_edges_115 = edges_base_set_115 & edges_pert_set_115
    
    ebc_dict_115 = nx.edge_betweenness_centrality(G_base_115, k=min(300, G_base_115.number_of_nodes()), seed=SEED)
    
    ebc_flipped_115 = [ebc_dict_115[e] for e in flipped_lost_115 if e in ebc_dict_115]
    ebc_preserved_115 = [ebc_dict_115[e] for e in preserved_edges_115 if e in ebc_dict_115]
    
    mean_ebc_flip_115 = float(np.mean(ebc_flipped_115)) if len(ebc_flipped_115) > 0 else 0.0
    mean_ebc_pres_115 = float(np.mean(ebc_preserved_115)) if len(ebc_preserved_115) > 0 else 0.0
    enrichment_115 = mean_ebc_flip_115 / (mean_ebc_pres_115 + 1e-12)
    
    if len(ebc_flipped_115) > 0 and len(ebc_preserved_115) > 0:
        u_stat_115, p_val_115 = stats.mannwhitneyu(ebc_flipped_115, ebc_preserved_115, alternative='greater')
    else:
        u_stat_115, p_val_115 = 0.0, 1.0
        
    sweep_rows_115.append({
        'theta': th,
        'total_baseline_edges': len(edges_base_set_115),
        'total_flipped_edges': len(all_flipped_115),
        'mean_ebc_flipped': mean_ebc_flip_115,
        'mean_ebc_preserved': mean_ebc_pres_115,
        'ebc_enrichment_ratio': enrichment_115,
        'mann_whitney_u': u_stat_115,
        'p_value': p_val_115
    })

df_test2 = pd.DataFrame(sweep_rows_115)
df_test2.to_csv(os.path.join(OUTPUT_DIR, 'test2_gse115513_density_matched_sweep.csv'), index=False)

print("\n--- TEST 2 RESULTS ---")
print(df_test2.to_string(index=False))

print("\n" + "=" * 80)
print(" BOTH EXPERIMENTS EXECUTED SUCCESSFULLY. ")
print("=" * 80 + "\n")
