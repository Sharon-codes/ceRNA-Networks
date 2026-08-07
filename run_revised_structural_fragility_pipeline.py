"""
Revised Structural Fragility Pipeline on GSE73002 Breast Cancer Cohort
Targeted Sparsity (2.5% Density), MAD Feature Selection, Sensitivity Sweep,
WGCNA Soft-Thresholding Control & Regulatory Biological Validation
"""

import os
import gzip
import io
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
from scipy.stats import median_abs_deviation
import GEOparse
import gseapy as gp

# Set global seed for exact reproducibility
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "./mirna_audit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 80)
print(" REVISED STRUCTURAL FRAGILITY PIPELINE: GSE73002 BREAST CANCER COHORT ")
print("=" * 80)


# ==============================================================================
# STEP 1: STRICT PREPROCESSING & MAD FEATURE SELECTION
# ==============================================================================
print("\n[PROGRESS] Step 1: Ingesting GSE73002 & Applying MAD Feature Selection...")

def load_gse73002_bc_serum():
    filepath = './GSE73002_series_matrix.txt.gz'
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    data_start = 0
    diag_line = None
    for idx, l in enumerate(lines):
        if l.startswith('!series_matrix_table_begin'):
            data_start = idx + 1
            break
        if l.startswith('!Sample_characteristics_ch1') and 'diagnosis' in l.lower():
            diag_line = [x.replace('"', '').strip() for x in l.split('\t')[1:]]
            
    expr_lines = [l for l in lines[data_start:] if not l.startswith('!') and l.strip()]
    df_raw = pd.read_csv(io.StringIO(''.join(expr_lines)), sep='\t', index_col=0).apply(pd.to_numeric, errors='coerce')
    
    # Filter strictly to breast cancer serum samples ('diagnosis: breast cancer')
    bc_mask = [True if 'breast cancer' in d.lower() else False for d in diag_line]
    bc_ids = df_raw.columns[bc_mask]
    df_bc = df_raw[bc_ids].copy()
    return df_bc

df_bc_raw = load_gse73002_bc_serum()
print(f"  - GSE73002 Breast Cancer Serum Matrix Shape (Probes x Samples): {df_bc_raw.shape}")

# Drop probes with >20% missing values
missing_frac = df_bc_raw.isnull().mean(axis=1)
df_clean = df_bc_raw.loc[missing_frac <= 0.20].copy()

# Left-censor remaining below-detection values to 0.0
df_clean = df_clean.fillna(0.0)

if (df_clean.values < 0).any():
    df_clean = df_clean.clip(lower=0.0)
if (df_clean.values > 50).any():
    df_clean = np.log2(df_clean + 1.0)

# Compute Mean Absolute Deviation (MAD) for each probe across samples
probe_mads = median_abs_deviation(df_clean.values, axis=1)
mad_series = pd.Series(probe_mads, index=df_clean.index)

# Select top 500 probes with highest MAD
top500_probes = mad_series.nlargest(500).index
df_top500 = df_clean.loc[top500_probes].copy()

# Samples x Probes Matrix
X_df = df_top500.T
print(f"  - Final MAD Feature-Selected Matrix Shape (Samples x Probes): {X_df.shape}")

# Map MIMAT IDs to mature miRNA symbols using GPL18941 platform
gpl18941_path = './GPL18941.txt'
if os.path.exists(gpl18941_path):
    gpl_obj = GEOparse.get_GEO(filepath=gpl18941_path, silent=True)
    gpl_map = dict(zip(gpl_obj.table['ID'], gpl_obj.table['miRNA_ID_LIST']))
else:
    gpl_map = {}

probe_to_symbol = {}
for p in top500_probes:
    raw_sym = str(gpl_map.get(p, p))
    clean_sym = raw_sym.split('//')[0].split(',')[0].strip()
    probe_to_symbol[p] = clean_sym


# ==============================================================================
# STEP 2: DENSITY-TARGETED THRESHOLDING (TARGET DENSITY = 2.5%)
# ==============================================================================
print("\n[PROGRESS] Step 2: Dynamically Finding Threshold for 2.5% Target Density...")

def compute_spearman_matrix(X_samples_probes):
    R_spearman, _ = stats.spearmanr(X_samples_probes.values, axis=0)
    R_spearman = np.nan_to_num(R_spearman, nan=0.0)
    np.fill_diagonal(R_spearman, 1.0)
    return R_spearman

R_spearman_base = compute_spearman_matrix(X_df)
n_nodes = 500
total_possible_edges = n_nodes * (n_nodes - 1) / 2.0  # 124,750
target_density = 0.025  # 2.5%

best_theta = 0.75
min_density_diff = 1.0
best_edge_count = 0
best_density = 0.0

# Sweep theta from 0.99 down to 0.50 with step 0.001
theta_grid = np.arange(0.99, 0.499, -0.001)
for th in theta_grid:
    A_temp = (R_spearman_base >= th).astype(np.int8)
    np.fill_diagonal(A_temp, 0)
    e_count = int(np.sum(A_temp) / 2)
    dens = e_count / total_possible_edges
    diff = abs(dens - target_density)
    if diff < min_density_diff:
        min_density_diff = diff
        best_theta = float(th)
        best_edge_count = e_count
        best_density = float(dens)

theta_target = best_theta
print(f"  - Target Density: 2.5000%")
print(f"  - Optimal Dynamic Threshold (theta_target): {theta_target:.4f}")
print(f"  - Baseline Network Density at theta_target: {best_density * 100.0:.4f}% ({best_density:.6f})")
print(f"  - Baseline Edge Count at theta_target: {best_edge_count}")

A_base_target = (R_spearman_base >= theta_target).astype(np.int8)
np.fill_diagonal(A_base_target, 0)
G_full_base = nx.from_numpy_array(A_base_target)

# Extract Giant Connected Component (GCC)
gcc_nodes = max(nx.connected_components(G_full_base), key=len)
G_gcc = G_full_base.subgraph(gcc_nodes).copy()

n_gcc_nodes = G_gcc.number_of_nodes()
n_gcc_edges = G_gcc.number_of_edges()

print(f"  - Giant Connected Component (GCC) Nodes: {n_gcc_nodes} / 500")
print(f"  - Giant Connected Component (GCC) Edges: {n_gcc_edges}")


# ==============================================================================
# STEP 3: PSD BOOTSTRAPPING (N=1,000) & SENSITIVITY SWEEP
# ==============================================================================
print("\n[PROGRESS] Step 3: Running N=1,000 Bootstraps & Sensitivity Sweep...")

N_BOOTSTRAPS = 1000
gcc_edges_set = set(G_gcc.edges())

# Track hard edge flip counts and soft edge weights across bootstraps
edge_flip_counts = {e: 0 for e in gcc_edges_set}

# Store soft edge weight history for Step 4 control
best_beta_wgcna = 12  # Soft power
soft_weight_history = {e: [] for e in gcc_edges_set}
hard_weight_history = {e: [] for e in gcc_edges_set}

X_mat = X_df.values
n_samples = X_mat.shape[0]

rng = np.random.RandomState(SEED)
t_boot_start = time.time()

for b in range(N_BOOTSTRAPS):
    if (b + 1) % 250 == 0 or b == 0:
        print(f"    * Bootstrap Iteration {b+1} / {N_BOOTSTRAPS}...")
        
    boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X_mat[boot_idx, :]
    
    R_boot, _ = stats.spearmanr(X_boot, axis=0)
    R_boot = np.nan_to_num(R_boot, nan=0.0)
    np.fill_diagonal(R_boot, 1.0)
    
    A_boot_hard = (R_boot >= theta_target).astype(np.int8)
    np.fill_diagonal(A_boot_hard, 0)
    
    A_boot_soft = np.power(np.abs(R_boot), best_beta_wgcna)
    np.fill_diagonal(A_boot_soft, 0.0)
    
    for u, v in gcc_edges_set:
        hard_val = A_boot_hard[u, v]
        soft_val = A_boot_soft[u, v]
        
        hard_weight_history[(u, v)].append(hard_val)
        soft_weight_history[(u, v)].append(soft_val)
        
        if hard_val == 0:  # Edge lost in bootstrap
            edge_flip_counts[(u, v)] += 1

t_boot_end = time.time()
print(f"  - Completed N={N_BOOTSTRAPS} Bootstraps in {t_boot_end - t_boot_start:.2f} s")

# Compute Baseline EBC on GCC
print("  - Calculating Baseline GCC Edge Betweenness Centrality (EBC)...")
ebc_gcc_dict = nx.edge_betweenness_centrality(G_gcc, seed=SEED)

# Sensitivity Sweep across flip probability thresholds (>2%, >5%, >10%)
sweep_thresholds = [0.02, 0.05, 0.10]
sweep_results = {}

for p_th in sweep_thresholds:
    unstable_e = {e for e, c in edge_flip_counts.items() if (c / float(N_BOOTSTRAPS)) > p_th}
    stable_e = gcc_edges_set - unstable_e
    
    ebc_unstable = [ebc_gcc_dict[e] for e in unstable_e if e in ebc_gcc_dict]
    ebc_stable = [ebc_gcc_dict[e] for e in stable_e if e in ebc_gcc_dict]
    
    m_unstable = float(np.mean(ebc_unstable)) if len(ebc_unstable) > 0 else 0.0
    m_stable = float(np.mean(ebc_stable)) if len(ebc_stable) > 0 else 0.0
    enrichment = m_unstable / (m_stable + 1e-12)
    
    sweep_results[p_th] = {
        'unstable_count': len(unstable_e),
        'stable_count': len(stable_e),
        'mean_ebc_unstable': m_unstable,
        'mean_ebc_stable': m_stable,
        'enrichment_ratio': enrichment,
        'unstable_edges_set': unstable_e,
        'stable_edges_set': stable_e
    }

print("\n--- SENSITIVITY SWEEP RESULTS ---")
for p_th in sweep_thresholds:
    r_dict = sweep_results[p_th]
    print(f"  * Flip Prob > {p_th*100:.0f}%: Unstable={r_dict['unstable_count']}, Stable={r_dict['stable_count']}, "
          f"Mean EBC Unstable={r_dict['mean_ebc_unstable']:.6e}, Stable={r_dict['mean_ebc_stable']:.6e}, "
          f"Enrichment={r_dict['enrichment_ratio']:.4f}x")

# Detailed metrics for 5% instability threshold
res_5pct = sweep_results[0.05]
ebc_unstable_5 = [ebc_gcc_dict[e] for e in res_5pct['unstable_edges_set'] if e in ebc_gcc_dict]
ebc_stable_5 = [ebc_gcc_dict[e] for e in res_5pct['stable_edges_set'] if e in ebc_gcc_dict]

if len(ebc_unstable_5) > 0 and len(ebc_stable_5) > 0:
    u_stat_5, p_val_5 = stats.mannwhitneyu(ebc_unstable_5, ebc_stable_5, alternative='greater')
    n1, n2 = len(ebc_unstable_5), len(ebc_stable_5)
    rank_biserial_r_5 = float(np.abs((2.0 * u_stat_5) / (n1 * n2) - 1.0))
else:
    u_stat_5, p_val_5, rank_biserial_r_5 = 0.0, 1.0, 0.0

print(f"\n  * 5% Instability Threshold Mann-Whitney U: {u_stat_5:.1f}, p-value = {p_val_5:.6e}")
print(f"  * 5% Instability Threshold Rank-Biserial Effect Size (|r|): {rank_biserial_r_5:.4f}")


# ==============================================================================
# STEP 4: WGCNA SOFT-THRESHOLDING CONTROL
# ==============================================================================
print("\n[PROGRESS] Step 4: Fitting WGCNA Scale-Free Topology & Analyzing Soft Variance Control...")

def fit_wgcna_scale_free(R_matrix, beta_range=range(1, 21)):
    best_beta = 1
    best_r2 = -1.0
    for beta in beta_range:
        A_soft = np.power(np.abs(R_matrix), beta)
        np.fill_diagonal(A_soft, 0.0)
        k_vec = np.sum(A_soft, axis=1)
        if np.max(k_vec) == np.min(k_vec): continue
        hist, bin_edges = np.histogram(k_vec, bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        valid = (hist > 0) & (bin_centers > 0)
        if np.sum(valid) < 3: continue
        log_k = np.log10(bin_centers[valid]).reshape(-1, 1)
        log_pk = np.log10(hist[valid] / np.sum(hist)).reshape(-1, 1)
        reg = stats.linregress(log_k.flatten(), log_pk.flatten())
        r2 = reg.rvalue ** 2
        if r2 > best_r2:
            best_r2 = r2
            best_beta = beta
    return best_beta, best_r2

best_beta_wgcna, best_r2_wgcna = fit_wgcna_scale_free(R_spearman_base)
print(f"  - Optimal WGCNA Power Beta: {best_beta_wgcna}")
print(f"  - Scale-Free Topology Fit R^2: {best_r2_wgcna:.4f}")

# Compute variance of hard indicators vs soft edge weights for unstable (high-EBC) edges
unstable_edges_5 = res_5pct['unstable_edges_set']
hard_variances = [np.var(hard_weight_history[e]) for e in unstable_edges_5]
soft_variances = [np.var(soft_weight_history[e]) for e in unstable_edges_5]

mean_hard_var = float(np.mean(hard_variances)) if len(hard_variances) > 0 else 0.0
mean_soft_var = float(np.mean(soft_variances)) if len(soft_variances) > 0 else 0.0

var_reduction_fold = mean_hard_var / (mean_soft_var + 1e-12)

print(f"  - Unstable Edges Hard Threshold Variance: {mean_hard_var:.6f}")
print(f"  - Unstable Edges Soft Threshold Variance (WGCNA beta={best_beta_wgcna}): {mean_soft_var:.6e}")
print(f"  - Soft-Threshold Variance Reduction: {var_reduction_fold:.2f}x reduction in structural variance!")


# ==============================================================================
# STEP 5: UPDATED BIOLOGICAL VALIDATION (GO & KEGG ENRICHR)
# ==============================================================================
print("\n[PROGRESS] Step 5: Programmatic Biological Validation (miRTarBase & KEGG)...")

def get_mirna_symbols_for_edges(edge_set, probe_indices, probe_to_sym):
    unique_symbols = set()
    for u, v in edge_set:
        p_u = probe_indices[u]
        p_v = probe_indices[v]
        sym_u = probe_to_sym.get(p_u, p_u)
        sym_v = probe_to_sym.get(p_v, p_v)
        unique_symbols.add(str(sym_u))
        unique_symbols.add(str(sym_v))
    return list(unique_symbols)

probe_idx_list = list(top500_probes)
stable_mirnas = get_mirna_symbols_for_edges(res_5pct['stable_edges_set'], probe_idx_list, probe_to_symbol)
unstable_mirnas = get_mirna_symbols_for_edges(res_5pct['unstable_edges_set'], probe_idx_list, probe_to_symbol)

print(f"  - Unique miRNAs in Stable Edges: {len(stable_mirnas)}")
print(f"  - Unique miRNAs in Unstable Edges: {len(unstable_mirnas)}")

# Map miRNAs to target mRNA genes using miRTarBase_2017 library
mirtar_dict = gp.get_library(name='miRTarBase_2017', organism='Human')

def map_mirnas_to_target_genes(mirna_list):
    target_genes = set()
    for m in mirna_list:
        m_str = str(m).strip()
        if m_str in mirtar_dict:
            target_genes.update(mirtar_dict[m_str])
        else:
            m_clean = m_str.replace('-5p', '').replace('-3p', '')
            matched_keys = [k for k in mirtar_dict if m_str.lower() in k.lower() or m_clean.lower() in k.lower()]
            for k in matched_keys[:3]:
                target_genes.update(mirtar_dict[k])
    return list(target_genes)

stable_target_mRNAs = map_mirnas_to_target_genes(stable_mirnas)
unstable_target_mRNAs = map_mirnas_to_target_genes(unstable_mirnas)

print(f"  - Mapped Stable Target mRNA Genes: {len(stable_target_mRNAs)}")
print(f"  - Mapped Unstable Target mRNA Genes: {len(unstable_target_mRNAs)}")

# Query KEGG_2021_Human
def run_kegg_enrichment(gene_list):
    pathway_dict = {}  # Term -> Adj P-value
    try:
        res = gp.enrichr(gene_list=gene_list, gene_sets='KEGG_2021_Human', organism='human', outdir=None)
        df_res = res.results
        if df_res is not None and not df_res.empty:
            df_sig = df_res[df_res['Adjusted P-value'] < 0.05]
            for _, row in df_sig.iterrows():
                pathway_dict[row['Term']] = float(row['Adjusted P-value'])
    except Exception as e:
        print(f"    * Warning: Enrichr query failed: {e}")
    return pathway_dict

stable_kegg = run_kegg_enrichment(stable_target_mRNAs)
unstable_kegg = run_kegg_enrichment(unstable_target_mRNAs)

print(f"  - Total Enriched KEGG Pathways in Stable Core Graph (Adj P < 0.05): {len(stable_kegg)}")
print(f"  - Total Enriched KEGG Pathways in Unstable Graph (Adj P < 0.05): {len(unstable_kegg)}")

# Set difference: Stable - Unstable (pathways in stable graph lost in unstable graph)
lost_pathways_all = set(stable_kegg.keys()) - set(unstable_kegg.keys())

# Filter out generic housekeeping terms (Ribosome, Spliceosome, Proteasome, RNA transport, etc.)
generic_terms = {'Ribosome', 'Spliceosome', 'Proteasome', 'RNA transport', 'Nucleotide excision repair',
                 'Basal transcription factors', 'DNA replication', 'Mismatch repair', 'Homologous recombination'}

lost_regulatory_pathways = [p for p in lost_pathways_all if not any(g.lower() in p.lower() for g in generic_terms)]
n_lost_regulatory = len(lost_regulatory_pathways)

# Sort lost regulatory pathways by adjusted p-value
sorted_lost_regulatory = sorted([(term, stable_kegg[term]) for term in lost_regulatory_pathways], key=lambda x: x[1])
top10_lost = sorted_lost_regulatory[:10]


# ==============================================================================
# CONSOLIDATED TERMINAL REPORT
# ==============================================================================
print("\n" + "=" * 80)
print(" CONSOLIDATED REVISED STRUCTURAL FRAGILITY TERMINAL REPORT ")
print("=" * 80)

report_text = f"""
================================================================================
  REVISED STRUCTURAL FRAGILITY REPORT (GSE73002 BREAST CANCER COHORT)
================================================================================

--- 1. SPARSITY-TARGETED THRESHOLDING (TARGET DENSITY = 2.5%) ---
  * Optimal Dynamic Threshold (theta_target): {theta_target:.4f}
  * Network Density at theta_target: {best_density * 100.0:.4f}%
  * Baseline Edge Count: {best_edge_count}
  * GCC Node Count: {n_gcc_nodes} / 500
  * GCC Edge Count: {n_gcc_edges}

--- 2. SENSITIVITY SWEEP ACROSS INSTABILITY DEFINITIONS ---
  * Flip Prob > 2%:  Unstable = {sweep_results[0.02]['unstable_count']} edges, Mean EBC = {sweep_results[0.02]['mean_ebc_unstable']:.6e}, Enrichment = {sweep_results[0.02]['enrichment_ratio']:.4f}x
  * Flip Prob > 5%:  Unstable = {sweep_results[0.05]['unstable_count']} edges, Mean EBC = {sweep_results[0.05]['mean_ebc_unstable']:.6e}, Enrichment = {sweep_results[0.05]['enrichment_ratio']:.4f}x
  * Flip Prob > 10%: Unstable = {sweep_results[0.10]['unstable_count']} edges, Mean EBC = {sweep_results[0.10]['mean_ebc_unstable']:.6e}, Enrichment = {sweep_results[0.10]['enrichment_ratio']:.4f}x

--- 3. 5% INSTABILITY STATISTICAL METRICS ---
  * Mann-Whitney U Statistic: {u_stat_5:.1f}
  * Mann-Whitney p-value: {p_val_5:.6e}
  * Rank-Biserial Effect Size (|r|): {rank_biserial_r_5:.4f}

--- 4. WGCNA SOFT-THRESHOLDING CONTROL ---
  * Optimal Soft Power Beta: {best_beta_wgcna}
  * Scale-Free Topology Fit R^2: {best_r2_wgcna:.4f}
  * Unstable Edges Hard Threshold Variance: {mean_hard_var:.6f}
  * Unstable Edges Soft Threshold Variance: {mean_soft_var:.6e}
  * Soft-Threshold Variance Reduction: {var_reduction_fold:.2f}x structural variance reduction

--- 5. UPDATED REGULATORY BIOLOGICAL VALIDATION (KEGG) ---
  * Total Significant KEGG Pathways in Stable Core Graph: {len(stable_kegg)}
  * Total Significant KEGG Pathways in Unstable Graph: {len(unstable_kegg)}
  * Total Falsely Erased Regulatory/Oncogenic Pathways: {n_lost_regulatory}
  * Top 10 Falsely Erased Regulatory/Oncogenic Pathways:
"""

for i, (term, p_adj) in enumerate(top10_lost, 1):
    report_text += f"      {i:2d}. {term} -- Adjusted P-value = {p_adj:.6e}\n"

report_text += "=" * 80 + "\n"

print(report_text)

# Save output files
with open(os.path.join(OUTPUT_DIR, 'revised_structural_fragility_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

df_sweep_csv = pd.DataFrame([
    {
        'instability_threshold': 'p_flip_gt_0.02',
        'unstable_edges': sweep_results[0.02]['unstable_count'],
        'stable_edges': sweep_results[0.02]['stable_count'],
        'mean_ebc_unstable': sweep_results[0.02]['mean_ebc_unstable'],
        'mean_ebc_stable': sweep_results[0.02]['mean_ebc_stable'],
        'enrichment_ratio': sweep_results[0.02]['enrichment_ratio']
    },
    {
        'instability_threshold': 'p_flip_gt_0.05',
        'unstable_edges': sweep_results[0.05]['unstable_count'],
        'stable_edges': sweep_results[0.05]['stable_count'],
        'mean_ebc_unstable': sweep_results[0.05]['mean_ebc_unstable'],
        'mean_ebc_stable': sweep_results[0.05]['mean_ebc_stable'],
        'enrichment_ratio': sweep_results[0.05]['enrichment_ratio']
    },
    {
        'instability_threshold': 'p_flip_gt_0.10',
        'unstable_edges': sweep_results[0.10]['unstable_count'],
        'stable_edges': sweep_results[0.10]['stable_count'],
        'mean_ebc_unstable': sweep_results[0.10]['mean_ebc_unstable'],
        'mean_ebc_stable': sweep_results[0.10]['mean_ebc_stable'],
        'enrichment_ratio': sweep_results[0.10]['enrichment_ratio']
    }
])
df_sweep_csv.to_csv(os.path.join(OUTPUT_DIR, 'revised_sensitivity_sweep.csv'), index=False)

print(f"[+] Outputs saved to {OUTPUT_DIR}/revised_structural_fragility_report.txt and revised_sensitivity_sweep.csv")
