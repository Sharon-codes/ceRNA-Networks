"""
Statistical Redesign Pipeline: Continuous Instability & Distance-Controlled Regression
Independently analyzes GSE115513 (Colorectal Tissue) and GSE73002 (Breast Cancer Serum)
Without Zero-Censoring, Continuous Flip Probabilities (N=1000), GLM Binomial Distance-Controlled Regression, and Corrected Dual-Axis Threshold Sweep
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
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# Global Random Seed
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "./mirna_audit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 90)
print(" STARTING STATISTICAL REDESIGN & DISTANCE-CONTROLLED REGRESSION PIPELINE ")
print("=" * 90)


def load_uncensored_cohort(filepath, filter_key, filter_val, top_k=500):
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    data_start = 0
    char_lines = []
    for idx, l in enumerate(lines):
        if l.startswith('!series_matrix_table_begin'):
            data_start = idx + 1
            break
        if l.startswith('!Sample_characteristics_ch1'):
            char_lines.append([x.replace('"', '').strip() for x in l.split('\t')[1:]])
            
    expr_lines = [l for l in lines[data_start:] if not l.startswith('!') and l.strip()]
    df_raw = pd.read_csv(io.StringIO(''.join(expr_lines)), sep='\t', index_col=0).apply(pd.to_numeric, errors='coerce')
    
    target_char = next(cl for cl in char_lines if any(filter_val.lower() in x.lower() for x in cl))
    mask = [True if filter_val.lower() in x.lower() else False for x in target_char]
    sub_ids = df_raw.columns[mask]
    df_sub = df_raw[sub_ids].copy()
    
    # 1. Drop probes with > 20% missing values
    missing_frac = df_sub.isnull().mean(axis=1)
    df_clean = df_sub.loc[missing_frac <= 0.20].copy()
    
    # 2. DO NOT left-censor remaining below-detection values to 0.0. Drop or fill missing rows to avoid point-mass zero ties
    df_clean = df_clean.dropna(axis=0).copy()
    
    if (df_clean.values > 50).any():
        df_clean = np.log2(df_clean + 1.0)
        
    # 3. Top 500 MAD probes
    probe_mads = median_abs_deviation(df_clean.values, axis=1)
    mad_series = pd.Series(probe_mads, index=df_clean.index)
    top_probes = mad_series.nlargest(top_k).index
    df_top = df_clean.loc[top_probes].copy()
    
    return df_top, df_sub.shape[1]


def run_cohort_regression_analysis(filepath, cohort_name, filter_key, filter_val):
    print(f"\n" + "=" * 90)
    print(f" PIPELINE EXECUTION FOR COHORT: {cohort_name} ")
    print("=" * 90)
    
    df_top, n_samples = load_uncensored_cohort(filepath, filter_key, filter_val, top_k=500)
    X_mat = df_top.T.values
    print(f"  * Homogeneous Un-censored Cohort Size (N): {n_samples}")
    print(f"  * Top 500 MAD Probes Matrix (Samples x Probes): {X_mat.shape}")
    
    # Spearman Rank Correlation Matrix
    R_spearman, _ = stats.spearmanr(X_mat, axis=0)
    R_spearman = np.nan_to_num(R_spearman, nan=0.0)
    np.fill_diagonal(R_spearman, 1.0)
    
    # Determine theta_target for ~2.5% density
    tot_possible = 500 * 499 / 2.0  # 124,750
    best_th, min_diff, best_e, best_dens = 0.75, 1.0, 0, 0.0
    for th in np.arange(0.99, 0.499, -0.001):
        A_temp = (R_spearman >= th).astype(np.int8)
        np.fill_diagonal(A_temp, 0)
        e_c = int(np.sum(A_temp) / 2)
        dens = e_c / tot_possible
        diff = abs(dens - 0.025)
        if diff < min_diff:
            min_diff = diff
            best_th = float(th)
            best_e = e_c
            best_dens = float(dens)
            
    theta_target = best_th
    print(f"  * Dynamic Threshold (theta_target): {theta_target:.4f}")
    print(f"  * Baseline Density: {best_dens*100.0:.4f}% ({best_e} total edges)")
    
    # Extract empirical GCC & EBC
    A_base = (R_spearman >= theta_target).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_full = nx.from_numpy_array(A_base)
    gcc_nodes = max(nx.connected_components(G_full), key=len)
    G_gcc = G_full.subgraph(gcc_nodes).copy()
    
    v_gcc = G_gcc.number_of_nodes()
    e_gcc = G_gcc.number_of_edges()
    print(f"  * GCC Nodes: {v_gcc} / 500, GCC Edges: {e_gcc}")
    
    ebc_dict = nx.edge_betweenness_centrality(G_gcc, seed=SEED)
    gcc_edges = list(G_gcc.edges())
    
    # Calculate Distance for each GCC edge: abs(abs(r_ij) - theta_target)
    distances = [abs(abs(R_spearman[u, v]) - theta_target) for u, v in gcc_edges]
    ebc_vals = [ebc_dict[e] for e in gcc_edges]
    
    # Continuous Bootstrap Instability (N=1000)
    print(f"  * Running N=1,000 Patient-Resampling Bootstraps for Continuous P_flip...")
    N_BOOT = 1000
    flips_dict = {e: 0 for e in gcc_edges}
    rng = np.random.RandomState(SEED)
    
    t_boot_start = time.time()
    for b in range(N_BOOT):
        b_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_b = X_mat[b_idx, :]
        R_b, _ = stats.spearmanr(X_b, axis=0)
        R_b = np.nan_to_num(R_b, nan=0.0)
        np.fill_diagonal(R_b, 1.0)
        
        A_b = (R_b >= theta_target).astype(np.int8)
        np.fill_diagonal(A_b, 0)
        
        for u, v in gcc_edges:
            if A_b[u, v] == 0:
                flips_dict[(u, v)] += 1
                
    t_boot_end = time.time()
    print(f"  * Completed N=1,000 Bootstraps in {t_boot_end - t_boot_start:.2f} s")
    
    p_flips = np.array([flips_dict[e] / float(N_BOOT) for e in gcc_edges])
    flips_count = np.array([flips_dict[e] for e in gcc_edges])
    
    # Standardize independent variables (z-scores)
    z_dist = (np.array(distances) - np.mean(distances)) / (np.std(distances) + 1e-12)
    z_ebc = (np.array(ebc_vals) - np.mean(ebc_vals)) / (np.std(ebc_vals) + 1e-12)
    
    # GLM Binomial Regression: dependent = (flips, N_BOOT - flips)
    endog = np.column_stack((flips_count, N_BOOT - flips_count))
    exog = pd.DataFrame({
        'const': 1.0,
        'z_Distance': z_dist,
        'z_EBC': z_ebc
    })
    
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    
    print("\n" + "-" * 80)
    print(f" STATSMODELS GLM BINOMIAL REGRESSION SUMMARY: {cohort_name} ")
    print("-" * 80)
    print(glm_binom.summary())
    
    # Extract Distance and EBC p-values & standardized coefficients
    coef_dist = glm_binom.params['z_Distance']
    pval_dist = glm_binom.pvalues['z_Distance']
    
    coef_ebc = glm_binom.params['z_EBC']
    pval_ebc = glm_binom.pvalues['z_EBC']
    se_ebc = glm_binom.bse['z_EBC']
    z_stat_ebc = glm_binom.tvalues['z_EBC']
    
    print("\n" + "=" * 80)
    print(f" CRITICAL REGRESSION HYPOTHESIS TEST RESULTS: {cohort_name} ")
    print("=" * 80)
    print(f"  * z_Distance Coefficient: {coef_dist:.4f} (p-value = {pval_dist:.6e})")
    print(f"  * z_EBC Coefficient: {coef_ebc:.4f} (SE = {se_ebc:.4f}, z = {z_stat_ebc:.4f}, p-value = {pval_ebc:.6e})")
    
    if pval_ebc > 0.05:
        verdict = "Hypothesis Dead: EBC does not predict instability independent of threshold proximity."
    else:
        verdict = f"Hypothesis Confirmed: EBC significantly predicts instability independent of threshold proximity (beta = {coef_ebc:.4f}, p = {pval_ebc:.6e})."
        
    print(f"  * CONCLUSION: {verdict}")
    print("=" * 80)
    
    return {
        'cohort': cohort_name,
        'N_samples': n_samples,
        'theta_target': theta_target,
        'v_gcc': v_gcc,
        'e_gcc': e_gcc,
        'coef_dist': coef_dist,
        'pval_dist': pval_dist,
        'coef_ebc': coef_ebc,
        'se_ebc': se_ebc,
        'z_stat_ebc': z_stat_ebc,
        'pval_ebc': pval_ebc,
        'verdict': verdict,
        'glm_summary_str': str(glm_binom.summary())
    }


# Execute Cohort 1: GSE115513 (Colorectal Tissue)
res_tissue = run_cohort_regression_analysis('./GSE115513_series_matrix.txt.gz', "GSE115513 Tissue", 'tissue', 'carcinoma')

# Execute Cohort 2: GSE73002 (Breast Cancer Serum)
res_serum = run_cohort_regression_analysis('./GSE73002_series_matrix.txt.gz', "GSE73002 Serum", 'diagnosis', 'breast cancer')


# ==============================================================================
# STEP 5: CORRECTED THRESHOLD SWEEP (GSE115513 TISSUE ONLY)
# ==============================================================================
print("\n" + "=" * 90)
print(" STEP 5: CORRECTED THRESHOLD SWEEP (GSE115513 TISSUE) ")
print("=" * 90)

df_top_115, n_115 = load_uncensored_cohort('./GSE115513_series_matrix.txt.gz', 'tissue', 'carcinoma', top_k=500)
X_115 = df_top_115.T.values
R_115, _ = stats.spearmanr(X_115, axis=0)
R_115 = np.nan_to_num(R_115, nan=0.0)
np.fill_diagonal(R_115, 1.0)

theta_range = np.arange(0.70, 0.851, 0.005)
edge_counts = []
ebc_enrichment_ratios = []

for th in theta_range:
    A_th = (R_115 >= th).astype(np.int8)
    np.fill_diagonal(A_th, 0)
    e_cnt = int(np.sum(A_th) / 2)
    edge_counts.append(e_cnt)
    
    if e_cnt < 10:
        ebc_enrichment_ratios.append(1.0)
        continue
        
    G_th_full = nx.from_numpy_array(A_th)
    gcc_nodes = max(nx.connected_components(G_th_full), key=len)
    G_th_gcc = G_th_full.subgraph(gcc_nodes).copy()
    
    # Quick 50 bootstraps to label unstable vs stable at this threshold
    N_SWEEP_BOOT = 50
    gcc_edges_set = set(G_th_gcc.edges())
    flips_th = {e: 0 for e in gcc_edges_set}
    rng = np.random.RandomState(SEED)
    for _ in range(N_SWEEP_BOOT):
        b_idx = rng.choice(n_115, size=n_115, replace=True)
        X_b = X_115[b_idx, :]
        R_b, _ = stats.spearmanr(X_b, axis=0)
        R_b = np.nan_to_num(R_b, nan=0.0)
        np.fill_diagonal(R_b, 1.0)
        A_b = (R_b >= th).astype(np.int8)
        np.fill_diagonal(A_b, 0)
        for u, v in gcc_edges_set:
            if A_b[u, v] == 0:
                flips_th[(u, v)] += 1
                
    unstable = {e for e, c in flips_th.items() if (c / float(N_SWEEP_BOOT)) > 0.05}
    stable = gcc_edges_set - unstable
    
    ebc_dict = nx.edge_betweenness_centrality(G_th_gcc, seed=SEED)
    ebc_u = [ebc_dict[e] for e in unstable if e in ebc_dict]
    ebc_s = [ebc_dict[e] for e in stable if e in ebc_dict]
    
    mean_u = float(np.mean(ebc_u)) if len(ebc_u) > 0 else 0.0
    mean_s = float(np.mean(ebc_s)) if len(ebc_s) > 0 else 0.0
    ratio = mean_u / (mean_s + 1e-12)
    ebc_enrichment_ratios.append(ratio)

# Plot Dual-Axis Corrected Threshold Sweep
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

color_edge = '#2980b9'  # Blue for Edge Count
ax1.set_xlabel('Correlation Threshold (θ)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('GCC Edge Count', fontsize=12, fontweight='bold', color=color_edge, labelpad=10)
line1 = ax1.plot(theta_range, edge_counts, color=color_edge, linewidth=2.5, marker='o', markersize=5, label='GCC Edge Count')
ax1.tick_params(axis='y', labelcolor=color_edge)
ax1.grid(axis='x', linestyle='--', alpha=0.4)

ax2 = ax1.twinx()
color_ratio = '#c0392b'  # Red for EBC Enrichment Ratio
ax2.set_ylabel('EBC Enrichment Ratio (Unstable / Stable)', fontsize=12, fontweight='bold', color=color_ratio, labelpad=10)
line2 = ax2.plot(theta_range, ebc_enrichment_ratios, color=color_ratio, linewidth=2.5, linestyle='--', marker='s', markersize=5, label='EBC Enrichment Ratio')
ax2.tick_params(axis='y', labelcolor=color_ratio)

# Highlight target threshold 0.8190
ax1.axvline(x=0.8190, color='#27ae60', linestyle=':', linewidth=2.0, label='Target Threshold (θ = 0.8190)')

plt.title('Corrected Threshold Sweep: Edge Count & EBC Enrichment (GSE115513)', fontsize=13, fontweight='bold', pad=12)

# Combined Legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'corrected_step4_threshold_sweep.png')
plt.savefig(plot_path)
plt.close()

print(f"  * Dual-axis plot successfully saved to {plot_path}")

# Sync plot to arghhh and arghhhh
import shutil
for d in ['./arghhh', './arghhhh']:
    os.makedirs(d, exist_ok=True)
    shutil.copy(plot_path, os.path.join(d, 'corrected_step4_threshold_sweep.png'))


# ==============================================================================
# CONSOLIDATED REGRESSION REPORT
# ==============================================================================
report_text = f"""
================================================================================
  DISTANCE-CONTROLLED GLM REGRESSION REPORT (UN-CENSORED PREPROCESSING)
================================================================================

1. COHORT 1: {res_tissue['cohort']}
--------------------------------------------------------------------------------
  * Sample Size (N): {res_tissue['N_samples']}
  * Dynamic Target Threshold (theta_target): {res_tissue['theta_target']:.4f}
  * GCC Node Count: {res_tissue['v_gcc']}, GCC Edge Count: {res_tissue['e_gcc']}
  * Distance Coefficient (z_Distance): {res_tissue['coef_dist']:.4f} (p-value = {res_tissue['pval_dist']:.6e})
  * EBC Coefficient (z_EBC): {res_tissue['coef_ebc']:.4f} (SE = {res_tissue['se_ebc']:.4f}, z = {res_tissue['z_stat_ebc']:.4f}, p-value = {res_tissue['pval_ebc']:.6e})
  * VERDICT: {res_tissue['verdict']}

--------------------------------------------------------------------------------
2. COHORT 2: {res_serum['cohort']}
--------------------------------------------------------------------------------
  * Sample Size (N): {res_serum['N_samples']}
  * Dynamic Target Threshold (theta_target): {res_serum['theta_target']:.4f}
  * GCC Node Count: {res_serum['v_gcc']}, GCC Edge Count: {res_serum['e_gcc']}
  * Distance Coefficient (z_Distance): {res_serum['coef_dist']:.4f} (p-value = {res_serum['pval_dist']:.6e})
  * EBC Coefficient (z_EBC): {res_serum['coef_ebc']:.4f} (SE = {res_serum['se_ebc']:.4f}, z = {res_serum['z_stat_ebc']:.4f}, p-value = {res_serum['pval_ebc']:.6e})
  * VERDICT: {res_serum['verdict']}
================================================================================
"""

print(report_text)

with open(os.path.join(OUTPUT_DIR, 'distance_controlled_regression_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

df_reg = pd.DataFrame([{
    'cohort': res_tissue['cohort'],
    'N_samples': res_tissue['N_samples'],
    'theta_target': res_tissue['theta_target'],
    'v_gcc': res_tissue['v_gcc'],
    'e_gcc': res_tissue['e_gcc'],
    'z_Distance_coef': res_tissue['coef_dist'],
    'z_Distance_pval': res_tissue['pval_dist'],
    'z_EBC_coef': res_tissue['coef_ebc'],
    'z_EBC_se': res_tissue['se_ebc'],
    'z_EBC_zstat': res_tissue['z_stat_ebc'],
    'z_EBC_pval': res_tissue['pval_ebc'],
    'verdict': res_tissue['verdict']
}, {
    'cohort': res_serum['cohort'],
    'N_samples': res_serum['N_samples'],
    'theta_target': res_serum['theta_target'],
    'v_gcc': res_serum['v_gcc'],
    'e_gcc': res_serum['e_gcc'],
    'z_Distance_coef': res_serum['coef_dist'],
    'z_Distance_pval': res_serum['pval_dist'],
    'z_EBC_coef': res_serum['coef_ebc'],
    'z_EBC_se': res_serum['se_ebc'],
    'z_EBC_zstat': res_serum['z_stat_ebc'],
    'z_EBC_pval': res_serum['pval_ebc'],
    'verdict': res_serum['verdict']
}])

df_reg.to_csv(os.path.join(OUTPUT_DIR, 'distance_controlled_regression_metrics.csv'), index=False)

print(f"[+] Output written to {OUTPUT_DIR}/distance_controlled_regression_report.txt and distance_controlled_regression_metrics.csv")
