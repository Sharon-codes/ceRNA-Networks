"""
Ultra-Fast Peer-Review Revision Analysis Pipeline
Ingests real GEO miRNA profiling datasets GSE73002 (Breast Cancer Serum, N=907) 
and GSE115513 (Colorectal Cancer Tissue, N=606).
Executes Steps 0 through 9 with exact empirical metrics.
"""

import os
import gzip
import io
import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg as la
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import GEOparse

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

# Global Seed for Exact Reproducibility
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "./mirna_audit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

print("\n" + "=" * 80)
print(" STARTING COMPLETE PEER-REVIEW MIRNA AUDIT PIPELINE ")
print("=" * 80)


# ==============================================================================
# STEP 0: DATA LOADING AND VERIFICATION
# ==============================================================================
print("\n[PROGRESS] Step 0: Starting Data Loading and Verification...")

def load_raw_series_matrix(filepath, n_samples):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file {filepath} not found.")
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    data_start = next(i for i, l in enumerate(lines) if l.startswith('!series_matrix_table_begin')) + 1
    expr_lines = [l for l in lines[data_start:] if not l.startswith('!') and l.strip()]
    df = pd.read_csv(io.StringIO(''.join(expr_lines)), sep='\t', index_col=0).apply(pd.to_numeric, errors='coerce')
    df_sub = df.iloc[:, :min(n_samples, df.shape[1])].copy()
    return df_sub

df73_raw = load_raw_series_matrix('./GSE73002_series_matrix.txt.gz', 907)
df115_raw = load_raw_series_matrix('./GSE115513_series_matrix.txt.gz', 606)

print(f"  - GSE73002 Raw Matrix Shape (Probes x Samples): {df73_raw.shape}")
print(f"  - GSE115513 Raw Matrix Shape (Probes x Samples): {df115_raw.shape}")

# Map GSE73002 MIMAT IDs to mature miRNA names using GPL18941 platform mapping
gpl18941_path = './GPL18941.txt'
if not os.path.exists(gpl18941_path):
    gpl_obj = GEOparse.get_GEO('GPL18941', destdir='.', silent=True)
    gpl_map = dict(zip(gpl_obj.table['ID'], gpl_obj.table['miRNA_ID_LIST']))
else:
    gpl_obj = GEOparse.get_GEO(filepath=gpl18941_path, silent=True)
    gpl_map = dict(zip(gpl_obj.table['ID'], gpl_obj.table['miRNA_ID_LIST']))

df73_raw.index = df73_raw.index.map(lambda x: gpl_map.get(x, x))
df73_raw = df73_raw.groupby(df73_raw.index).mean()

def clean_and_impute_matrix(df):
    missing_ratio = df.isnull().mean(axis=1)
    df_clean = df.loc[missing_ratio <= 0.20].copy()
    df_clean = df_clean.apply(lambda row: row.fillna(row.median()), axis=1)
    if (df_clean.values <= 0).any():
        df_clean = np.log2(df_clean.clip(lower=0) + 1.0)
    else:
        df_clean = np.log2(df_clean + 1.0)
    return df_clean

df73_clean = clean_and_impute_matrix(df73_raw)
df115_clean = clean_and_impute_matrix(df115_raw)

print(f"  - GSE73002 Cleaned Matrix Shape: {df73_clean.shape}")
print(f"  - GSE115513 Cleaned Matrix Shape: {df115_clean.shape}")

overlapping_probes = df73_clean.index.intersection(df115_clean.index)
print(f"  - Overlapping miRNA Probes (Intersection): {len(overlapping_probes)}")

df73_overlap = df73_clean.loc[overlapping_probes]
df115_overlap = df115_clean.loc[overlapping_probes]

top500_probes_73 = df73_clean.std(axis=1).nlargest(500).index
top500_probes_115 = df115_clean.std(axis=1).nlargest(500).index

df73_top500 = df73_clean.loc[top500_probes_73]
df115_top500 = df115_clean.loc[top500_probes_115]

print("[PROGRESS] Step 0: Completed Data Loading and Verification.")


# ==============================================================================
# STEP 1: EMPIRICAL CROSS-COHORT DELTA-R DISTRIBUTION
# ==============================================================================
print("\n[PROGRESS] Step 1: Starting Empirical Cross-Cohort Delta-R Distribution Analysis...")

R73_overlap = np.corrcoef(df73_overlap.values)
R73_overlap = np.nan_to_num(R73_overlap, nan=0.0)
np.fill_diagonal(R73_overlap, 1.0)

R115_overlap = np.corrcoef(df115_overlap.values)
R115_overlap = np.nan_to_num(R115_overlap, nan=0.0)
np.fill_diagonal(R115_overlap, 1.0)

Delta_R_matrix = R73_overlap - R115_overlap
triu_indices = np.triu_indices(len(overlapping_probes), k=1)

delta_r_vals = Delta_R_matrix[triu_indices]
abs_delta_r_vals = np.abs(delta_r_vals)

mean_delta_r = float(np.mean(delta_r_vals))
mean_abs_delta_r = float(np.mean(abs_delta_r_vals))
std_delta_r = float(np.std(delta_r_vals))

pct_5 = float(np.percentile(abs_delta_r_vals, 5))
pct_25 = float(np.percentile(abs_delta_r_vals, 25))
pct_50 = float(np.percentile(abs_delta_r_vals, 50))
pct_75 = float(np.percentile(abs_delta_r_vals, 75))
pct_90 = float(np.percentile(abs_delta_r_vals, 90))
pct_95 = float(np.percentile(abs_delta_r_vals, 95))
pct_99 = float(np.percentile(abs_delta_r_vals, 99))

EMPIRICAL_MAX_DELTA_R = pct_95
print(f"  - Calculated EMPIRICAL_MAX_DELTA_R (95th percentile of |Delta R|): {EMPIRICAL_MAX_DELTA_R:.6f}")

frac_001 = float(np.mean(abs_delta_r_vals > 0.01))
frac_002 = float(np.mean(abs_delta_r_vals > 0.02))
frac_0035 = float(np.mean(abs_delta_r_vals > 0.035))
frac_005 = float(np.mean(abs_delta_r_vals > 0.05))
frac_010 = float(np.mean(abs_delta_r_vals > 0.10))

df_step1_pct = pd.DataFrame([{
    'metric': 'delta_r',
    'mean_delta_r': mean_delta_r,
    'mean_abs_delta_r': mean_abs_delta_r,
    'std_delta_r': std_delta_r,
    'pct_5th': pct_5,
    'pct_25th': pct_25,
    'pct_50th_median': pct_50,
    'pct_75th': pct_75,
    'pct_90th': pct_90,
    'pct_95th_EMPIRICAL_MAX_DELTA_R': pct_95,
    'pct_99th': pct_99,
    'frac_gt_0.01': frac_001,
    'frac_gt_0.02': frac_002,
    'frac_gt_0.035': frac_0035,
    'frac_gt_0.05': frac_005,
    'frac_gt_0.10': frac_010
}])
df_step1_pct.to_csv(os.path.join(OUTPUT_DIR, 'step1_delta_r_percentiles.csv'), index=False)

fig1, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)
sns.histplot(delta_r_vals, bins=80, kde=True, color='#1f77b4', ax=ax1, edgecolor='black', alpha=0.6)
ax1.axvline(pct_95, color='#c23b22', linestyle='--', linewidth=2.2, label=f'95th Percentile (|$\\Delta r$| = {pct_95:.4f})')
ax1.axvline(-pct_95, color='#c23b22', linestyle='--', linewidth=2.2)

ax1.set_xlabel(r'Empirical $\Delta r_{ij} = r_{ij}(\text{GSE73002}) - r_{ij}(\text{GSE115513})$', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probe Pair Density', fontsize=12, fontweight='bold')
ax1.set_title(r'Empirical Cross-Cohort Correlation Shift ($\Delta r$) Distribution', fontsize=13, fontweight='bold', pad=12)
ax1.legend(loc='upper right', fontsize=11, frameon=True)

ax1.text(pct_95 * 1.05, ax1.get_ylim()[1] * 0.75, f'EMPIRICAL_MAX_DELTA_R\n= {pct_95:.4f}',
         fontsize=10, fontweight='bold', color='#c23b22', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.8))

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'step1_delta_r_distribution.png'), format='png', dpi=300, bbox_inches='tight')
plt.close(fig1)

print("[PROGRESS] Step 1: Completed Empirical Cross-Cohort Delta-R Distribution Analysis.")


# ==============================================================================
# STEP 2: EMPIRICAL BRIDGE-EDGE VULNERABILITY TEST WITH REAL DELTA-R PERTURBATION
# ==============================================================================
print("\n[PROGRESS] Step 2: Starting Empirical Bridge-Edge Vulnerability Test with Real Delta-R Perturbation...")

theta = 0.75
A73_base = (R73_overlap >= theta).astype(np.int8)
np.fill_diagonal(A73_base, 0)
G73_base = nx.from_numpy_array(A73_base)

A_real_pert = (R115_overlap >= theta).astype(np.int8)
np.fill_diagonal(A_real_pert, 0)
G_real_pert = nx.from_numpy_array(A_real_pert)

edges_base_set = set(G73_base.edges())
edges_pert_set = set(G_real_pert.edges())

flipped_lost = edges_base_set - edges_pert_set
flipped_gained = edges_pert_set - edges_base_set
total_flipped = flipped_lost | flipped_gained
preserved_edges = edges_base_set & edges_pert_set

# Fast sampling for EBC calculation on large graph
ebc_dict = nx.edge_betweenness_centrality(G73_base, k=min(250, G73_base.number_of_nodes()), seed=SEED)

ebc_flipped_vals = [ebc_dict[e] for e in flipped_lost if e in ebc_dict]
ebc_preserved_vals = [ebc_dict[e] for e in preserved_edges if e in ebc_dict]

mean_ebc_flip = float(np.mean(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
std_ebc_flip = float(np.std(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
mean_ebc_pres = float(np.mean(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
std_ebc_pres = float(np.std(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0

enrichment_ratio_step2 = mean_ebc_flip / (mean_ebc_pres + 1e-12)

if len(ebc_flipped_vals) > 0 and len(ebc_preserved_vals) > 0:
    u_stat_s2, p_val_s2 = stats.mannwhitneyu(ebc_flipped_vals, ebc_preserved_vals, alternative='greater')
    n1, n2 = len(ebc_flipped_vals), len(ebc_preserved_vals)
    rank_biserial_r_s2 = float(1.0 - (2.0 * u_stat_s2) / (n1 * n2))
else:
    u_stat_s2, p_val_s2, rank_biserial_r_s2 = 0.0, 1.0, 0.0

df_step2 = pd.DataFrame([{
    'dataset': 'GSE73002_vs_GSE115513_Real_Perturbation',
    'total_baseline_edges': len(edges_base_set),
    'flipped_lost_edges': len(flipped_lost),
    'flipped_gained_edges': len(flipped_gained),
    'total_flipped_edges': len(total_flipped),
    'preserved_edges': len(preserved_edges),
    'mean_ebc_flipped': mean_ebc_flip,
    'std_ebc_flipped': std_ebc_flip,
    'mean_ebc_preserved': mean_ebc_pres,
    'std_ebc_preserved': std_ebc_pres,
    'ebc_enrichment_ratio': enrichment_ratio_step2,
    'mann_whitney_u': u_stat_s2,
    'p_value': p_val_s2,
    'rank_biserial_r': rank_biserial_r_s2
}])
df_step2.to_csv(os.path.join(OUTPUT_DIR, 'step2_real_perturbation_ebc.csv'), index=False)

fig2, ax2 = plt.subplots(figsize=(6.5, 5), dpi=300)
bars = ax2.bar(['Flipped Edges\n(Bridge-Vulnerable)', 'Preserved Edges'], [mean_ebc_flip, mean_ebc_pres],
               yerr=[std_ebc_flip / np.sqrt(max(1, len(ebc_flipped_vals))), std_ebc_pres / np.sqrt(max(1, len(ebc_preserved_vals)))],
               color=['#c23b22', '#2ca02c'], edgecolor='black', capsize=5, width=0.45)

ax2.set_ylabel('Edge Betweenness Centrality (EBC)', fontsize=12, fontweight='bold')
ax2.set_title('Real Observed Cross-Cohort Perturbation EBC Vulnerability', fontsize=13, fontweight='bold', pad=12)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.grid(axis='y', linestyle='--', alpha=0.5)

ax2.text(0, mean_ebc_flip * 1.15, f'Enrichment = {enrichment_ratio_step2:.2f}x\nMann-Whitney $p = {p_val_s2:.2e}$',
         ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.9))

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'step2_real_perturbation_ebc.png'), format='png', dpi=300, bbox_inches='tight')
plt.close(fig2)

print("[PROGRESS] Step 2: Completed Empirical Bridge-Edge Vulnerability Test with Real Delta-R Perturbation.")


# ==============================================================================
# STEP 3: CALIBRATED HETEROSCEDASTIC MODEL AT EMPIRICAL_MAX_DELTA_R
# ==============================================================================
print("\n[PROGRESS] Step 3: Starting Calibrated Heteroscedastic Model Analysis...")

def run_calibrated_heteroscedastic_audit(df_matrix, name, max_delta_r=EMPIRICAL_MAX_DELTA_R, theta=0.75, seed=SEED):
    R_base = np.corrcoef(df_matrix.values)
    R_base = np.nan_to_num(R_base, nan=0.0)
    np.fill_diagonal(R_base, 1.0)
    
    A_base = (R_base >= theta).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_base = nx.from_numpy_array(A_base)
    
    rng = np.random.RandomState(seed)
    n = R_base.shape[0]
    raw_noise_std = (1.0 - np.abs(R_base))
    noise = rng.normal(0.0, raw_noise_std, size=(n, n))
    noise = (noise + noise.T) / 2.0
    
    max_noise_abs = np.max(np.abs(noise))
    if max_noise_abs > 0:
        noise = noise * (max_delta_r / max_noise_abs)
        
    R_pert = np.clip(R_base + noise, -1.0, 1.0)
    np.fill_diagonal(R_pert, 1.0)
    
    A_pert = (R_pert >= theta).astype(np.int8)
    np.fill_diagonal(A_pert, 0)
    G_pert = nx.from_numpy_array(A_pert)
    
    edges_base_set = set(G_base.edges())
    edges_pert_set = set(G_pert.edges())
    
    flipped_lost = edges_base_set - edges_pert_set
    flipped_gained = edges_pert_set - edges_base_set
    total_flipped = flipped_lost | flipped_gained
    preserved_edges = edges_base_set & edges_pert_set
    
    ebc_dict = nx.edge_betweenness_centrality(G_base, k=min(250, G_base.number_of_nodes()), seed=seed)
    ebc_flipped_vals = [ebc_dict[e] for e in flipped_lost if e in ebc_dict]
    ebc_preserved_vals = [ebc_dict[e] for e in preserved_edges if e in ebc_dict]
    
    mean_ebc_flip = float(np.mean(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
    std_ebc_flip = float(np.std(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
    mean_ebc_pres = float(np.mean(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
    std_ebc_pres = float(np.std(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
    
    enrichment = mean_ebc_flip / (mean_ebc_pres + 1e-12)
    
    if len(ebc_flipped_vals) > 0 and len(ebc_preserved_vals) > 0:
        u_stat, p_val = stats.mannwhitneyu(ebc_flipped_vals, ebc_preserved_vals, alternative='greater')
        n1, n2 = len(ebc_flipped_vals), len(ebc_preserved_vals)
        rank_biserial_r = float(1.0 - (2.0 * u_stat) / (n1 * n2))
    else:
        u_stat, p_val, rank_biserial_r = 0.0, 1.0, 0.0
        
    return {
        'dataset': name,
        'total_baseline_edges': len(edges_base_set),
        'flipped_lost_edges': len(flipped_lost),
        'flipped_gained_edges': len(flipped_gained),
        'total_flipped_edges': len(total_flipped),
        'preserved_edges': len(preserved_edges),
        'mean_ebc_flipped': mean_ebc_flip,
        'std_ebc_flipped': std_ebc_flip,
        'mean_ebc_preserved': mean_ebc_pres,
        'std_ebc_preserved': std_ebc_pres,
        'ebc_enrichment_ratio': enrichment,
        'mann_whitney_u': u_stat,
        'p_value': p_val,
        'rank_biserial_r': rank_biserial_r
    }, ebc_flipped_vals, ebc_preserved_vals

res_s3_73, ebc_flip_73_s3, ebc_pres_73_s3 = run_calibrated_heteroscedastic_audit(df73_top500, 'GSE73002')
res_s3_115, ebc_flip_115_s3, ebc_pres_115_s3 = run_calibrated_heteroscedastic_audit(df115_top500, 'GSE115513')

pd.DataFrame([res_s3_73]).to_csv(os.path.join(OUTPUT_DIR, 'step3_calibrated_heteroscedastic_ebc_GSE73002.csv'), index=False)
pd.DataFrame([res_s3_115]).to_csv(os.path.join(OUTPUT_DIR, 'step3_calibrated_heteroscedastic_ebc_GSE115513.csv'), index=False)

fig3, ax3 = plt.subplots(figsize=(7.5, 5.5), dpi=300)
x_pos = np.arange(2)
bar_w = 0.35

means_flip = [res_s3_73['mean_ebc_flipped'], res_s3_115['mean_ebc_flipped']]
sems_flip = [
    res_s3_73['std_ebc_flipped'] / np.sqrt(max(1, len(ebc_flip_73_s3))),
    res_s3_115['std_ebc_flipped'] / np.sqrt(max(1, len(ebc_flip_115_s3)))
]

means_pres = [res_s3_73['mean_ebc_preserved'], res_s3_115['mean_ebc_preserved']]
sems_pres = [
    res_s3_73['std_ebc_preserved'] / np.sqrt(max(1, len(ebc_pres_73_s3))),
    res_s3_115['std_ebc_preserved'] / np.sqrt(max(1, len(ebc_pres_115_s3)))
]

ax3.bar(x_pos - bar_w/2, means_flip, bar_w, yerr=sems_flip, label='Flipped Edges (Bridge-Vulnerable)', color='#c23b22', edgecolor='black', capsize=4)
ax3.bar(x_pos + bar_w/2, means_pres, bar_w, yerr=sems_pres, label='Preserved Edges', color='#2ca02c', edgecolor='black', capsize=4)

ax3.set_ylabel('Edge Betweenness Centrality (EBC)', fontsize=12, fontweight='bold')
ax3.set_title(f'Calibrated Heteroscedastic EBC Vulnerability (Max $\\Delta r$ = {EMPIRICAL_MAX_DELTA_R:.4f})', fontsize=13, fontweight='bold', pad=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['GSE73002\n(Breast Cancer)', 'GSE115513\n(Colorectal Cancer)'], fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10, frameon=True)
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.grid(axis='y', linestyle='--', alpha=0.5)

ax3.text(0 - bar_w/2, means_flip[0] * 1.12, f"Enrichment = {res_s3_73['ebc_enrichment_ratio']:.2f}x\nMW $p = {res_s3_73['p_value']:.2e}$", ha='center', va='bottom', fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffcc', alpha=0.8))
ax3.text(1 - bar_w/2, means_flip[1] * 1.12, f"Enrichment = {res_s3_115['ebc_enrichment_ratio']:.2f}x\nMW $p = {res_s3_115['p_value']:.2e}$", ha='center', va='bottom', fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffcc', alpha=0.8))

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'step3_calibrated_heteroscedastic_ebc.png'), format='png', dpi=300, bbox_inches='tight')
plt.close(fig3)

print("[PROGRESS] Step 3: Completed Calibrated Heteroscedastic Model Analysis.")


# ==============================================================================
# STEP 4: EMPIRICAL THRESHOLD SWEEP WITH EMPIRICAL_MAX_DELTA_R
# ==============================================================================
print("\n[PROGRESS] Step 4: Starting Empirical Threshold Sweep Analysis...")

def get_community_count(G):
    if G.number_of_edges() == 0:
        return G.number_of_nodes()
    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G, random_state=SEED)
        return len(set(partition.values()))
    else:
        communities = list(nx.community.louvain_communities(G, seed=SEED))
        return len(communities)

R73_top500 = np.corrcoef(df73_top500.values)
R73_top500 = np.nan_to_num(R73_top500, nan=0.0)
np.fill_diagonal(R73_top500, 1.0)

thetas = [0.70, 0.725, 0.75, 0.775, 0.80]
sweep_rows = []

for th in thetas:
    A_base = (R73_top500 >= th).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_base = nx.from_numpy_array(A_base)
    
    rng = np.random.RandomState(SEED)
    n = R73_top500.shape[0]
    raw_noise_std = (1.0 - np.abs(R73_top500))
    noise = rng.normal(0.0, raw_noise_std, size=(n, n))
    noise = (noise + noise.T) / 2.0
    max_noise_abs = np.max(np.abs(noise))
    if max_noise_abs > 0:
        noise = noise * (EMPIRICAL_MAX_DELTA_R / max_noise_abs)
        
    R_pert = np.clip(R73_top500 + noise, -1.0, 1.0)
    np.fill_diagonal(R_pert, 1.0)
    
    A_pert = (R_pert >= th).astype(np.int8)
    np.fill_diagonal(A_pert, 0)
    G_pert = nx.from_numpy_array(A_pert)
    
    edges_base_set = set(G_base.edges())
    edges_pert_set = set(G_pert.edges())
    
    flipped_lost = edges_base_set - edges_pert_set
    flipped_gained = edges_pert_set - edges_base_set
    all_flipped = flipped_lost | flipped_gained
    preserved_edges = edges_base_set & edges_pert_set
    
    ged = float(np.sum(np.abs(A_base - A_pert)) / 2.0)
    
    ebc_dict = nx.edge_betweenness_centrality(G_base, k=min(250, G_base.number_of_nodes()), seed=SEED)
    ebc_flipped_vals = [ebc_dict[e] for e in flipped_lost if e in ebc_dict]
    ebc_preserved_vals = [ebc_dict[e] for e in preserved_edges if e in ebc_dict]
    
    mean_ebc_flip = float(np.mean(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
    mean_ebc_pres = float(np.mean(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
    enrichment = mean_ebc_flip / (mean_ebc_pres + 1e-12)
    
    if len(ebc_flipped_vals) > 0 and len(ebc_preserved_vals) > 0:
        u_stat, p_val = stats.mannwhitneyu(ebc_flipped_vals, ebc_preserved_vals, alternative='greater')
    else:
        u_stat, p_val = 0.0, 1.0
        
    c_before = get_community_count(G_base)
    c_after = get_community_count(G_pert)
    abs_delta_C = abs(c_before - c_after)
    
    sweep_rows.append({
        'theta': th,
        'baseline_edge_count': len(edges_base_set),
        'flipped_edge_count': len(all_flipped),
        'graph_edit_distance': ged,
        'mean_ebc_flipped': mean_ebc_flip,
        'mean_ebc_preserved': mean_ebc_pres,
        'ebc_enrichment_ratio': enrichment,
        'mann_whitney_p_value': p_val,
        'community_count_before': c_before,
        'community_count_after': c_after,
        'abs_delta_C': abs_delta_C
    })

df_step4 = pd.DataFrame(sweep_rows)
df_step4.to_csv(os.path.join(OUTPUT_DIR, 'step4_threshold_sweep.csv'), index=False)

fig4, ax4_left = plt.subplots(figsize=(8, 5), dpi=300)
color1 = '#1f77b4'
color2 = '#d62728'

ax4_left.plot(df_step4['theta'], df_step4['flipped_edge_count'], marker='o', color=color1, linewidth=2.2, label='Flipped Edge Count')
ax4_left.set_xlabel(r'Binarisation Threshold ($\theta$)', fontsize=12, fontweight='bold')
ax4_left.set_ylabel('Flipped Edge Count', color=color1, fontsize=12, fontweight='bold')
ax4_left.tick_params(axis='y', labelcolor=color1)
ax4_left.grid(True, linestyle='--', alpha=0.5)

ax4_right = ax4_left.twinx()
ax4_right.plot(df_step4['theta'], df_step4['ebc_enrichment_ratio'], marker='s', linestyle='--', color=color2, linewidth=2.2, label='EBC Enrichment Ratio')
ax4_right.set_ylabel('EBC Enrichment Ratio (Flipped / Preserved)', color=color2, fontsize=12, fontweight='bold')
ax4_right.tick_params(axis='y', labelcolor=color2)

ax4_left.set_title(f'Empirical Threshold Sweep Dynamics (Max $\\Delta r$ = {EMPIRICAL_MAX_DELTA_R:.4f})', fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'step4_threshold_sweep.png'), format='png', dpi=300, bbox_inches='tight')
plt.close(fig4)

print("[PROGRESS] Step 4: Completed Empirical Threshold Sweep Analysis.")


# ==============================================================================
# STEP 5: PERMUTATION ROBUSTNESS TEST WITH EMPIRICAL_MAX_DELTA_R
# ==============================================================================
print("\n[PROGRESS] Step 5: Starting Permutation Robustness Test...")

perm_seeds = [42, 43, 44, 45, 46]
perm_rows = []

for idx, pseed in enumerate(perm_seeds):
    rng_perm = np.random.RandomState(pseed)
    n_genes = R73_top500.shape[0]
    perm_idx = rng_perm.permutation(n_genes)
    
    R_perm = R73_top500[perm_idx, :][:, perm_idx]
    
    A_perm_base = (R_perm >= 0.75).astype(np.int8)
    np.fill_diagonal(A_perm_base, 0)
    G_perm_base = nx.from_numpy_array(A_perm_base)
    
    rng_noise = np.random.RandomState(pseed)
    raw_noise_std = (1.0 - np.abs(R_perm))
    noise = rng_noise.normal(0.0, raw_noise_std, size=(n_genes, n_genes))
    noise = (noise + noise.T) / 2.0
    max_noise_abs = np.max(np.abs(noise))
    if max_noise_abs > 0:
        noise = noise * (EMPIRICAL_MAX_DELTA_R / max_noise_abs)
        
    R_perm_pert = np.clip(R_perm + noise, -1.0, 1.0)
    np.fill_diagonal(R_perm_pert, 1.0)
    
    A_perm_pert = (R_perm_pert >= 0.75).astype(np.int8)
    np.fill_diagonal(A_perm_pert, 0)
    G_perm_pert = nx.from_numpy_array(A_perm_pert)
    
    edges_pbase = set(G_perm_base.edges())
    edges_ppert = set(G_perm_pert.edges())
    
    flipped_lost = edges_pbase - edges_ppert
    preserved_edges = edges_pbase & edges_ppert
    
    ebc_dict = nx.edge_betweenness_centrality(G_perm_base, k=min(250, G_perm_base.number_of_nodes()), seed=pseed)
    ebc_flipped_vals = [ebc_dict[e] for e in flipped_lost if e in ebc_dict]
    ebc_preserved_vals = [ebc_dict[e] for e in preserved_edges if e in ebc_dict]
    
    mean_ebc_flip = float(np.mean(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
    mean_ebc_pres = float(np.mean(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
    enrichment = mean_ebc_flip / (mean_ebc_pres + 1e-12)
    
    if len(ebc_flipped_vals) > 0 and len(ebc_preserved_vals) > 0:
        u_stat, p_val = stats.mannwhitneyu(ebc_flipped_vals, ebc_preserved_vals, alternative='greater')
    else:
        u_stat, p_val = 0.0, 1.0
        
    perm_rows.append({
        'permutation_run': idx + 1,
        'seed': pseed,
        'mean_ebc_flipped': mean_ebc_flip,
        'mean_ebc_preserved': mean_ebc_pres,
        'ebc_enrichment_ratio': enrichment,
        'mann_whitney_p_value': p_val
    })

df_step5 = pd.DataFrame(perm_rows)
df_step5.to_csv(os.path.join(OUTPUT_DIR, 'step5_permutation_robustness.csv'), index=False)

print("[PROGRESS] Step 5: Completed Permutation Robustness Test.")


# ==============================================================================
# STEP 6: HARD VS SOFT THRESHOLDING SPECTRAL COMPARISON WITH EMPIRICAL_MAX_DELTA_R
# ==============================================================================
print("\n[PROGRESS] Step 6: Starting Hard vs Soft Thresholding Spectral Comparison...")

def compute_normalized_laplacian_spectral_distance(A_base, A_pert):
    n = A_base.shape[0]
    def norm_laplacian_eigs(adj):
        deg = np.sum(adj, axis=1).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
        return la.eigvalsh(L)

    eigs_base = norm_laplacian_eigs(A_base)
    eigs_pert = norm_laplacian_eigs(A_pert)
    return float(np.sqrt(np.sum((eigs_base - eigs_pert) ** 2)) / n)

A_hard_base = (R73_top500 >= 0.75).astype(np.float32)
np.fill_diagonal(A_hard_base, 0.0)

rng = np.random.RandomState(SEED)
n = R73_top500.shape[0]
raw_noise_std = (1.0 - np.abs(R73_top500))
noise = rng.normal(0.0, raw_noise_std, size=(n, n))
noise = (noise + noise.T) / 2.0
max_noise_abs = np.max(np.abs(noise))
if max_noise_abs > 0:
    noise = noise * (EMPIRICAL_MAX_DELTA_R / max_noise_abs)
    
R73_pert = np.clip(R73_top500 + noise, -1.0, 1.0)
np.fill_diagonal(R73_pert, 1.0)

A_hard_pert = (R73_pert >= 0.75).astype(np.float32)
np.fill_diagonal(A_hard_pert, 0.0)

dist_hard = compute_normalized_laplacian_spectral_distance(A_hard_base, A_hard_pert)

beta = 6
A_soft_base = np.power(np.abs(R73_top500), beta).astype(np.float32)
np.fill_diagonal(A_soft_base, 0.0)

A_soft_pert = np.power(np.abs(R73_pert), beta).astype(np.float32)
np.fill_diagonal(A_soft_pert, 0.0)

dist_soft = compute_normalized_laplacian_spectral_distance(A_soft_base, A_soft_pert)
fold_reduction = dist_hard / (dist_soft + 1e-12)

df_step6 = pd.DataFrame([{
    'dataset': 'GSE73002',
    'hard_threshold_spectral_distance': dist_hard,
    'soft_threshold_wgcna_spectral_distance': dist_soft,
    'spectral_distance_fold_reduction': fold_reduction
}])
df_step6.to_csv(os.path.join(OUTPUT_DIR, 'step6_hard_vs_soft_spectral.csv'), index=False)

print("[PROGRESS] Step 6: Completed Hard vs Soft Thresholding Spectral Comparison.")


# ==============================================================================
# STEP 7: BOUNDARY DENSITY AUDIT WITH EMPIRICAL_MAX_DELTA_R
# ==============================================================================
print("\n[PROGRESS] Step 7: Starting Boundary Density Audit...")

boundary_thetas = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
triu_idx_500 = np.triu_indices(500, k=1)
r_top500_vals = R73_top500[triu_idx_500]
total_pairs_500 = len(r_top500_vals)

boundary_rows = []

for th in boundary_thetas:
    count_empirical_margin = np.sum(np.abs(r_top500_vals - th) <= EMPIRICAL_MAX_DELTA_R)
    pct_empirical_margin = float(count_empirical_margin / total_pairs_500 * 100.0)
    
    count_fixed_margin = np.sum(np.abs(r_top500_vals - th) <= 0.02)
    pct_fixed_margin = float(count_fixed_margin / total_pairs_500 * 100.0)
    
    boundary_rows.append({
        'theta': th,
        'empirical_margin_delta_r': EMPIRICAL_MAX_DELTA_R,
        'count_within_empirical_margin': count_empirical_margin,
        'pct_within_empirical_margin': pct_empirical_margin,
        'count_within_fixed_0.02_margin': count_fixed_margin,
        'pct_within_fixed_0.02_margin': pct_fixed_margin
    })

df_step7 = pd.DataFrame(boundary_rows)
df_step7.to_csv(os.path.join(OUTPUT_DIR, 'step7_boundary_density_audit.csv'), index=False)

print("[PROGRESS] Step 7: Completed Boundary Density Audit.")


# ==============================================================================
# STEP 8: CROSS-CANCER MIRNA MODALITY STATEMENT
# ==============================================================================
print("\n[PROGRESS] Step 8: Writing Cross-Cancer miRNA Modality Statement...")

mean_corr_gse73 = float(np.mean(R73_overlap[np.triu_indices(len(overlapping_probes), k=1)]))
mean_corr_gse115 = float(np.mean(R115_overlap[np.triu_indices(len(overlapping_probes), k=1)]))

statement_text = f"""================================================================================
CROSS-CANCER MIRNA DATASET CHARACTERISATION STATEMENT (STEP 8)
================================================================================
- Overlapping miRNA Probes Count: {len(overlapping_probes)}
- Primary Cohort (GSE73002): Serum miRNA profiling, Breast Cancer (N=907 samples)
  * Mean Pairwise Pearson Correlation: {mean_corr_gse73:.6f}
- Replication Cohort (GSE115513): Tissue miRNA profiling, Colorectal Cancer (N=606 samples)
  * Mean Pairwise Pearson Correlation: {mean_corr_gse115:.6f}
- Cross-Cohort Correlation Shift (|Delta r|):
  * Mean Absolute Delta r Across Overlapping Pairs: {mean_abs_delta_r:.6f}
  * 95th Percentile Empirical Max Delta r: {EMPIRICAL_MAX_DELTA_R:.6f}

MODALITY STATEMENT:
Both datasets evaluated in this audit comprise non-coding RNA microRNA (miRNA) co-expression 
profiling by microarray platforms. GSE73002 captures circulating serum miRNA expression 
in a liquid biopsy breast cancer cohort, whereas GSE115513 captures tissue miRNA expression 
in a colorectal cancer cohort. All structural, topological, and betweenness centrality 
vulnerabilities audited herein pertain specifically to empirical miRNA co-expression 
networks derived from raw, non-synthetic clinical expression matrices.
================================================================================
"""

with open(os.path.join(OUTPUT_DIR, 'step8_dataset_characterisation.txt'), 'w', encoding='utf-8') as f:
    f.write(statement_text)

print("[PROGRESS] Step 8: Completed Cross-Cancer miRNA Modality Statement.")


# ==============================================================================
# STEP 9: CONSOLIDATED RESULTS SUMMARY
# ==============================================================================
print("\n[PROGRESS] Step 9: Compiling Consolidated Results Summary...")

summary_text = f"""================================================================================
          CONSOLIDATED PEER-REVIEW REVISION AUDIT RESULTS SUMMARY (STEP 9)
================================================================================

--- STEP 1: EMPIRICAL DELTA-R DISTRIBUTION ---
  * Overlapping miRNA Probes: {len(overlapping_probes)}
  * Mean Delta r: {mean_delta_r:.6f} | Mean |Delta r|: {mean_abs_delta_r:.6f} | Std Delta r: {std_delta_r:.6f}
  * Absolute Delta r Percentiles:
      - 5th: {pct_5:.6f} | 25th: {pct_25:.6f} | 50th (Median): {pct_50:.6f} | 75th: {pct_75:.6f}
      - 90th: {pct_90:.6f} | 95th (EMPIRICAL_MAX_DELTA_R): {pct_95:.6f} | 99th: {pct_99:.6f}
  * Fraction Pairs with |Delta r| > 0.01: {frac_001:.4f} | > 0.02: {frac_002:.4f} | > 0.035: {frac_0035:.4f} | > 0.05: {frac_005:.4f} | > 0.10: {frac_010:.4f}
  * EMPIRICAL_MAX_DELTA_R (95th Percentile) = {EMPIRICAL_MAX_DELTA_R:.6f}

--- STEP 2: EMPIRICAL BRIDGE-EDGE VULNERABILITY (REAL DELTA-R PERTURBATION) ---
  * Total Baseline Edges (GSE73002): {df_step2.iloc[0]['total_baseline_edges']}
  * Flipped Lost Edges: {df_step2.iloc[0]['flipped_lost_edges']} | Flipped Gained Edges: {df_step2.iloc[0]['flipped_gained_edges']} | Total Flipped: {df_step2.iloc[0]['total_flipped_edges']}
  * Mean EBC Flipped Edges: {df_step2.iloc[0]['mean_ebc_flipped']:.6e} ± {df_step2.iloc[0]['std_ebc_flipped']:.6e}
  * Mean EBC Preserved Edges: {df_step2.iloc[0]['mean_ebc_preserved']:.6e} ± {df_step2.iloc[0]['std_ebc_preserved']:.6e}
  * EBC Enrichment Ratio (Flipped / Preserved): {df_step2.iloc[0]['ebc_enrichment_ratio']:.4f}x
  * Mann-Whitney U Test: Statistic = {df_step2.iloc[0]['mann_whitney_u']:.1f}, p-value = {df_step2.iloc[0]['p_value']:.6e}, Rank-Biserial r = {df_step2.iloc[0]['rank_biserial_r']:.4f}

--- STEP 3: CALIBRATED HETEROSCEDASTIC EBC VULNERABILITY (MAX DELTA-R = {EMPIRICAL_MAX_DELTA_R:.4f}) ---
  * GSE73002 (Breast Cancer):
      - Flipped Edges: {res_s3_73['total_flipped_edges']} | Preserved Edges: {res_s3_73['preserved_edges']}
      - Mean EBC Flipped: {res_s3_73['mean_ebc_flipped']:.6e} | Preserved: {res_s3_73['mean_ebc_preserved']:.6e}
      - Enrichment Ratio: {res_s3_73['ebc_enrichment_ratio']:.4f}x | Mann-Whitney p-value: {res_s3_73['p_value']:.6e}
  * GSE115513 (Colorectal Cancer):
      - Flipped Edges: {res_s3_115['total_flipped_edges']} | Preserved Edges: {res_s3_115['preserved_edges']}
      - Mean EBC Flipped: {res_s3_115['mean_ebc_flipped']:.6e} | Preserved: {res_s3_115['mean_ebc_preserved']:.6e}
      - Enrichment Ratio: {res_s3_115['ebc_enrichment_ratio']:.4f}x | Mann-Whitney p-value: {res_s3_115['p_value']:.6e}

--- STEP 4: EMPIRICAL THRESHOLD SWEEP (GSE73002) ---
{df_step4.to_string(index=False)}

--- STEP 5: PERMUTATION ROBUSTNESS TEST ---
{df_step5.to_string(index=False)}

--- STEP 6: HARD VS SOFT THRESHOLDING SPECTRAL DISTANCE ---
  * Hard Threshold (theta = 0.75) Spectral Distance: {dist_hard:.6f}
  * Soft Threshold (WGCNA beta = 6) Spectral Distance: {dist_soft:.6f}
  * Spectral Distance Reduction Fold: {fold_reduction:.4f}x stability improvement

--- STEP 7: BOUNDARY DENSITY AUDIT (GSE73002) ---
{df_step7.to_string(index=False)}

================================================================================
"""

with open(os.path.join(OUTPUT_DIR, 'step9_consolidated_results.txt'), 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(summary_text)
print("[PROGRESS] Step 9: Completed Consolidated Results Summary.")
print("\n" + "=" * 80)
print(" ALL 10 PIPELINE STEPS SUCCESSFULLY EXECUTED. RESULTS SAVED IN mirna_audit_results/ ")
print("=" * 80 + "\n")
