"""
Weighted-Network (Soft-Thresholding) Robustness Comparison on GSE115513 Tissue
Generates Figure 4: 'fuck it/fig4_weighted_robustness.png' at 300 DPI
Comparing Hard Binarization Variance vs Soft-Thresholding (WGCNA beta=6) Variance across distance-to-threshold bins
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
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# Enable publication font styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Global Random Seed
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "./fuck it"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 90)
print(" STARTING WEIGHTED-NETWORK (SOFT-THRESHOLDING) ROBUSTNESS COMPARISON ")
print("=" * 90)

# Load GSE115513 Un-censored Data
gse115_path = './GSE115513_series_matrix.txt.gz'
with gzip.open(gse115_path, 'rt', encoding='utf-8', errors='ignore') as f:
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

target_char = next(cl for cl in char_lines if any('carcinoma' in x.lower() for x in cl))
mask = [True if 'carcinoma' in x.lower() else False for x in target_char]
sub_ids = df_raw.columns[mask]
df_sub = df_raw[sub_ids].copy()

missing_frac = df_sub.isnull().mean(axis=1)
df_clean = df_sub.loc[missing_frac <= 0.20].dropna(axis=0).copy()

if (df_clean.values > 50).any():
    df_clean = np.log2(df_clean + 1.0)

probe_mads = median_abs_deviation(df_clean.values, axis=1)
mad_series = pd.Series(probe_mads, index=df_clean.index)
top500_probes = mad_series.nlargest(500).index
df_top500 = df_clean.loc[top500_probes].copy()

X_mat = df_top500.T.values
n_samples = X_mat.shape[0]

# Spearman Rank Correlation Matrix
R_spearman, _ = stats.spearmanr(X_mat, axis=0)
R_spearman = np.nan_to_num(R_spearman, nan=0.0)
np.fill_diagonal(R_spearman, 1.0)

theta_target = 0.8190
beta_power = 6.0

A_base = (R_spearman >= theta_target).astype(np.int8)
np.fill_diagonal(A_base, 0)
G_full = nx.from_numpy_array(A_base)

gcc_nodes = max(nx.connected_components(G_full), key=len)
G_gcc = G_full.subgraph(gcc_nodes).copy()

gcc_edges = list(G_gcc.edges())
n_gcc_edges = len(gcc_edges)

distances = np.array([abs(abs(R_spearman[u, v]) - theta_target) for u, v in gcc_edges])

print(f"  * Homogeneous GSE115513 Tissue Size (N): {n_samples}")
print(f"  * GCC Node Count: {G_gcc.number_of_nodes()}, GCC Edge Count: {n_gcc_edges}")
print(f"  * Hard Threshold (theta_target): {theta_target}, Soft Power (beta): {beta_power}")

# ------------------------------------------------------------------------------
# STEP 2: Bootstrap Edge Variance Comparison (N=1,000)
# ------------------------------------------------------------------------------
print("\n[STEP 2] Running N=1,000 Patient-Resampling Bootstraps for Hard vs Soft Variance...")

N_BOOT = 1000
rng = np.random.RandomState(SEED)

hard_states = np.zeros((N_BOOT, n_gcc_edges), dtype=np.float32)
soft_weights = np.zeros((N_BOOT, n_gcc_edges), dtype=np.float32)

t_start = time.time()

for b in range(N_BOOT):
    b_idx = rng.choice(n_samples, size=n_samples, replace=True)
    X_b = X_mat[b_idx, :]
    R_b, _ = stats.spearmanr(X_b, axis=0)
    R_b = np.nan_to_num(R_b, nan=0.0)
    np.fill_diagonal(R_b, 1.0)
    
    for idx_e, (u, v) in enumerate(gcc_edges):
        r_val = abs(R_b[u, v])
        # Hard binary state: 1 if r_val >= theta_target else 0
        hard_states[b, idx_e] = 1.0 if r_val >= theta_target else 0.0
        # Soft edge weight: W_ij = |r_ij|^beta
        soft_weights[b, idx_e] = r_val ** beta_power

t_end = time.time()
print(f"  * Completed N=1,000 Bootstraps in {t_end - t_start:.2f} s")

# Compute continuous variance across 1,000 bootstraps for each edge
hard_var = np.var(hard_states, axis=0)  # Equals P_flip * (1 - P_flip)
soft_var = np.var(soft_weights, axis=0)

# ------------------------------------------------------------------------------
# STEP 3: Group into 10 Distance Bins & Plot Figure 4
# ------------------------------------------------------------------------------
print("\n[STEP 3] Grouping GCC Edges into 10 Distance Bins & Plotting Figure 4...")

df_edges = pd.DataFrame({
    'Distance': distances,
    'Hard_Var': hard_var,
    'Soft_Var': soft_var
})

# Quantile-based 10 bins for balanced edge counts per bin
df_edges['Distance_Bin'] = pd.qcut(df_edges['Distance'], q=10, duplicates='drop')

bin_stats = df_edges.groupby('Distance_Bin', observed=False).agg(
    mean_dist=('Distance', 'mean'),
    mean_hard_var=('Hard_Var', 'mean'),
    mean_soft_var=('Soft_Var', 'mean'),
    count=('Distance', 'count')
).reset_index()

print(bin_stats)

fig, ax1 = plt.subplots(figsize=(8.5, 6), dpi=300)

x_centers = bin_stats['mean_dist'].values
y_hard = bin_stats['mean_hard_var'].values
y_soft = bin_stats['mean_soft_var'].values

# Plot Hard Binarization Variance (Red Line)
color_hard = '#c0392b'
ax1.plot(x_centers, y_hard, color=color_hard, linewidth=2.8, marker='o', markersize=7, label='Hard Binarization Variance')
ax1.set_xlabel('Absolute Distance from Threshold (|r| - θ)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Hard Binarization Edge Variance', fontsize=12, fontweight='bold', color=color_hard, labelpad=10)
ax1.tick_params(axis='y', labelcolor=color_hard)
ax1.grid(True, linestyle='--', alpha=0.4)

# Plot Soft-Thresholding Variance (Blue Line) on secondary axis or primary
ax2 = ax1.twinx()
color_soft = '#2980b9'
ax2.plot(x_centers, y_soft, color=color_soft, linewidth=2.8, linestyle='--', marker='s', markersize=7, label='Soft-Thresholding Variance (WGCNA β=6)')
ax2.set_ylabel('Soft-Thresholding Edge Variance', fontsize=12, fontweight='bold', color=color_soft, labelpad=10)
ax2.tick_params(axis='y', labelcolor=color_soft)

plt.title('Soft-Thresholding Mitigates Boundary Fragility', fontsize=14, fontweight='bold', pad=15)

# Combined Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=11)

plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "fig4_weighted_robustness.png")
fig.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\n[+] Successfully generated Figure 4: '{fig4_path}' at 300 DPI!")
