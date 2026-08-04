"""
Empirical Bridge-Edge Vulnerability Audit Pipeline
Executes 4 complete empirical analyses directly on raw transcriptomic correlation matrices
from GSE73002 (N=907) and GSE115513 (N=606).
"""

import os
import gzip
import io
import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg as la
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import community as community_louvain
    HAS_PYTHON_LOUVAIN = True
except ImportError:
    HAS_PYTHON_LOUVAIN = False

# Set global random seed for exact reproducibility
SEED = 42
np.random.seed(SEED)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']


# ==============================================================================
# Helper Functions: Data Loading & Graph Metrics
# ==============================================================================

def load_geo_expression(file_path: str, n_samples: int = 907, top_n_genes: int = 500) -> pd.DataFrame:
    """Loads raw series matrix file, extracts top N variable genes, and subsets N samples."""
    print(f"[*] Ingesting raw expression data from {file_path}...")
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    data_start = next(i for i, l in enumerate(lines) if l.startswith('!series_matrix_table_begin')) + 1
    expr_lines = [l for l in lines[data_start:] if not l.startswith('!') and l.strip()]
    
    df_raw = pd.read_csv(io.StringIO(''.join(expr_lines)), sep='\t', index_col=0)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0).T
    
    # Subset to requested N samples
    df_sub = df_raw.iloc[:min(n_samples, len(df_raw))].copy()
    
    # Select top N most variable genes by standard deviation
    gene_stds = df_sub.std(axis=0)
    top_genes = gene_stds.nlargest(top_n_genes).index
    expr_top = df_sub[top_genes]
    
    print(f"[+] Loaded {file_path}: Shape = {expr_top.shape} ({expr_top.shape[0]} samples x {expr_top.shape[1]} genes)")
    return expr_top


def get_louvain_communities(G: nx.Graph) -> int:
    """Computes Louvain community count."""
    if G.number_of_edges() == 0:
        return G.number_of_nodes()
    if HAS_PYTHON_LOUVAIN:
        partition = community_louvain.best_partition(G, random_state=SEED)
        return len(set(partition.values()))
    else:
        communities = list(nx.community.louvain_communities(G, seed=SEED))
        return len(communities)


def compute_spectral_distance_normalized(A_base: np.ndarray, A_pert: np.ndarray) -> float:
    """
    Computes normalised Laplacian Spectral Distance:
    L = D^{-1/2} A D^{-1/2}
    Distance = sqrt(sum((lambda_i_base - lambda_i_pert)^2)) / n
    """
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
    
    dist = float(np.sqrt(np.sum((eigs_base - eigs_pert) ** 2)) / n)
    return dist


def apply_heteroscedastic_noise(R_base: np.ndarray, seed: int = SEED) -> np.ndarray:
    """
    Applies heteroscedastic batch shift noise:
    std = 0.035 * (1 - |r_ij|)
    Clip to [-1, 1], re-symmetrise, diagonal = 1.0
    """
    rng = np.random.RandomState(seed)
    n = R_base.shape[0]
    noise_std = 0.035 * (1.0 - np.abs(R_base))
    noise = rng.normal(0.0, noise_std, size=(n, n))
    noise = (noise + noise.T) / 2.0
    
    R_pert = np.clip(R_base + noise, -1.0, 1.0)
    np.fill_diagonal(R_pert, 1.0)
    return R_pert


# ==============================================================================
# DATA INGESTION & INTERMEDIATE SHAPE CHECKS
# ==============================================================================
print("\n" + "=" * 80)
print(" EMPIRICAL TRANSCRIPTOMIC DATASET INGESTION ")
print("=" * 80)

expr_gse73002 = load_geo_expression('./GSE73002_series_matrix.txt.gz', n_samples=907, top_n_genes=500)
expr_gse115513 = load_geo_expression('./GSE115513_series_matrix.txt.gz', n_samples=606, top_n_genes=500)

# Compute 500x500 empirical Pearson correlation matrices directly from expression data
R_gse73002 = np.corrcoef(expr_gse73002.values.T)
R_gse73002 = np.nan_to_num(R_gse73002, nan=0.0)
np.fill_diagonal(R_gse73002, 1.0)
R_gse73002 = np.clip((R_gse73002 + R_gse73002.T) / 2.0, -1.0, 1.0)

R_gse115513 = np.corrcoef(expr_gse115513.values.T)
R_gse115513 = np.nan_to_num(R_gse115513, nan=0.0)
np.fill_diagonal(R_gse115513, 1.0)
R_gse115513 = np.clip((R_gse115513 + R_gse115513.T) / 2.0, -1.0, 1.0)

print(f"[Check] R_gse73002 correlation matrix shape: {R_gse73002.shape}, Symmetric: {np.allclose(R_gse73002, R_gse73002.T)}")
print(f"[Check] R_gse115513 correlation matrix shape: {R_gse115513.shape}, Symmetric: {np.allclose(R_gse115513, R_gse115513.T)}")


# ==============================================================================
# ANALYSIS 1 — Empirical Bridge-Edge Vulnerability Test (PRIMARY)
# ==============================================================================
print("\n" + "=" * 80)
print(" ANALYSIS 1: EMPIRICAL BRIDGE-EDGE VULNERABILITY TEST ")
print("=" * 80)

def run_analysis_1(R_matrix: np.ndarray, name: str, theta: float = 0.75, seed: int = SEED):
    # Step 2: Binarise at theta = 0.75
    A_base = (R_matrix >= theta).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_base = nx.from_numpy_array(A_base)
    
    # Step 3: Apply heteroscedastic batch shift
    R_pert = apply_heteroscedastic_noise(R_matrix, seed=seed)
    
    # Step 4: Binarise perturbed matrix at theta = 0.75
    A_pert = (R_pert >= theta).astype(np.int8)
    np.fill_diagonal(A_pert, 0)
    G_pert = nx.from_numpy_array(A_pert)
    
    # Step 5: Identify flipped lost, flipped gained, and preserved edges
    edges_base_set = set(G_base.edges())
    edges_pert_set = set(G_pert.edges())
    
    flipped_lost = edges_base_set - edges_pert_set
    flipped_gained = edges_pert_set - edges_base_set
    preserved_edges = edges_base_set & edges_pert_set
    
    all_flipped = flipped_lost | flipped_gained
    
    # Step 6: Compute Edge Betweenness Centrality (EBC) for ALL edges in G_base
    ebc_base = nx.edge_betweenness_centrality(G_base)
    
    # Step 7: Compare EBC of flipped edges vs preserved edges
    # Extract EBC for lost edges present in G_base
    ebc_flipped_vals = [ebc_base[e] for e in flipped_lost if e in ebc_base]
    ebc_preserved_vals = [ebc_base[e] for e in preserved_edges if e in ebc_base]
    
    mean_ebc_flipped = float(np.mean(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
    std_ebc_flipped = float(np.std(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
    
    mean_ebc_preserved = float(np.mean(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
    std_ebc_preserved = float(np.std(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
    
    if len(ebc_flipped_vals) > 0 and len(ebc_preserved_vals) > 0:
        stat, p_val = stats.mannwhitneyu(ebc_flipped_vals, ebc_preserved_vals, alternative='greater')
        n1, n2 = len(ebc_flipped_vals), len(ebc_preserved_vals)
        # Rank-biserial correlation r = 1 - (2U / (n1 * n2))
        rank_biserial_r = float(1.0 - (2.0 * stat) / (n1 * n2))
    else:
        stat, p_val, rank_biserial_r = 0.0, 1.0, 0.0
        
    # Step 8: Community count change
    c_before = get_louvain_communities(G_base)
    c_after = get_louvain_communities(G_pert)
    delta_C = abs(c_before - c_after)
    
    results = {
        "dataset": name,
        "total_edges_base": len(edges_base_set),
        "flipped_lost": len(flipped_lost),
        "flipped_gained": len(flipped_gained),
        "total_flipped": len(all_flipped),
        "preserved_edges": len(preserved_edges),
        "mean_ebc_flipped": mean_ebc_flipped,
        "std_ebc_flipped": std_ebc_flipped,
        "mean_ebc_preserved": mean_ebc_preserved,
        "std_ebc_preserved": std_ebc_preserved,
        "mann_whitney_u": stat,
        "p_value": p_val,
        "rank_biserial_r": rank_biserial_r,
        "community_count_before": c_before,
        "community_count_after": c_after,
        "abs_delta_C": delta_C
    }
    return results, ebc_flipped_vals, ebc_preserved_vals

res_a1_gse73002, ebc_flip_73, ebc_pres_73 = run_analysis_1(R_gse73002, "GSE73002")
res_a1_gse115513, ebc_flip_115, ebc_pres_115 = run_analysis_1(R_gse115513, "GSE115513")

df_a1_gse73002 = pd.DataFrame([res_a1_gse73002])
df_a1_gse115513 = pd.DataFrame([res_a1_gse115513])

df_a1_gse73002.to_csv("empirical_bridge_edge_GSE73002.csv", index=False)
df_a1_gse115513.to_csv("empirical_bridge_edge_GSE115513.csv", index=False)

print("\n--- Analysis 1 Results: GSE73002 ---")
print(df_a1_gse73002.to_string())
print("\n--- Analysis 1 Results: GSE115513 ---")
print(df_a1_gse115513.to_string())


# ==============================================================================
# ANALYSIS 2 — Threshold Sweep on Empirical Data
# ==============================================================================
print("\n" + "=" * 80)
print(" ANALYSIS 2: THRESHOLD SWEEP ON EMPIRICAL DATA (GSE73002) ")
print("=" * 80)

thetas = [0.70, 0.725, 0.75, 0.775, 0.80]
sweep_records = []

for th in thetas:
    A_base = (R_gse73002 >= th).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_base = nx.from_numpy_array(A_base)
    
    R_pert = apply_heteroscedastic_noise(R_gse73002, seed=SEED)
    A_pert = (R_pert >= th).astype(np.int8)
    np.fill_diagonal(A_pert, 0)
    G_pert = nx.from_numpy_array(A_pert)
    
    edges_base_set = set(G_base.edges())
    edges_pert_set = set(G_pert.edges())
    
    flipped_lost = edges_base_set - edges_pert_set
    flipped_gained = edges_pert_set - edges_base_set
    all_flipped = flipped_lost | flipped_gained
    preserved_edges = edges_base_set & edges_pert_set
    
    # Graph Edit Distance: 0.5 * sum(|A_base - A_pert|)
    ged = float(np.sum(np.abs(A_base - A_pert)) / 2.0)
    
    ebc_base = nx.edge_betweenness_centrality(G_base)
    ebc_flipped_vals = [ebc_base[e] for e in flipped_lost if e in ebc_base]
    ebc_preserved_vals = [ebc_base[e] for e in preserved_edges if e in ebc_base]
    
    mean_ebc_flipped = float(np.mean(ebc_flipped_vals)) if len(ebc_flipped_vals) > 0 else 0.0
    mean_ebc_preserved = float(np.mean(ebc_preserved_vals)) if len(ebc_preserved_vals) > 0 else 0.0
    
    if len(ebc_flipped_vals) > 0 and len(ebc_preserved_vals) > 0:
        stat, p_val = stats.mannwhitneyu(ebc_flipped_vals, ebc_preserved_vals, alternative='greater')
    else:
        stat, p_val = 0.0, 1.0
        
    c_before = get_louvain_communities(G_base)
    c_after = get_louvain_communities(G_pert)
    delta_C = abs(c_before - c_after)
    
    sweep_records.append({
        "theta": th,
        "edges_G_base": len(edges_base_set),
        "flipped_edge_count": len(all_flipped),
        "GED": ged,
        "mean_EBC_flipped": mean_ebc_flipped,
        "mean_EBC_preserved": mean_ebc_preserved,
        "mann_whitney_p_value": p_val,
        "community_before": c_before,
        "community_after": c_after,
        "abs_delta_C": delta_C
    })

df_a2_sweep = pd.DataFrame(sweep_records)
df_a2_sweep.to_csv("empirical_threshold_sweep.csv", index=False)

print("\n--- Empirical Table 2: Threshold Sweep Results ---")
print(df_a2_sweep.to_string())


# ==============================================================================
# ANALYSIS 3 — Hard vs Soft Thresholding on Empirical Data
# ==============================================================================
print("\n" + "=" * 80)
print(" ANALYSIS 3: HARD VS SOFT THRESHOLDING ON EMPIRICAL DATA (GSE73002) ")
print("=" * 80)

# Hard Thresholding (theta = 0.75)
A_hard_base = (R_gse73002 >= 0.75).astype(np.float32)
np.fill_diagonal(A_hard_base, 0.0)

R_pert_gse73002 = apply_heteroscedastic_noise(R_gse73002, seed=SEED)
A_hard_pert = (R_pert_gse73002 >= 0.75).astype(np.float32)
np.fill_diagonal(A_hard_pert, 0.0)

dist_hard = compute_spectral_distance_normalized(A_hard_base, A_hard_pert)

# Soft Thresholding (WGCNA beta = 6)
beta = 6
A_soft_base = np.power(np.abs(R_gse73002), beta).astype(np.float32)
np.fill_diagonal(A_soft_base, 0.0)

A_soft_pert = np.power(np.abs(R_pert_gse73002), beta).astype(np.float32)
np.fill_diagonal(A_soft_pert, 0.0)

dist_soft = compute_spectral_distance_normalized(A_soft_base, A_soft_pert)

df_a3_spectral = pd.DataFrame([{
    "dataset": "GSE73002",
    "hard_threshold_spectral_distance": dist_hard,
    "soft_threshold_wgcna_spectral_distance": dist_soft,
    "reduction_ratio": dist_hard / (dist_soft + 1e-12)
}])
df_a3_spectral.to_csv("hard_vs_soft_spectral.csv", index=False)

print("\n--- Hard vs Soft Normalised Laplacian Spectral Distance ---")
print(df_a3_spectral.to_string())


# ==============================================================================
# ANALYSIS 4 — Robustness Across Permuted Architectures
# ==============================================================================
print("\n" + "=" * 80)
print(" ANALYSIS 4: ROBUSTNESS ACROSS PERMUTED ARCHITECTURES (GSE73002) ")
print("=" * 80)

perm_records = []
for k in range(5):
    perm_seed = SEED + k
    rng_perm = np.random.RandomState(perm_seed)
    
    n_genes = R_gse73002.shape[0]
    perm_indices = rng_perm.permutation(n_genes)
    
    # Simultaneously permute rows and columns
    R_perm = R_gse73002[perm_indices, :][:, perm_indices]
    
    # Execute Analysis 1 pipeline at theta = 0.75
    A_perm_base = (R_perm >= 0.75).astype(np.int8)
    np.fill_diagonal(A_perm_base, 0)
    G_perm_base = nx.from_numpy_array(A_perm_base)
    
    R_perm_pert = apply_heteroscedastic_noise(R_perm, seed=perm_seed)
    A_perm_pert = (R_perm_pert >= 0.75).astype(np.int8)
    np.fill_diagonal(A_perm_pert, 0)
    G_perm_pert = nx.from_numpy_array(A_perm_pert)
    
    edges_pbase = set(G_perm_base.edges())
    edges_ppert = set(G_perm_pert.edges())
    
    flipped_p_lost = edges_pbase - edges_ppert
    preserved_p = edges_pbase & edges_ppert
    
    ebc_pbase = nx.edge_betweenness_centrality(G_perm_base)
    
    ebc_pflipped = [ebc_pbase[e] for e in flipped_p_lost if e in ebc_pbase]
    ebc_ppreserved = [ebc_pbase[e] for e in preserved_p if e in ebc_pbase]
    
    mean_ebc_pflipped = float(np.mean(ebc_pflipped)) if len(ebc_pflipped) > 0 else 0.0
    mean_ebc_ppreserved = float(np.mean(ebc_ppreserved)) if len(ebc_ppreserved) > 0 else 0.0
    
    if len(ebc_pflipped) > 0 and len(ebc_ppreserved) > 0:
        stat, p_val = stats.mannwhitneyu(ebc_pflipped, ebc_ppreserved, alternative='greater')
    else:
        stat, p_val = 0.0, 1.0
        
    perm_records.append({
        "permutation_id": k + 1,
        "seed": perm_seed,
        "mean_ebc_flipped": mean_ebc_pflipped,
        "mean_ebc_preserved": mean_ebc_ppreserved,
        "mann_whitney_p_value": p_val
    })

df_a4_perm = pd.DataFrame(perm_records)
df_a4_perm.to_csv("permutation_robustness.csv", index=False)

print("\n--- Robustness Across Permuted Gene Architectures ---")
print(df_a4_perm.to_string())


# ==============================================================================
# PUBLICATION-READY FIGURE GENERATION
# ==============================================================================
print("\n" + "=" * 80)
print(" GENERATING PUBLICATION-READY PDF FIGURES ")
print("=" * 80)

# Figure A: Bar Chart — Mean EBC of Flipped vs Preserved Edges (GSE73002 & GSE115513)
fig_a, ax_a = plt.subplots(figsize=(7.5, 5), dpi=300)

cohorts = ['GSE73002\n(N=907)', 'GSE115513\n(N=606)']
x = np.arange(len(cohorts))
width = 0.35

flipped_means = [res_a1_gse73002['mean_ebc_flipped'], res_a1_gse115513['mean_ebc_flipped']]
flipped_stds = [res_a1_gse73002['std_ebc_flipped'], res_a1_gse115513['std_ebc_flipped']]

preserved_means = [res_a1_gse73002['mean_ebc_preserved'], res_a1_gse115513['mean_ebc_preserved']]
preserved_stds = [res_a1_gse73002['std_ebc_preserved'], res_a1_gse115513['std_ebc_preserved']]

rects1 = ax_a.bar(x - width/2, flipped_means, width, yerr=flipped_stds, label='Flipped Edges (Bridge Vulnerable)', color='#c23b22', edgecolor='black', capsize=5)
rects2 = ax_a.bar(x + width/2, preserved_means, width, yerr=preserved_stds, label='Preserved Edges', color='#2ca02c', edgecolor='black', capsize=5)

ax_a.set_ylabel('Edge Betweenness Centrality (EBC)', fontsize=12, fontweight='bold')
ax_a.set_title('Empirical Bridge-Edge Centrality Vulnerability Across Cohorts', fontsize=13, fontweight='bold', pad=14)
ax_a.set_xticks(x)
ax_a.set_xticklabels(cohorts, fontsize=11, fontweight='semibold')
ax_a.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

# Annotate with Mann-Whitney p-values
p_val_73 = res_a1_gse73002['p_value']
p_val_115 = res_a1_gse115513['p_value']

ax_a.text(0, max(flipped_means[0] + flipped_stds[0], preserved_means[0]) * 1.15,
          f"Mann-Whitney p = {p_val_73:.2e}", ha='center', va='bottom', fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
ax_a.text(1, max(flipped_means[1] + flipped_stds[1], preserved_means[1]) * 1.15,
          f"Mann-Whitney p = {p_val_115:.2e}", ha='center', va='bottom', fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax_a.set_ylim(0, max(flipped_means) * 1.6)
plt.tight_layout()
fig_a.savefig("empirical_EBC_comparison.pdf", format='pdf', bbox_inches='tight')
plt.close(fig_a)
print("[+] Saved empirical_EBC_comparison.pdf")

# Figure B: Table / Heatmap of Threshold Sweep Results (Analysis 2)
fig_b, ax_b = plt.subplots(figsize=(9, 4), dpi=300)
ax_b.axis('tight')
ax_b.axis('off')

# Format DataFrame for visual table display
df_table_display = df_a2_sweep.copy()
df_table_display['theta'] = df_table_display['theta'].map('{:.3f}'.format)
df_table_display['GED'] = df_table_display['GED'].map('{:.1f}'.format)
df_table_display['mean_EBC_flipped'] = df_table_display['mean_EBC_flipped'].map('{:.6f}'.format)
df_table_display['mean_EBC_preserved'] = df_table_display['mean_EBC_preserved'].map('{:.6f}'.format)
df_table_display['mann_whitney_p_value'] = df_table_display['mann_whitney_p_value'].map('{:.2e}'.format)

table_data = [df_table_display.columns.tolist()] + df_table_display.values.tolist()
table = ax_b.table(cellText=table_data, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.4)

# Format header row
for i in range(len(df_table_display.columns)):
    cell = table[0, i]
    cell.set_facecolor('#1f77b4')
    cell.get_text().set_color('white')
    cell.get_text().set_weight('bold')

plt.title("Empirical Table 2: Threshold Sweep Results (GSE73002)", fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
fig_b.savefig("empirical_threshold_sweep.pdf", format='pdf', bbox_inches='tight')
plt.close(fig_b)
print("[+] Saved empirical_threshold_sweep.pdf")

print("\n" + "=" * 80)
print(" PIPELINE EXECUTION COMPLETE. ALL 5 CSVs AND 2 PDFs GENERATED. ")
print("=" * 80 + "\n")
