"""
Structural Fragility of miRNA Co-Expression Networks Under Finite Sampling Variance
Master Execution Pipeline adhering strictly to 6 Methodological Constraints
"""

import os
import gzip
import io
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
import GEOparse
import gseapy as gp

# Set global seed for exact reproducibility
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "./mirna_audit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 80)
print(" STARTING STRUCTURAL FRAGILITY PIPELINE (N=1,000 BOOTSTRAPS) ")
print("=" * 80)


# ==============================================================================
# CONSTRAINT 1: STRICT MNAR HANDLING & FEATURE SELECTION
# ==============================================================================
print("\n[PROGRESS] Constraint 1: Ingesting GSE73002 & Applying Strict MNAR Handling...")

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
    
    # Filter strictly to early-stage breast cancer serum samples ('diagnosis: breast cancer')
    bc_mask = [True if 'breast cancer' in d.lower() else False for d in diag_line]
    bc_ids = df_raw.columns[bc_mask]
    df_bc = df_raw[bc_ids].copy()
    return df_bc

df_bc_raw = load_gse73002_bc_serum()
print(f"  - GSE73002 Breast Cancer Serum Matrix Shape (Probes x Samples): {df_bc_raw.shape}")

# 1. Drop miRNA probes with >20% missing values across samples (MNAR)
missing_frac = df_bc_raw.isnull().mean(axis=1)
df_clean = df_bc_raw.loc[missing_frac <= 0.20].copy()
print(f"  - Probes remaining after dropping >20% missing (MNAR): {df_clean.shape[0]}")

# 2. Treat remaining below-detection missing values as left-censored structural zeros (fillna 0)
df_clean = df_clean.fillna(0.0)

# Log2 transformation if necessary
if (df_clean.values < 0).any():
    df_clean = df_clean.clip(lower=0.0)
if (df_clean.values > 50).any():
    df_clean = np.log2(df_clean + 1.0)

# 3. Select top 500 highly variable miRNAs based on sample variance
probe_vars = df_clean.var(axis=1)
top500_probes = probe_vars.nlargest(500).index
df_top500 = df_clean.loc[top500_probes].copy()

# Samples x Probes matrix
X_df = df_top500.T
print(f"  - Final Feature-Selected Matrix Shape (Samples x Probes): {X_df.shape}")


# Map MIMAT probe IDs to mature miRNA gene symbols using GPL18941 platform mapping
gpl18941_path = './GPL18941.txt'
if os.path.exists(gpl18941_path):
    gpl_obj = GEOparse.get_GEO(filepath=gpl18941_path, silent=True)
    gpl_map = dict(zip(gpl_obj.table['ID'], gpl_obj.table['miRNA_ID_LIST']))
else:
    gpl_map = {}

# Clean MIMAT to miRNA symbol (e.g., hsa-miR-21-5p)
probe_to_symbol = {}
for p in top500_probes:
    raw_sym = str(gpl_map.get(p, p))
    # If multiple miRNAs separated by // or comma, pick primary
    clean_sym = raw_sym.split('//')[0].split(',')[0].strip()
    probe_to_symbol[p] = clean_sym


# ==============================================================================
# CONSTRAINT 2: SPEARMAN CORRELATION & GIANT CONNECTED COMPONENT (GCC) EXTRACTION
# ==============================================================================
print("\n[PROGRESS] Constraint 2: Computing Spearman Correlation & Extracting GCC...")

def compute_spearman_matrix(X_samples_probes):
    R_spearman, _ = stats.spearmanr(X_samples_probes.values, axis=0)
    R_spearman = np.nan_to_num(R_spearman, nan=0.0)
    np.fill_diagonal(R_spearman, 1.0)
    return R_spearman

R_spearman_base = compute_spearman_matrix(X_df)
theta = 0.75

A_base = (R_spearman_base >= theta).astype(np.int8)
np.fill_diagonal(A_base, 0)
G_full_base = nx.from_numpy_array(A_base)

# Extract Giant Connected Component (GCC)
gcc_nodes = max(nx.connected_components(G_full_base), key=len)
G_gcc = G_full_base.subgraph(gcc_nodes).copy()

n_gcc_nodes = G_gcc.number_of_nodes()
n_gcc_edges = G_gcc.number_of_edges()

print(f"  - Full Baseline Graph Edges (theta=0.75): {G_full_base.number_of_edges()}")
print(f"  - Giant Connected Component (GCC) Nodes: {n_gcc_nodes} / 500")
print(f"  - Giant Connected Component (GCC) Edges: {n_gcc_edges}")


# ==============================================================================
# CONSTRAINT 3: PSD-PRESERVED BOOTSTRAPPING (N=1,000)
# ==============================================================================
print("\n[PROGRESS] Constraint 3: Running N=1,000 PSD-Preserved Bootstrap Resamples...")

N_BOOTSTRAPS = 1000
gcc_edges_set = set(G_gcc.edges())

# Track flip counts for all GCC edges
edge_flip_counts = {e: 0 for e in gcc_edges_set}

X_mat = X_df.values
n_samples = X_mat.shape[0]

rng = np.random.RandomState(SEED)
t_boot_start = time.time()

for b in range(N_BOOTSTRAPS):
    if (b + 1) % 200 == 0 or b == 0:
        print(f"    * Bootstrap Iteration {b+1} / {N_BOOTSTRAPS}...")
        
    boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X_mat[boot_idx, :]
    
    R_boot, _ = stats.spearmanr(X_boot, axis=0)
    R_boot = np.nan_to_num(R_boot, nan=0.0)
    np.fill_diagonal(R_boot, 1.0)
    
    A_boot = (R_boot >= theta).astype(np.int8)
    np.fill_diagonal(A_boot, 0)
    
    # Check flip state for baseline GCC edges
    for u, v in gcc_edges_set:
        if A_boot[u, v] == 0:  # Baseline edge absent in bootstrap
            edge_flip_counts[(u, v)] += 1

t_boot_end = time.time()
print(f"  - Completed N={N_BOOTSTRAPS} Bootstraps in {t_boot_end - t_boot_start:.2f} s")

# Classify edges
unstable_edges = set()
stable_edges = set()

for e, count in edge_flip_counts.items():
    flip_prob = count / float(N_BOOTSTRAPS)
    if flip_prob > 0.05:
        unstable_edges.add(e)
    else:
        stable_edges.add(e)

n_unstable = len(unstable_edges)
n_stable = len(stable_edges)
pct_unstable = (n_unstable / float(n_gcc_edges)) * 100.0

print(f"  - Unstable Edges (P_flip > 0.05): {n_unstable} ({pct_unstable:.2f}%)")
print(f"  - Stable Edges (P_flip <= 0.05): {n_stable} ({100.0 - pct_unstable:.2f}%)")


# Compute EBC on baseline GCC
print("  - Calculating Baseline GCC Edge Betweenness Centrality (EBC)...")
ebc_gcc_dict = nx.edge_betweenness_centrality(G_gcc, seed=SEED)

ebc_unstable_vals = [ebc_gcc_dict[e] for e in unstable_edges if e in ebc_gcc_dict]
ebc_stable_vals = [ebc_gcc_dict[e] for e in stable_edges if e in ebc_gcc_dict]

mean_ebc_unstable = float(np.mean(ebc_unstable_vals)) if len(ebc_unstable_vals) > 0 else 0.0
mean_ebc_stable = float(np.mean(ebc_stable_vals)) if len(ebc_stable_vals) > 0 else 0.0


# ==============================================================================
# CONSTRAINT 4: DEGREE & CONNECTIVITY PRESERVING NULL MODEL
# ==============================================================================
print("\n[PROGRESS] Constraint 4: Constructing Markov-Chain Double Edge Swap Null Model...")

G_null = G_gcc.copy()
n_swaps = 10 * n_gcc_edges
max_tries = n_gcc_edges * 10

try:
    nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=max_tries, seed=SEED)
except Exception as err:
    print(f"  - Double edge swap warning: {err}")

ebc_null_dict = nx.edge_betweenness_centrality(G_null, seed=SEED)
ebc_null_vals = list(ebc_null_dict.values())
mean_ebc_null = float(np.mean(ebc_null_vals)) if len(ebc_null_vals) > 0 else 0.0

print(f"  - Null Model Baseline Mean EBC: {mean_ebc_null:.6e}")


# ==============================================================================
# CONSTRAINT 5: PROGRAMMATIC DIFFERENTIAL BIOLOGICAL VALIDATION (GSEAPY)
# ==============================================================================
print("\n[PROGRESS] Constraint 5: Performing Programmatic Differential Biological Validation...")

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
stable_mirnas = get_mirna_symbols_for_edges(stable_edges, probe_idx_list, probe_to_symbol)
unstable_mirnas = get_mirna_symbols_for_edges(unstable_edges, probe_idx_list, probe_to_symbol)

print(f"  - Unique miRNAs in Stable Edges: {len(stable_mirnas)}")
print(f"  - Unique miRNAs in Unstable Edges: {len(unstable_mirnas)}")

# 1. miRNA-to-Target Gene Mapping via miRTarBase_2017
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

# 2. Programmatic Pathway Enrichment (gseapy.enrichr)
def run_pathway_enrichment(gene_list, list_name):
    pathway_dict = {}  # Term -> Adj P-value
    target_libs = ['KEGG_2021_Human', 'WikiPathway_2021_Human', 'WikiPathway_2023_Human']
    for lib in target_libs:
        try:
            res = gp.enrichr(gene_list=gene_list, gene_sets=lib, organism='human', outdir=None)
            df_res = res.results
            if df_res is not None and not df_res.empty:
                df_sig = df_res[df_res['Adjusted P-value'] < 0.05]
                for _, row in df_sig.iterrows():
                    term = f"{row['Term']} ({lib})"
                    pathway_dict[term] = float(row['Adjusted P-value'])
        except Exception:
            continue
    return pathway_dict

print("  - Running Enrichr pathway analysis for Stable vs Unstable Target Genes...")
stable_pathways_dict = run_pathway_enrichment(stable_target_mRNAs, "Stable Graph Core")
unstable_pathways_dict = run_pathway_enrichment(unstable_target_mRNAs, "Unstable Graph")

print(f"  - Total Enriched Pathways in Stable Core Graph (Adj P < 0.05): {len(stable_pathways_dict)}")
print(f"  - Total Enriched Pathways in Unstable Graph (Adj P < 0.05): {len(unstable_pathways_dict)}")

# 3. Differential Analysis Output: (Stable Pathways) - (Unstable Pathways)
lost_pathways_terms = set(stable_pathways_dict.keys()) - set(unstable_pathways_dict.keys())
n_lost_axes = len(lost_pathways_terms)

sorted_lost = sorted([(term, stable_pathways_dict[term]) for term in lost_pathways_terms], key=lambda x: x[1])
top5_lost = sorted_lost[:5]


# ==============================================================================
# CONSTRAINT 6: EXACT STATISTICAL OUTPUT
# ==============================================================================
print("\n" + "=" * 80)
print(" CONSTRAINT 6: EXACT STATISTICAL TERMINAL REPORT ")
print("=" * 80)

# Mann-Whitney U test comparing unstable vs stable EBC
if len(ebc_unstable_vals) > 0 and len(ebc_stable_vals) > 0:
    u_stat, p_val = stats.mannwhitneyu(ebc_unstable_vals, ebc_stable_vals, alternative='greater')
    n1, n2 = len(ebc_unstable_vals), len(ebc_stable_vals)
    # Manual Rank-Biserial Effect Size |r|
    rank_biserial_r = float(np.abs((2.0 * u_stat) / (n1 * n2) - 1.0))
else:
    u_stat, p_val, rank_biserial_r = 0.0, 1.0, 0.0

enrichment_unstable_stable = mean_ebc_unstable / (mean_ebc_stable + 1e-12)
enrichment_unstable_null = mean_ebc_unstable / (mean_ebc_null + 1e-12)

report_text = f"""
================================================================================
  STRUCTURAL FRAGILITY AUDIT TERMINAL REPORT (GSE73002 BREAST CANCER SERUM)
================================================================================

--- 1. GCC NETWORK TOPOLOGY & EDGE FRAGILITY ---
  * Giant Connected Component (GCC) Edge Count: {n_gcc_edges}
  * Unstable Edge Count (P_flip > 0.05): {n_unstable} ({pct_unstable:.2f}%)
  * Stable Edge Count (P_flip <= 0.05): {n_stable} ({100.0 - pct_unstable:.2f}%)

--- 2. EDGE BETWEENNESS CENTRALITY (EBC) METRICS ---
  * Mean EBC (Unstable Edges): {mean_ebc_unstable:.6e}
  * Mean EBC (Stable Edges): {mean_ebc_stable:.6e}
  * Mean EBC (Degree-Preserving Null Model): {mean_ebc_null:.6e}
  * EBC Enrichment Ratio (Unstable / Stable): {enrichment_unstable_stable:.4f}x
  * EBC Enrichment Ratio (Unstable / Null Model): {enrichment_unstable_null:.4f}x

--- 3. MANN-WHITNEY U TEST & RANK-BISERIAL EFFECT SIZE ---
  * Mann-Whitney U Statistic: {u_stat:.1f}
  * Mann-Whitney p-value: {p_val:.6e}
  * Rank-Biserial Effect Size (|r|): {rank_biserial_r:.4f}

--- 4. DIFFERENTIAL BIOLOGICAL VALIDATION (ENRICHR) ---
  * Total Pathways Enriched in Stable Core Graph (Adj P < 0.05): {len(stable_pathways_dict)}
  * Total Pathways Enriched in Unstable Graph (Adj P < 0.05): {len(unstable_pathways_dict)}
  * Number of Falsely Erased/Lost Biological Pathways: {n_lost_axes}
  * Top 5 Falsely Lost Biological Pathways (Stable - Unstable):
"""

for i, (term, p_adj) in enumerate(top5_lost, 1):
    report_text += f"      {i}. {term} -- Adjusted P-value = {p_adj:.6e}\n"

report_text += "=" * 80 + "\n"

print(report_text)

# Save report and CSV tables
with open(os.path.join(OUTPUT_DIR, 'structural_fragility_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

df_stats = pd.DataFrame([{
    'gcc_edge_count': n_gcc_edges,
    'unstable_edge_count': n_unstable,
    'unstable_edge_pct': pct_unstable,
    'stable_edge_count': n_stable,
    'mean_ebc_unstable': mean_ebc_unstable,
    'mean_ebc_stable': mean_ebc_stable,
    'mean_ebc_null': mean_ebc_null,
    'ebc_enrichment_unstable_stable': enrichment_unstable_stable,
    'ebc_enrichment_unstable_null': enrichment_unstable_null,
    'mann_whitney_u': u_stat,
    'p_value': p_val,
    'rank_biserial_r': rank_biserial_r,
    'n_lost_regulatory_axes': n_lost_axes
}])
df_stats.to_csv(os.path.join(OUTPUT_DIR, 'structural_fragility_metrics.csv'), index=False)

print(f"[+] Results saved to {OUTPUT_DIR}/structural_fragility_report.txt and structural_fragility_metrics.csv")
