"""
Dual-Cohort Mathematically Strict Structural Fragility Pipeline
Independently analyzes GSE73002 (Breast Cancer Serum) and GSE115513 (Colorectal Tissue)
Targeted 2.5% Sparsity, MAD Feature Selection, Edge-Label Permutation Test (N=1000), Honest WGCNA & Pathway Validation
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
print(" STARTING DUAL-COHORT MATHEMATICALLY STRICT STRUCTURAL FRAGILITY PIPELINE ")
print("=" * 80)


# Fetch miRTarBase_2017 library dictionary for miRNA target mapping
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


def run_cohort_fragility_pipeline(filepath, cohort_name, filter_key, filter_val, gpl_path='./GPL18941.txt'):
    print(f"\n" + "=" * 80)
    print(f" PIPELINE EXECUTION FOR COHORT: {cohort_name} ")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # 1. Preprocessing & Sparsity Targeting
    # --------------------------------------------------------------------------
    print(f"\n[1/5] Ingesting {cohort_name} & Applying MAD Selection...")
    
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
    
    # Filter to homogeneous phenotype
    target_char = next(cl for cl in char_lines if any(filter_val.lower() in x.lower() for x in cl))
    mask = [True if filter_val.lower() in x.lower() else False for x in target_char]
    sub_ids = df_raw.columns[mask]
    df_sub = df_raw[sub_ids].copy()
    
    print(f"  - Strictly Subsampled Homogeneous Cohort Size (N): {df_sub.shape[1]}")
    
    # Drop >20% missing, fillna 0.0 (left-censored structural zeros)
    missing_frac = df_sub.isnull().mean(axis=1)
    df_clean = df_sub.loc[missing_frac <= 0.20].fillna(0.0).copy()
    
    if (df_clean.values < 0).any():
        df_clean = df_clean.clip(lower=0.0)
    if (df_clean.values > 50).any():
        df_clean = np.log2(df_clean + 1.0)
        
    # Select top 500 miRNAs using MAD
    probe_mads = median_abs_deviation(df_clean.values, axis=1)
    mad_series = pd.Series(probe_mads, index=df_clean.index)
    top500_probes = mad_series.nlargest(500).index
    df_top500 = df_clean.loc[top500_probes].copy()
    
    X_df = df_top500.T  # Samples x Probes
    print(f"  - Final Feature Matrix (Samples x Probes): {X_df.shape}")
    
    # Probe to mature miRNA symbol mapping
    if os.path.exists(gpl_path):
        gpl_obj = GEOparse.get_GEO(filepath=gpl_path, silent=True)
        gpl_map = dict(zip(gpl_obj.table['ID'], gpl_obj.table['miRNA_ID_LIST']))
    else:
        gpl_map = {}
        
    probe_to_symbol = {}
    for p in top500_probes:
        raw_sym = str(gpl_map.get(p, p))
        clean_sym = raw_sym.split('//')[0].split(',')[0].strip()
        probe_to_symbol[p] = clean_sym
        
    # Dynamic Thresholding to achieve 2.5% Target Density
    R_spearman = np.corrcoef(df_top500.values.T)  # or stats.spearmanr
    R_spearman, _ = stats.spearmanr(X_df.values, axis=0)
    R_spearman = np.nan_to_num(R_spearman, nan=0.0)
    np.fill_diagonal(R_spearman, 1.0)
    
    n_nodes = 500
    total_possible_edges = n_nodes * (n_nodes - 1) / 2.0  # 124,750
    target_density = 0.025  # 2.5%
    
    best_theta = 0.75
    min_diff = 1.0
    best_edge_count = 0
    best_density = 0.0
    
    for th in np.arange(0.99, 0.499, -0.001):
        A_temp = (R_spearman >= th).astype(np.int8)
        np.fill_diagonal(A_temp, 0)
        e_count = int(np.sum(A_temp) / 2)
        dens = e_count / total_possible_edges
        diff = abs(dens - target_density)
        if diff < min_diff:
            min_diff = diff
            best_theta = float(th)
            best_edge_count = e_count
            best_density = float(dens)
            
    theta_target = best_theta
    print(f"  - Dynamic Optimal Threshold (theta_target): {theta_target:.4f}")
    print(f"  - Baseline Density at theta_target: {best_density * 100.0:.4f}%")
    print(f"  - Baseline Edge Count: {best_edge_count}")
    
    A_base = (R_spearman >= theta_target).astype(np.int8)
    np.fill_diagonal(A_base, 0)
    G_full = nx.from_numpy_array(A_base)
    
    gcc_nodes = max(nx.connected_components(G_full), key=len)
    G_gcc = G_full.subgraph(gcc_nodes).copy()
    
    v_gcc = G_gcc.number_of_nodes()
    e_gcc = G_gcc.number_of_edges()
    n_disconnected = 500 - v_gcc
    
    print(f"  - GCC Node Count: {v_gcc} / 500")
    print(f"  - GCC Edge Count: {e_gcc}")
    print(f"  - Disconnected Nodes Excluded: {n_disconnected}")
    
    # --------------------------------------------------------------------------
    # 2. Bootstrapping (N=1000) & Phase Collapse
    # --------------------------------------------------------------------------
    print(f"\n[2/5] Running N=1,000 Patient-Resampling Bootstraps at theta_target={theta_target:.4f}...")
    
    N_BOOT = 1000
    gcc_edges_set = set(G_gcc.edges())
    edge_flip_counts = {e: 0 for e in gcc_edges_set}
    
    X_mat = X_df.values
    n_samples = X_mat.shape[0]
    rng = np.random.RandomState(SEED)
    
    t_boot_start = time.time()
    for b in range(N_BOOT):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_mat[boot_idx, :]
        
        R_boot, _ = stats.spearmanr(X_boot, axis=0)
        R_boot = np.nan_to_num(R_boot, nan=0.0)
        np.fill_diagonal(R_boot, 1.0)
        
        A_boot = (R_boot >= theta_target).astype(np.int8)
        np.fill_diagonal(A_boot, 0)
        
        for u, v in gcc_edges_set:
            if A_boot[u, v] == 0:
                edge_flip_counts[(u, v)] += 1
                
    t_boot_end = time.time()
    print(f"  - Completed N={N_BOOT} Bootstraps in {t_boot_end - t_boot_start:.2f} s")
    
    unstable_edges = {e for e, c in edge_flip_counts.items() if (c / float(N_BOOT)) > 0.05}
    stable_edges = gcc_edges_set - unstable_edges
    
    n_unstable = len(unstable_edges)
    n_stable = len(stable_edges)
    pct_unstable = (n_unstable / float(e_gcc)) * 100.0
    
    print(f"  - Unstable Edges (P_flip > 0.05): {n_unstable} ({pct_unstable:.2f}%)")
    print(f"  - Stable Edges (P_flip <= 0.05): {n_stable} ({100.0 - pct_unstable:.2f}%)")
    
    # --------------------------------------------------------------------------
    # 3. Empirical Permutation Test for Non-Independent EBC
    # --------------------------------------------------------------------------
    print(f"\n[3/5] Calculating GCC EBC & Edge-Label Permutation Test (N=1000)...")
    
    ebc_gcc_dict = nx.edge_betweenness_centrality(G_gcc, seed=SEED)
    
    ebc_unstable_vals = [ebc_gcc_dict[e] for e in unstable_edges if e in ebc_gcc_dict]
    ebc_stable_vals = [ebc_gcc_dict[e] for e in stable_edges if e in ebc_gcc_dict]
    
    mean_ebc_unstable = float(np.mean(ebc_unstable_vals)) if len(ebc_unstable_vals) > 0 else 0.0
    mean_ebc_stable = float(np.mean(ebc_stable_vals)) if len(ebc_stable_vals) > 0 else 0.0
    enrichment_ratio = mean_ebc_unstable / (mean_ebc_stable + 1e-12)
    
    if len(ebc_unstable_vals) > 0 and len(ebc_stable_vals) > 0:
        u_obs, _ = stats.mannwhitneyu(ebc_unstable_vals, ebc_stable_vals, alternative='greater')
        n1, n2 = len(ebc_unstable_vals), len(ebc_stable_vals)
        rank_biserial_r = float(np.abs((2.0 * u_obs) / (n1 * n2) - 1.0))
    else:
        u_obs, n1, n2, rank_biserial_r = 0.0, 1, 1, 0.0
        
    # Edge-Label Permutation Test across exact GCC topology (N=1000 shuffles)
    all_gcc_edges = list(G_gcc.edges())
    all_ebc_array = np.array([ebc_gcc_dict[e] for e in all_gcc_edges])
    
    N_PERM = 1000
    u_perm_count = 0
    rng_perm = np.random.RandomState(SEED)
    
    for _ in range(N_PERM):
        shuffled_indices = rng_perm.permutation(len(all_gcc_edges))
        perm_unstable_ebc = all_ebc_array[shuffled_indices[:n1]]
        perm_stable_ebc = all_ebc_array[shuffled_indices[n1:]]
        
        u_perm, _ = stats.mannwhitneyu(perm_unstable_ebc, perm_stable_ebc, alternative='greater')
        if u_perm >= u_obs:
            u_perm_count += 1
            
    empirical_p_val = float((1.0 + u_perm_count) / (N_PERM + 1.0))
    
    print(f"  - Mean EBC Unstable: {mean_ebc_unstable:.6e}")
    print(f"  - Mean EBC Stable: {mean_ebc_stable:.6e}")
    print(f"  - EBC Enrichment Ratio (Unstable / Stable): {enrichment_ratio:.4f}x")
    print(f"  - Rank-Biserial Effect Size (|r|): {rank_biserial_r:.4f}")
    print(f"  - Edge-Label Permutation Empirical p-value (N=1000): {empirical_p_val:.6e}")
    
    # --------------------------------------------------------------------------
    # 4. Honest WGCNA Evaluation
    # --------------------------------------------------------------------------
    print(f"\n[4/5] Evaluating WGCNA Scale-Free Topology Fit R^2 across Power Beta 1 to 20...")
    
    best_beta = 1
    best_r2 = -1.0
    for beta in range(1, 21):
        A_soft = np.power(np.abs(R_spearman), beta)
        np.fill_diagonal(A_soft, 0.0)
        k_vec = np.sum(A_soft, axis=1)
        if np.max(k_vec) == np.min(k_vec): continue
        hist, bin_edges = np.histogram(k_vec, bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        valid = (hist > 0) & (bin_centers > 0)
        if np.sum(valid) < 3: continue
        log_k = np.log10(bin_centers[valid])
        log_pk = np.log10(hist[valid] / np.sum(hist))
        reg = stats.linregress(log_k, log_pk)
        r2 = reg.rvalue ** 2
        if r2 > best_r2:
            best_r2 = r2
            best_beta = beta
            
    wgcna_passed = (best_r2 >= 0.85)
    wgcna_status_str = f"PASSED (R^2 = {best_r2:.4f} >= 0.85)" if wgcna_passed else f"FAILED (R^2 = {best_r2:.4f} < 0.85)"
    print(f"  - Optimal WGCNA Power Beta: {best_beta}")
    print(f"  - WGCNA Scale-Free Topology Fit Evaluation: {wgcna_status_str}")
    
    # --------------------------------------------------------------------------
    # 5. Updated Biological Validation
    # --------------------------------------------------------------------------
    print(f"\n[5/5] Differential Biological Pathway Validation (KEGG)...")
    
    probe_idx_list = list(top500_probes)
    
    def get_mirna_symbols(edge_set):
        syms = set()
        for u, v in edge_set:
            p_u = probe_idx_list[u]
            p_v = probe_idx_list[v]
            syms.add(probe_to_symbol.get(p_u, p_u))
            syms.add(probe_to_symbol.get(p_v, p_v))
        return list(syms)
        
    stable_mirnas = get_mirna_symbols(stable_edges)
    unstable_mirnas = get_mirna_symbols(unstable_edges)
    
    stable_target_genes = map_mirnas_to_target_genes(stable_mirnas)
    unstable_target_genes = map_mirnas_to_target_genes(unstable_mirnas)
    
    stable_kegg_dict = run_kegg_enrichment(stable_target_genes)
    unstable_kegg_dict = run_kegg_enrichment(unstable_target_genes)
    
    # Set difference: Pathways in Stable Graph Core lost in Unstable Graph
    lost_pathways_all = set(stable_kegg_dict.keys()) - set(unstable_kegg_dict.keys())
    
    generic_terms = {'Ribosome', 'Spliceosome', 'Proteasome', 'RNA transport', 'Nucleotide excision repair',
                     'Basal transcription factors', 'DNA replication', 'Mismatch repair', 'Homologous recombination'}
    
    lost_regulatory_pathways = [p for p in lost_pathways_all if not any(g.lower() in p.lower() for g in generic_terms)]
    n_lost_regulatory = len(lost_regulatory_pathways)
    
    sorted_lost = sorted([(p, stable_kegg_dict[p]) for p in lost_regulatory_pathways], key=lambda x: x[1])
    top5_lost = sorted_lost[:5]
    
    print(f"  - Enriched KEGG Pathways in Stable Core: {len(stable_kegg_dict)}")
    print(f"  - Enriched KEGG Pathways in Unstable Graph: {len(unstable_kegg_dict)}")
    print(f"  - Falsely Lost Disease-Relevant Regulatory Pathways: {n_lost_regulatory}")
    
    return {
        'cohort': cohort_name,
        'N_samples': n_samples,
        'theta_target': theta_target,
        'baseline_density': best_density,
        'baseline_edges': best_edge_count,
        'gcc_nodes': v_gcc,
        'gcc_edges': e_gcc,
        'disconnected_nodes': n_disconnected,
        'unstable_edges': n_unstable,
        'unstable_pct': pct_unstable,
        'stable_edges': n_stable,
        'mean_ebc_unstable': mean_ebc_unstable,
        'mean_ebc_stable': mean_ebc_stable,
        'ebc_enrichment': enrichment_ratio,
        'rank_biserial_r': rank_biserial_r,
        'empirical_p_val': empirical_p_val,
        'wgcna_beta': best_beta,
        'wgcna_r2': best_r2,
        'wgcna_passed': wgcna_passed,
        'n_lost_regulatory_pathways': n_lost_regulatory,
        'top5_lost': top5_lost
    }


# Execute for Cohort 1: GSE73002 (Breast Cancer Serum)
res_gse73 = run_cohort_fragility_pipeline(
    filepath='./GSE73002_series_matrix.txt.gz',
    cohort_name="GSE73002 (Breast Cancer Serum)",
    filter_key='diagnosis',
    filter_val='breast cancer'
)

# Execute for Cohort 2: GSE115513 (Colorectal Carcinoma Tissue)
res_gse115 = run_cohort_fragility_pipeline(
    filepath='./GSE115513_series_matrix.txt.gz',
    cohort_name="GSE115513 (Colorectal Carcinoma Tissue)",
    filter_key='tissue',
    filter_val='carcinoma'
)


# ==============================================================================
# CONSOLIDATED DUAL-COHORT REPORT
# ==============================================================================
print("\n" + "=" * 80)
print(" CONSOLIDATED DUAL-COHORT MATHEMATICALLY STRICT REPORT ")
print("=" * 80)

report_text = f"""
================================================================================
     DUAL-COHORT MATHEMATICALLY STRICT STRUCTURAL FRAGILITY REPORT
================================================================================

--------------------------------------------------------------------------------
1. COHORT 1: {res_gse73['cohort']}
--------------------------------------------------------------------------------
  * Homogeneous Sample Size (N): {res_gse73['N_samples']}
  * Dynamic Target Threshold (theta_target): {res_gse73['theta_target']:.4f}
  * Network Density at theta_target: {res_gse73['baseline_density']*100.0:.4f}%
  * GCC Edge Count: {res_gse73['gcc_edges']} (Disconnected Nodes Excluded: {res_gse73['disconnected_nodes']})
  * Unstable Edges (P_flip > 0.05): {res_gse73['unstable_edges']} ({res_gse73['unstable_pct']:.2f}%)
  * Stable Edges (P_flip <= 0.05): {res_gse73['stable_edges']}
  * Mean EBC Unstable Edges: {res_gse73['mean_ebc_unstable']:.6e}
  * Mean EBC Stable Edges: {res_gse73['mean_ebc_stable']:.6e}
  * EBC Enrichment Ratio (Unstable / Stable): {res_gse73['ebc_enrichment']:.4f}x
  * Rank-Biserial Effect Size (|r|): {res_gse73['rank_biserial_r']:.4f}
  * Edge-Label Permutation Empirical p-value (N=1000): {res_gse73['empirical_p_val']:.6e}
  * WGCNA Scale-Free Topology Fit (beta={res_gse73['wgcna_beta']}): R^2 = {res_gse73['wgcna_r2']:.4f} -> {'PASSED (>= 0.85)' if res_gse73['wgcna_passed'] else 'FAILED (< 0.85)'}
  * Falsely Erased Regulatory/Oncogenic Pathways: {res_gse73['n_lost_regulatory_pathways']}
  * Top 5 Falsely Lost Biological Pathways:
"""

for i, (term, p_adj) in enumerate(res_gse73['top5_lost'], 1):
    report_text += f"      {i}. {term} -- Adjusted P-value = {p_adj:.6e}\n"

report_text += f"""
--------------------------------------------------------------------------------
2. COHORT 2: {res_gse115['cohort']}
--------------------------------------------------------------------------------
  * Homogeneous Sample Size (N): {res_gse115['N_samples']}
  * Dynamic Target Threshold (theta_target): {res_gse115['theta_target']:.4f}
  * Network Density at theta_target: {res_gse115['baseline_density']*100.0:.4f}%
  * GCC Edge Count: {res_gse115['gcc_edges']} (Disconnected Nodes Excluded: {res_gse115['disconnected_nodes']})
  * Unstable Edges (P_flip > 0.05): {res_gse115['unstable_edges']} ({res_gse115['unstable_pct']:.2f}%)
  * Stable Edges (P_flip <= 0.05): {res_gse115['stable_edges']}
  * Mean EBC Unstable Edges: {res_gse115['mean_ebc_unstable']:.6e}
  * Mean EBC Stable Edges: {res_gse115['mean_ebc_stable']:.6e}
  * EBC Enrichment Ratio (Unstable / Stable): {res_gse115['ebc_enrichment']:.4f}x
  * Rank-Biserial Effect Size (|r|): {res_gse115['rank_biserial_r']:.4f}
  * Edge-Label Permutation Empirical p-value (N=1000): {res_gse115['empirical_p_val']:.6e}
  * WGCNA Scale-Free Topology Fit (beta={res_gse115['wgcna_beta']}): R^2 = {res_gse115['wgcna_r2']:.4f} -> {'PASSED (>= 0.85)' if res_gse115['wgcna_passed'] else 'FAILED (< 0.85)'}
  * Falsely Erased Regulatory/Oncogenic Pathways: {res_gse115['n_lost_regulatory_pathways']}
  * Top 5 Falsely Lost Biological Pathways:
"""

for i, (term, p_adj) in enumerate(res_gse115['top5_lost'], 1):
    report_text += f"      {i}. {term} -- Adjusted P-value = {p_adj:.6e}\n"

report_text += "=" * 80 + "\n"

print(report_text)

# Save report text and CSV
with open(os.path.join(OUTPUT_DIR, 'dual_cohort_structural_fragility_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

df_dual = pd.DataFrame([{
    'cohort': res_gse73['cohort'],
    'N_samples': res_gse73['N_samples'],
    'theta_target': res_gse73['theta_target'],
    'density_pct': res_gse73['baseline_density'] * 100.0,
    'gcc_edges': res_gse73['gcc_edges'],
    'disconnected_nodes': res_gse73['disconnected_nodes'],
    'unstable_edges': res_gse73['unstable_edges'],
    'unstable_pct': res_gse73['unstable_pct'],
    'ebc_enrichment': res_gse73['ebc_enrichment'],
    'rank_biserial_r': res_gse73['rank_biserial_r'],
    'empirical_p_val': res_gse73['empirical_p_val'],
    'wgcna_beta': res_gse73['wgcna_beta'],
    'wgcna_r2': res_gse73['wgcna_r2'],
    'wgcna_passed': res_gse73['wgcna_passed'],
    'lost_regulatory_pathways': res_gse73['n_lost_regulatory_pathways']
}, {
    'cohort': res_gse115['cohort'],
    'N_samples': res_gse115['N_samples'],
    'theta_target': res_gse115['theta_target'],
    'density_pct': res_gse115['baseline_density'] * 100.0,
    'gcc_edges': res_gse115['gcc_edges'],
    'disconnected_nodes': res_gse115['disconnected_nodes'],
    'unstable_edges': res_gse115['unstable_edges'],
    'unstable_pct': res_gse115['unstable_pct'],
    'ebc_enrichment': res_gse115['ebc_enrichment'],
    'rank_biserial_r': res_gse115['rank_biserial_r'],
    'empirical_p_val': res_gse115['empirical_p_val'],
    'wgcna_beta': res_gse115['wgcna_beta'],
    'wgcna_r2': res_gse115['wgcna_r2'],
    'wgcna_passed': res_gse115['wgcna_passed'],
    'lost_regulatory_pathways': res_gse115['n_lost_regulatory_pathways']
}])

df_dual.to_csv(os.path.join(OUTPUT_DIR, 'dual_cohort_structural_fragility_metrics.csv'), index=False)

print(f"[+] Output written to {OUTPUT_DIR}/dual_cohort_structural_fragility_report.txt and dual_cohort_structural_fragility_metrics.csv")
