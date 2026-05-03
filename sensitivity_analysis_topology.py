"""
Sensitivity Analysis: Topology Feature Distribution Comparison
==============================================================
Compares feature distributions between GSE73002 (Real mRNA) and 
other datasets (Imputed) to validate robustness and report discrepancies.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from pathlib import Path

# --- Paths ---
BASE = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV = BASE / "features" / "feature_matrix_combat.csv"
OUT_DIR = BASE / "analysis_output" / "sensitivity_topology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
print(f"Loading {FEAT_CSV}...")
df = pd.read_csv(FEAT_CSV, index_col=0)

# Identify topology features
with open(BASE / "models" / "topology_feature_names.json") as f:
    topo_cols = json.load(f)

# Reference vs Others
REF_DS = "GSE73002"
IMPUTED_DS = [ds for ds in df["dataset"].unique() if ds != REF_DS]

print(f"Reference dataset: {REF_DS}")
print(f"Other datasets: {IMPUTED_DS}")

results = []

# --- Statistical Comparison (Kruskal-Wallis) ---
for col in topo_cols:
    groups = []
    labels = []
    
    # 1. Collect data for Reference
    ref_data = df.loc[df["dataset"] == REF_DS, col].dropna()
    if len(ref_data) > 0:
        groups.append(ref_data)
        labels.append(REF_DS)
    
    # 2. Collect data for Others
    for ds in IMPUTED_DS:
        ds_data = df.loc[df["dataset"] == ds, col].dropna()
        if len(ds_data) > 0:
            groups.append(ds_data)
            labels.append(ds)
    
    if len(groups) < 2:
        continue
        
    stat, pval = kruskal(*groups)
    
    # Median comparison
    ref_median = ref_data.median()
    
    results.append({
        "feature": col,
        "kruskal_stat": round(stat, 4),
        "p_value": pval,
        "ref_median": round(ref_median, 4),
        "is_different_p005": pval < 0.05,
        "is_different_p001": pval < 0.001
    })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "topology_sensitivity_stats.csv", index=False)

print("\nTOPOLOGY SENSITIVITY ANALYSIS SUMMARY (p < 0.001)")
print(res_df[res_df["is_different_p001"]].to_string(index=False))

# --- Visualization ---
# Plot top 6 most different features
top_diff = res_df.sort_values("kruskal_stat", ascending=False).head(6)["feature"].tolist()

plt.figure(figsize=(15, 10))
for i, col in enumerate(top_diff, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df[df["dataset"].isin([REF_DS] + IMPUTED_DS)], x="dataset", y=col)
    plt.title(f"Distribution: {col}\nKW-p: {res_df.loc[res_df['feature']==col, 'p_value'].values[0]:.2e}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(OUT_DIR / "topology_distribution_comparison.png")
print(f"\nPlots saved to {OUT_DIR / 'topology_distribution_comparison.png'}")

# --- Final Verdict ---
n_diff = res_df["is_different_p001"].sum()
pct_diff = (n_diff / len(res_df)) * 100
print(f"\nVerdict: {n_diff}/{len(res_df)} ({pct_diff:.1f}%) topology features show significant distribution shifts across datasets (p < 0.001).")
if pct_diff < 20:
    print("REASSURING: Most topology features are stable across real and imputed datasets.")
else:
    print("WARNING: Significant distribution shifts detected. Dataset-stratified CV is critical.")
