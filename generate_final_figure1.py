import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Paths ---
BASE = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV = BASE / "features" / "feature_matrix_combat.csv"
OUT_DIR = BASE / "FINAL FIGURES"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
print(f"Loading {FEAT_CSV}...")
df = pd.read_csv(FEAT_CSV, index_col=0)

# Features requested by user
target_features = [
    "community_count", 
    "modularity", 
    "diameter", 
    "spectral_entropy", 
    "betti_0", 
    "mean_expression"
]

# Reference vs Others
REF_DS = "GSE73002"
OTHER_DS = [ds for ds in df["dataset"].unique() if ds != REF_DS]

print(f"Reference dataset: {REF_DS}")
print(f"Other datasets: {OTHER_DS}")

# --- Visualization ---
plt.figure(figsize=(15, 10))
for i, col in enumerate(target_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df[df["dataset"].isin([REF_DS] + OTHER_DS)], x="dataset", y=col, palette="Set2")
    plt.title(f"Distribution: {col}", fontweight="bold")
    plt.xticks(rotation=45)

plt.tight_layout()
out_path = OUT_DIR / "figure1_topology_sensitivity.eps"
plt.savefig(out_path, format='eps', dpi=300, bbox_inches='tight')
print(f"Figure 1 saved to {out_path}")
