import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

# --- Configuration ---
BASE_DIR = "c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection"
DOWNLOADS_DIR = "C:/Users/Samsunh/Downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Set premium styling rcParams
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#222222'
plt.rcParams['axes.labelcolor'] = '#222222'
plt.rcParams['xtick.color'] = '#444444'
plt.rcParams['ytick.color'] = '#444444'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# Load features
feat_path = os.path.join(BASE_DIR, "features/feature_matrix.csv")
print(f"Loading features from {feat_path}...")
df = pd.read_csv(feat_path, index_col=0)

# Clean/filter datasets
df = df[df["dataset"].notna() & (df["dataset"] != "unknown")]

# Target features to plot
target_features = [
    "community_count", 
    "modularity", 
    "diameter", 
    "spectral_entropy", 
    "betti_0", 
    "mean_expression"
]

feature_titles = {
    "community_count": "Community Count (Louvain)",
    "modularity": "Network Modularity",
    "diameter": "Network Diameter",
    "spectral_entropy": "Spectral Entropy",
    "betti_0": "Betti-0 Homology (Components)",
    "mean_expression": "Mean CPM Expression (Control)"
}

# --- Plotting ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Custom color palette for the 4 GEO datasets
palette = {
    "GSE73002": "#1f77b4", 
    "GSE115513": "#ff7f0e", 
    "GSE126094": "#2ca02c", 
    "GSE101684": "#d62728"
}
datasets_order = ["GSE73002", "GSE115513", "GSE126094", "GSE101684"]

for i, feature in enumerate(target_features):
    ax = axes[i]
    ax.set_facecolor("#ffffff")
    ax.grid(axis='y', color='#f0f0f0', linestyle='-', linewidth=0.6)
    
    # Calculate Kruskal-Wallis p-value
    groups_data = [df[df["dataset"] == ds][feature].dropna().values for ds in datasets_order if len(df[df["dataset"] == ds]) > 0]
    if len(groups_data) > 1:
        stat, p_val = kruskal(*groups_data)
        if p_val < 0.0001:
            p_text = "Kruskal-Wallis p < 0.0001"
        else:
            p_text = f"Kruskal-Wallis p = {p_val:.4f}"
    else:
        p_text = "Kruskal-Wallis N/A"
        
    # Generate Boxplot
    sns.boxplot(
        data=df,
        x="dataset",
        y=feature,
        order=datasets_order,
        palette=palette,
        ax=ax,
        width=0.5,
        linewidth=1.2,
        fliersize=2.5,
        hue="dataset",
        legend=False
    )
    
    ax.set_title(f"{feature_titles[feature]}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Z-score Normalized Value", fontsize=9.5)
    
    # Annotate p-value
    ax.text(
        0.5, 0.93, p_text, 
        transform=ax.transAxes, 
        fontsize=9, 
        fontweight="semibold",
        color="#d62728" if "p < 0.0001" in p_text else "#444444",
        ha="center",
        bbox=dict(facecolor="#fbfbfb", edgecolor="#cccccc", boxstyle="round,pad=0.3", alpha=0.9)
    )
    
    ax.tick_params(axis='x', rotation=30, labelsize=9.5)
    ax.tick_params(axis='y', labelsize=9.5)

plt.suptitle("Cross-Dataset Distributional Shift of Patient-Specific Graph Topology Features\n(Highlighting Confounding Batch Effects vs. Biological Control)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save outputs
out_png = os.path.join(DOWNLOADS_DIR, "topology_feature_comparison.png")
plt.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Comparison figure successfully saved to: {out_png}")
