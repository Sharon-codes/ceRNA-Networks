import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# --- Configuration ---
BASE = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE / "models"
FEAT_CSV = BASE / "features" / "feature_matrix.csv"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Batch-sensitive features to be starred
SENSITIVE_FEATURES = {
    "n_nodes", "n_edges", "mean_degree", "max_degree", "top5_hub_degree",
    "top5_hub_betweenness", "community_count", "modularity", "diameter",
    "graph_entropy", "spectral_entropy", "betti_0", "betti_1"
}

def generate_shap_with_stars():
    shap_path = MODELS_DIR / "shap_values.npy"
    base_path = MODELS_DIR / "shap_base_values.npy"
    names_path = MODELS_DIR / "topology_feature_names.json"

    if not shap_path.exists():
        print("shap_values.npy missing. Please run the model first.")
        return

    vals = np.load(shap_path)
    base = np.load(base_path)
    with open(names_path, encoding="utf-8") as f:
        names = json.load(f)

    # Load data for the summary plot
    df = pd.read_csv(FEAT_CSV, index_col=0)
    X = df[names].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    # Modify names to add a star to sensitive features
    starred_names = [f"{n} \u2605" if n in SENSITIVE_FEATURES else n for n in names]

    # Create SHAP Explanation
    base_val = float(base.ravel()[0])
    expl = shap.Explanation(values=vals, base_values=base_val, data=X, feature_names=starred_names)

    # Plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(expl, max_display=20, show=False)
    plt.title("SHAP Beeswarm with Batch-Sensitive Features Marked (\u2605)", fontweight="bold", pad=12)
    
    out_path = OUT_DIR / "figure6_shap_beeswarm.png"
    plt.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 6 saved to {out_path}")

if __name__ == "__main__":
    generate_shap_with_stars()
