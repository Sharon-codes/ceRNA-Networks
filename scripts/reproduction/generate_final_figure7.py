import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# --- Configuration ---
BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
RES_FILE = BASE / "scratch" / "robustness_results.json"
with open(RES_FILE, "r") as f:
    res = json.load(f)

def generate_robustness_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Protocol 1 (Naive) ---
    models = ["XGB (Hybrid)", "MLP (Baseline)", "LR (Baseline)", "XGB (Robust)"]
    p1_means = [res["P1_XGB_All"][0], res["P1_MLP_All"][0], res["P1_LR_All"][0], res["P1_XGB_Robust"][0]]
    p1_stds = [res["P1_XGB_All"][1], res["P1_MLP_All"][1], res["P1_LR_All"][1], res["P1_XGB_Robust"][1]]
    
    colors1 = ['#1f77b4', '#9467bd', '#7f7f7f', '#aec7e8']
    bars1 = ax1.bar(models, p1_means, yerr=p1_stds, color=colors1, capsize=7, alpha=0.9)
    ax1.set_title("(A) Naive CV: Impact of Model & Batch Features", fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel("AUROC", fontweight='bold')
    ax1.set_ylim(0.9, 1.01)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- Panel B: Protocol 2 (Stratified) ---
    models2 = ["XGB (Hybrid)", "XGB (Robust)"]
    p2_means = [res["P2_XGB_All"][0], res["P2_XGB_Robust"][0]]
    p2_stds = [res["P2_XGB_All"][1], res["P2_XGB_Robust"][1]]
    
    colors2 = ['#ff7f0e', '#ffbb78']
    bars2 = ax2.bar(models2, p2_means, yerr=p2_stds, color=colors2, capsize=7, alpha=0.9)
    ax2.set_title("(B) Stratified CV: Generalization Penalty", fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel("AUROC", fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle("Figure 7. Baseline Comparison and Feature Robustness Analysis", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_path = OUT_DIR / "figure7_robustness_baseline.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 7 updated: {out_path}")

if __name__ == "__main__":
    generate_robustness_figure()
