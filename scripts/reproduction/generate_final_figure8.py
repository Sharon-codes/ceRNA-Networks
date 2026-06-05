import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
FEAT_CSV = BASE / "features" / "feature_matrix.csv"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_violin_plots():
    df = pd.read_csv(FEAT_CSV, index_col=0)
    
    # Filter out unknown datasets if any exist
    df = df[df['dataset'].notna() & (df['dataset'] != 'unknown')]
    
    plt.figure(figsize=(10, 6))
    
    sns.violinplot(
        data=df,
        x="dataset",
        y="community_count",
        inner="quartile",
        hue="dataset",
        palette="viridis",
        legend=False
    )
    
    plt.title("Cross-Dataset Distributional Shift of Community Count\n(Highlighting Structural Batch Effects)", fontweight="bold", pad=12)
    plt.ylabel("Community Count (Normalized)", fontweight="bold")
    plt.xlabel("GEO Cohort", fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path_eps = OUT_DIR / "figure8_community_count_violin.eps"
    out_path_png = OUT_DIR / "figure8_community_count_violin.png"
    plt.savefig(out_path_eps, format='eps')
    plt.savefig(out_path_png, format='png', dpi=300)
    plt.close()
    print(f"Figure saved to {out_path_eps} and {out_path_png}")

if __name__ == "__main__":
    generate_violin_plots()
