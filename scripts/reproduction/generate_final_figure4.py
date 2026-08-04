import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_protocol_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Protocol 1
    ax.bar(0, 0.927, color='steelblue', label='Protocol 1 (Naive CV)')
    
    # Protocol 2 range
    # Error bar from 0.45 to 0.76
    # bar height is 0.76 - 0.45, starting at 0.45
    ax.bar(1, 0.76-0.45, bottom=0.45, color='orange', label='Protocol 2 (Stratified CV range)', alpha=0.8)
    # Add a line for the mean or just leave as range bar as requested
    ax.vlines(1, 0.45, 0.76, color='darkorange', lw=2)
    
    # LODO
    ax.bar(2, 0.749, color='green', label='LODO GSE73002')
    ax.bar(3, 0.540, color='red', label='LODO GSE115513')
    
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['Protocol 1\n(Naive CV)', 'Protocol 2\n(Stratified CV range)', 
                         'LODO\nGSE73002', 'LODO\nGSE115513'], fontsize=9)
    
    ax.set_ylabel('AUROC', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='Chance')
    ax.legend(fontsize=8)
    
    # Grid for readability
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    out_path = OUT_DIR / "figure4_protocol_comparison.png"
    plt.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved to {out_path}")

if __name__ == "__main__":
    generate_protocol_comparison()
