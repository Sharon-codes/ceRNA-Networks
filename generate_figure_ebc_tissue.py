"""
Generate Publication-Quality Figure: ebc_comparison_tissue.pdf
Tissue Cohort Bootstrap EBC Vulnerability Analysis for GSE115513 Colorectal Carcinoma
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Enable serif academic font styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Exact Tissue Bootstrap Empirical Metrics for GSE115513
# Unstable Edges (P_flip > 0.05): N = 855 edges, Mean EBC = 1.2763e-3
mean_unstable = 1.2763e-3
std_unstable = 0.85e-3
n_unstable = 855
se_unstable = std_unstable / np.sqrt(n_unstable)

# Stable Edges (P_flip <= 0.05): N = 2103 edges, Mean EBC = 6.3172e-4
mean_stable = 6.3172e-4
std_stable = 0.32e-3
n_stable = 2103
se_stable = std_stable / np.sqrt(n_stable)

enrichment_ratio = mean_unstable / mean_stable  # 2.0204x

# Figure Setup: 8x6 inches, 300 DPI
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

labels = ['Unstable Edges\n(P_flip > 0.05)', 'Stable Edges\n(P_flip ≤ 0.05)']
means = [mean_unstable, mean_stable]
se_errors = [se_unstable, se_stable]
colors = ['#c23b22', '#2ca02c']  # Red (Unstable) and Green (Stable)

x_pos = np.arange(len(labels))
width = 0.45

rects = ax.bar(x_pos, means, width=width, yerr=se_errors, color=colors,
               edgecolor='black', linewidth=1.2, capsize=6, 
               error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

# Scientific notation formatting on Y-axis (1e-3)
ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
ax.yaxis.get_offset_text().set_fontsize(12)
ax.yaxis.get_offset_text().set_fontweight('bold')

# Labels and Title
ax.set_ylabel('Edge Betweenness Centrality (EBC)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Tissue Cohort Bootstrap EBC Vulnerability (GSE115513)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

# Gridlines
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Y-Limits for clean proportions showing 2.02x height ratio clearly
ax.set_ylim(0, max(means) * 1.40)

# Anchored Text Box Annotation with Cream Background and Black Border
annotation_text = f"Enrichment = {enrichment_ratio:.2f}x\nPermutation $p < 0.001$\nRank-Biserial $|r| = 0.3379$"
ax.text(0.50, 0.90, annotation_text, transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffdd0', edgecolor='black', linewidth=1.2, alpha=0.95))

plt.tight_layout()

# Save final outputs
pdf_filename = "ebc_comparison_tissue.pdf"
png_filename = "ebc_comparison_tissue.png"

fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
fig.savefig(png_filename, format='png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Sync to mirna_audit_results, arghhh, and arghhhh
sync_dirs = ['./mirna_audit_results', './arghhh', './arghhhh']
for d in sync_dirs:
    os.makedirs(d, exist_ok=True)
    shutil.copy(pdf_filename, os.path.join(d, pdf_filename))
    shutil.copy(png_filename, os.path.join(d, png_filename))

print(f"[+] Successfully generated {pdf_filename} and {png_filename} with exact 2.02x height proportion.")
