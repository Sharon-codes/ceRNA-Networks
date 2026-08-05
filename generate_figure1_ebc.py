import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Enable serif academic font styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Exact Empirical Values
mean_flipped = 1.47e-5
std_flipped = 9.05e-5
n_flipped = 2708
se_flipped = std_flipped / np.sqrt(n_flipped)

mean_preserved = 0.83e-5
std_preserved = 1.90e-5
n_preserved = 113822
se_preserved = std_preserved / np.sqrt(n_preserved)

# Figure Setup: 8x6 inches, 300 DPI
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

labels = ['Flipped Edges\n(Bridge-Vulnerable)', 'Preserved Edges']
means = [mean_flipped, mean_preserved]
se_errors = [se_flipped, se_preserved]
colors = ['#c23b22', '#2ca02c']  # Red and Green

x_pos = np.arange(len(labels))
width = 0.45

rects = ax.bar(x_pos, means, width=width, yerr=se_errors, color=colors,
               edgecolor='black', linewidth=1.2, capsize=6, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

# Scientific notation formatting on Y-axis
ax.ticklabel_format(style='sci', axis='y', scilimits=(-5, -5))
ax.yaxis.get_offset_text().set_fontsize(12)
ax.yaxis.get_offset_text().set_fontweight('bold')

# Labels and Title
ax.set_ylabel('Edge Betweenness Centrality (EBC)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Real Observed Cross-Cohort Perturbation EBC Vulnerability', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

# Gridlines
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Y-Limits for clean proportions showing 1.76x height clearly
ax.set_ylim(0, max(means) * 1.45)

# Anchored Text Box Annotation with Cream Background and Black Border
annotation_text = "Enrichment = 1.76x\nMann-Whitney $p < 10^{-15}$"
ax.text(0.48, 0.90, annotation_text, transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffdd0', edgecolor='black', linewidth=1.2, alpha=0.95))

plt.tight_layout()

# Save final outputs
pdf_filename = "figure_1_ebc_vulnerability.pdf"
png_filename = "figure_1_ebc_vulnerability.png"

fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
fig.savefig(png_filename, format='png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Sync to arghhh, arghhhh, and mirna_audit_results folders
sync_dirs = ['./arghhh', './arghhhh', './mirna_audit_results']
for d in sync_dirs:
    os.makedirs(d, exist_ok=True)
    shutil.copy(pdf_filename, os.path.join(d, pdf_filename))
    shutil.copy(png_filename, os.path.join(d, png_filename))

print(f"[+] Successfully generated {pdf_filename} and {png_filename} with exact 1.76x height proportion.")
