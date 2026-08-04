import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load saved metrics from CSVs
df_gse73 = pd.read_csv('./empirical_bridge_edge_GSE73002.csv').iloc[0]
df_gse115 = pd.read_csv('./empirical_bridge_edge_GSE115513.csv').iloc[0]
df_sweep = pd.read_csv('./empirical_threshold_sweep.csv')

# Set publication font and styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# ==============================================================================
# FIGURE A: Empirical EBC Comparison (GSE73002 & GSE115513)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

cohorts = ['GSE73002\n(N=907 Samples)', 'GSE115513\n(N=606 Samples)']
x = np.arange(len(cohorts))
width = 0.32

flipped_means = [df_gse73['mean_ebc_flipped'], df_gse115['mean_ebc_flipped']]
preserved_means = [df_gse73['mean_ebc_preserved'], df_gse115['mean_ebc_preserved']]

# Compute Standard Error of the Mean (SEM = std / sqrt(N)) for clean, readable error bars
sem_flipped = [
    df_gse73['std_ebc_flipped'] / np.sqrt(df_gse73['total_flipped']),
    df_gse115['std_ebc_flipped'] / np.sqrt(df_gse115['total_flipped'])
]
sem_preserved = [
    df_gse73['std_ebc_preserved'] / np.sqrt(df_gse73['preserved_edges']),
    df_gse115['std_ebc_preserved'] / np.sqrt(df_gse115['preserved_edges'])
]

rects1 = ax.bar(x - width/2, flipped_means, width, yerr=sem_flipped, label='Flipped Edges (Bridge-Vulnerable)',
               color='#c23b22', edgecolor='black', linewidth=1.1, capsize=4, error_kw={'elinewidth': 1.2})
rects2 = ax.bar(x + width/2, preserved_means, width, yerr=sem_preserved, label='Preserved Edges',
               color='#2ca02c', edgecolor='black', linewidth=1.1, capsize=4, error_kw={'elinewidth': 1.2})

# Aesthetics
ax.set_ylabel('Edge Betweenness Centrality (EBC)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_title('Empirical Bridge-Edge Centrality Vulnerability Across Cohorts', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(cohorts, fontsize=12, fontweight='bold')
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11)

# Enable subtle y-axis grid
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Set tight y-limits cleanly avoiding huge whitespace top
max_val = max(flipped_means[0] + sem_flipped[0], flipped_means[1] + sem_flipped[1])
ax.set_ylim(0, max_val * 1.35)

# Format y-axis to scientific / clean float
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(11)

# Annotate Mann-Whitney p-values cleanly right above the bars
p73 = df_gse73['p_value']
p115 = df_gse115['p_value']

ax.text(x[0] - width/2, flipped_means[0] + sem_flipped[0] + max_val * 0.04,
        f"Mann-Whitney\n$p = {p73:.2e}$", ha='center', va='bottom', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='#cccc99', alpha=0.9))

ax.text(x[1] - width/2, flipped_means[1] + sem_flipped[1] + max_val * 0.04,
        f"Mann-Whitney\n$p = {p115:.2e}$", ha='center', va='bottom', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='#cccc99', alpha=0.9))

plt.tight_layout()

# Save PDF and PNG versions
fig.savefig("empirical_EBC_comparison.pdf", format='pdf', bbox_inches='tight')
fig.savefig("empirical_EBC_comparison.png", format='png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("[+] Saved fixed empirical_EBC_comparison.pdf and .png")


# ==============================================================================
# FIGURE B: Empirical Threshold Sweep Results Table & Plot
# ==============================================================================
fig, (ax_plot, ax_tbl) = plt.subplots(2, 1, figsize=(9, 7.5), gridspec_kw={'height_ratios': [1.3, 1]}, dpi=300)

# Top Subplot: GED & Flipped Edge Count vs Threshold
thetas = df_sweep['theta']
color1 = '#1f77b4'
color2 = '#d62728'

ax_plot.plot(thetas, df_sweep['flipped_edge_count'], marker='o', linewidth=2.2, color=color1, label='Flipped Edge Count')
ax_plot.set_xlabel(r'Binarisation Threshold ($\theta$)', fontsize=12, fontweight='bold')
ax_plot.set_ylabel('Flipped Edge Count', color=color1, fontsize=12, fontweight='bold')
ax_plot.tick_params(axis='y', labelcolor=color1)
ax_plot.grid(True, linestyle='--', alpha=0.5)

ax_sec = ax_plot.twinx()
ax_sec.plot(thetas, df_sweep['mean_EBC_flipped'] / df_sweep['mean_EBC_preserved'], marker='s', linestyle='--', linewidth=2.2, color=color2, label='EBC Ratio (Flipped/Preserved)')
ax_sec.set_ylabel('EBC Ratio (Flipped / Preserved)', color=color2, fontsize=12, fontweight='bold')
ax_sec.tick_params(axis='y', labelcolor=color2)

ax_plot.set_title('Empirical Table 2: Threshold Sweep Dynamics (GSE73002)', fontsize=13, fontweight='bold', pad=12)

# Bottom Subplot: Formatted Data Table
ax_tbl.axis('off')
df_display = df_sweep.copy()
df_display['theta'] = df_display['theta'].map('{:.3f}'.format)
df_display['GED'] = df_display['GED'].map('{:.0f}'.format)
df_display['mean_EBC_flipped'] = df_display['mean_EBC_flipped'].map('{:.6f}'.format)
df_display['mean_EBC_preserved'] = df_display['mean_EBC_preserved'].map('{:.6f}'.format)
df_display['mann_whitney_p_value'] = df_display['mann_whitney_p_value'].map('{:.2e}'.format)

col_labels = [r'$\theta$', r'Edges ($G_{\text{base}}$)', 'Flipped', 'GED', 'Mean EBC (Flip)', 'Mean EBC (Pres)', 'MW $p$-value', r'$C_{\text{before}}$', r'$C_{\text{after}}$', r'$|\Delta C|$']
table_vals = df_display.values.tolist()

tbl = ax_tbl.table(cellText=table_vals, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.4)

for i in range(len(col_labels)):
    cell = tbl[0, i]
    cell.set_facecolor('#2c3e50')
    cell.get_text().set_color('white')
    cell.get_text().set_weight('bold')

plt.tight_layout()
fig.savefig("empirical_threshold_sweep.pdf", format='pdf', bbox_inches='tight')
fig.savefig("empirical_threshold_sweep.png", format='png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("[+] Saved fixed empirical_threshold_sweep.pdf and .png")
