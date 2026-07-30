import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("./images", exist_ok=True)
    
    # Load occlusion scores for Rank 1
    scores_path = "./Evaluation/XAI/occlusion_scores.pkl"
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Occlusion scores not found at {scores_path}")
        
    with open(scores_path, "rb") as f:
        data = pickle.load(f)
        
    entry = data[0]
    pep_chars = entry["pephla_chars"][:10]
    scores_w1 = entry["pephla_scores_w1"][:10]
    scores_w3 = entry["pephla_scores_w3"][:10]
    matrix = np.array([scores_w1, scores_w3])
    
    # Set up styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Figure dimensions exactly 3.25 inches by 1.75 inches as required by ACS
    fig = plt.figure(figsize=(3.25, 1.75))
    
    # Main schematic axes (fills the entire canvas)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Left schematic drawing
    ax.text(0.03, 0.72, "TCR\n(ESM-2)", size=6, ha='left', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", fc='#F2F2F2', ec='#CCCCCC', lw=0.8))
            
    ax.text(0.03, 0.28, "Peptide-HLA\n(ESM-2)", size=6, ha='left', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", fc='#F2F2F2', ec='#CCCCCC', lw=0.8))
            
    ax.text(0.21, 0.50, "Direct\nConcat", size=6, ha='left', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", fc='#E5FFE5', ec='#99FF99', lw=0.8))
            
    ax.text(0.35, 0.50, "MLP", size=6, ha='left', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", fc='#E5FFE5', ec='#99FF99', lw=0.8))
            
    ax.text(0.44, 0.50, "Binding\nProb: 0.99", size=5.5, ha='left', va='center', weight='bold', color='#006600')
    
    # Arrows
    # TCR to Concat
    ax.annotate('', xy=(0.20, 0.56), xytext=(0.12, 0.72),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=0.8, mutation_scale=6))
    # Peptide to Concat
    ax.annotate('', xy=(0.20, 0.44), xytext=(0.14, 0.28),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=0.8, mutation_scale=6))
    # Concat to MLP
    ax.annotate('', xy=(0.34, 0.50), xytext=(0.29, 0.50),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=0.8, mutation_scale=6))
    # MLP to Binding
    ax.annotate('', xy=(0.43, 0.50), xytext=(0.39, 0.50),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=0.8, mutation_scale=6))
                
    # Dotted arrow to XAI map
    ax.annotate('', xy=(0.57, 0.38), xytext=(0.49, 0.38),
                arrowprops=dict(arrowstyle="->", color='grey', linestyle=':', lw=0.8, mutation_scale=6))
    ax.text(0.53, 0.44, "Sensitivity\nMapping", size=5, ha='center', color='grey')
    
    # Embed micro-heatmap
    ax_heat = fig.add_axes([0.60, 0.20, 0.36, 0.50])
    sns.heatmap(
        matrix,
        xticklabels=pep_chars,
        yticklabels=["W1", "W3"],
        cmap="Reds",
        ax=ax_heat,
        cbar=False,
        vmin=min(0.0, matrix.min()),
        vmax=max(0.1, matrix.max()),
        linewidths=0.2,
        linecolor='grey'
    )
    
    # Micro labels styling
    ax_heat.set_xticklabels(pep_chars, rotation=0, fontsize=7, fontweight='bold', fontfamily='monospace')
    ax_heat.set_yticklabels(["W1", "W3"], rotation=0, fontsize=6)
    ax_heat.tick_params(axis='both', which='both', length=0, pad=2)
    
    # Micro heatmap title
    ax_heat.set_title("Influenza XAI Map", fontsize=7, fontweight='bold', pad=4)
    
    # Save files
    png_path = "./images/Graphical_TOC.png"
    pdf_path = "./images/Graphical_TOC.pdf"
    
    # Save with exact dimensions (no tight layout here to prevent dimension shifts)
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, format='pdf')
    plt.close()
    
    print(f"Successfully generated Graphical TOC at {png_path} and {pdf_path}")

if __name__ == "__main__":
    main()
