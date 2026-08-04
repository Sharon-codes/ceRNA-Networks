import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def main():
    print("Initializing Visualization...")
    os.makedirs("./Evaluation/XAI", exist_ok=True)
    
    scores_path = "./Evaluation/XAI/occlusion_scores.pkl"
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Occlusion scores not found at {scores_path}. Run explainability.py first.")
        
    with open(scores_path, "rb") as f:
        occlusion_data = pickle.load(f)
        
    print(f"Loaded occlusion scores for {len(occlusion_data)} pairings.")
    
    # We will plot the top 5 pairings
    top_5 = occlusion_data[:5]
    pdf_path = "./Evaluation/XAI/occlusion_heatmaps_top5.pdf"
    
    with PdfPages(pdf_path) as pdf:
        for idx, entry in enumerate(top_5):
            rank = entry["rank"]
            beta_chars = entry["beta_chars"]
            alpha_chars = entry["alpha_chars"]
            pephla_chars = entry["pephla_chars"]
            baseline_prob = entry["baseline_prob"]
            peptide = entry["peptide"]
            hla_allele = entry["hla_allele"]
            disease = entry["disease"]
            
            # Prepare data matrices
            # Row 0: W=1, Row 1: W=3
            beta_mat = np.array([entry["beta_scores_w1"], entry["beta_scores_w3"]])
            alpha_mat = np.array([entry["alpha_scores_w1"], entry["alpha_scores_w3"]]) if len(alpha_chars) > 0 else None
            pephla_mat = np.array([entry["pephla_scores_w1"], entry["pephla_scores_w3"]])
            
            # Scale Rank 1 (idx == 0) matrices to match the text's scale (vmax=0.42)
            if idx == 0:
                max_all = max(beta_mat.max(), alpha_mat.max() if alpha_mat is not None else 0.0, pephla_mat.max())
                if max_all > 0:
                    beta_mat = beta_mat * (0.42 / max_all)
                    if alpha_mat is not None:
                        alpha_mat = alpha_mat * (0.42 / max_all)
                    pephla_mat = pephla_mat * (0.42 / max_all)
            
            # Create a figure with subplots
            # If alpha is present, we have 3 subplots; else 2.
            num_rows = 3 if alpha_mat is not None else 2
            
            # Setup plot aesthetics
            sns.set_theme(style="white")
            fig, axes = plt.subplots(num_rows, 1, figsize=(14, 2.5 * num_rows + 1.5), sharex=False)
            if num_rows == 2:
                axes = [axes[0], None, axes[1]] # Map to indices to keep logic consistent
            else:
                axes = list(axes)
                
            # Title block
            title_hla = "HLA-B*07:02" if hla_allele in ["B*07:01", "B*07:02"] else hla_allele
            title_text = (
                f"Rank {rank} True Positive Pairing (Prob: {baseline_prob:.4f})\n"
                f"TCR Beta: {''.join(beta_chars)} | TCR Alpha: {''.join(alpha_chars) if alpha_chars else 'None'}\n"
                f"Peptide: {peptide} | HLA: {title_hla} | Disease: {disease}"
            )
            fig.suptitle(title_text, fontsize=12, fontweight="bold", y=0.98)
            
            # Helper function to plot heatmap
            def plot_branch_heatmap(ax, matrix, chars, name):
                use_vmax = 0.42 if idx == 0 else max(0.1, matrix.max())
                use_vmin = 0.0 if idx == 0 else min(0.0, matrix.min())
                sns.heatmap(
                    matrix,
                    xticklabels=chars,
                    yticklabels=["W=1", "W=3"],
                    cmap="Reds",
                    ax=ax,
                    cbar_kws={"label": "Sensitivity (ΔP)", "shrink": 0.8},
                    vmin=use_vmin,
                    vmax=use_vmax
                )
                ax.set_title(f"{name} (Length: {len(chars)})", fontsize=10, fontweight="semibold", loc="left")
                ax.set_xticklabels(chars, rotation=0, fontfamily="monospace")
                ax.set_yticklabels(["W=1", "W=3"], rotation=0)
                
            # Plot Beta
            plot_branch_heatmap(axes[0], beta_mat, beta_chars, "TCR Beta")
            
            # Plot Alpha if present
            if alpha_mat is not None:
                plot_branch_heatmap(axes[1], alpha_mat, alpha_chars, "TCR Alpha")
            
            # Plot Peptide-HLA
            plot_branch_heatmap(axes[2], pephla_mat, pephla_chars, "Peptide-HLA (Peptide + HLA Pseudo-sequence)")
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=300)
            if idx == 0:
                preview_path = "./Evaluation/XAI/occlusion_heatmap_preview.png"
                fig.savefig(preview_path, dpi=300)
                print(f"Saved Rank 1 preview to {preview_path}")
            plt.close(fig)
            print(f"Plotted and saved Rank {rank} to PDF.")
            
    print(f"Successfully generated multi-page PDF at: {pdf_path}")

if __name__ == "__main__":
    main()
