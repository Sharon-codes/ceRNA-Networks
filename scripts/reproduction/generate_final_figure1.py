import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# --- Configuration ---
BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_workflow_diagram():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # Helper to draw boxes
    def draw_box(x, y, w, h, text, color='#e3f2fd', edgecolor='#1976d2'):
        # Using a rectangle with rounded corners
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                       facecolor=color, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', wrap=True)

    # (A) Data Retrieval
    draw_box(1, 11, 8, 1, "(A) Raw GEO Data Retrieval & Quality Filtering\n(GSE73002, GSE115513, GSE126094, GSE101684)")
    
    # (B) Graph Construction
    draw_box(1, 9.5, 8, 1, "(B) Patient-Specific Tripartite ceRNA Graph Construction\n(miRTarBase v9.0, CircInteractome)")

    # (C) Feature Extraction
    draw_box(1, 8, 8, 1, "(C) 21-Feature Extraction\n(4 Expression, 17 Topology including TDA)")

    # (D) Preprocessing
    draw_box(1, 6.5, 8, 1, "(D) Preprocessing & ComBat Batch Harmonisation\n(Z-score scaling, expression correction)")

    # (E) Evaluation Protocols
    ax.text(5, 5.8, "(E) Contrasted Evaluation Protocols", ha='center', fontsize=12, fontweight='bold')
    draw_box(0.5, 4.2, 4, 1.2, "Protocol 1\nDataset-Naive CV\n(Standard Practice)", color='#fffde7', edgecolor='#fbc02d')
    draw_box(5.5, 4.2, 4, 1.2, "Protocol 2\nDataset-Stratified CV\n(Recommended Practice)", color='#fffde7', edgecolor='#fbc02d')

    # (F/G) Diagnostics
    draw_box(0.5, 2.2, 4, 1, "(F) Topology Batch\nSensitivity Diagnostic", color='#f1f8e9', edgecolor='#558b2f')
    draw_box(5.5, 2.2, 4, 1, "(G) LODO Validation &\nSHAP Interpretability", color='#f1f8e9', edgecolor='#558b2f')

    # Final Output
    draw_box(2.5, 0.5, 5, 0.8, "Publication-Ready\nResearch Figures", color='#f3e5f5', edgecolor='#7b1fa2')

    # Draw Arrows
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

    arrow(5, 11, 5, 10.7)
    arrow(5, 9.5, 5, 9.2)
    arrow(5, 8, 5, 7.7)
    arrow(5, 6.5, 5, 6.2)
    
    # Split to Protocols
    ax.plot([5, 5], [6.2, 5.8], color='#424242', lw=2)
    arrow(5, 5.8, 2.5, 5.5)
    arrow(5, 5.8, 7.5, 5.5)
    
    # From Protocols to Diagnostics
    arrow(2.5, 4.2, 2.5, 3.4)
    arrow(7.5, 4.2, 7.5, 3.4)
    
    # Final Arrows
    arrow(2.5, 2.2, 5, 1.5)
    arrow(7.5, 2.2, 5, 1.5)

    plt.title("Figure 1. Complete Methodological Workflow", fontsize=14, fontweight='bold', pad=20)
    
    out_path = OUT_DIR / "figure1_workflow.png"
    plt.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 1 saved to {out_path}")

if __name__ == "__main__":
    generate_workflow_diagram()
