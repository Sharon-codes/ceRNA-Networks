import os
import matplotlib.pyplot as plt

def main():
    os.makedirs("./images", exist_ok=True)
    
    # Set up styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    color_grey = '#F2F2F2'
    color_red = '#FFE5E5'
    color_green = '#E5FFE5'
    border_grey = '#CCCCCC'
    border_red = '#FF9999'
    border_green = '#99FF99'
    
    # Title formatting
    title_font = {'weight': 'bold', 'size': 13}
    text_font = {'size': 11, 'ha': 'center', 'va': 'center'}
    
    # Column A: Flawed Standard
    ax.text(2.5, 9.6, "A. Flawed Standard (Representation Collapse)", title_font, ha='center', color='#800000')
    
    ax.text(2.5, 8.2, "Frozen ESM-2 Tensors\nTCR & Peptide-HLA\n(320-dim Embeddings)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_grey, ec=border_grey, lw=1.5))
            
    ax.text(2.5, 5.2, "Transformer/Mamba Encoders\n(Severe Oversmoothing/\nRepresentation Flatline)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_red, ec=border_red, lw=1.5))
            
    ax.text(2.5, 2.2, "Cross Attention\n(Softmax Flattening &\nGradient Deadlock)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_red, ec=border_red, lw=1.5))
            
    # Arrows A
    # L1 to L2: source Y=7.3, target Y=6.0
    ax.annotate('', xy=(2.5, 6.0), xytext=(2.5, 7.3),
                arrowprops=dict(arrowstyle="-|>", color='#CC0000', lw=2, mutation_scale=15))
    # L2 to L3: source Y=4.4, target Y=3.0
    ax.annotate('', xy=(2.5, 3.0), xytext=(2.5, 4.4),
                arrowprops=dict(arrowstyle="-|>", color='#CC0000', lw=2, mutation_scale=15))
                
    # Column B: Our Framework
    ax.text(7.5, 9.6, "B. Our Framework (Direct Concatenation)", title_font, ha='center', color='#006600')
    
    ax.text(7.5, 8.6, "Frozen ESM-2 Tensors\nTCR & Peptide-HLA\n(320-dim Embeddings)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_grey, ec=border_grey, lw=1.5))
            
    ax.text(7.5, 6.8, "Linear Projection\n(Dimension reduction to d_model)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    ax.text(7.5, 5.0, "Direct Concatenation\n(Preserve spatial sequence structures)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    ax.text(7.5, 3.2, "MLP Position-Weight Matrix\n(Implicit spatial positioning & prediction)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    ax.text(7.5, 1.4, "Binding Probability\n(No sigmoid/LN bottleneck)", text_font,
            bbox=dict(boxstyle="round,pad=0.5", fc=color_green, ec=border_green, lw=1.5))
            
    # Arrows B
    ax.annotate('', xy=(7.5, 7.5), xytext=(7.5, 8.0),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
    ax.annotate('', xy=(7.5, 5.7), xytext=(7.5, 6.2),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
    ax.annotate('', xy=(7.5, 3.9), xytext=(7.5, 4.4),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
    ax.annotate('', xy=(7.5, 2.1), xytext=(7.5, 2.6),
                arrowprops=dict(arrowstyle="-|>", color='#006600', lw=2, mutation_scale=15))
                
    # Divider line
    ax.plot([5.0, 5.0], [0.5, 9.8], color='#999999', linestyle='--', lw=1.5)
    
    plt.tight_layout()
    
    # Save files
    png_path = "./images/Figure1_Architecture.png"
    pdf_path = "./images/Figure1_Architecture.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated Figure 1 at {png_path} and {pdf_path}")

if __name__ == "__main__":
    main()
