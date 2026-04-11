"""
Generate visualizations for the global and patient-specific ceRNA networks.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("visualize")

def plot_global_network(global_g: nx.Graph, output_path: Path):
    """Plot a simplified version of the global network."""
    plt.figure(figsize=(12, 12))
    
    # Filter for nodes with some degree to avoid clutter
    nodes_to_plot = [n for n in global_g.nodes if global_g.degree(n) > 5]
    if not nodes_to_plot:
        nodes_to_plot = list(global_g.nodes)[:200]
        
    sub = global_g.subgraph(nodes_to_plot)
    
    colors = []
    for n in sub.nodes:
        ntype = sub.nodes[n].get("node_type", "unknown")
        if ntype == "circRNA": colors.append("gold")
        elif ntype == "miRNA": colors.append("skyblue")
        elif ntype == "mRNA": colors.append("lightgreen")
        else: colors.append("gray")
        
    pos = nx.spring_layout(sub, k=0.15, seed=42)
    nx.draw_networkx_nodes(sub, pos, node_size=50, node_color=colors, alpha=0.8)
    nx.draw_networkx_edges(sub, pos, width=0.5, alpha=0.3)
    
    plt.title("Global ceRNA Interaction Network (High-degree Nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    log.info("Saved global network plot to %s", output_path)

def main():
    graphs_path = cfg.network_dir / "graphs.pkl"
    outputs_dir = cfg.models_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if graphs_path.exists():
        with open(graphs_path, "rb") as f:
            graphs = pickle.load(f)
        
        if graphs:
            # Using the first graph as a template for global structure
            # (In build_cerna, global_g is reconstructed)
            # Actually, let's just use the first patient graph as a demo
            g = graphs[0]
            pid = g.graph.get("patient_id", "P1")
            
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(g, k=0.3, seed=42)
            
            colors = []
            for n in g.nodes:
                ntype = g.nodes[n].get("node_type", "unknown")
                if ntype == "circRNA": colors.append("gold")
                elif ntype == "miRNA": colors.append("skyblue")
                elif ntype == "mRNA": colors.append("lightgreen")
                else: colors.append("gray")
                
            nx.draw_networkx_nodes(g, pos, node_size=100, node_color=colors)
            nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(g, pos, font_size=8)
            
            plt.title(f"Patient-specific ceRNA Network ({pid})")
            plt.axis("off")
            plt.savefig(outputs_dir / f"patient_network_{pid}.png", dpi=300)
            log.info("Saved patient network plot.")

if __name__ == "__main__":
    main()
