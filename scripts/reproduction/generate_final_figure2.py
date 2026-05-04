import pickle
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np

# --- Configuration ---
BASE = Path(__file__).resolve().parent.parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from config.config import cfg

OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_COLORS = {"circRNA": "#1a9e9e", "miRNA": "#f4a261", "mRNA": "#adb5bd"}

def _representative_subgraph(patient_ids, patient_graphs, max_nodes=30):
    graph_keys_str = {str(k): k for k in patient_graphs.keys()}
    ids = [graph_keys_str[str(p)] for p in patient_ids if str(p) in graph_keys_str]
    if not ids:
        return nx.DiGraph()
    edge_counts = {}
    for pid in ids:
        g = patient_graphs[pid]
        edges = {(u, v) for u, v in g.edges()}
        for e in edges:
            edge_counts[e] = edge_counts.get(e, 0) + 1
    n = len(ids)
    thr = max(1.0, 0.1 * n) # Using 10% as a more robust threshold for representative edges
    freq = [e for e, c in edge_counts.items() if c >= thr]
    
    if not freq: return nx.DiGraph()

    ref_id = ids[0]
    ref = patient_graphs[ref_id]
    H = nx.DiGraph()
    for u, v in freq:
        if u in ref:
            H.add_node(u, **ref.nodes[u])
        if v in ref:
            H.add_node(v, **ref.nodes[v])
        H.add_edge(u, v)

    und = H.to_undirected()
    if und.number_of_nodes() == 0: return H
    lcc = max(nx.connected_components(und), key=len)
    sub = H.subgraph(lcc).copy()
    
    if sub.number_of_nodes() > max_nodes:
        bc = nx.betweenness_centrality(sub.to_undirected())
        top = sorted(bc, key=bc.get, reverse=True)[:max_nodes]
        sub = sub.subgraph(top).copy()
    return sub

def generate_network_comparison():
    gp = cfg.network_dir / "graphs.pkl"
    with open(gp, "rb") as f:
        graphs_raw = pickle.load(f)
        if isinstance(graphs_raw, list):
            patient_graphs = {str(g.graph.get("patient_id", f"unknown_{i}")): g for i, g in enumerate(graphs_raw)}
        else:
            patient_graphs = graphs_raw

    metadata = pd.read_csv(cfg.processed_dir / "metadata.csv", index_col=0)
    
    healthy_ids_all = metadata[metadata["cancer_type"].astype(str).str.lower() == "healthy"].index.astype(str).tolist()
    stage_col = metadata["stage"].astype(str).str.upper() if "stage" in metadata.columns else pd.Series("UNKNOWN", index=metadata.index)
    cancer_ids_all = metadata[(metadata["cancer_type"].astype(str).str.lower() != "healthy") & (stage_col == "I")].index.astype(str).tolist()

    # Sample to speed up processing
    import random
    random.seed(42)
    healthy_ids = random.sample(healthy_ids_all, min(100, len(healthy_ids_all)))
    cancer_ids = random.sample(cancer_ids_all, min(100, len(cancer_ids_all)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, ids, title in zip(
        axes,
        [healthy_ids, cancer_ids],
        [
            "Healthy — Dense, Modular ceRNA Network",
            "Stage-I Cancer — Fragmented ceRNA Network",
        ],
    ):
        G = _representative_subgraph(ids, patient_graphs, max_nodes=30)
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        bc = nx.betweenness_centrality(G.to_undirected(), normalized=True)
        cols = [NODE_COLORS.get(G.nodes[n].get("node_type", "mRNA"), "#adb5bd") for n in G.nodes]
        sizes = [max(100.0, bc.get(n, 0.0) * 5000.0) for n in G.nodes]
        pos = nx.spring_layout(G, seed=42, k=1.5)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, node_size=sizes, alpha=0.85)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", alpha=0.4, arrows=True, arrowsize=15, width=1.0)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.axis("off")

    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in NODE_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    out_path = OUT_DIR / "figure2_network_comparison.png"
    plt.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved to {out_path}")

if __name__ == "__main__":
    generate_network_comparison()
