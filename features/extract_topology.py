"""
Extract topology and summary features from per-patient ceRNA graphs. (Fast & High Signal)
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg, write_phase_metrics

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(cfg.logs_dir / "extract_topology.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("extract_topology")

from joblib import Parallel, delayed

def extract_features_from_graph(patient_id: str, g: nx.Graph) -> Optional[Dict[str, Any]]:
    if g.number_of_nodes() < 2:
        return None
    
    row = {"patient_id": patient_id}
    try:
        import scipy.stats as stats
        row["n_nodes"] = float(g.number_of_nodes())
        row["n_edges"] = float(g.number_of_edges())
        
        circ_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "circRNA"]
        row["n_circ_expressed"] = float(len(circ_nodes))
        
        exprs = [d.get("expression", 0.0) for _, d in g.nodes(data=True)]
        row["mean_expression"] = float(np.mean(exprs)) if exprs else 0.0
        row["max_expression"] = float(np.max(exprs)) if exprs else 0.0
        row["expression_std"] = float(np.std(exprs)) if exprs else 0.0
        
        degs = dict(g.degree())
        deg_vals = list(degs.values())
        row["mean_degree"] = float(np.mean(deg_vals))
        row["max_degree"] = float(np.max(deg_vals))
        
        top5_deg = sorted(deg_vals, reverse=True)[:5]
        row["top5_hub_degree"] = float(np.mean(top5_deg)) if top5_deg else 0.0
        
        circ_degs = sorted([degs.get(n, 0) for n in circ_nodes], reverse=True)
        row["top1_circ_degree"] = float(circ_degs[0]) if circ_degs else 0.0
        
        try:
            pr = nx.pagerank(g, alpha=0.85, max_iter=30)
            circ_pr = sorted([pr.get(n, 0.0) for n in circ_nodes], reverse=True)
            row["max_circ_pagerank"] = float(circ_pr[0]) if circ_pr else 0.0
        except Exception:
            row["max_circ_pagerank"] = 0.0
        
        # We need a small k for betweenness, else it is too slow
        bc = nx.betweenness_centrality(g, k=min(20, g.number_of_nodes()))
        top5_bc = sorted(list(bc.values()), reverse=True)[:5]
        row["top5_hub_betweenness"] = float(np.mean(top5_bc)) if top5_bc else 0.0
        
        row["avg_clustering"] = float(nx.average_clustering(g))
        
        lcc_nodes = max(nx.connected_components(g), key=len)
        lcc = g.subgraph(lcc_nodes)
        row["diameter"] = float(nx.diameter(lcc)) if lcc.number_of_nodes() > 0 else 0.0
        
        deg_counts = np.bincount(deg_vals)
        p_deg = deg_counts[deg_counts > 0] / np.sum(deg_counts)
        row["graph_entropy"] = float(stats.entropy(p_deg))
        
        try:
            adj = nx.adjacency_matrix(g).todense()
            eigenvalues = np.linalg.eigvalsh(adj)
            eig_pos = np.abs(eigenvalues)
            eig_sum = np.sum(eig_pos)
            if eig_sum > 0:
                p_eig = eig_pos / eig_sum
                row["spectral_entropy"] = float(stats.entropy(p_eig))
            else:
                row["spectral_entropy"] = 0.0
        except Exception:
            row["spectral_entropy"] = 0.0
            
        try:
            import community.community_louvain as cl
            part = cl.best_partition(g)
            row["community_count"] = float(len(set(part.values())))
            row["modularity"] = float(cl.modularity(part, g))
        except Exception:
            row["community_count"] = 0.0
            row["modularity"] = 0.0
            
        row["betti_0"] = float(nx.number_connected_components(g))
        row["betti_1"] = float(g.number_of_edges() - g.number_of_nodes() + row["betti_0"])
        
        try:
            import gudhi
            st = gudhi.SimplexTree()
            for u, v, d in g.edges(data=True):
                w = d.get("weight", 0.1)
                st.insert([u, v], filtration=1.0 / (w + 1e-5))
            st.persistence()
            intervals = st.persistence_intervals_in_dimension(0)
            if len(intervals) > 0:
                finite = intervals[intervals[:, 1] != np.inf]
                if len(finite) > 0:
                    lengths = finite[:, 1] - finite[:, 0]
                    p_len = lengths / np.sum(lengths)
                    row["persistence_entropy"] = float(stats.entropy(p_len))
                else:
                    row["persistence_entropy"] = 0.0
            else:
                row["persistence_entropy"] = 0.0
        except Exception:
            row["persistence_entropy"] = 0.0
            
    except Exception:
        return None
    return row

def build_feature_matrix(patient_graphs: list[nx.Graph]) -> pd.DataFrame:
    log.info("Starting parallel feature extraction on %d CPU cores...", multiprocessing.cpu_count())
    
    # helper for parallel
    def _wrap(g):
        pid = g.graph.get("patient_id", "unknown")
        return extract_features_from_graph(pid, g)

    try:
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_wrap)(g) for g in tqdm(patient_graphs, desc="Extracting Features")
        )
    except PermissionError:
        log.warning("Parallel backend unavailable; falling back to serial extraction.")
        results = [_wrap(g) for g in tqdm(patient_graphs, desc="Extracting Features")]
    
    rows = [r for r in results if r is not None]
    log.info("Extracted features from %d/%d valid graphs.", len(rows), len(patient_graphs))
            
    df = pd.DataFrame(rows).set_index("patient_id")
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].apply(lambda s: stats.zscore(s, nan_policy="omit")).fillna(0.0)

    meta_path = cfg.processed_dir / "metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, index_col=0)
        have = [c for c in ["cancer_type", "stage", "age", "sex", "dataset"] if c in meta.columns]
        df = df.join(meta[have], how="left")
    return df

import multiprocessing
def main() -> None:
    t0 = time.time()
    try:
        graphs_path = cfg.network_dir / "graphs.pkl"
        log.info("Loading graphs from %s...", graphs_path)
        with open(graphs_path, "rb") as f:
            graphs = pickle.load(f)
        log.info("Graphs loaded. Building feature matrix...")
        fm = build_feature_matrix(graphs)
        fm.to_csv(cfg.features_dir / "feature_matrix.csv")
        log.info("Feature matrix saved: %s", fm.shape)
        write_phase_metrics(4, {"status": "ok", "shape": list(fm.shape)}, elapsed_sec=time.time()-t0)
    except Exception:
        log.exception("extract_topology failed")
        write_phase_metrics(4, {"status": "error"}, elapsed_sec=time.time()-t0)
        raise

if __name__ == "__main__":
    main()
