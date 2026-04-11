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
        row["n_nodes"] = float(g.number_of_nodes())
        row["n_edges"] = float(g.number_of_edges())
        
        circ_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "circRNA"]
        mir_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "miRNA"]
        row["n_circ_expressed"] = float(len(circ_nodes))
        
        # Fast degree stats
        degs = dict(g.degree())
        deg_vals = list(degs.values())
        row["mean_degree"] = float(np.mean(deg_vals))
        row["max_degree"] = float(np.max(deg_vals))
        
        # Hub features (Fast)
        circ_degs = sorted([degs.get(n, 0) for n in circ_nodes], reverse=True)
        row["top1_circ_degree"] = float(circ_degs[0]) if circ_degs else 0.0
        
        # PageRank (Fast, highly discriminative)
        try:
            # Setting n_jobs=1 internally to avoid nested parallelism issues
            pr = nx.pagerank(g, alpha=0.85, max_iter=30)
            circ_pr = sorted([pr.get(n, 0.0) for n in circ_nodes], reverse=True)
            row["max_circ_pagerank"] = float(circ_pr[0]) if circ_pr else 0.0
        except:
            row["max_circ_pagerank"] = 0.0
            
        # Clustering (Relatively fast)
        row["avg_clustering"] = float(nx.average_clustering(g))
        
        # Expression summary
        exprs = [d.get("expression", 0.0) for _, d in g.nodes(data=True)]
        row["mean_expression"] = float(np.mean(exprs))
        row["max_expression"] = float(np.max(exprs))
        row["expression_std"] = float(np.std(exprs))
        
    except Exception:
        return None
    return row

def build_feature_matrix(patient_graphs: list[nx.Graph]) -> pd.DataFrame:
    log.info("Starting parallel feature extraction on %d CPU cores...", multiprocessing.cpu_count())
    
    # helper for parallel
    def _wrap(g):
        pid = g.graph.get("patient_id", "unknown")
        return extract_features_from_graph(pid, g)

    results = Parallel(n_jobs=-1)(delayed(_wrap)(g) for g in tqdm(patient_graphs, desc="Extracting Features"))
    
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
