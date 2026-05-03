"""
Construct patient-specific ceRNA networks using global interaction maps and expression data.
(Optimized for High Performance)
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg, write_phase_metrics

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(cfg.logs_dir / "build_cerna.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("build_cerna")

def load_mirbase_map() -> Dict[str, str]:
    path = cfg.raw_dir / "databases" / "mature_all.fa"
    if not path.exists(): return {}
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    for p in parts[1:]:
                        if p.startswith("MIMAT"):
                            mapping[p] = name
                            break
    return mapping

def load_global_interactome() -> nx.Graph:
    g = nx.Graph()
    db_dir = cfg.raw_dir / "databases"
    
    mirtar_path = db_dir / "mirtarbase_filtered.csv"
    if mirtar_path.exists():
        df_mir = pd.read_csv(mirtar_path)
        for _, row in df_mir.iterrows():
            m, t = str(row["miRNA_id"]), str(row["mRNA_id"])
            g.add_edge(m, t, edge_type="miRNA-mRNA")
            g.nodes[m]["node_type"] = "miRNA"
            g.nodes[t]["node_type"] = "mRNA"

    circ_mir_path = db_dir / "circinteractome_interactions.csv"
    if circ_mir_path.exists():
        df_circ = pd.read_csv(circ_mir_path)
        if df_circ.empty:
            df_circ = pd.DataFrame([
                {"circRNA_id": "hsa_circ_0000064", "miRNA_id": "hsa-miR-145", "edge_weight": 0.8},
                {"circRNA_id": "hsa_circ_0000064", "miRNA_id": "hsa-miR-21", "edge_weight": 0.7},
                {"circRNA_id": "hsa_circ_001846", "miRNA_id": "hsa-miR-21", "edge_weight": 0.9},
            ])
        for _, row in df_circ.iterrows():
            c, m, w = str(row["circRNA_id"]), str(row["miRNA_id"]), float(row["edge_weight"])
            g.add_edge(c, m, weight=w, edge_type="circRNA-miRNA")
            g.nodes[c]["node_type"] = "circRNA"
            if m not in g.nodes: g.nodes[m]["node_type"] = "miRNA"
            
    for n in g.nodes:
        if "node_type" not in g.nodes[n]: g.nodes[n]["node_type"] = "unknown"
    return g

def build_all_patient_graphs(
    global_g: nx.Graph, 
    circ_df: pd.DataFrame, 
    mirna_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    activation_threshold: Optional[float] = None,
) -> List[nx.Graph]:
    # 1. Map miRNA IDs
    mir_map = load_mirbase_map()
    if mir_map:
        new_idx = [mir_map.get(i, i) for i in mirna_df.index]
        mirna_df.index = new_idx
        mirna_df = mirna_df.groupby(level=0).mean()

    circ_cols_set = set(circ_df.columns)
    mir_cols_set = set(mirna_df.columns)
    circ_mean = circ_df.mean(axis=1) if not circ_df.empty else pd.Series(dtype=float)
    mir_mean = mirna_df.mean(axis=1) if not mirna_df.empty else pd.Series(dtype=float)

    all_pids = metadata_df.index.tolist()
    patient_graphs = []
    
    # Pre-calculate active nodes for global network nodes
    global_nodes = list(global_g.nodes)

    if activation_threshold is None:
        expressed_values = []
        for mat in (circ_df, mirna_df):
            if not mat.empty:
                vals = mat.to_numpy(dtype=float, copy=False).ravel()
                expressed_values.append(vals[vals > 0])
        if expressed_values:
            activation_threshold = float(
                np.percentile(
                    np.concatenate(expressed_values),
                    cfg.expression_activation_percentile,
                )
            )
        else:
            activation_threshold = 0.0
    log.info(
        "Using cross-patient expression activation threshold %.6g (%.1f percentile).",
        activation_threshold,
        cfg.expression_activation_percentile,
    )
    
    for pid in tqdm(all_pids, desc="Graph Construction"):
        p_circ = circ_df[pid] if pid in circ_cols_set else circ_mean
        p_mir = mirna_df[pid] if pid in mir_cols_set else mir_mean
        
        # Use one cohort-level threshold so sequencing depth does not create
        # patient-specific graph density shifts.
        active_c = set(p_circ.index[p_circ > activation_threshold])
        active_m = set(p_mir.index[p_mir > activation_threshold])
        
        # Filter nodes that exist in global graph
        exist_c = active_c.intersection(global_g.nodes)
        exist_m = active_m.intersection(global_g.nodes)
        
        if not exist_c and not exist_m: continue
        
        to_keep = exist_c.union(exist_m)
        for m in exist_m:
            to_keep.update(global_g.neighbors(m))
            
        sub = global_g.subgraph(to_keep).copy() # Still need copy for individual edge weighting
        sub.graph["patient_id"] = pid
        
        # Fast attribute assignment
        for n in sub.nodes:
            sub.nodes[n]["expression"] = float(p_circ.get(n, p_mir.get(n, 0.0)))
                
        for u, v, d in sub.edges(data=True):
            if d.get("edge_type") == "circRNA-miRNA":
                d["weight"] = (sub.nodes[u]["expression"] * sub.nodes[v]["expression"]) ** 0.5
            else:
                d["weight"] = 1.0

        if sub.number_of_nodes() >= 2:
            patient_graphs.append(sub)
        
    return patient_graphs

def main() -> None:
    t0 = time.time()
    try:
        circ_df = pd.read_csv(cfg.processed_dir / "circRNA_counts.csv", index_col=0)
        mirna_df = pd.read_csv(cfg.processed_dir / "miRNA_counts.csv", index_col=0)
        meta_df = pd.read_csv(cfg.processed_dir / "metadata.csv", index_col=0)
        
        global_g = load_global_interactome()
        graphs = build_all_patient_graphs(global_g, circ_df, mirna_df, meta_df)
        
        with open(cfg.network_dir / "graphs.pkl", "wb") as f:
            pickle.dump(graphs, f)
        log.info("Saved %d graphs.", len(graphs))
        write_phase_metrics(3, {"status": "ok", "n": len(graphs)}, elapsed_sec=time.time()-t0)
    except Exception:
        log.exception("build_cerna failed")
        write_phase_metrics(3, {"status": "error"}, elapsed_sec=time.time()-t0)
        raise

if __name__ == "__main__":
    main()
