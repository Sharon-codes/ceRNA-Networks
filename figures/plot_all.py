"""
Generate manuscript figures (network sketch, ROC, SHAP, stage plot).
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import auc, roc_curve

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg, write_phase_metrics

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(cfg.logs_dir / "plot_all.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("plot_all")

NODE_COLORS = {"circRNA": "#1a9e9e", "miRNA": "#f4a261", "mRNA": "#adb5bd"}
CANCER_COLORS = {
    "CRC": "#e63946",
    "Lung": "#457b9d",
    "HCC": "#2a9d8f",
    "Breast": "#e9c46a",
    "Gastric": "#f4a261",
    "Healthy": "#6c757d",
}

FIGURE_DPI = 300
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _representative_subgraph(
    patient_ids: List[str],
    patient_graphs: Dict[str, nx.DiGraph],
    max_nodes: int = 30,
) -> nx.DiGraph:
    """Edges present in >50% of cohort graphs; induced on LCC, capped at ``max_nodes``.

    Args:
        patient_ids: Cohort identifiers with matching graphs.
        patient_graphs: All personalised graphs.
        max_nodes: Max nodes to draw.

    Returns:
        Representative ``DiGraph`` for layout.
    """
    ids = [p for p in patient_ids if p in patient_graphs]
    if not ids:
        return nx.DiGraph()
    edge_counts: Dict[Tuple[str, str], int] = {}
    for pid in ids:
        g = patient_graphs[pid]
        edges = {(u, v) for u, v in g.edges()}
        for e in edges:
            edge_counts[e] = edge_counts.get(e, 0) + 1
    n = len(ids)
    thr = max(1.0, 0.5 * n)
    freq = [e for e, c in edge_counts.items() if c >= thr]

    ref_id = ids[0]
    ref = patient_graphs[ref_id]
    H = nx.DiGraph()
    for u, v in freq:
        if not ref.has_edge(u, v):
            for pid in ids:
                gg = patient_graphs[pid]
                if gg.has_edge(u, v):
                    ref = gg
                    break
            else:
                continue
        d_edge = ref.get_edge_data(u, v, default={})
        if u in ref:
            H.add_node(u, **ref.nodes[u])
        if v in ref:
            H.add_node(v, **ref.nodes[v])
        H.add_edge(u, v, **d_edge)

    und = H.to_undirected()
    if und.number_of_nodes() == 0:
        return H
    lcc = max(nx.connected_components(und), key=len)
    sub = H.subgraph(lcc).copy()
    und2 = sub.to_undirected()
    if und2.number_of_nodes() > max_nodes:
        bc = nx.betweenness_centrality(und2)
        top = sorted(bc, key=bc.get, reverse=True)[:max_nodes]
        sub = sub.subgraph(top).copy()
    return sub


def plot_network_comparison(patient_graphs: dict, metadata: pd.DataFrame) -> None:
    """Fig 2: healthy vs stage-I representative networks."""
    log.info("Fig 2: network comparison")
    meta = metadata.copy()
    if "cancer_type" not in meta.columns:
        log.warning("metadata missing cancer_type — skip Fig 2")
        return
    healthy_ids = meta[meta["cancer_type"].astype(str).str.lower() == "healthy"].index.tolist()
    stage_col = meta["stage"].astype(str).str.upper() if "stage" in meta.columns else pd.Series("unknown", index=meta.index)
    cancer_ids = meta[(meta["cancer_type"].astype(str).str.lower() != "healthy") & (stage_col == "I")].index.tolist()

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
        bc = nx.betweenness_centrality(G, normalized=True)
        cols = [NODE_COLORS.get(G.nodes[n].get("node_type", "mRNA"), "#adb5bd") for n in G.nodes]
        sizes = [max(50.0, bc.get(n, 0.0) * 3000.0) for n in G.nodes]
        pos = nx.spring_layout(G, seed=42, k=1.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, node_size=sizes, alpha=0.85)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", alpha=0.4, arrows=True, arrowsize=10, width=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in NODE_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = cfg.figures_dir / "Fig2_network_comparison.png"
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_roc_curves(all_preds: np.ndarray, y_bin: np.ndarray, y_multi: np.ndarray) -> None:
    """Fig 3: per cancer type OVR ROC with bootstrap CI."""
    log.info("Fig 3: ROC curves (n_bootstrap=%s)", cfg.bootstrap_roc_samples)
    cancers = [c for c in np.unique(y_multi) if str(c).lower() not in ("healthy", "unknown", "nan")]

    n_cols = min(3, max(len(cancers), 1))
    n_rows = int(np.ceil(len(cancers) / n_cols)) if cancers else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    rng = np.random.RandomState(42)

    def boot(y_t: np.ndarray, y_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        tprs: list[np.ndarray] = []
        aucs: list[float] = []
        mean_fpr = np.linspace(0, 1, 100)
        for _ in range(cfg.bootstrap_roc_samples):
            idx = rng.randint(0, len(y_t), len(y_t))
            if len(np.unique(y_t[idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_t[idx], y_s[idx])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
        ta = np.array(tprs)
        return mean_fpr, ta.mean(axis=0), ta.std(axis=0), float(np.mean(aucs)), float(np.std(aucs))

    for ax_idx, cancer in enumerate(cancers):
        ax = axes[ax_idx]
        y_ov = (np.asarray(y_multi) == cancer).astype(int)
        if y_ov.sum() < 5 or len(np.unique(y_ov)) < 2:
            ax.text(0.5, 0.5, f"{cancer}: insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue
        fpr_m, tpr_m, tpr_sd, m_auc, s_auc = boot(y_ov, all_preds)
        col = CANCER_COLORS.get(str(cancer), "#333333")
        ax.plot(fpr_m, tpr_m, color=col, lw=2, label=f"AUC = {m_auc:.3f} ± {s_auc:.3f}")
        ax.fill_between(
            fpr_m,
            tpr_m - 1.96 * tpr_sd,
            tpr_m + 1.96 * tpr_sd,
            alpha=0.2,
            color=col,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(f"{cancer} vs Rest", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)

    for j in range(len(cancers), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Pan-Cancer ROC Curves (topology model scores)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = cfg.figures_dir / "Fig3_roc_curves.png"
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_shap_beeswarm(df: pd.DataFrame) -> None:
    """Fig 4: SHAP beeswarm for topology features."""
    log.info("Fig 4: SHAP beeswarm")
    shap_path = cfg.models_dir / "shap_values.npy"
    base_path = cfg.models_dir / "shap_base_values.npy"
    names_path = cfg.models_dir / "topology_feature_names.json"
    if not shap_path.exists():
        log.warning("shap_values.npy missing — run models/classify.py")
        return

    vals = np.load(shap_path)
    base = np.load(base_path)
    names: List[str]
    if names_path.exists():
        with open(names_path, encoding="utf-8") as f:
            names = json.load(f)
    else:
        meta = ["cancer_type", "stage", "age", "sex", "dataset"]
        names = [c for c in df.columns if c not in meta]

    X = df[names].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    if vals.shape[1] != X.shape[1]:
        log.warning("SHAP/features mismatch — trimming to min width")
        w = min(vals.shape[1], X.shape[1], len(names))
        vals = vals[:, :w]
        X = X[:, :w]
        names = names[:w]

    base_val = float(base.ravel()[0])
    expl = shap.Explanation(values=vals, base_values=base_val, data=X, feature_names=names)
    fig, _ = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(expl, max_display=20, show=False)
    plt.title("SHAP — topology features", fontweight="bold", pad=12)
    plt.tight_layout()
    out = cfg.figures_dir / "Fig4_shap_beeswarm.png"
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_stage_stratification(
    all_preds: np.ndarray,
    y_true: np.ndarray,
    stages: np.ndarray,
) -> None:
    """Fig 5: boxplot score by stage."""
    log.info("Fig 5: stage stratification")
    df = pd.DataFrame(
        {
            "model_score": all_preds,
            "stage": [str(s).upper() for s in stages],
            "true_label": np.where(y_true, "Cancer", "Healthy"),
        }
    )
    order = ["I", "II", "III", "IV"]
    df = df[df["stage"].isin(order)]
    if df.empty:
        log.warning("No staged samples — skip Fig 5")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="stage",
        y="model_score",
        hue="true_label",
        order=order,
        palette={"Cancer": "#e63946", "Healthy": "#6c757d"},
        width=0.6,
        linewidth=0.8,
        ax=ax,
    )
    ax.axhline(0.5, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Disease Stage", fontsize=12)
    ax.set_ylabel("Model Probability Score", fontsize=12)
    ax.set_title("Detection score by stage", fontweight="bold")
    ax.legend(title="True label", fontsize=9)
    plt.tight_layout()
    out = cfg.figures_dir / "Fig5_stage_stratification.png"
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def main() -> None:
    """CLI entry."""
    import time

    t0 = time.time()
    metrics: Dict[str, object] = {}
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        graphs = {}
        gp = cfg.network_dir / "graphs.pkl"
        if gp.exists():
            with open(gp, "rb") as f:
                graphs = pickle.load(f)
        meta_path = cfg.processed_dir / "metadata.csv"
        metadata = pd.read_csv(meta_path, index_col=0) if meta_path.exists() else pd.DataFrame()

        feat_path = cfg.features_dir / "feature_matrix.csv"
        if not feat_path.exists():
            raise FileNotFoundError("feature_matrix.csv missing")
        df = pd.read_csv(feat_path, index_col=0)

        meta_cols = ["cancer_type", "stage", "age", "sex", "dataset"]
        names_path = cfg.models_dir / "topology_feature_names.json"
        model_path = cfg.models_dir / "best_model.pkl"
        if model_path.exists() and names_path.exists():
            with open(names_path, encoding="utf-8") as f:
                topo_names = json.load(f)
            X_infer = df.reindex(columns=topo_names).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        else:
            topo_names = [c for c in df.columns if c not in meta_cols]
            X_infer = df[topo_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            log.warning("topology_feature_names.json missing — using all numeric non-metadata columns.")

        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            preds = model.predict_proba(X_infer.values)[:, 1]
        else:
            log.warning("best_model.pkl missing — using random scores for layout only")
            preds = np.random.RandomState(42).rand(len(df))

        y_bin = (
            df["cancer_type"].fillna("unknown").astype(str).str.lower() != "healthy"
        ).astype(int).values
        y_multi = df["cancer_type"].fillna("Unknown").values
        stages = df["stage"].fillna("unknown").values if "stage" in df.columns else np.array(["unknown"] * len(df))

        if graphs and not metadata.empty:
            plot_network_comparison(graphs, metadata)
        plot_roc_curves(preds, y_bin, y_multi)
        plot_shap_beeswarm(df)
        plot_stage_stratification(preds, y_bin, stages)

        metrics = {
            "status": "ok",
            "figures": [
                str(cfg.figures_dir / "Fig2_network_comparison.png"),
                str(cfg.figures_dir / "Fig3_roc_curves.png"),
                str(cfg.figures_dir / "Fig4_shap_beeswarm.png"),
                str(cfg.figures_dir / "Fig5_stage_stratification.png"),
            ],
        }
    except Exception:
        metrics["status"] = "error"
        log.exception("plot_all failed")
        write_phase_metrics(6, metrics, elapsed_sec=time.time() - t0)
        raise
    write_phase_metrics(6, metrics, elapsed_sec=time.time() - t0)


if __name__ == "__main__":
    main()
