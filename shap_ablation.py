"""
SHAP Feature Importance + Ablation Study
==========================================
Answers the reviewer: "Which graph features actually drive performance?"

Produces:
  1. SHAP summary plot (top 20 features) — publication figure
  2. Ablation table: AUC when removing each feature group
  3. Feature importance CSV with SHAP values

Output:
  analysis_output/shap_summary.png
  analysis_output/shap_ablation.csv
  analysis_output/top_features.csv
"""

import json, pickle, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE       = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV   = BASE / "features" / "feature_matrix.csv"
FEAT_NAMES = BASE / "models"   / "topology_feature_names.json"
MODEL_PKL  = BASE / "models"   / "best_model.pkl"
OUT_DIR    = BASE / "analysis_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(FEAT_CSV, index_col=0)
with open(FEAT_NAMES) as f:
    topo_cols = json.load(f)

X = df.reindex(columns=topo_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)

with open(MODEL_PKL, "rb") as f:
    model = pickle.load(f)

# ── SHAP values ───────────────────────────────────────────────────────────────
logger.info("Computing SHAP values (TreeExplainer)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Mean absolute SHAP per feature
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df   = pd.DataFrame({"feature": topo_cols, "mean_abs_shap": mean_shap})
shap_df   = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_df.to_csv(OUT_DIR / "top_features.csv", index=False)

logger.info("Top 10 features by SHAP:")
print(shap_df.head(10).to_string(index=False))

# ── SHAP summary plot ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, feature_names=topo_cols,
                  max_display=20, show=False)
plt.title("SHAP Feature Importance — ceRNA Network Topology", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()
logger.info(f"SHAP plot saved: {OUT_DIR / 'shap_summary.png'}")

# ── Ablation by feature group ──────────────────────────────────────────────────
# Define feature groups (adjust these to match your actual feature names)
FEATURE_GROUPS = {
    "degree_centrality":     [c for c in topo_cols if "degree" in c.lower()],
    "betweenness_centrality":[c for c in topo_cols if "between" in c.lower()],
    "clustering":            [c for c in topo_cols if "cluster" in c.lower()],
    "pagerank":              [c for c in topo_cols if "pagerank" in c.lower() or "page_rank" in c.lower()],
    "eigenvector":           [c for c in topo_cols if "eigen" in c.lower()],
    "connectivity":          [c for c in topo_cols if "connect" in c.lower()],
    "path_length":           [c for c in topo_cols if "path" in c.lower()],
    "hub_score":             [c for c in topo_cols if "hub" in c.lower()],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_auc(X_sub, y):
    aucs = []
    for tr, te in cv.split(X_sub, y):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42,
                               use_label_encoder=False, eval_metric="logloss",
                               scale_pos_weight=int((y==0).sum())/max(int(y.sum()),1))
        m.fit(X_sub.iloc[tr], y.iloc[tr], verbose=False)
        prob = m.predict_proba(X_sub.iloc[te])[:, 1]
        try:
            aucs.append(roc_auc_score(y.iloc[te], prob))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")

logger.info("\nRunning ablation study (remove one feature group at a time)...")
ablation_results = [{"removed_group": "none (full model)", "n_features": len(topo_cols),
                     "auc": cv_auc(X, y)}]

for grp_name, grp_cols in FEATURE_GROUPS.items():
    present = [c for c in grp_cols if c in topo_cols]
    if not present:
        logger.info(f"  Group '{grp_name}': no matching features — skipping")
        continue
    remaining = [c for c in topo_cols if c not in present]
    if len(remaining) < 2:
        continue
    X_sub = X[remaining]
    auc   = cv_auc(X_sub, y)
    drop  = ablation_results[0]["auc"] - auc
    logger.info(f"  Remove '{grp_name}' ({len(present)} features): AUC={auc:.4f}  Δ={drop:+.4f}")
    ablation_results.append({
        "removed_group": grp_name,
        "n_features_removed": len(present),
        "n_features_remaining": len(remaining),
        "auc": round(auc, 4),
        "auc_drop": round(drop, 4),
    })

abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv(OUT_DIR / "shap_ablation.csv", index=False)
logger.info(f"\nAblation table saved: {OUT_DIR / 'shap_ablation.csv'}")
print(abl_df.to_string(index=False))
