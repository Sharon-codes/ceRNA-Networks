"""
Fix: Separate topology features from expression features properly.

The current model mixes expression features with topology features,
which means SHAP shows expression dominating — undermining the paper's
core claim that TOPOLOGY adds signal over expression alone.

This script:
  1. Splits feature_matrix into topology-only and expression-only
  2. Trains 3 models: topology-only, expression-only, combined
  3. Reports AUC for each — THIS is your core Table 1 for the paper
  4. Saves separate feature name lists

The correct claim structure for the paper:
  - Expression-only AUC:  X.XXX  (your current "baseline")
  - Topology-only AUC:    X.XXX  (new — proves topology has signal alone)
  - Combined AUC:         X.XXX  (should be highest — topology adds value)
"""

import json, pickle, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE       = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV   = BASE / "features" / "feature_matrix.csv"
FEAT_NAMES = BASE / "models"   / "topology_feature_names.json"
OUT_DIR    = BASE / "models"
SEED       = 42

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(FEAT_CSV, index_col=0)
with open(FEAT_NAMES) as f:
    all_topo_cols = json.load(f)

X_all = df.reindex(columns=all_topo_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y     = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)

# ── Define feature groups ──────────────────────────────────────────────────────
# Expression features — these are NOT topology, they are raw abundance
EXPRESSION_KEYWORDS = [
    "expression", "count", "tpm", "rpkm", "fpkm",
    "mean_expr", "max_expr", "std_expr", "median_expr",
    "log_expr", "norm_expr", "abundance"
]

# Topology features — graph structural properties
TOPOLOGY_KEYWORDS = [
    "degree", "betweenness", "closeness", "eigenvector", "pagerank",
    "clustering", "modularity", "community", "diameter", "path_length",
    "hub", "entropy", "spectral", "centrality", "connectivity",
    "transitivity", "assortativity", "density", "component"
]

def classify_feature(name: str) -> str:
    name_lo = name.lower()
    if any(k in name_lo for k in EXPRESSION_KEYWORDS):
        return "expression"
    if any(k in name_lo for k in TOPOLOGY_KEYWORDS):
        return "topology"
    return "other"

feat_classes = {col: classify_feature(col) for col in all_topo_cols}

expr_cols  = [c for c, t in feat_classes.items() if t == "expression"]
topo_cols  = [c for c, t in feat_classes.items() if t == "topology"]
other_cols = [c for c, t in feat_classes.items() if t == "other"]

logger.info(f"Expression features: {len(expr_cols)} — {expr_cols}")
logger.info(f"Topology features:   {len(topo_cols)} — {topo_cols}")
logger.info(f"Other features:      {len(other_cols)} — {other_cols}")

# Save separated feature name lists
with open(OUT_DIR / "expression_feature_names.json", "w") as f:
    json.dump(expr_cols, f)
with open(OUT_DIR / "topology_only_feature_names.json", "w") as f:
    json.dump(topo_cols, f)

# ── Model comparison function ─────────────────────────────────────────────────
def sens_at_spec(y_true, y_score, target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where((1 - fpr) >= target)[0]
    return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

def evaluate_model(X_sub, y, label):
    if X_sub.shape[1] == 0:
        logger.warning(f"  {label}: NO FEATURES — skipping")
        return None

    n_pos = int(y.sum()); n_neg = int((y==0).sum())
    spw   = n_neg / max(n_pos, 1)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, use_label_encoder=False,
        eval_metric="logloss", random_state=SEED
    )
    oof_proba = cross_val_predict(model, X_sub, y, cv=cv, method="predict_proba")[:, 1]

    auc  = float(roc_auc_score(y, oof_proba))
    ap   = float(average_precision_score(y, oof_proba))
    sens = sens_at_spec(y.values, oof_proba, 0.95)

    # Bootstrap CI
    np.random.seed(SEED)
    boot_aucs = []
    for _ in range(2000):
        idx = np.random.choice(len(y), len(y), replace=True)
        if len(np.unique(y.values[idx])) > 1:
            boot_aucs.append(roc_auc_score(y.values[idx], oof_proba[idx]))
    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))

    logger.info(f"\n  {label}")
    logger.info(f"    Features: {X_sub.shape[1]}")
    logger.info(f"    AUC:      {auc:.4f} (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
    logger.info(f"    Sens@95%: {sens:.4f}")
    logger.info(f"    AP:       {ap:.4f}")

    # Train final model on all data and save
    model.fit(X_sub, y, verbose=False)
    safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    pkl_path = OUT_DIR / f"model_{safe_label}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "model":    label,
        "n_features": X_sub.shape[1],
        "auc":      round(auc, 4),
        "ci_lo":    round(ci_lo, 4),
        "ci_hi":    round(ci_hi, 4),
        "sens_95spec": round(sens, 4),
        "ap":       round(ap, 4),
    }

# ── Run all three comparisons ─────────────────────────────────────────────────
logger.info("\n" + "="*55)
logger.info("MODEL COMPARISON: Expression vs Topology vs Combined")
logger.info("="*55)

results = []

r1 = evaluate_model(X_all[expr_cols] if expr_cols else pd.DataFrame(), y, "Expression only")
if r1: results.append(r1)

r2 = evaluate_model(X_all[topo_cols] if topo_cols else pd.DataFrame(), y, "Topology only")
if r2: results.append(r2)

# Combined = all features (your original model)
r3 = evaluate_model(X_all, y, "Expression + Topology (combined)")
if r3: results.append(r3)

# Topology + expression together to see delta
if expr_cols and topo_cols:
    r4 = evaluate_model(X_all[topo_cols + expr_cols], y, "Topology + Expression")
    if r4: results.append(r4)

# ── Summary table ─────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

logger.info("\n" + "="*55)
logger.info("FINAL COMPARISON TABLE (this goes in your paper as Table 1)")
logger.info("="*55)
print(res_df[["model", "n_features", "auc", "ci_lo", "ci_hi", "sens_95spec"]].to_string(index=False))
logger.info("="*55)
logger.info("\nInterpretation guide:")
logger.info("  If Topology-only AUC > 0.80: topology has independent signal — strong claim")
logger.info("  If Combined AUC > Expr-only: topology adds value — paper holds")
logger.info("  If Topology-only AUC < 0.70: topology alone is weak — reframe paper")
