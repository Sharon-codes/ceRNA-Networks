"""
Generate Precision-Recall (PR) Curves for binary and multi-cancer classification.
"""

import json, pickle, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE       = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV   = BASE / "features" / "feature_matrix.csv"
FEAT_NAMES = BASE / "models"   / "topology_feature_names.json"
OUT_DIR    = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(FEAT_CSV, index_col=0)
with open(FEAT_NAMES) as f:
    topo_cols = json.load(f)

X = df.reindex(columns=topo_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_bin = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)
y_multi = df["cancer_type"].fillna("Unknown")

# ── Get OOF Predictions ───────────────────────────────────────────────────────
logger.info("Computing OOF predictions for PR curves...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Using the same params as in models/cv_results.csv if possible, or reasonable defaults
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    use_label_encoder=False, eval_metric="logloss", random_state=42
)

# Binary PR
probs_bin = cross_val_predict(model, X, y_bin, cv=cv, method="predict_proba")[:, 1]
precision, recall, _ = precision_recall_curve(y_bin, probs_bin)
ap_bin = average_precision_score(y_bin, probs_bin)

# ── Plotting ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 7))
plt.plot(recall, precision, color='steelblue', lw=2, label=f'Binary (AP={ap_bin:.3f})')

# Multi-class PR (OVR)
cancers = [c for c in np.unique(y_multi) if str(c).lower() not in ("healthy", "unknown", "nan")]
colors = plt.cm.get_cmap("tab10", len(cancers))

for i, cancer in enumerate(cancers):
    y_ovr = (y_multi == cancer).astype(int)
    if y_ovr.sum() < 5: continue
    
    # We need OVR probabilities
    probs_ovr = cross_val_predict(model, X, y_ovr, cv=cv, method="predict_proba")[:, 1]
    p, r, _ = precision_recall_curve(y_ovr, probs_ovr)
    ap = average_precision_score(y_ovr, probs_ovr)
    plt.plot(r, p, lw=1.5, alpha=0.7, label=f'{cancer} (AP={ap:.3f})')

plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curves", fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=9, bbox_to_anchor=(1, 0.5))
plt.grid(alpha=0.3)
plt.tight_layout()

out_path = OUT_DIR / "PR_curves.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
logger.info(f"PR curves saved to {out_path}")
