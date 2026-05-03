"""
Model Calibration Check
========================
Reviewer requirement: show the model is not just high AUC but also
well-calibrated (predicted probabilities match actual frequencies).

Produces:
  analysis_output/calibration_curve.png   — reliability diagram
  analysis_output/calibration_metrics.json
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

BASE       = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV   = BASE / "features" / "feature_matrix.csv"
FEAT_NAMES = BASE / "models"   / "topology_feature_names.json"
MODEL_PKL  = BASE / "models"   / "best_model.pkl"
OUT_DIR    = BASE / "analysis_output"; OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(FEAT_CSV, index_col=0)
with open(FEAT_NAMES) as f:
    topo_cols = json.load(f)
X = df.reindex(columns=topo_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)

with open(MODEL_PKL, "rb") as f:
    model = pickle.load(f)

# OOF probabilities for calibration (avoids overfitting bias)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

# Calibration curve
fraction_pos, mean_pred = calibration_curve(y, oof_proba, n_bins=10, strategy="quantile")
brier = brier_score_loss(y, oof_proba)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
ax.plot(mean_pred, fraction_pos, "s-", color="steelblue", lw=2,
        markersize=7, label=f"Topology model (Brier={brier:.4f})")
ax.set_xlabel("Mean predicted probability", fontsize=12)
ax.set_ylabel("Fraction of positives", fontsize=12)
ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13)
ax.legend(fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "calibration_curve.png", dpi=300, bbox_inches="tight")
plt.close()

metrics = {
    "brier_score": round(brier, 5),
    "note": "Brier < 0.10 = well calibrated for clinical use. "
            "Brier < 0.20 = acceptable. Brier > 0.25 = needs Platt/isotonic recalibration."
}
with open(OUT_DIR / "calibration_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Brier score: {brier:.5f}")
print(f"Calibration plot saved: {OUT_DIR / 'calibration_curve.png'}")
print("If Brier > 0.15, add Platt scaling: CalibratedClassifierCV(model, cv=5, method='sigmoid')")
