"""
Sensitivity Analysis on Feature Selection Threshold
===================================================
Reviewer question: Is the model performance brittle to the feature selection threshold?
We test this by varying the percentile of features kept based on variance.
"""

import json, pickle, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE       = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV   = BASE / "features" / "feature_matrix.csv"
FEAT_NAMES = BASE / "models"   / "topology_feature_names.json"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(FEAT_CSV, index_col=0)
with open(FEAT_NAMES) as f:
    topo_cols = json.load(f)

X = df.reindex(columns=topo_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)

# ── Run Sensitivity Analysis on One Fold ──────────────────────────────────────
# We vary the 'percentile' of features removed by variance
# (The user mentioned 25th percentile)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X, y))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

results = []
thresholds = [0, 0.1, 0.25, 0.4, 0.5] # Percentiles of low-variance features to remove

logger.info("Running sensitivity analysis on variance-based feature pruning...")

for p in thresholds:
    # Calculate variance threshold for the p-th percentile
    variances = X_train.var()
    if p > 0:
        thresh_val = np.percentile(variances, p * 100)
    else:
        thresh_val = 0.0
        
    selector = VarianceThreshold(threshold=thresh_val)
    X_tr_sub = selector.fit_transform(X_train)
    X_te_sub = selector.transform(X_test)
    
    n_feat = X_tr_sub.shape[1]
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
        scale_pos_weight=(y_train==0).sum()/y_train.sum()
    )
    model.fit(X_tr_sub, y_train)
    probs = model.predict_proba(X_te_sub)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    logger.info(f"Threshold Percentile: {p:.2f} | Features Kept: {n_feat} | AUC: {auc:.4f}")
    results.append({"percentile_removed": p, "n_features": n_feat, "auc": auc})

res_df = pd.DataFrame(results)
out_path = BASE / "analysis_output" / "sensitivity_analysis_25th_percentile.csv"
res_df.to_csv(out_path, index=False)
logger.info(f"Sensitivity analysis results saved to {out_path}")
print(res_df.to_string(index=False))
