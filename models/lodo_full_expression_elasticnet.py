"""
Leave-One-Dataset-Out (LODO) Validation for Elastic Net
=======================================================
Evaluation of the log2(CPM+1) full expression model on unseen datasets.
"""

import os, json, pickle, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import optuna

# ── Configuration ─────────────────────────────────────────────────────────────
BASE         = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
COUNTS_CSV   = BASE / "data" / "processed" / "circRNA_counts.csv"
META_CSV     = BASE / "data" / "processed" / "metadata.csv"

OUT_DIR      = BASE / "models" / "elastic_net_lodo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV      = OUT_DIR / "lodo_results.csv"
OUT_METRICS  = OUT_DIR / "metrics.json"

SEED         = 42
INNER_FOLDS  = 3
N_TRIALS     = 5  # Keeping it fast for now

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Load and Prepare Data ─────────────────────────────────────────────────────
logger.info("Loading counts and metadata...")
counts = pd.read_csv(COUNTS_CSV, index_col=0).fillna(0.0)
meta   = pd.read_csv(META_CSV, index_col=0)

available_samples = counts.columns
meta_subset = meta.reindex(available_samples)
valid_mask = ~meta_subset["cancer_type"].fillna("unknown").str.lower().isin(["unknown", "unknown cancer"])
valid_samples = meta_subset.index[valid_mask]

X_raw = counts[valid_samples].T
y_all = (meta.reindex(valid_samples)["cancer_type"].str.lower() != "healthy").astype(int)
groups = meta.reindex(valid_samples)["dataset"]

# ── Normalization: log2(CPM + 1) ──────────────────────────────────────────────
logger.info("Normalizing...")
sample_sums = X_raw.sum(axis=1)
sample_sums[sample_sums == 0] = 1.0
X = np.log2((X_raw.div(sample_sums, axis=0) * 1e6) + 1).fillna(0.0)

# Filter for top 2000 features by variance (same as optimized CV)
logger.info("Filtering features...")
variances = X.var().sort_values(ascending=False)
X = X[variances.head(2000).index]

# ── Metrics Helper ────────────────────────────────────────────────────────────
def sensitivity_at_specificity(y_true, y_score, target_spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0: return 0.0
    return float(tpr[idx[-1]])

# ── LODO Loop ─────────────────────────────────────────────────────────────────
datasets = groups.unique()
logger.info(f"Datasets for LODO: {datasets}")

lodo_results = []

for held_out in datasets:
    test_mask = (groups == held_out)
    train_mask = ~test_mask
    
    if test_mask.sum() < 5 or y_all[test_mask].nunique() < 2:
        logger.warning(f"Skipping {held_out} (not enough samples or classes)")
        continue
        
    logger.info(f"\n--- LODO Holdout: {held_out} ({test_mask.sum()} samples) ---")
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]
    
    # ── INNER: Tune ──
    def objective(trial):
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
        C = trial.suggest_float("C", 1e-3, 10, log=True)
        
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        inner_aucs = []
        for tr_i, val_i in inner_cv.split(X_train, y_train):
            m = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, C=C,
                    max_iter=500, class_weight='balanced', random_state=SEED
                ))
            ])
            m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
            prob = m.predict_proba(X_train.iloc[val_i])[:, 1]
            inner_aucs.append(roc_auc_score(y_train.iloc[val_i], prob))
        return float(np.mean(inner_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
    
    best = study.best_params
    
    # ── OUTER: Evaluate ──
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=best["l1_ratio"], C=best["C"],
            max_iter=500, class_weight='balanced', random_state=SEED
        ))
    ])
    final_model.fit(X_train, y_train)
    
    proba = final_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    sens = sensitivity_at_specificity(y_test, proba, 0.95)
    
    logger.info(f"  Result for {held_out}: AUC={auc:.4f}, Sens@95={sens:.4f}")
    lodo_results.append({
        "dataset": held_out,
        "n_test": int(test_mask.sum()),
        "auc": auc,
        "sens_95spec": sens,
        "best_l1": best["l1_ratio"],
        "best_C": best["C"]
    })

# ── Save ──
df_res = pd.DataFrame(lodo_results)
df_res.to_csv(OUT_CSV, index=False)

summary = {
    "mean_lodo_auc": float(df_res["auc"].mean()),
    "weighted_mean_auc": float((df_res["auc"] * df_res["n_test"]).sum() / df_res["n_test"].sum()),
    "results": lodo_results
}
with open(OUT_METRICS, "w") as f:
    json.dump(summary, f, indent=2)

logger.info("\nLODO RESULTS SUMMARY:")
logger.info(df_res.to_string(index=False))
