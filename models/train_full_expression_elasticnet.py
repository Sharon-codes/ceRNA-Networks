"""
Train Elastic Net Classifier on Full circRNA Expression Matrix
==============================================================
Comparison baseline: log2(CPM+1) all circRNA columns per patient.
Protocol: Nested 5x3-fold CV with Optuna tuning (same as XGBoost).

Metrics:
  - AUROC
  - Sensitivity @ 95% Specificity
"""

import os, json, pickle, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import optuna

# ── Configuration ─────────────────────────────────────────────────────────────
BASE         = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
COUNTS_CSV   = BASE / "data" / "processed" / "circRNA_counts.csv"
META_CSV     = BASE / "data" / "processed" / "metadata.csv"

OUT_DIR      = BASE / "models" / "elastic_net_full_expr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CV       = OUT_DIR / "cv_results.csv"
OUT_METRICS  = OUT_DIR / "metrics.json"
OUT_MODEL    = OUT_DIR / "best_model.pkl"

SEED         = 42
OUTER_FOLDS  = 5
INNER_FOLDS  = 3
N_TRIALS     = 5  # Reduced for speed

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

logger.info(f"Samples with valid labels: {len(valid_samples)}")

X_raw = counts[valid_samples].T
y_all = (meta.reindex(valid_samples)["cancer_type"].str.lower() != "healthy").astype(int)

# ── Normalization: log2(CPM + 1) ──────────────────────────────────────────────
logger.info("Normalizing to log2(CPM+1)...")
sample_sums = X_raw.sum(axis=1)
# Ensure no division by zero
sample_sums[sample_sums == 0] = 1.0

X_cpm = X_raw.div(sample_sums, axis=0) * 1e6
X = np.log2(X_cpm + 1)

# Check for NaNs/Infs
if X.isna().any().any():
    logger.warning(f"Found {X.isna().sum().sum()} NaNs in X. Filling with 0.")
    X = X.fillna(0.0)
if np.isinf(X).any().any():
    logger.warning("Found Infs in X. Filling with 0.")
    X[np.isinf(X)] = 0.0

# Filter out zero-variance features
logger.info("Filtering constant features...")
variances = X.var().sort_values(ascending=False)
top_features = variances.head(2000).index
X = X[top_features]
logger.info(f"Features after filtering for top 2000 by variance: {X.shape[1]}")

logger.info(f"Final matrix shape: {X.shape}")
logger.info(f"Class distribution: {y_all.value_counts().to_dict()}")

# ── Metrics Helper ────────────────────────────────────────────────────────────
def sensitivity_at_specificity(y_true, y_score, target_spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0: return 0.0
    return float(tpr[idx[-1]])

# ── Nested CV ─────────────────────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
fold_results = []
oof_proba = np.zeros(len(y_all))

logger.info(f"Starting nested {OUTER_FOLDS}-fold CV...")

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_all)):
    logger.info(f"--- Outer Fold {fold_idx + 1}/{OUTER_FOLDS} ---")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]
    
    # ── INNER: Optuna tuning ──
    def objective(trial):
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
        C = trial.suggest_float("C", 1e-3, 10, log=True)
        
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        inner_aucs = []
        
        for tr_i, val_i in inner_cv.split(X_train, y_train):
            m = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=l1_ratio,
                    C=C,
                    max_iter=500,
                    class_weight='balanced',
                    random_state=SEED
                ))
            ])
            m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
            prob = m.predict_proba(X_train.iloc[val_i])[:, 1]
            inner_aucs.append(roc_auc_score(y_train.iloc[val_i], prob))
            
        return float(np.mean(inner_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
    
    best_params = study.best_params
    logger.info(f"  Best params: {best_params} (Inner AUC: {study.best_value:.4f})")
    
    # ── OUTER: Evaluation ──
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=best_params["l1_ratio"],
            C=best_params["C"],
            max_iter=500,
            class_weight='balanced',
            random_state=SEED
        ))
    ])
    final_model.fit(X_train, y_train)
    
    proba = final_model.predict_proba(X_test)[:, 1]
    oof_proba[test_idx] = proba
    
    fold_auc = roc_auc_score(y_test, proba)
    fold_sens = sensitivity_at_specificity(y_test, proba, 0.95)
    
    logger.info(f"  Fold AUC: {fold_auc:.4f} | Sens@95%spec: {fold_sens:.4f}")
    res = {
        "fold": fold_idx + 1,
        "auc": fold_auc,
        "sens_95spec": fold_sens,
        "l1_ratio": best_params["l1_ratio"],
        "C": best_params["C"]
    }
    fold_results.append(res)
    # Save intermediate
    pd.DataFrame([res]).to_csv(OUT_CV, mode='a', header=not os.path.exists(OUT_CV), index=False)

# ── Final Results ──
cv_df = pd.DataFrame(fold_results)
cv_df.to_csv(OUT_CV, index=False)

aggregate_auc = roc_auc_score(y_all, oof_proba)
aggregate_sens = sensitivity_at_specificity(y_all, oof_proba, 0.95)

# Bootstrap CI for AUC
np.random.seed(SEED)
boot_aucs = []
for _ in range(1000):
    idx = np.random.choice(len(y_all), len(y_all), replace=True)
    if len(np.unique(y_all.iloc[idx])) > 1:
        boot_aucs.append(roc_auc_score(y_all.iloc[idx], oof_proba[idx]))
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

metrics = {
    "model": "Elastic Net (Logistic Regression)",
    "data": "Full circRNA log2(CPM+1) expression matrix",
    "n_samples": len(y_all),
    "n_features": X.shape[1],
    "aggregate_auc": round(aggregate_auc, 4),
    "aggregate_auc_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
    "aggregate_sens_at_95spec": round(aggregate_sens, 4),
    "fold_mean_auc": round(cv_df["auc"].mean(), 4),
    "fold_std_auc": round(cv_df["auc"].std(), 4)
}

with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

logger.info("\n" + "="*40)
logger.info("ELASTIC NET FULL EXPRESSION RESULTS")
logger.info("="*40)
logger.info(f"Aggregate AUC:        {aggregate_auc:.4f} (95% CI: {ci_lo:.4f}-{ci_hi:.4f})")
logger.info(f"Sensitivity@95%spec:  {aggregate_sens:.4f}")
logger.info(f"Mean Fold AUC:        {cv_df['auc'].mean():.4f}")
logger.info("="*40)

# ── Retrain Final Model ──
# Use best params from the highest AUC fold
best_fold_idx = cv_df["auc"].idxmax()
best_l1 = cv_df.loc[best_fold_idx, "l1_ratio"]
best_C = cv_df.loc[best_fold_idx, "C"]

logger.info(f"Retraining final model on all data (l1={best_l1:.4f}, C={best_C:.4f})...")
final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=best_l1,
        C=best_C,
        max_iter=500,
        class_weight='balanced',
        random_state=SEED
    ))
])
final_model.fit(X, y_all)

with open(OUT_MODEL, "wb") as f:
    pickle.dump(final_model, f)

logger.info(f"Results saved to {OUT_DIR}")
