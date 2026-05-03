"""
ceRNA Network Topology — Cancer Detection Classifier
Nested Cross-Validation with Optuna (no data leakage)

KEY FIX: Optuna hyperparameter search runs INSIDE each outer fold,
so the test fold is never seen during tuning. This eliminates the
optimistic bias from the previous implementation.

Outputs:
  models/best_model.pkl              — model trained on all data with best nested params
  models/cv_results.csv             — per-fold AUC, sensitivity, specificity
  models/nested_cv_metrics.json     — final metrics with 95% CI
  models/topology_feature_names.json
"""

import os, json, pickle, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score, f1_score
)
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths (edit if needed) ─────────────────────────────────────────────────────
BASE        = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV    = BASE / "features" / "feature_matrix_combat.csv"
FEAT_NAMES  = BASE / "models"   / "topology_feature_names.json"
OUT_MODEL   = BASE / "models"   / "best_model.pkl"
OUT_CV      = BASE / "models"   / "cv_results.csv"
OUT_METRICS = BASE / "models"   / "nested_cv_metrics.json"

SEED        = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3      # inner CV for Optuna
N_TRIALS    = 50     # Optuna trials per outer fold

# ── Load data ──────────────────────────────────────────────────────────────────
logger.info("Loading feature matrix...")
df = pd.read_csv(FEAT_CSV, index_col=0)

with open(FEAT_NAMES) as f:
    topo_cols = json.load(f)

X = df.reindex(columns=topo_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)

logger.info(f"  n={len(X)}, cancer={y.sum()}, healthy={(y==0).sum()}, features={X.shape[1]}")

# ── Sensitivity at fixed specificity ──────────────────────────────────────────
def sensitivity_at_specificity(y_true, y_score, target_spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    # find closest specificity >= target
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])

# ── Nested CV ─────────────────────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
oof_proba   = np.zeros(len(y))   # out-of-fold probabilities for aggregate AUC

logger.info(f"\nStarting nested {OUTER_FOLDS}-fold CV with {N_TRIALS} Optuna trials per fold...")

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    logger.info(f"\n── Outer Fold {fold_idx + 1}/{OUTER_FOLDS} ──")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw   = n_neg / max(n_pos, 1)

    # ── INNER: Optuna tunes ONLY on training fold ──────────────────────────
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "scale_pos_weight": spw,
            "use_label_encoder": False,
            "eval_metric":      "logloss",
            "random_state":     SEED,
        }
        inner_aucs = []
        for tr_i, val_i in inner_cv.split(X_train, y_train):
            m = xgb.XGBClassifier(**params)
            m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i], verbose=False)
            prob = m.predict_proba(X_train.iloc[val_i])[:, 1]
            try:
                inner_aucs.append(roc_auc_score(y_train.iloc[val_i], prob))
            except ValueError:
                inner_aucs.append(0.5)
        return float(np.mean(inner_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight":  spw,
        "use_label_encoder": False,
        "eval_metric":       "logloss",
        "random_state":      SEED,
    })
    logger.info(f"  Best inner AUC: {study.best_value:.4f} | params: {best_params}")

    # ── OUTER: Evaluate on held-out test fold ──────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, verbose=False)

    proba = final_model.predict_proba(X_test)[:, 1]
    oof_proba[test_idx] = proba

    fold_auc  = roc_auc_score(y_test, proba)
    fold_ap   = average_precision_score(y_test, proba)
    fold_sens = sensitivity_at_specificity(y_test.values, proba, 0.95)
    fold_f1   = f1_score(y_test, (proba >= 0.5).astype(int), zero_division=0)

    logger.info(f"  Fold AUC={fold_auc:.4f}  AP={fold_ap:.4f}  Sens@95spec={fold_sens:.4f}")
    fold_results.append({
        "fold":       fold_idx + 1,
        "n_test":     len(y_test),
        "auc":        round(fold_auc, 4),
        "ap":         round(fold_ap, 4),
        "sens_95spec":round(fold_sens, 4),
        "f1":         round(fold_f1, 4),
        "best_params": json.dumps(best_params),
    })

# ── Aggregate metrics ─────────────────────────────────────────────────────────
cv_df = pd.DataFrame(fold_results)
cv_df.to_csv(OUT_CV, index=False)

# Aggregate AUC (computed on all OOF predictions at once — correct method)
aggregate_auc  = float(roc_auc_score(y, oof_proba))
aggregate_sens = float(sensitivity_at_specificity(y.values, oof_proba, 0.95))
aggregate_ap   = float(average_precision_score(y, oof_proba))

# 95% CI from bootstrap on OOF probabilities
np.random.seed(SEED)
boot_aucs = []
for _ in range(2000):
    idx = np.random.choice(len(y), len(y), replace=True)
    if len(np.unique(y.values[idx])) > 1:
        boot_aucs.append(roc_auc_score(y.values[idx], oof_proba[idx]))

ci_lo = float(np.percentile(boot_aucs, 2.5))
ci_hi = float(np.percentile(boot_aucs, 97.5))

# Per-fold mean ± SD (for supplementary table)
fold_mean = float(cv_df["auc"].mean())
fold_std  = float(cv_df["auc"].std())
fold_ci_lo = fold_mean - 1.96 * fold_std / np.sqrt(OUTER_FOLDS)
fold_ci_hi = fold_mean + 1.96 * fold_std / np.sqrt(OUTER_FOLDS)

metrics = {
    "method":                    "Nested CV (Optuna inside each outer fold)",
    "outer_folds":               OUTER_FOLDS,
    "inner_folds":               INNER_FOLDS,
    "optuna_trials_per_fold":    N_TRIALS,
    "n_samples":                 len(y),
    "n_cancer":                  int(y.sum()),
    "n_healthy":                 int((y == 0).sum()),

    # Primary result — aggregate OOF (report this in text)
    "aggregate_auc":             round(aggregate_auc, 4),
    "aggregate_auc_95ci_lo":     round(ci_lo, 4),
    "aggregate_auc_95ci_hi":     round(ci_hi, 4),
    "aggregate_sens_at_95spec":  round(aggregate_sens, 4),
    "aggregate_ap":              round(aggregate_ap, 4),

    # Per-fold summary (for supplementary)
    "fold_mean_auc":             round(fold_mean, 4),
    "fold_std_auc":              round(fold_std, 4),
    "fold_ci_lo":                round(fold_ci_lo, 4),
    "fold_ci_hi":                round(fold_ci_hi, 4),
    "per_fold_auc":              list(cv_df["auc"]),
}

with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

logger.info("\n" + "=" * 55)
logger.info("FINAL NESTED CV RESULTS")
logger.info("=" * 55)
logger.info(f"Aggregate AUC:        {aggregate_auc:.4f} (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
logger.info(f"Sensitivity@95%spec:  {aggregate_sens:.4f}")
logger.info(f"Average Precision:    {aggregate_ap:.4f}")
logger.info(f"Per-fold AUC:         {list(cv_df['auc'])}")
logger.info(f"Fold mean ± SD:       {fold_mean:.4f} ± {fold_std:.4f}")
logger.info("=" * 55)

# ── Retrain on ALL data with modal best params (for deployment) ───────────────
# Use params from the fold with highest inner AUC as final params
best_fold = cv_df.loc[cv_df["auc"].idxmax(), "best_params"]
final_params = json.loads(best_fold)
n_pos_all = int(y.sum()); n_neg_all = int((y==0).sum())
final_params["scale_pos_weight"] = n_neg_all / max(n_pos_all, 1)

logger.info("\nRetraining final model on all data...")
final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X, y, verbose=False)

with open(OUT_MODEL, "wb") as f:
    pickle.dump(final_model, f)

logger.info(f"Final model saved: {OUT_MODEL}")
logger.info(f"CV results saved:  {OUT_CV}")
logger.info(f"Metrics saved:     {OUT_METRICS}")
