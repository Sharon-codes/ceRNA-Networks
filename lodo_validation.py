"""
Leave-One-Dataset-Out (LODO) Validation
========================================
Trains on 3 of the 4 GEO datasets, tests on the held-out one.
This is the external validation reviewers want.

Each iteration is a completely independent train/test split —
no sample from the test dataset is ever seen during training or tuning.

Output:
  models/lodo_results.json   — per-dataset AUC, sensitivity, CI
  models/lodo_results.csv    — table for supplementary
"""

import json, pickle, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE        = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV    = BASE / "features" / "feature_matrix_combat.csv"
META_CSV    = BASE / "data"     / "processed" / "metadata.csv"
FEAT_NAMES  = BASE / "models"   / "topology_feature_names.json"
OUT_JSON    = BASE / "models"   / "lodo_results.json"
OUT_CSV     = BASE / "models"   / "lodo_results.csv"

SEED      = 42
N_TRIALS  = 30   # fewer trials since we run this 4 times

# ── Load ──────────────────────────────────────────────────────────────────────
df   = pd.read_csv(FEAT_CSV, index_col=0)
meta = pd.read_csv(META_CSV, index_col=0)

with open(FEAT_NAMES) as f:
    topo_cols = json.load(f)

X = df.reindex(columns=topo_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)

# Attach dataset column
if "dataset" not in df.columns:
    df["dataset"] = meta.reindex(df.index)["dataset"]

datasets = df["dataset"].dropna().unique()
logger.info(f"Datasets found: {list(datasets)}")

# ── Sensitivity at fixed specificity ──────────────────────────────────────────
def sens_at_spec(y_true, y_score, target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target)[0]
    return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

# ── LODO loop ─────────────────────────────────────────────────────────────────
results = []

for held_out in datasets:
    logger.info(f"\n══ Held-out dataset: {held_out} ══")

    test_mask  = df["dataset"] == held_out
    train_mask = ~test_mask

    X_train = X[train_mask]; y_train = y[train_mask]
    X_test  = X[test_mask];  y_test  = y[test_mask]

    if len(np.unique(y_test)) < 2:
        logger.warning(f"  Skipping {held_out}: only one class in test set")
        continue

    logger.info(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    logger.info(f"  Test cancer: {y_test.sum()} | healthy: {(y_test==0).sum()}")

    n_pos = int(y_train.sum()); n_neg = int((y_train==0).sum())
    spw   = n_neg / max(n_pos, 1)

    # Tune on training datasets only (3-fold inner CV on training data)
    from sklearn.model_selection import StratifiedKFold
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": spw,
            "use_label_encoder": False,
            "eval_metric":      "logloss",
            "random_state":     SEED,
        }
        aucs = []
        for tr_i, val_i in inner_cv.split(X_train, y_train):
            m = xgb.XGBClassifier(**params)
            m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i], verbose=False)
            prob = m.predict_proba(X_train.iloc[val_i])[:, 1]
            try:
                aucs.append(roc_auc_score(y_train.iloc[val_i], prob))
            except ValueError:
                aucs.append(0.5)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    best.update({"scale_pos_weight": spw, "use_label_encoder": False,
                 "eval_metric": "logloss", "random_state": SEED})

    # Train on ALL training datasets, evaluate on held-out
    model = xgb.XGBClassifier(**best)
    model.fit(X_train, y_train, verbose=False)
    proba = model.predict_proba(X_test)[:, 1]

    auc  = float(roc_auc_score(y_test, proba))
    ap   = float(average_precision_score(y_test, proba))
    sens = sens_at_spec(y_test.values, proba, 0.95)

    # Bootstrap CI for this fold
    np.random.seed(SEED)
    boot = []
    for _ in range(1000):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        yt, yp = y_test.values[idx], proba[idx]
        if len(np.unique(yt)) > 1:
            boot.append(roc_auc_score(yt, yp))
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    logger.info(f"  AUC: {auc:.4f} (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
    logger.info(f"  Sensitivity@95%spec: {sens:.4f}")

    results.append({
        "held_out_dataset": held_out,
        "n_test":           len(y_test),
        "n_cancer_test":    int(y_test.sum()),
        "n_healthy_test":   int((y_test==0).sum()),
        "auc":              round(auc, 4),
        "auc_ci_lo":        round(ci_lo, 4),
        "auc_ci_hi":        round(ci_hi, 4),
        "average_precision":round(ap, 4),
        "sensitivity_95spec":round(sens, 4),
    })

# ── Save ──────────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_CSV, index=False)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

logger.info("\n" + "=" * 55)
logger.info("LEAVE-ONE-DATASET-OUT RESULTS")
logger.info("=" * 55)
print(res_df[["held_out_dataset", "n_test", "auc", "auc_ci_lo", "auc_ci_hi",
              "sensitivity_95spec"]].to_string(index=False))
logger.info("=" * 55)
logger.info(f"Saved: {OUT_CSV}")
logger.info(f"Saved: {OUT_JSON}")
