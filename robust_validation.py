"""
Reviewer-grade validation for ceRNA cancer detection.

This runner keeps dataset identity out of ordinary CV leakage paths:

* outer and inner CV are blocked by GEO dataset
* topology and expression features are harmonised by dataset inside each split
* dataset/platform dummy features are learned only for training datasets
* LODO reports sample size, confidence interval, platform, and same-platform support
* topology batch sensitivity is tested with Kruskal-Wallis diagnostics

The script consumes the current feature matrix. Re-run phases 3 and 4 first if
you want graph features rebuilt after changing the activation threshold.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import kruskal
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robust_validation")

META_COLS = {"cancer_type", "stage", "age", "sex", "dataset", "platform"}
SEED = cfg.xgb_seed
OUTER_FOLDS = cfg.n_cv_folds
INNER_FOLDS = cfg.optuna_inner_splits
N_TRIALS = int(os.environ.get("N_TRIALS", cfg.optuna_trials))
BOOTSTRAPS = int(os.environ.get("BOOTSTRAPS", cfg.bootstrap_roc_samples))


def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where((1.0 - fpr) >= target)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0


def safe_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    y_arr = np.asarray(list(y_true))
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, np.asarray(list(y_score))))


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = BOOTSTRAPS) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    aucs: List[float] = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_path = cfg.features_dir / "feature_matrix.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}")
    df = pd.read_csv(feature_path, index_col=0)
    if "dataset" not in df.columns:
        meta = pd.read_csv(cfg.processed_dir / "metadata.csv", index_col=0)
        df["dataset"] = meta.reindex(df.index)["dataset"]
    df = df[~df["cancer_type"].astype(str).str.lower().eq("unknown")].copy()
    df["platform"] = df["dataset"].map(cfg.dataset_platforms).fillna("unknown")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = (~df["cancer_type"].astype(str).str.lower().eq("healthy")).astype(int)
    groups = df["dataset"].astype(str)
    return df, X, y, groups


def harmonize_train_test(
    X_train: pd.DataFrame,
    train_groups: pd.Series,
    X_test: pd.DataFrame,
    test_groups: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dataset-wise feature harmonisation without cancer-label covariates."""
    train_groups = pd.Series(train_groups, index=X_train.index).astype(str)
    grand_mean = X_train.mean(axis=0)
    grand_std = X_train.std(axis=0).replace(0, 1.0).fillna(1.0)

    X_train_h = X_train.copy()
    for ds in train_groups.unique():
        mask = train_groups == ds
        batch = X_train.loc[mask]
        batch_mean = batch.mean(axis=0)
        batch_std = batch.std(axis=0).replace(0, 1.0).fillna(1.0)
        X_train_h.loc[mask] = ((batch - batch_mean) / batch_std) * grand_std + grand_mean

    X_test_h = X_test.copy()
    if test_groups is None:
        X_test_h = (X_test - grand_mean) / grand_std
    else:
        test_groups = pd.Series(test_groups, index=X_test.index).astype(str)
        for ds in test_groups.unique():
            mask = test_groups == ds
            batch = X_test.loc[mask]
            batch_mean = batch.mean(axis=0)
            batch_std = batch.std(axis=0).replace(0, 1.0).fillna(1.0)
            X_test_h.loc[mask] = ((batch - batch_mean) / batch_std) * grand_std + grand_mean

    return X_train_h.fillna(0.0), X_test_h.fillna(0.0)


def add_platform_covariates(
    X_train: pd.DataFrame,
    train_groups: pd.Series,
    X_test: pd.DataFrame,
    test_groups: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_groups = pd.Series(train_groups, index=X_train.index).astype(str)
    train_dummies = pd.get_dummies(train_groups, prefix="dataset")
    train_dummies.index = X_train.index

    test_dummies = pd.DataFrame(0, index=X_test.index, columns=train_dummies.columns)
    if test_groups is not None:
        test_groups = pd.Series(test_groups, index=X_test.index).astype(str)
        for col in train_dummies.columns:
            ds = col.replace("dataset_", "", 1)
            test_dummies[col] = (test_groups == ds).astype(int)

    return (
        pd.concat([X_train, train_dummies], axis=1),
        pd.concat([X_test, test_dummies], axis=1),
    )


def prepare_split(
    X: pd.DataFrame,
    groups: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    g_train = groups.iloc[train_idx]
    g_test = groups.iloc[test_idx]
    X_train, X_test = harmonize_train_test(X_train, g_train, X_test, g_test)
    return add_platform_covariates(X_train, g_train, X_test, g_test)


def make_params(trial: optuna.Trial, scale_pos_weight: float) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 80, 260),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 8.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 8.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": 1,
    }


def tune_params(X_train_raw: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series) -> Dict[str, Any]:
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / max(n_pos, 1)
    n_group_splits = min(INNER_FOLDS, groups_train.nunique())

    if n_group_splits >= 2:
        splitter = StratifiedGroupKFold(n_splits=n_group_splits, shuffle=True, random_state=SEED)
        split_iter = lambda: splitter.split(X_train_raw, y_train, groups_train)
    else:
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        split_iter = lambda: splitter.split(X_train_raw, y_train)

    def objective(trial: optuna.Trial) -> float:
        params = make_params(trial, spw)
        aucs: List[float] = []
        for tr_i, val_i in split_iter():
            X_tr, X_val = prepare_split(X_train_raw, groups_train, tr_i, val_i)
            y_tr = y_train.iloc[tr_i]
            y_val = y_train.iloc[val_i]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            proba = model.predict_proba(X_val)[:, 1]
            auc = safe_auc(y_val, proba)
            aucs.append(0.5 if np.isnan(auc) else auc)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = dict(study.best_params)
    best.update(
        {
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": SEED,
            "n_jobs": 1,
        }
    )
    return best


def run_grouped_nested_cv(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Dict[str, Any]:
    n_splits = min(OUTER_FOLDS, groups.nunique())
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    rows: List[Dict[str, Any]] = []
    per_dataset_rows: List[Dict[str, Any]] = []

    log.info("Running dataset-blocked nested CV: outer=%s inner=%s trials=%s", n_splits, INNER_FOLDS, N_TRIALS)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        test_datasets = sorted(groups.iloc[test_idx].unique())
        log.info("Outer fold %s/%s, held dataset(s): %s", fold, n_splits, test_datasets)
        best = tune_params(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx])
        X_train, X_test = prepare_split(X, groups, train_idx, test_idx)
        model = xgb.XGBClassifier(**best)
        model.fit(X_train, y.iloc[train_idx], verbose=False)
        proba = model.predict_proba(X_test)[:, 1]
        oof[test_idx] = proba

        fold_auc = safe_auc(y.iloc[test_idx], proba)
        rows.append(
            {
                "fold": fold,
                "held_out_datasets": ",".join(test_datasets),
                "n_test": int(len(test_idx)),
                "n_cancer": int(y.iloc[test_idx].sum()),
                "n_healthy": int((y.iloc[test_idx] == 0).sum()),
                "auc": round(fold_auc, 4),
                "average_precision": round(float(average_precision_score(y.iloc[test_idx], proba)), 4),
                "sens_95spec": round(sensitivity_at_specificity(y.iloc[test_idx].to_numpy(), proba), 4),
                "f1": round(float(f1_score(y.iloc[test_idx], (proba >= 0.5).astype(int), zero_division=0)), 4),
                "best_params": json.dumps(best),
            }
        )

        for ds in test_datasets:
            ds_mask = groups.iloc[test_idx].to_numpy() == ds
            if ds_mask.sum() >= 10:
                ds_y = y.iloc[test_idx].to_numpy()[ds_mask]
                ds_p = proba[ds_mask]
                per_dataset_rows.append(
                    {
                        "fold": fold,
                        "dataset": ds,
                        "platform": cfg.dataset_platforms.get(ds, "unknown"),
                        "n": int(ds_mask.sum()),
                        "n_cancer": int(ds_y.sum()),
                        "n_healthy": int((ds_y == 0).sum()),
                        "auc": round(safe_auc(ds_y, ds_p), 4),
                    }
                )

    ci_lo, ci_hi = bootstrap_auc_ci(y.to_numpy(), oof)
    aggregate_auc = safe_auc(y, oof)
    metrics = {
        "method": "Nested CV with StratifiedGroupKFold by dataset, split-local harmonisation, and train-dataset platform covariates",
        "outer_folds": int(n_splits),
        "inner_folds": int(INNER_FOLDS),
        "optuna_trials_per_fold": int(N_TRIALS),
        "n_samples": int(len(y)),
        "n_cancer": int(y.sum()),
        "n_healthy": int((y == 0).sum()),
        "aggregate_auc": round(aggregate_auc, 4),
        "aggregate_auc_95ci_lo": round(ci_lo, 4),
        "aggregate_auc_95ci_hi": round(ci_hi, 4),
        "aggregate_ap": round(float(average_precision_score(y, oof)), 4),
        "aggregate_sens_at_95spec": round(sensitivity_at_specificity(y.to_numpy(), oof), 4),
        "fold_mean_auc": round(float(pd.DataFrame(rows)["auc"].mean()), 4),
        "fold_std_auc": round(float(pd.DataFrame(rows)["auc"].std()), 4),
    }

    pd.DataFrame(rows).to_csv(cfg.models_dir / "robust_nested_cv_folds.csv", index=False)
    pd.DataFrame(per_dataset_rows).to_csv(cfg.models_dir / "robust_nested_cv_by_dataset.csv", index=False)
    with open(cfg.models_dir / "robust_nested_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(cfg.models_dir / "robust_nested_cv_oof_predictions.csv", "w", encoding="utf-8") as f:
        pred_df = df[["cancer_type", "dataset", "platform"]].copy()
        pred_df["y_true"] = y
        pred_df["y_score"] = oof
        pred_df.to_csv(f)
    return metrics


def run_lodo(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    datasets = list(groups.value_counts().index)
    log.info("Running LODO over %s datasets; primary table requires n >= %s", len(datasets), cfg.min_lodo_samples)

    for held_out in datasets:
        test_mask = groups == held_out
        train_mask = ~test_mask
        if int(test_mask.sum()) < cfg.min_lodo_samples:
            results.append(
                {
                    "held_out_dataset": held_out,
                    "platform": cfg.dataset_platforms.get(held_out, "unknown"),
                    "n_test": int(test_mask.sum()),
                    "status": "skipped_n_below_minimum",
                }
            )
            continue
        if y[test_mask].nunique() < 2:
            results.append(
                {
                    "held_out_dataset": held_out,
                    "platform": cfg.dataset_platforms.get(held_out, "unknown"),
                    "n_test": int(test_mask.sum()),
                    "status": "skipped_single_class",
                }
            )
            continue

        held_platform = cfg.dataset_platforms.get(held_out, "unknown")
        train_platforms = {cfg.dataset_platforms.get(ds, "unknown") for ds in groups[train_mask].unique()}
        log.info("LODO holdout %s (%s), n=%s", held_out, held_platform, int(test_mask.sum()))

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        best = tune_params(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx])
        X_train, X_test = prepare_split(X, groups, train_idx, test_idx)
        model = xgb.XGBClassifier(**best)
        model.fit(X_train, y.iloc[train_idx], verbose=False)
        proba = model.predict_proba(X_test)[:, 1]
        y_test = y.iloc[test_idx].to_numpy()
        ci_lo, ci_hi = bootstrap_auc_ci(y_test, proba)
        auc = safe_auc(y_test, proba)

        results.append(
            {
                "held_out_dataset": held_out,
                "platform": held_platform,
                "same_platform_in_training": bool(held_platform in train_platforms),
                "n_test": int(len(test_idx)),
                "n_cancer_test": int(y_test.sum()),
                "n_healthy_test": int((y_test == 0).sum()),
                "auc": round(auc, 4),
                "auc_ci_lo": round(ci_lo, 4),
                "auc_ci_hi": round(ci_hi, 4),
                "ci_excludes_0_5": bool(ci_lo > 0.5 or ci_hi < 0.5),
                "average_precision": round(float(average_precision_score(y_test, proba)), 4),
                "sensitivity_95spec": round(sensitivity_at_specificity(y_test, proba), 4),
                "status": "ok",
                "best_params": json.dumps(best),
            }
        )

    out = pd.DataFrame(results)
    out.to_csv(cfg.models_dir / "robust_lodo_results.csv", index=False)
    with open(cfg.models_dir / "robust_lodo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out


def run_topology_batch_diagnostics(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, feature in enumerate(X.columns):
        cancer_groups = [
            X.loc[(y == 1) & (groups == ds), feature].to_numpy()
            for ds in groups.unique()
            if int(((y == 1) & (groups == ds)).sum()) > 10
        ]
        healthy_groups = [
            X.loc[(y == 0) & (groups == ds), feature].to_numpy()
            for ds in groups.unique()
            if int(((y == 0) & (groups == ds)).sum()) > 10
        ]
        label_groups = [X.loc[y == cls, feature].to_numpy() for cls in [0, 1] if int((y == cls).sum()) > 10]

        def kw(groups_list: List[np.ndarray]) -> Tuple[float, float]:
            if len(groups_list) < 2:
                return float("nan"), float("nan")
            try:
                stat, pval = kruskal(*groups_list)
                return float(stat), float(pval)
            except ValueError:
                return float("nan"), float("nan")

        cancer_stat, cancer_p = kw(cancer_groups)
        healthy_stat, healthy_p = kw(healthy_groups)
        label_stat, label_p = kw(label_groups)
        rows.append(
            {
                "feature": feature,
                "dataset_kw_stat_cancer_only": cancer_stat,
                "dataset_kw_p_cancer_only": cancer_p,
                "dataset_kw_stat_healthy_only": healthy_stat,
                "dataset_kw_p_healthy_only": healthy_p,
                "label_kw_stat_all_samples": label_stat,
                "label_kw_p_all_samples": label_p,
                "batch_sensitive_p_lt_0_001": bool(
                    (not np.isnan(cancer_p) and cancer_p < 0.001)
                    or (not np.isnan(healthy_p) and healthy_p < 0.001)
                ),
            }
        )

    out = pd.DataFrame(rows)
    cfg.project_root.joinpath("analysis_output").mkdir(exist_ok=True)
    out.to_csv(cfg.project_root / "analysis_output" / "topology_batch_stability.csv", index=False)
    return out


def main() -> None:
    df, X, y, groups = load_inputs()
    log.info("Loaded n=%s, cancer=%s, healthy=%s, features=%s", len(y), int(y.sum()), int((y == 0).sum()), X.shape[1])
    log.info("Dataset distribution:\n%s", pd.crosstab(groups, df["cancer_type"]).to_string())

    diagnostics = run_topology_batch_diagnostics(df, X, y, groups)
    nested = run_grouped_nested_cv(df, X, y, groups)
    lodo = run_lodo(df, X, y, groups)

    n_pos_all = int(y.sum())
    n_neg_all = int((y == 0).sum())
    final_params = cfg.xgb_base.copy()
    final_params.update(
        {
            "scale_pos_weight": n_neg_all / max(n_pos_all, 1),
            "eval_metric": "logloss",
            "random_state": SEED,
            "n_jobs": 1,
        }
    )
    X_h, _ = harmonize_train_test(X, groups, X.iloc[:0], groups.iloc[:0])
    X_final, _ = add_platform_covariates(X_h, groups, X_h.iloc[:0], groups.iloc[:0])
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_final, y, verbose=False)
    with open(cfg.models_dir / "robust_best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nDATASET-BLOCKED NESTED CV")
    print(json.dumps(nested, indent=2))
    print("\nROBUST LODO")
    cols = [
        "held_out_dataset",
        "platform",
        "same_platform_in_training",
        "n_test",
        "auc",
        "auc_ci_lo",
        "auc_ci_hi",
        "ci_excludes_0_5",
        "sensitivity_95spec",
        "status",
    ]
    print(lodo.reindex(columns=cols).to_string(index=False))
    print("\nBATCH-SENSITIVE FEATURES (p < 0.001)")
    print(diagnostics.loc[diagnostics["batch_sensitive_p_lt_0_001"], ["feature", "dataset_kw_p_cancer_only", "dataset_kw_p_healthy_only"]].to_string(index=False))


if __name__ == "__main__":
    main()
