"""
Binary and multi-class cancer classification with nested CV and Stage-specific evaluation.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg, write_phase_metrics

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(cfg.logs_dir / "classify.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("classify")


def load_feature_matrix() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    path = cfg.features_dir / "feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError("feature_matrix.csv not found")
    df = pd.read_csv(path, index_col=0)
    
    if "cancer_type" in df.columns:
        df = df[~df["cancer_type"].astype(str).str.lower().eq("unknown")]

    expr_cols = ["n_circ_expressed", "n_mirna_expressed", "mean_expression", "max_expression", "expression_std"]
    meta_cols = ["cancer_type", "stage", "age", "sex", "dataset"]
    expr_have = [c for c in expr_cols if c in df.columns]
    topo_cols = [c for c in df.columns if c not in meta_cols]

    X_topo = df[topo_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_expr = df[expr_have].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    cancer_col = df["cancer_type"]
    y_binary = (~cancer_col.astype(str).str.lower().eq("healthy")).astype(int)
    y_multi = cancer_col.copy()
    stages = df["stage"].fillna("unknown")

    return X_topo, X_expr, y_binary, y_multi, stages


def sensitivity_at_specificity(y_true, y_score, spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= (1.0 - spec)
    return float(np.max(tpr[mask])) if np.any(mask) else 0.0


import optuna

def run_classification_pipeline(X: pd.DataFrame, y_bin: pd.Series, y_mul: pd.Series, stages: pd.Series, tag: str) -> Dict[str, Any]:
    log.info("=== Training Pipeline: %s ===", tag)
    skf = StratifiedKFold(n_splits=cfg.n_cv_folds, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "eval_metric": "auc",
            "scale_pos_weight": (y_bin == 0).sum() / max(1, (y_bin == 1).sum())
        }
        aucs = []
        for tr, te in StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y_bin):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X.iloc[tr], y_bin.iloc[tr])
            aucs.append(roc_auc_score(y_bin.iloc[te], clf.predict_proba(X.iloc[te])[:, 1]))
        return np.mean(aucs)
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    log.info("Running Optuna tuning (50 trials)...")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    log.info("Best params: %s", best_params)
    
    bin_params = best_params.copy()
    bin_params["scale_pos_weight"] = (y_bin == 0).sum() / max(1, (y_bin == 1).sum())
    bin_params["random_state"] = 42
    bin_params["eval_metric"] = "auc"
    
    mul_params = best_params.copy()
    mul_params["random_state"] = 42
    
    binary_aucs, binary_sens, multi_accs, stage_I_aucs, stage_I_sens = [], [], [], [], []
    
    for fold, (tr, te) in enumerate(skf.split(X, y_mul)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_bin.iloc[tr], y_bin.iloc[te]
        ym_tr, ym_te = y_mul.iloc[tr], y_mul.iloc[te]
        st_te = stages.iloc[te]
        
        clf_bin = xgb.XGBClassifier(**bin_params)
        clf_bin.fit(X_tr, y_tr)
        probs = clf_bin.predict_proba(X_te)[:, 1]
        
        binary_aucs.append(roc_auc_score(y_te, probs))
        binary_sens.append(sensitivity_at_specificity(y_te, probs, 0.95))
        
        le = LabelEncoder()
        ym_tr_enc = le.fit_transform(ym_tr)
        ym_te_enc = le.transform(ym_te)
        clf_mul = xgb.XGBClassifier(**mul_params)
        clf_mul.fit(X_tr, ym_tr_enc)
        multi_accs.append(accuracy_score(ym_te_enc, clf_mul.predict(X_te)))
        
        s1_mask = st_te.astype(str).str.upper().isin(["I", "STAGE I", "1"])
        h_mask = ym_te.astype(str).str.lower().eq("healthy")
        
        eval_mask = s1_mask | h_mask
        if s1_mask.any() and h_mask.any():
            y_eval = y_te[eval_mask]
            p_eval = probs[eval_mask]
            if len(np.unique(y_eval)) > 1:
                stage_I_aucs.append(roc_auc_score(y_eval, p_eval))
                stage_I_sens.append(sensitivity_at_specificity(y_eval, p_eval, 0.95))

    res = {
        "tag": tag,
        "mean_auc": np.mean(binary_aucs),
        "mean_sensitivity_95spec": np.mean(binary_sens),
        "mean_multi_accuracy": np.mean(multi_accs),
        "mean_stage_I_auc": np.mean(stage_I_aucs) if stage_I_aucs else 0.0,
        "mean_stage_I_sensitivity": np.mean(stage_I_sens) if stage_I_sens else 0.0,
        "best_params": bin_params
    }
    log.info("Results for %s: AUC=%.3f, Acc=%.3f, Stage-I AUC=%.3f", 
             tag, res['mean_auc'], res['mean_multi_accuracy'], res['mean_stage_I_auc'])
    return res

def main() -> None:
    t0 = time.time()
    try:
        X_t, X_e, y_b, y_m, stages = load_feature_matrix()
        
        res_topo = run_classification_pipeline(X_t, y_b, y_m, stages, "topology_xgboost")
        res_expr = run_classification_pipeline(X_e, y_b, y_m, stages, "expression_xgboost")
        
        pd.DataFrame([{k: v for k, v in r.items() if k != "best_params"} for r in [res_topo, res_expr]]).to_csv(cfg.models_dir / "cv_results.csv", index=False)
        
        final_model = xgb.XGBClassifier(**res_topo["best_params"])
        final_model.fit(X_t, y_b)
        
        import pickle
        import shap
        import json
        with open(cfg.models_dir / "best_model.pkl", "wb") as f:
            pickle.dump(final_model, f)
            
        with open(cfg.models_dir / "topology_feature_names.json", "w", encoding="utf-8") as f:
            json.dump(list(X_t.columns), f)
            
        # Compute SHAP
        explainer = shap.TreeExplainer(final_model)
        shap_vals = explainer.shap_values(X_t)
        np.save(cfg.models_dir / "shap_values.npy", shap_vals)
        np.save(cfg.models_dir / "shap_base_values.npy", np.array([explainer.expected_value]))
        
        write_phase_metrics(5, {"status": "ok", "runs": [res_topo, res_expr]}, elapsed_sec=time.time()-t0)
        
    except Exception:
        log.exception("classify failed")
        write_phase_metrics(5, {"status": "error"}, elapsed_sec=time.time()-t0)
        raise

if __name__ == "__main__":
    main()
