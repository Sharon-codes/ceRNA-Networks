import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from pathlib import Path

# --- Configuration ---
BASE = Path(__file__).resolve().parent.parent.parent
FEAT_CSV = BASE / "features" / "feature_matrix_combat.csv"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
print(f"Loading {FEAT_CSV}...")
df = pd.read_csv(FEAT_CSV, index_col=0)

# Define hybrid features (Topology + Expression)
# Exclude metadata and non-feature columns
meta_cols = ["patient_id", "cancer_type", "stage", "age", "sex", "dataset"]
feature_cols = [c for c in df.columns if c not in meta_cols]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

target_cancers = ["Breast", "CRC", "Prostate"]
labels = {"Breast": "Breast Cancer", "CRC": "Colorectal Cancer", "Prostate": "Prostate Cancer"}
colors = {"Breast": "#1f77b4", "CRC": "#ff7f0e", "Prostate": "#2ca02c"}

def get_roc_data(cancer_name, X, df):
    # One-versus-rest as specified in caption
    y = (df["cancer_type"] == cancer_name).astype(int)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Using fixed reasonable parameters for XGBoost
    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "scale_pos_weight": (y == 0).sum() / max(1, (y == 1).sum())
    }
    
    for train, test in skf.split(X, y):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X.iloc[train], y.iloc[train], verbose=False)
        probs = clf.predict_proba(X.iloc[test])[:, 1]
        fpr, tpr, _ = roc_curve(y.iloc[test], probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
        
    return mean_fpr, tprs, aucs

# --- Visualization ---
plt.figure(figsize=(10, 8))

for cancer in target_cancers:
    print(f"Processing {cancer} (n={len(df[df['cancer_type']==cancer])})...")
    mean_fpr, tprs, aucs = get_roc_data(cancer, X, df)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    n_samples = len(df[df['cancer_type']==cancer])
    
    plt.plot(mean_fpr, mean_tpr, color=colors[cancer],
             label=f'{labels[cancer]} (AUROC {mean_auc:.3f} ± {std_auc:.3f}, n = {n_samples:,})',
             lw=2.5, alpha=0.9)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[cancer], alpha=0.15)

plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', label='Chance', alpha=0.6)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Per-Cancer-Type ROC Curves (Protocol 1)', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
plt.grid(alpha=0.2, linestyle='--')

out_path = OUT_DIR / "figure5_roc_curves.png"
plt.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 5 saved to {out_path}")
