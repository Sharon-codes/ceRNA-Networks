"""
ComBat Batch Correction for Multi-GEO Meta-Analysis
=====================================================
The LODO collapse (GSE115513 AUC=0.46) shows batch effects between datasets.
This script applies ComBat correction to harmonise expression features across
the 4 GEO datasets before feature extraction.

After correction, re-run the nested CV and LODO — the LODO AUC should improve.

Requires: pip install neuroCombat
  (neuroCombat is the Python port of the original R ComBat)
"""

import json, warnings, logging
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE     = Path("c:/Users/Samsunh/Desktop/Amity University/Research/ceRNA-cancer-detection")
FEAT_CSV = BASE / "features" / "feature_matrix.csv"
META_CSV = BASE / "data" / "processed" / "metadata.csv"
OUT_CSV  = BASE / "features" / "feature_matrix_combat.csv"

try:
    from neuroCombat import neuroCombat
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False
    logger.warning("neuroCombat not installed. Run: pip install neuroCombat --user")

df   = pd.read_csv(FEAT_CSV, index_col=0)
meta = pd.read_csv(META_CSV, index_col=0)

# Attach dataset info
if "dataset" not in df.columns:
    df["dataset"] = meta.reindex(df.index)["dataset"]

EXPR_KEYWORDS = ["expression", "count", "tpm", "rpkm", "mean_expr",
                  "max_expr", "std_expr", "abundance"]
expr_cols = [c for c in df.columns
             if any(k in c.lower() for k in EXPR_KEYWORDS)
             and c != "dataset"]
meta_cols = ["cancer_type", "stage", "age", "sex", "dataset"]
feat_cols = [c for c in df.columns if c not in meta_cols]

logger.info(f"Expression columns to correct: {len(expr_cols)}")
logger.info(f"Dataset distribution:\n{df['dataset'].value_counts()}")

if not COMBAT_AVAILABLE:
    logger.error("Cannot run batch correction — neuroCombat not installed.")
    logger.error("Run: pip install neuroCombat --user")
    logger.error("Then re-run this script.")
    exit(1)

# ComBat requires (features × samples) format
X_expr = df[expr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Batch vector — integer-encoded dataset IDs
dataset_map = {d: i for i, d in enumerate(df["dataset"].unique())}
batch       = df["dataset"].map(dataset_map).values.astype(int)

# Biological covariates to preserve (cancer vs healthy)
y_bin = (df["cancer_type"].fillna("unknown").str.lower() != "healthy").astype(int)
covars = pd.DataFrame({"cancer": y_bin.values, "batch": batch})

logger.info("Running ComBat batch correction...")
corrected = neuroCombat(
    dat=X_expr.T.values,     # (features × samples)
    covars=covars,
    batch_col="batch",
    categorical_cols=["cancer"],
)["data"]  # returns (features × samples)

X_corrected = pd.DataFrame(
    corrected.T,
    index=df.index,
    columns=expr_cols
)

# Rebuild full feature matrix with corrected expression
df_corrected = df.copy()
df_corrected[expr_cols] = X_corrected.values

df_corrected.to_csv(OUT_CSV)
logger.info(f"Batch-corrected feature matrix saved: {OUT_CSV}")
logger.info("Next: re-run classify_nested_cv.py and lodo_validation.py")
logger.info("      pointing FEAT_CSV to feature_matrix_combat.csv")
logger.info("Expected improvement: LODO GSE115513 AUC should rise from 0.46 → 0.65+")
