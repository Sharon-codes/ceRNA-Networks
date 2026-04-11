"""
Central configuration for the ceRNA pan-cancer detection pipeline.

All paths, thresholds, and model hyperparameters are defined here.
Other modules must import ``cfg`` and must not hardcode paths or thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Config:
    """Pipeline configuration: filesystem layout, GEO IDs, URLs, and ML settings."""

    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    db_subdir: Path = raw_dir / "databases"
    network_dir: Path = project_root / "network"
    features_dir: Path = project_root / "features"
    models_dir: Path = project_root / "models"
    figures_dir: Path = project_root / "figures"
    logs_dir: Path = project_root / "logs"

    geo_datasets: Dict[str, str] = field(
        default_factory=lambda: {
            "circRNA_CRC": "GSE126094",
            "circRNA_lung": "GSE101684",
            "circRNA_atlas": "GSE126094",
            "miRNA_CRC": "GSE115513",
            "miRNA_multicancer": "GSE73002",
        }
    )

    mirtarbase_urls: Tuple[str, ...] = (
        "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/cache/download/miRTarBase_MTI.xlsx",
        "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2022/cache/download/miRTarBase_MTI.xlsx",
    )
    mirbase_mature_fa_urls: Tuple[str, ...] = (
        "https://www.mirbase.org/download/CURRENT/mature.fa.gz",
        "https://www.mirbase.org/download/CURRENT/mature.fa",
    )
    circinteractome_api_url: str = (
        "https://circinteractome.nia.nih.gov/api/v2/mirnasponge?format=tsv"
    )

    min_cpm: float = 1.0
    circrna_min_sample_frac: float = 0.30
    mirna_min_sample_frac: float = 0.20

    n_cv_folds: int = 5
    optuna_trials: int = 10
    optuna_inner_splits: int = 3
    bootstrap_roc_samples: int = 500
    xgb_seed: int = 42

    xgb_base: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "eval_metric": "auc",
            "random_state": 42,
        }
    )

    rf_n_estimators: int = 500
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 2

    log_level: str = "INFO"


cfg = Config()

for _d in (
    cfg.raw_dir,
    cfg.db_subdir,
    cfg.processed_dir,
    cfg.network_dir,
    cfg.features_dir,
    cfg.models_dir,
    cfg.figures_dir,
    cfg.logs_dir,
):
    _d.mkdir(parents=True, exist_ok=True)


def write_phase_metrics(
    phase_num: int,
    metrics: Dict[str, Any],
    elapsed_sec: Optional[float] = None,
) -> Path:
    """Persist per-phase metrics JSON for the pipeline dashboard.

    Args:
        phase_num: Phase index (1–6).
        metrics: Serializable dict of phase-specific metrics.
        elapsed_sec: Optional wall-clock seconds for this phase.

    Returns:
        Path to the written JSON file.

    Raises:
        OSError: If the file cannot be written.
    """
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "phase": phase_num,
        "metrics": metrics,
    }
    if elapsed_sec is not None:
        payload["elapsed_sec"] = round(float(elapsed_sec), 3)
    out = cfg.logs_dir / f"phase_{phase_num}_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def load_all_phase_metrics() -> List[Dict[str, Any]]:
    """Load every ``phase_*_metrics.json`` present under ``logs/``.

    Returns:
        Sorted list of metric payloads (by phase number).
    """
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cfg.logs_dir.glob("phase_*_metrics.json"))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda x: x.get("phase", 0))
    return rows


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("config")
    log.info("project_root=%s", cfg.project_root)
    log.info("paths: processed=%s logs=%s", cfg.processed_dir, cfg.logs_dir)
