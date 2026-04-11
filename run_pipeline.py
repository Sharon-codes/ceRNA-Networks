#!/usr/bin/env python3
"""
Master orchestrator for the ceRNA pan-cancer detection pipeline.

Examples:
    python run_pipeline.py
    python run_pipeline.py --start=3
    python run_pipeline.py --phase=4
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from config.config import cfg, load_all_phase_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(cfg.logs_dir / "run_pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("run_pipeline")

TOTAL_PHASES = 6

PHASES: List[Tuple[int, str, str]] = [
    (1, "data/download_dbs.py", "Download external databases"),
    (2, "data/load_geo.py", "Download GEO + preprocess"),
    (3, "network/build_cerna.py", "Build ceRNA graphs"),
    (4, "features/extract_topology.py", "Extract topology features"),
    (5, "models/classify.py", "Train classifiers + SHAP"),
    (6, "figures/plot_all.py", "Generate figures"),
]


def write_progress_snapshot(
    last_completed: Optional[int],
    status: str,
    error_phase: Optional[int] = None,
) -> None:
    """Write ``logs/pipeline_progress.json`` summarising overall completion.

    Args:
        last_completed: Highest successfully finished phase number, or None.
        status: ``running`` | ``completed`` | ``failed``.
        error_phase: Phase that failed when ``status`` is ``failed``.
    """
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    done = last_completed or 0
    overall_pct = round(100.0 * done / TOTAL_PHASES, 1) if status != "failed" else round(100.0 * (done) / TOTAL_PHASES, 1)
    snap: Dict[str, Any] = {
        "total_phases": TOTAL_PHASES,
        "phases_completed": done,
        "overall_progress_percent": overall_pct,
        "remaining_percent": round(100.0 - overall_pct, 1),
        "phase_metrics": load_all_phase_metrics(),
        "status": status,
        "failed_at_phase": error_phase,
    }
    out = cfg.logs_dir / "pipeline_progress.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    log.info(
        "Pipeline dashboard: %.1f%% complete (%s/%s phases) - snapshot -> %s",
        overall_pct,
        done,
        TOTAL_PHASES,
        out,
    )


def run_phase(phase_num: int, script: str, desc: str) -> bool:
    """Execute one phase script as a subprocess.

    Args:
        phase_num: Phase id (1–6).
        script: Path relative to project root.
        desc: Human-readable label.

    Returns:
        True on exit code 0.
    """
    log.info("%s", "=" * 60)
    log.info("Phase %s: %s", phase_num, desc)
    log.info(
        "Before Phase %s: overall pipeline ~%.1f%% done (%s phases fully finished)",
        phase_num,
        100.0 * (phase_num - 1) / TOTAL_PHASES,
        phase_num - 1,
    )
    t0 = time.time()
    cmd = [sys.executable, str(BASE / script)]
    result = subprocess.run(cmd, cwd=str(BASE))
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error("Phase %s FAILED after %.1f s", phase_num, elapsed)
        return False
    log.info("Phase %s finished in %.1f s", phase_num, elapsed)
    write_progress_snapshot(last_completed=phase_num, status="running")
    log.info(
        "After Phase %s: overall pipeline ~%.1f%% complete",
        phase_num,
        100.0 * phase_num / TOTAL_PHASES,
    )
    return True


def main() -> None:
    """Parse CLI flags and run selected phases."""
    parser = argparse.ArgumentParser(description="ceRNA detection pipeline orchestrator")
    parser.add_argument("--start", type=int, default=1, help="Start from this phase (1–6)")
    parser.add_argument("--phase", type=int, default=None, help="Run only this phase")
    args = parser.parse_args()

    phases_to_run = PHASES
    if args.phase is not None:
        phases_to_run = [p for p in PHASES if p[0] == args.phase]
        if not phases_to_run:
            log.error("Invalid --phase=%s", args.phase)
            sys.exit(1)
    elif args.start > 1:
        phases_to_run = [p for p in PHASES if p[0] >= args.start]

    log.info("Planned phases: %s", [p[0] for p in phases_to_run])
    write_progress_snapshot(last_completed=args.start - 1, status="running")

    last_ok = args.start - 1
    for phase_num, script, desc in phases_to_run:
        ok = run_phase(phase_num, script, desc)
        if not ok:
            log.error(
                "Stopped at phase %s. Fix the error, then re-run with: python run_pipeline.py --start=%s",
                phase_num,
                phase_num,
            )
            write_progress_snapshot(last_completed=last_ok, status="failed", error_phase=phase_num)
            sys.exit(1)
        last_ok = phase_num

    final_status = "completed" if last_ok >= TOTAL_PHASES else "completed_partial"
    write_progress_snapshot(last_completed=last_ok, status=final_status)
    log.info(
        "Finished requested phases through %s / %s (see logs/pipeline_progress.json).",
        last_ok,
        TOTAL_PHASES,
    )


if __name__ == "__main__":
    main()
