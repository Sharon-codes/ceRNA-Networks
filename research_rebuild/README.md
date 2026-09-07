# miRNA network rebuild and evidence audit

This directory contains the corrected analysis and two subsequent verification stages. It evaluates conditional network variability, hub/module recovery, held-out edge disappearance and external reproducibility. These analyses do not establish a validated cancer detector, a ceRNA mechanism, or a universal advantage of soft weighting.

| Directory | Purpose |
| --- | --- |
| `src/` | Original corrected rebuild, input downloads and historical numerical audit |
| `followup_src/` | Independent reconciliation, fixed-support interval audit, repeated held-out comparisons and graph-failure accounting |
| `biosystems_src/` | Affine CV control, hub stability, known-structure simulation, fixed modules, additional external cohort, honest predictive baselines and integrated reporting |
| `configs/` | Original primary configuration and archived configuration snapshots for the final evidence run |

Start with [the original reproduction instructions](reproducibility_README.md). The later stages depend on the actual saved outputs of their predecessors; they are not standalone scripts that can reconstruct missing empirical caches from summary statistics.

## Environment and inputs

The recorded execution used Python 3.14.3 on Windows. Install `requirements.lock.txt`; the final evidence stage additionally requires scikit-learn and markdown-it-py, pinned in `requirements-evidence.lock.txt`. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `PYTHONUTF8=1` before numerical execution.

Processed GEO matrices, individual sample metadata, manuscript/reviewer documents, generated reports, simulation checkpoints and large correlation caches are not committed with this code update. Public input download logic begins in `src/download_inputs.py`. The final evidence stage also requires the selected external matrices/platform annotations and the source/research files listed by its scripts. Missing files must be obtained or restored; do not fabricate substitute data. Literature retrieval and cohort screening were partly conducted outside these numerical scripts, so a source-only clone does not recreate their archived research inputs automatically.

The run paths are defined in `followup_src/common.py` and `biosystems_src/base.py`. Report rendering uses local Node/Playwright/Edge paths in `biosystems_src/render_report.cjs`; adjust these installation paths on another machine. The freeze script also records the supplied manuscript's original local path.

## Executed stages

The preserved run IDs are `20260907_rebuild_v1`, `20260907_empirical_followup_v1` and `20260907_biosystems_evidence_v1`, under `outputs/` locally. Run the original rebuild first, then the follow-up. The final evidence stage consumes both verified output inventories, node/correlation caches and held-out model checkpoints.

For the final evidence stage, the executed numerical sequence was:

```text
empirical.py inventory
empirical.py modules_external
empirical.py controls_hubs
empirical.py graph_consequences
prediction.py
additional_cohort.py
external48267.py
simulation.py
synthesis.py
plots.py
report.py
checks.py
```

These filenames are under `biosystems_src/`; run Python from the repository root. `additional_cohort.py` preserves GSE29622's failed mapping gate; `external48267.py` evaluates the next metadata-selected eligible cohort. The simulation count was frozen at 30 independent datasets per condition after runtime pilots, yielding 300 datasets across 10 conditions. Resume uses saved checkpoints rather than replacing scientific failures.

`render_report.cjs` renders the self-contained HTML to PDF; `visual_capture.py` supports layout inspection. `finalize.py` requires actual test, rendering and visual-inspection receipts before verifying hashes and producing the evidence archive. Do not invent a visual-inspection receipt. Freeze scripts record original design/provenance: do not rerun them over a completed evidence directory. Choose a new run directory for a new independent run.

## Verification and interpretation

The completed local evidence run passed eight targeted checks, loaded nine embedded scientific figures, and preserved all 862 earlier artifact hashes and 26 earlier source-file hashes. Those are recorded-run results; tests requiring its data/caches will not pass on a source-only clone until the dependencies are supplied.

Lower CV did not consistently imply more stable hub selection or better recovery. EBC did not consistently improve held-out predictions; external comparisons did not establish soft-network superiority. Failed estimators, limited mapping, absent functional truth and unexecuted population-EBC inference remain explicit. The report's contribution assessment is methodological; distinct biological insight remains unestablished.
