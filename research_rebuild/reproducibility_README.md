# Reproduce the research rebuild

Repository root: `C:\Users\Samsunh\Desktop\Amity University\Research\Sunno Madarchodo`.
Run ID: `20260907_rebuild_v1`. Original tracked baseline: `4a70722cf5eb5e1a8b0bfc03ff97ca03ac8f63c1`.
Initial analysis-plan/code commit: `613abc7`; subsequent code commits and dirty hashes are recorded in manifests and logs. No remote writes were made.

The canonical implementation is `research_rebuild/src/`. Outputs are under `research_rebuild/outputs/20260907_rebuild_v1/`. Reports, tables, SVGs/300-DPI PNGs, model designs/coefficients, hashes and replicate caches are separate. Do not import historical root scripts: many run immediately and overwrite prior figures.

## Environment

Python 3.14.3 on Windows; exact package versions and hardware are in `manifests/environment_initial.json` and `requirements.lock.txt`. Install the pinned requirements into an isolated Python environment. PowerShell commands below run from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONUTF8='1'
python -m pip install -r research_rebuild/requirements.lock.txt
```

For a fresh clone without the original matrix files, first download them without modifying any existing copies:

```powershell
python research_rebuild/src/download_inputs.py
```

## Full execution

```powershell
python research_rebuild/src/provenance.py
python research_rebuild/src/manuscript_audit.py
python -m unittest discover -s research_rebuild/tests -v
python research_rebuild/src/pipeline.py qc
python research_rebuild/src/pipeline.py primary
python research_rebuild/src/historical.py
python research_rebuild/src/pipeline.py sensitivity
python research_rebuild/src/pipeline.py split
python research_rebuild/src/pipeline.py full
python research_rebuild/src/nested_pilot.py
python research_rebuild/src/simulations.py
python research_rebuild/src/external.py
python research_rebuild/src/biology.py
python research_rebuild/src/reference_audit.py
python research_rebuild/src/report.py
python research_rebuild/src/figures.py
python -m unittest discover -s research_rebuild/tests -v
python research_rebuild/src/finalize.py
```

The supplied PDF is external to Git. `manuscript_audit.py --manuscript PATH` overrides its original supplied location. The analysis can run without the PDF, but manuscript provenance cannot. Full clean reproduction takes materially less than the six-hour limit on the recorded machine; actual runtimes are in `logs/` and `reports/execution_inventory.md`. The scripts do not silently consider all requested analyses successful: for example, tissue 1000-feature settings fail when no full-matrix replicate is valid.

The prospective config and plan were frozen before corrected outcomes. Do not rewrite their historical freeze receipt when reproducing. A changed analysis requires a new run ID/config and amendment. Paths are currently computed from the source-tree root; the run ID in `provenance.py` must be changed alongside the config for an independent run directory. This explicit manual coordination is a remaining usability limitation, not automatic run discovery.

## Resume / figure-only commands

```powershell
python research_rebuild/src/pipeline.py primary --cohort GSE115513
python research_rebuild/src/pipeline.py sensitivity --cohort GSE115513
python research_rebuild/src/pipeline.py split --cohort GSE73002
python research_rebuild/src/pipeline.py stability
python research_rebuild/src/simulations.py
python research_rebuild/src/report.py
python research_rebuild/src/figures.py
```

Conditional caches are indexed by full expression hash, shape, seed label, config hash, core-source hash, B and block size. Every compressed block is SHA-256 validated and atomically committed. Corrupt blocks are recomputed; changed input/config/core creates a distinct cache. Per-replicate streams are independent SeedSequence streams, not a mutable global seed. All invalid replicate IDs/reasons are saved. Changing only plotting/report code need not rerun numerical bootstraps. Simulation checkpoint rows resume by scenario/dataset ID; do not reuse that checkpoint across source/config changes without a new run directory. Nested pilot and full-pipeline stages currently rerun their stage deterministically rather than resume individual outer replicates.

Binary bitwise equality is expected on this pinned environment, but BLAS/platform changes may move least significant bits. Verify Spearman agreement at 1e-12, coefficients at 1e-6 relative/absolute, and exact selected IDs/counts/seed mapping. Cutoff decisions sufficiently close to floating-point ties need explicit review instead of relaxing tolerances to obtain desired graph counts. Input, code, output and figure-source hashes are recorded.

## Evidence boundaries

`historical/` = historical specification executed separately. Primary tables = EMPIRICAL conditional results. Model prediction/contrast files = MODEL-PREDICTED. `simulation_*.csv` = SIMULATED known latent-factor experiments. Unit-test fixtures are synthetic numerical tests, never empirical results. Plotting reads saved result tables only.

Main intervals are Monte Carlo uncertainty conditional on observed data; they are not finite-patient population confidence intervals. The nested 20x100 pilot and correlation simulations do not establish a calibrated EBC null. Do not relabel the package as a complete confirmatory study. The final readiness report lists this and other unresolved issues.

All originals, intermediate failures and prior inconsistent stability summaries are preserved under their original paths or `logs/preserved_pre_fix/`. Numerical arrays are stored locally and ignored by Git to avoid adding multi-GB data to version control. No push, publish, submission or external contact is part of these commands.
