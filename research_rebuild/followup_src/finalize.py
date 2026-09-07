from common import *
import re

def main():
    start=time.perf_counter()
    tests=(RUN/'logs/test_results.txt').read_text(encoding='utf-8-sig');assert '\nOK' in tests and 'FAILED' not in tests
    ledger=pd.read_csv(RUN/'tables/verified_results_ledger.csv')
    new=ledger.underlying_file.map(lambda p:Path(p).is_relative_to(RUN))
    ledger.loc[new,'source_run']=RUN.name;ledger.loc[new,'config_sha256']=sha(RUN/'followup_analysis_amendment.md')
    csv('verified_results_ledger',ledger)
    # Receipt migration only after independent numerical/paired-score tests pass.
    for acc in ACCS:
        for s in range(20):
            p=RUN/'models'/f'{acc}_split{s:02d}.json';q=p.with_suffix('.npz');assert p.exists() and q.exists()
            write_json(p.with_suffix('.receipt.json'),{'json_sha256':sha(p),'npz_sha256':sha(q),'split_manifest_sha256':sha(RUN/'manifests/splits.json'),'amendment_sha256':sha(RUN/'followup_analysis_amendment.md')})
    stages=[]
    for p in (RUN/'logs').glob('status_*.json'):
        s=json.loads(p.read_text());stages.append({'stage':p.stem.removeprefix('status_'),'status':s['status'],'runtime_seconds':s['runtime_seconds'],'receipt':str(p),'recorded_split_seconds':s.get('total_recorded_split_seconds',None)})
    csv('stage_runtime_summary',stages)
    checks=[]
    for f in json.loads((RUN/'figures/figure_manifest.json').read_text()):
        for s in f['source_tables']:checks.append({'figure':f['figure'],'source':s['path'],'matches':sha(Path(s['path']))==s['sha256']})
    assert all(c['matches'] for c in checks);csv('final_figure_source_checks',checks)
    # Verify preservation again after all new analyses.
    preserved=[]
    for row in json.loads((OLD/'output_inventory.json').read_text()):
        p=OLD/row['path'];preserved.append({'path':str(p),'matches':sha(p)==row['sha256']})
    assert all(c['matches'] for c in preserved);csv('final_previous_run_preservation',preserved)
    text='''# Reproduction and resume

Follow-up run: `research_rebuild/outputs/20260907_empirical_followup_v1`. This directory is separate from and preserves `20260907_rebuild_v1`. The dated amendment is at the run root; protocols/drafts/audits/readiness are in `reports/`; deliverable CSVs are in `tables/`; PNG/SVG and the source manifest are in `figures/`. This maps the requested deliverable names without duplicating inconsistent files.

Use repository root in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONUTF8='1'
C:\\Python314\\python.exe research_rebuild\\followup_src\\freeze.py
C:\\Python314\\python.exe research_rebuild\\followup_src\\audit.py statuses
C:\\Python314\\python.exe research_rebuild\\followup_src\\audit.py primary
C:\\Python314\\python.exe research_rebuild\\followup_src\\audit.py degeneracy
C:\\Python314\\python.exe research_rebuild\\followup_src\\cv_audit.py
C:\\Python314\\python.exe research_rebuild\\followup_src\\external_verify.py
C:\\Python314\\python.exe research_rebuild\\followup_src\\verify_old_heldout.py
C:\\Python314\\python.exe research_rebuild\\followup_src\\heldout.py
C:\\Python314\\python.exe research_rebuild\\followup_src\\test_followup.py *> research_rebuild\\outputs\\20260907_empirical_followup_v1\\logs\\test_results.txt
C:\\Python314\\python.exe research_rebuild\\followup_src\\paper.py
C:\\Python314\\python.exe research_rebuild\\followup_src\\figures.py
C:\\Python314\\python.exe research_rebuild\\followup_src\\finalize.py
```

Run only the affected command when resuming. No jobs remain unfinished. The40 held-out checkpoints and retained-edge train-correlation caches avoid repeating250-draw fits. Split reuse validates JSON/NPZ checksums, frozen split IDs and amendment SHA; original cache reuse validates expression identity and every chunk hash. Raw data and official annotations remain in the original locations. Do not run the old production pipeline into the completed old run. The degeneracy stage reconstructs exact old attempts because the original complete-graph cache retained no partial correlation matrices. The primary verification's interrupted first implementation used slow Python reductions; its receipt is preserved and the completed equivalent NumPy implementation is recorded. No completed scientific stage was discarded to select favorable results.

The initial freeze records commit/status, environment, package versions, old source hashes, config and input hashes. final_manifest.json records final code and every artifact hash. Runtime tables report completed stage time, with original sensitivity receipts explicitly caveated when they measure cache reads; original logs preserve production stage elapsed times. Whole-run elapsed includes implementation and figure review and must not be mistaken for compute time.

Intervals: coefficient/CV intervals are conditional Monte Carlo only; graph-count percentiles are resampling-distribution summaries; overlapping-split distributions are descriptive. None is a calibrated population EBC interval. New primary held-out Brier gains are without-minus-with EBC (positive favors EBC); old logloss differences use the opposite sign and a different bootstrap-test outcome. Keep separate.
'''
    report('reproduction_resume',text)
    for p in (RUN/'reports').glob('*.md'):p.write_text(clean_prose(p.read_text(encoding='utf-8')),encoding='utf-8')
    src={str(p.relative_to(ROOT)):sha(p) for p in (BASE/'followup_src').glob('*.py')}
    artifact=[{'path':str(p.relative_to(RUN)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(RUN.rglob('*')) if p.is_file() and p.name!='final_manifest.json']
    write_json(RUN/'final_manifest.json',{'run_id':RUN.name,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'commit':git('rev-parse','HEAD'),'git_status':git('status','--short'),'source_sha256':src,'amendment_sha256':sha(RUN/'followup_analysis_amendment.md'),'split_manifest_sha256':sha(RUN/'manifests/splits.json'),'prior_artifacts_preserved':len(preserved),'tests':'8 targeted tests passed; logs/test_results.txt','figures_visually_inspected':6,'stages':stages,'artifact_inventory':artifact,'unfinished_jobs':0,'note':'All new source files remain in working tree; no push/publish/submit performed.'})
    print('Finalized',len(artifact),'artifacts;',len(preserved),'prior artifacts preserved; source files',len(src),flush=True)
if __name__=='__main__':main()
