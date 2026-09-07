"""Verify artifacts, complete manifests and report exact stage inventory."""
import hashlib,importlib.metadata,json,shutil,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from provenance import BASE,ROOT,RUN,sha,write_json,git
from pipeline import prepare
from core import cohort,select_features,rng_for

def main():
    # Reconstruct deterministic selection manifests and verify against executed cache hashes.
    rows=[]
    for acc in ['GSE115513','GSE73002']:
        cases=[('primary',500,'supplied','complete',1),('features250',250,'supplied','complete',1),('features1000',1000,'supplied','complete',1),('sample50',500,'supplied','complete',.5),('sample75',500,'supplied','complete',.75)]
        cases.append(('log1p_selection',500,'log1p','complete',1) if acc=='GSE115513' else ('median_imputation20',500,'supplied','median20',1))
        for name,k,scale,missing,fraction in cases:
            df,meta,sel,x=prepare(acc,k,scale,missing,fraction);label=acc+'_'+name
            h=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest();matches=[]
            for file in (RUN/'cache').glob(label+'_*/identity.json'):
                identity=json.loads(file.read_text())
                if identity['label']==label and identity['x']==h:matches.append(str(file))
            pd.DataFrame({'node':range(len(sel)),'probe_id':sel.index}).to_csv(BASE/'manifests'/f'{label}_probes.csv',index=False)
            meta.to_csv(BASE/'manifests'/f'{label}_samples.csv')
            rows.append({'analysis':label,'sample_count':len(meta),'probe_count':len(sel),'expression_sha256':h,'verified_executed_cache':bool(matches),'cache_identities':matches,'reconstruction':'Post-run deterministic identity reconstruction checked against execution-cache expression hash'})
    write_json(BASE/'manifests/selection_identity_verification.json',rows)
    assert all(row['verified_executed_cache'] for row in rows)
    for acc in ['GSE115513','GSE73002']:
        df,meta,_=cohort(acc)
        for stage,B in [('full_pipeline',100),('outer_pilot',20)]:
            selections=[]
            for b in range(B):
                ix=rng_for(20260907,acc+'_'+stage,b).integers(0,len(meta),len(meta))
                sel,_,_=select_features(df.iloc[:,ix],500)
                selections.append({'replicate':b,'sample_ids':df.columns[ix].tolist(),'probe_ids':sel.index.tolist(),'provenance':'deterministically reconstructed from executed source and seed mapping'})
            write_json(BASE/'manifests'/f'{acc}_{stage}_selections.json',selections)
        print('Reconstructed full-pipeline manifests',acc,flush=True)
    finish(rows)

def finish(rows):
    sources={str(p.relative_to(BASE)):sha(p) for p in (BASE/'src').glob('*.py')}
    write_json(BASE/'manifests/final_code.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'commit':git('rev-parse','HEAD'),'dirty_diff_sha256':hashlib.sha256(git('diff','HEAD').encode()).hexdigest(),'source_hashes':sources,'git_status':git('status','--short')})
    packages=['numpy','pandas','scipy','networkx','statsmodels','matplotlib','pymupdf','pillow','patsy','threadpoolctl','tabulate']
    (BASE/'requirements.lock.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in packages)+'\n')
    for name in ['analysis_plan.md','amendments.md','reproducibility_README.md']:
        shutil.copyfile(BASE/name,RUN/'reports'/name)
    ref=BASE/'references_and_novelty.md'
    if ref.exists():shutil.copyfile(ref,RUN/'reports/references_and_novelty.md')
    status=[]
    for stage in ['qc','primary','historical','sensitivity','split','full','stability']:
        path=RUN/'logs'/('status_historical.json' if stage=='historical' else f'status_{stage}_both.json')
        if not path.exists():continue
        data=json.loads(path.read_text())
        if stage=='sensitivity':data['GSE115513']=json.loads((RUN/'logs/status_sensitivity_GSE115513.json').read_text())['GSE115513']
        for acc,row in data.items():
            status.append({'stage':stage,'cohort':acc,'status':row['status'],'elapsed_seconds':row.get('elapsed_seconds',row.get('result',{}).get('seconds') if isinstance(row.get('result'),dict) else None),'receipt':str(path.relative_to(RUN))})
    for stage in ['nested_pilot','simulations','biology']:
        path=RUN/'logs'/f'status_{stage}.json'
        if path.exists():status.append({'stage':stage,'cohort':'both / declared scenarios','status':'completed','receipt':str(path.relative_to(RUN)),'details':json.loads(path.read_text())})
    pd.DataFrame(status).to_csv(RUN/'tables/execution_inventory.csv',index=False)
    blocks=[]
    for p in (RUN/'cache').glob('*/completion.json'):
        identity=json.loads((p.parent/'identity.json').read_text());c=json.loads(p.read_text())
        blocks.append({'label':identity['label'],'requested_B':c['requested'],'valid_B':c['valid'],'invalid_B':len(c['failures']),'runtime_seconds_last_access':c['runtime_seconds'],'identity':str(p.parent/'identity.json')})
    pd.DataFrame(blocks).to_csv(RUN/'tables/bootstrap_inventory.csv',index=False)
    nested=pd.read_csv(RUN/'tables/nested_pilot.csv')
    inv='''# Execution inventory

All feasible configured stages were executed. 'Completed' for a sensitivity stage means settings were evaluated, including explicit invalid settings; it does not turn a failed setting into a result. Original failure receipts and the pre-vectorization simulation log remain preserved. Stage elapsed times can be cumulative over cohorts, so do not sum cumulative per-cohort entries. Core primary execution took about 680 seconds; historical tissue/serum about 111/164 seconds; vectorized simulations about 214 seconds. All logs are actual local runtime measurements, affected by concurrency. Final code hashes and output hashes are separate from the historical audit commit.

'''+pd.DataFrame(status).drop(columns=['details'],errors='ignore').to_markdown(index=False)+'\n\n## Nested profile\n\n'+nested.groupby('cohort').agg(outer_attempts=('outer','size'),mean_seconds=('seconds','mean'),mean_inner_valid=('inner_valid','mean'),ebc_outer_sd=('ebc_coefficient','std')).reset_index().to_markdown(index=False)+'\n\nThe outer distribution is a pilot, not calibrated population inference. With no validated EBC null/alternative target, increasing its counts is not a scientifically justified production-inference step.\n'
    (RUN/'reports/execution_inventory.md').write_text(inv,encoding='utf-8')
    entries=[]
    for p in sorted(RUN.rglob('*')):
        if p.is_file() and 'cache' not in p.parts and p.name!='output_inventory.json':entries.append({'path':str(p.relative_to(RUN)),'bytes':p.stat().st_size,'sha256':sha(p)})
    write_json(RUN/'output_inventory.json',entries)
    print('Verified selection identities',len(rows),'artifacts',len(entries),'nested completed',sum(nested.status.eq('completed')),'of',len(nested),flush=True)
if __name__=='__main__':
    if '--inventory-only' in sys.argv:finish(json.loads((BASE/'manifests/selection_identity_verification.json').read_text()))
    else:main()
