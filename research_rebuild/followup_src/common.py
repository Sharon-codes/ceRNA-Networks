"""Read-only access to the completed rebuild; separate follow-up outputs."""
import os, sys, json, hashlib, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parent
sys.path.insert(0,str(BASE/'src'))
import numpy as np
import pandas as pd
from provenance import sha,write_json,read_geo,git
from core import rng_for,select_features,spearman,graph_from,node_sums
OLD=BASE/'outputs/20260907_rebuild_v1'
RUN=BASE/'outputs/20260907_empirical_followup_v1'
for sub in ['tables','reports','figures','models','logs','manifests','cache']:(RUN/sub).mkdir(parents=True,exist_ok=True)
SEED=20260907
ACCS=['GSE115513','GSE73002']
def csv(name,rows):
    p=RUN/'tables'/f'{name}.csv';pd.DataFrame(rows).to_csv(p,index=False);return p
def clean_prose(text):
    import re
    text=re.sub(r'\b(all|All|original|Original|of|on|from|in|with|and|per|tissue|serum|discovery|Tissue|Serum|Discovery|than|only|after|retains|adds|uses|selects|each|the|for|power|beta|df)(?=\d)',r'\1 ',text)
    return re.sub(r'(?<=\d)(?=(?:arrays|nodes|probes|draws|streams|splits|attempts|identities|selected|reported)\b)',' ',text)
def report(name,text): (RUN/'reports'/f'{name}.md').write_text(clean_prose(text),encoding='utf-8')
def cohort(acc):
    df,meta,_=read_geo(ROOT/f'{acc}_series_matrix.txt.gz')
    mask=meta.tissue.eq('Carcinoma') if acc==ACCS[0] else meta.diagnosis.eq('breast cancer')&meta.tissue.eq('serum')
    return df.loc[:,mask],meta.loc[mask]
def cache_folder(label):
    folders=list((OLD/'cache').glob(label+'_*'))
    assert len(folders)==1,(label,folders)
    return folders[0]
def cache_blocks(label,x=None):
    folder=cache_folder(label);identity=json.loads((folder/'identity.json').read_text())
    if x is not None: assert hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()==identity['x']
    for p in sorted(folder.glob('*.npz')):
        assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'],p
        with np.load(p) as z:yield z['replicate'],z['r']
def cv(a):
    m=a.mean(axis=0);s=a.std(axis=0,ddof=1)
    return np.divide(s,m,out=np.full(m.shape,np.nan),where=m>1e-12)
def paircorr_partial(x):
    from scipy.stats import rankdata
    ranks=rankdata(x,axis=0);ranks-=ranks.mean(axis=0)
    norms=np.sqrt((ranks*ranks).sum(axis=0));valid=np.isfinite(x).all(axis=0)&(norms>0)
    with np.errstate(divide='ignore',invalid='ignore'):r=(ranks.T@ranks)/np.outer(norms,norms)
    r=np.clip(r,-1,1);r[~valid,:]=np.nan;r[:,~valid]=np.nan
    return r,valid
def status(stage,start,**kwargs):
    write_json(RUN/'logs'/f'status_{stage}.json',{'status':'completed','runtime_seconds':time.perf_counter()-start,**kwargs})
