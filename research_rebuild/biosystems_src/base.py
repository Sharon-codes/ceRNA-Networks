import os,sys,json,time,hashlib
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
sys.path.insert(0,str(BASE/'src'))
import numpy as np,pandas as pd
from provenance import sha,write_json,read_geo,git
from core import rng_for,select_features,spearman,node_sums
from scipy.stats import rankdata,spearmanr
OLD=BASE/'outputs/20260907_rebuild_v1';PREV=BASE/'outputs/20260907_empirical_followup_v1';RUN=BASE/'outputs/20260907_biosystems_evidence_v1'
for s in ['tables','models','logs','figures','research','data','manifests','report_parts']:(RUN/s).mkdir(parents=True,exist_ok=True)
ACCS=['GSE115513','GSE73002'];SEED=2026090717
def csv(name,rows):p=RUN/'tables'/f'{name}.csv';pd.DataFrame(rows).to_csv(p,index=False);return p
def txt(name,s):p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8');return p
def cv(z):
    m=z.mean(axis=0);return np.divide(z.std(axis=0,ddof=1),m,out=np.full(m.shape,np.nan),where=m>1e-12)
def top(z,k):return np.argsort(-np.asarray(z),kind='stable')[:k]
def rho(x,y):return float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else np.nan
def cohort(acc):
    df,meta,_=read_geo(ROOT/f'{acc}_series_matrix.txt.gz');mask=meta.tissue.eq('Carcinoma') if acc==ACCS[0] else meta.diagnosis.eq('breast cancer')&meta.tissue.eq('serum');return df.loc[:,mask],meta.loc[mask]
def blocks(acc):
    folders=list((OLD/'cache').glob(acc+'_primary_*'));assert len(folders)==1
    for p in sorted(folders[0].glob('*.npz')):
        assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];z=np.load(p);yield z['replicate'],z['r']
def status(name,t,state='completed',**k):write_json(RUN/'logs'/f'status_{name}.json',{'status':state,'seconds':time.perf_counter()-t,**k})
