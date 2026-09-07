"""Read-only source inspection and official-file verification; no historical imports."""
import csv, gzip, hashlib, io, json, os, pathlib, platform, subprocess, sys, time, urllib.request
from collections import Counter
import importlib.metadata
import pandas as pd
import numpy as np
import pymupdf

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / 'research_rebuild'
RUN = BASE / 'outputs' / '20260907_rebuild_v1'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()
def write_json(p, obj):
    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.ndarray):return clean(v.tolist())
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    tmp.write_text(json.dumps(clean(obj), indent=2, default=str,allow_nan=False),encoding='utf-8'); os.replace(tmp,p)
def read_geo(p):
    meta=[]
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'): break
            row=next(csv.reader([line],delimiter='\t'))
            if row: meta.append(row)
        body=[]
        for line in f:
            if line.startswith('!series_matrix_table_end'): break
            body.append(line)
    df=pd.read_csv(io.StringIO(''.join(body)),sep='\t',index_col=0)
    ids=next(r[1:] for r in meta if r[0]=='!Sample_geo_accession')
    if list(df.columns)!=ids or len(set(ids))!=len(ids): raise ValueError('Sample alignment or duplicate IDs')
    samples=pd.DataFrame(index=ids); samples.index.name='sample_id'
    for row in meta:
        if row[0]=='!Sample_characteristics_ch1':
            parsed=[v.split(': ',1) for v in row[1:]]
            keys={v[0].lower() for v in parsed}
            if len(keys)!=1 or any(len(v)!=2 for v in parsed): raise ValueError('Ambiguous metadata field')
            key=next(iter(keys))
            if key in samples: raise ValueError('Duplicate metadata field')
            samples[key]=[v[1] for v in parsed]
        elif row[0] in ['!Sample_title','!Sample_source_name_ch1','!Sample_platform_id']:
            samples[row[0][8:]]=row[1:]
    return df,samples,meta
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8',errors='replace').strip()
def main():
    for d in ['configs','src','tests','manifests','data']: (BASE/d).mkdir(parents=True,exist_ok=True)
    for d in ['tables','figures','models','logs','reports','historical','cache']: (RUN/d).mkdir(parents=True,exist_ok=True)
    env={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'root':str(ROOT),'branch':git('branch','--show-current'),'commit':git('rev-parse','HEAD'),'status':git('status','--short'),'diff_sha256':hashlib.sha256(git('diff','HEAD').encode()).hexdigest(),'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','networkx','statsmodels','matplotlib','pymupdf','patsy']}}
    write_json(BASE/'manifests/environment_initial.json',env)
    inventory=[]
    for p in ROOT.glob('*.pdf'):
        doc=pymupdf.open(p); title=doc[0].get_text()[:400]
        inventory.append({'file':str(p),'sha256':sha(p),'pages':len(doc),'first_page':title})
        if len(doc)>1:
            out=RUN/'historical'/p.stem;out.mkdir(exist_ok=True)
            (out/'complete_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+page.get_text() for i,page in enumerate(doc)),encoding='utf-8')
            for i,page in enumerate(doc):
                if page.get_images(): page.get_pixmap(matrix=pymupdf.Matrix(1,1)).save(str(out/f'page_{i+1}.png'))
    write_json(BASE/'manifests/pdf_inventory.json',inventory)
    records=[]
    for acc in ['GSE115513','GSE73002']:
        p=ROOT/f'{acc}_series_matrix.txt.gz'
        df,samples,meta=read_geo(p)
        samples.to_csv(BASE/'manifests'/f'{acc}_all_samples.csv')
        write_json(BASE/'manifests'/f'{acc}_metadata.json',{'fields':{c:dict(Counter(samples[c])) for c in samples},'processing':[r[1] for r in meta if r[0]=='!Sample_data_processing'],'series':[r for r in meta if r[0].startswith('!Series_') and r[0]!='!Series_sample_id']})
        url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/matrix/{p.name}'
        rec={'accession':acc,'local_path':str(p),'local_sha256':sha(p),'url':url,'retrieval_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'shape':df.shape}
        try:
            dest=BASE/'data'/p.name
            if not dest.exists():
                with urllib.request.urlopen(url,timeout=60) as response,open(str(dest)+'.tmp','wb') as output:
                    rec['http_headers']=dict(response.headers)
                    while chunk:=response.read(2**20):output.write(chunk)
                os.replace(str(dest)+'.tmp',dest)
            rec['official_sha256']=sha(dest);rec['identical']=rec['local_sha256']==rec['official_sha256']
        except Exception as e:rec['verification_error']=repr(e)
        records.append(rec)
        write_json(BASE/'manifests/input_provenance.json',records)
        print(acc,df.shape,'fields',samples.columns.tolist(),'verified',rec.get('identical'),flush=True)
    scans=[]
    terms=['ratio_th =','0.218','0.035','0.257','0.093','0.704','0.094','nan_to_num','dropna(axis=0)','> 50','N_BOOT','N_SWEEP_BOOT','cov_type','Zero Partial','Zero Effect','Zero effect','Zero Partial','Zero','Zero']
    for p in list(ROOT.glob('generate_*.py'))+[ROOT/'run_statistical_redesign_pipeline.py']:
        for n,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
            if any(t in line for t in terms):scans.append({'file':p.name,'line':n,'source':line.strip()})
    pd.DataFrame(scans).to_csv(RUN/'tables/source_audit_locations.csv',index=False)
if __name__=='__main__':main()
