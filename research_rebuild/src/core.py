"""Canonical numerical primitives. No execution at import."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, median_abs_deviation
from threadpoolctl import threadpool_limits
from provenance import BASE, ROOT, RUN, read_geo, sha, write_json

def rng_for(seed, label, replicate):
    token=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')
    return np.random.default_rng(np.random.SeedSequence([seed,token,replicate]))

def spearman(x):
    if not np.isfinite(x).all(): raise ValueError('Nonfinite expression')
    ranks=rankdata(x,axis=0,method='average')
    ranks-=ranks.mean(axis=0)
    norm=np.sqrt(np.sum(ranks*ranks,axis=0))
    if np.any(norm==0): raise ValueError('Undefined constant-probe correlation')
    ranks/=norm
    with threadpool_limits(limits=1): r=ranks.T@ranks
    r=np.clip((r+r.T)/2,-1,1); np.fill_diagonal(r,1)
    return r

def select_features(df,k,scale='supplied',missing='complete'):
    x=df.copy()
    finite=np.isfinite(x.to_numpy())
    frac=1-finite.mean(axis=1)
    allowed=frac==0 if missing=='complete' else frac<=.2
    x=x.loc[allowed].copy()
    if missing=='median20':
        x=x.where(np.isfinite(x)).T.fillna(x.median(axis=1)).T
    if scale=='log1p':
        if np.any(x.to_numpy()<0):raise ValueError('log1p selection forbidden for negative supplied values')
        x=np.log2(x+1)
    variable=x.max(axis=1)>x.min(axis=1)
    x=x.loc[variable]
    mad=median_abs_deviation(x.to_numpy(),axis=1)
    order=np.argsort(-mad,kind='stable')[:min(k,len(x))]
    return x.iloc[order],pd.Series(mad,index=x.index),frac

def cohort(acc):
    df,meta,raw=read_geo(ROOT/f'{acc}_series_matrix.txt.gz')
    mask=meta.tissue.eq('Carcinoma') if acc=='GSE115513' else meta.diagnosis.eq('breast cancer') & meta.tissue.eq('serum')
    meta['included']=mask;meta['reason']=np.where(mask,'exact eligible diagnosis/tissue','other tissue or diagnosis')
    meta.to_csv(BASE/'manifests'/f'{acc}_sample_selection.csv')
    if 'individual' in meta and meta.loc[mask,'individual'].duplicated().any():
        raise ValueError('Repeated patients require block sampling; no independent-row fallback')
    return df.loc[:,mask],meta.loc[mask].copy(),raw

def transformed(r,sign): return np.abs(r) if sign=='unsigned' else np.maximum(r,0)
def graph_from(r,density=.025,sign='unsigned',theta=None,scope='full'):
    n=len(r);u,v=np.triu_indices(n,1)
    s=transformed(r,sign)
    theta=float(np.quantile(s[u,v],1-density,method='linear')) if theta is None else float(theta)
    a=s>=theta;np.fill_diagonal(a,False)
    if sign=='positive': a &= r>=theta
    g=nx.from_numpy_array(a)
    components=sorted(nx.connected_components(g),key=lambda c:(-len(c),min(c)))
    gcc=sorted(components[0]); gg=g.subgraph(gcc).copy()
    focal=g if scope=='full' else gg
    ebc=nx.edge_betweenness_centrality(focal,normalized=True,weight=None)
    edges=np.array(sorted(tuple(sorted(e)) for e in focal.edges()),dtype=int).reshape(-1,2)
    lookup={(int(i),int(j)):z for z,(i,j) in enumerate(zip(u,v))}
    idx=np.array([lookup[tuple(e)] for e in edges],dtype=int)
    bridge=set(tuple(sorted(e)) for e in nx.bridges(focal))
    values=np.array([ebc[tuple(e)] for e in edges])
    summary={'nodes':n,'eligible_pairs':len(u),'full_edges':g.number_of_edges(),'full_density':nx.density(g),'components':len(components),'isolates':len(list(nx.isolates(g))),'gcc_nodes':len(gcc),'gcc_edges':gg.number_of_edges(),'gcc_density':nx.density(gg),'theta':theta,'cutoff_ties':int(np.sum(s[u,v]==theta)),'sign':sign,'scope':scope,'retained_edges':len(edges),'bridges':len(bridge)}
    return {'summary':summary,'edges':edges,'pair_indices':idx,'ebc':values,'bridge':[tuple(e) in bridge for e in edges],'gcc':gcc,'adjacency':a,'margin':s[u,v][idx]-theta,'r':r}

def bootstrap(x,B,label,seed,config_hash,chunk=50):
    """Independent per-replicate streams; float64 pair correlations; chunk hashes."""
    p=x.shape[1];u,v=np.triu_indices(p,1)
    identity={'x':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),'shape':x.shape,'seed':seed,'label':label,'config':config_hash,'core':sha(Path(__file__)),'chunk':chunk,'B':B}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=RUN/'cache'/f'{label}_{key[:12]}';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'identity.json',identity)
    blocks=[];valid=[];failures=[];start=time.perf_counter()
    for start_b in range(0,B,chunk):
        stop=min(B,start_b+chunk);pfile=folder/f'{start_b:05d}.npz';receipt=pfile.with_suffix('.json')
        if pfile.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==sha(pfile):
            saved=np.load(pfile);vals=saved['r'];ids=saved['replicate'];failures.extend(json.loads(receipt.read_text())['failures'])
        else:
            vals=[];ids=[];bad=[]
            for b in range(start_b,stop):
                rng=rng_for(seed,label,b);rows=rng.integers(0,len(x),len(x))
                try: vals.append(spearman(x[rows])[u,v]);ids.append(b)
                except ValueError as e:bad.append({'replicate':b,'error':str(e)})
            vals=np.asarray(vals).reshape(-1,len(u));ids=np.array(ids)
            with open(str(pfile)+'.tmp','wb') as f:np.savez_compressed(f,r=vals,replicate=ids)
            os.replace(str(pfile)+'.tmp',pfile)
            write_json(receipt,{'sha256':sha(pfile),'failures':bad});failures.extend(bad)
        blocks.append(vals);valid.extend(ids.tolist())
        print(f'{label}: {stop}/{B} attempts; {len(valid)} valid; {time.perf_counter()-start:.1f}s',flush=True)
    result=np.concatenate(blocks)
    write_json(folder/'completion.json',{'requested':B,'valid':len(valid),'failures':failures,'runtime_seconds':time.perf_counter()-start,'replicate_ids':valid})
    return result,folder

def cv(a,ddof=1):
    mean=np.mean(a,axis=0);std=np.std(a,axis=0,ddof=ddof)
    return np.divide(std,mean,out=np.full_like(mean,np.nan,dtype=float),where=mean>1e-12)

def node_sums(pair_values,n):
    u,v=np.triu_indices(n,1)
    incidence=csr_matrix((np.ones(2*len(u)),(np.r_[u,v],np.tile(np.arange(len(u)),2))),shape=(n,len(u)))
    return np.asarray(incidence@pair_values.T).T
