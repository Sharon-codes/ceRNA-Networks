"""SIMULATED expression-level PSD latent-factor experiments. No empirical stand-ins."""
import json,time,traceback
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata
from core import spearman,graph_from,rng_for
from provenance import BASE,RUN,write_json

CFG=json.loads((BASE/'configs/primary.json').read_text())
def batch_spearman(x,indices):
    ranks=rankdata(x[indices],axis=1,method='average')
    ranks-=ranks.mean(axis=1,keepdims=True)
    norms=np.sqrt(np.sum(ranks*ranks,axis=1,keepdims=True))
    valid=np.all(norms>0,axis=(1,2));ranks=ranks[valid]/norms[valid]
    return np.clip(ranks.transpose(0,2,1)@ranks,-1,1),int(sum(~valid))
def generator(rng,n,scenario,p=40):
    # Four independent normal module factors, independent errors, optional common noise.
    load=np.linspace(.45,.9,p);load[0]=0 # a declared truly independent node in Gaussian scenarios
    if scenario.startswith('weak'):load*=.55
    modules=np.arange(p)//10;L=np.zeros((p,4));L[np.arange(p),modules]=load
    var=1-load**2;cov=L@L.T+np.diag(var)
    x=rng.normal(size=(n,4))@L.T+rng.normal(size=(n,p))*np.sqrt(var)
    if scenario.startswith('correlated_noise'):
        loading=np.linspace(0,.7,p);loading[0]=0
        x+=rng.normal(size=(n,1))*loading;cov+=np.outer(loading,loading)
    if scenario.startswith('ties'):x=np.round(x)
    if scenario.startswith('missing'):x[rng.random(x.shape)<.01]=np.nan
    if scenario.startswith('outliers'):
        flags=rng.random(n)<.05;x[flags]+=rng.normal(size=(sum(flags),1))*5
    if scenario.startswith('batch'):x+=(rng.random((n,1))>.5)*np.linspace(0,3,p)
    cor=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    target=6/np.pi*np.arcsin(cor/2)
    return x,target

def wilson(k,n):
    if n==0:return [None,None]
    z=1.95996398454;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def main():
    checkpoint=RUN/'tables/simulation_datasets.csv'
    rows=pd.read_csv(checkpoint).to_dict('records') if checkpoint.exists() else []
    done={(r['scenario'],r['replicate']) for r in rows};start=time.perf_counter();status={}
    for scenario in CFG['simulation_scenarios']:
        n=250 if '250' in scenario else 80
        known=not any(scenario.startswith(s) for s in ['ties','outliers','batch'])
        for rep in range(CFG['simulation_datasets_per_scenario']):
            if (scenario,rep) in done:continue
            rng=rng_for(CFG['seed'],'SIMULATED_'+scenario,rep);x,target=generator(rng,n,scenario)
            keep=np.isfinite(x).all(axis=0);ids=np.flatnonzero(keep);x=x[:,keep];target=target[np.ix_(keep,keep)]
            r=spearman(x);u,v=np.triu_indices(len(r),1);theta=float(np.quantile(np.abs(r[u,v]),.9));baseline=np.abs(r[u,v])>=theta
            indices=rng.integers(0,n,(CFG['simulation_bootstraps'],n))
            matrices,failed=batch_spearman(x,indices);draws=matrices[:,u,v]
            lo,hi=np.quantile(draws,[.025,.975],axis=0);sd=draws.std(axis=0,ddof=1)
            truth=target[u,v];covered=(lo<=truth)&(truth<=hi)
            null=(ids[u]==0);alt=(ids[u]==1)&(ids[v]==2)
            # Average per dataset then aggregate datasets, avoiding treating pairs as independent trials.
            row={'content':'SIMULATED','scenario':scenario,'replicate':rep,'n':n,'features':len(r),'valid_B':len(draws),'failed_B':failed,'known_spearman_target':known,'mean_correlation_bias':float(np.mean(r[u,v]-truth)) if known else None,'mean_interval_coverage':float(covered.mean()) if known else None,'mean_bootstrap_sd':float(sd.mean()),'edge_retention':float(np.mean(np.abs(draws[:,baseline])>=theta)),'ebc_null_calibrated':False}
            if known and any(null):
                # Choose one prespecified available pair involving independent node 0.
                j=np.flatnonzero(null)[0];row.update(null_reject=bool(lo[j]>0 or hi[j]<0),null_error=float(r[u,v][j]),null_bootstrap_sd=float(sd[j]))
            if known and any(alt):
                j=np.flatnonzero(alt)[0];row.update(alt_reject_zero=bool(lo[j]>0 or hi[j]<0),alt_covered=bool(covered[j]),alt_target=float(truth[j]))
            if known:
                oracle=np.abs(truth)>=np.quantile(np.abs(truth),.9)
                row['oracle_edge_jaccard']=float(np.sum(oracle&baseline)/np.sum(oracle|baseline))
            rows.append(row)
            if (rep+1)%50==0:
                print('SIMULATED',scenario,rep+1,'total_seconds',time.perf_counter()-start,flush=True)
                pd.DataFrame(rows).to_csv(checkpoint,index=False)
        pd.DataFrame(rows).to_csv(RUN/'tables/simulation_datasets.csv',index=False)
        status[scenario]={'status':'completed','datasets':CFG['simulation_datasets_per_scenario']};write_json(RUN/'logs/status_simulations.json',status)
    summary=[]
    for scenario,g in pd.DataFrame(rows).groupby('scenario'):
        row={'content':'SIMULATED','scenario':scenario,'datasets':len(g),'known_target':bool(g.known_spearman_target.iloc[0]),'coverage_target':'Spearman correlation, NOT adjusted EBC coefficient','mean_edge_retention':g.edge_retention.mean()}
        for c in ['mean_interval_coverage','mean_correlation_bias','oracle_edge_jaccard']:
            z=g[c].dropna();row[c]=z.mean() if len(z) else None;row[c+'_dataset_mcse']=z.std(ddof=1)/np.sqrt(len(z)) if len(z)>1 else None
        for c in ['null_reject','alt_reject_zero','alt_covered']:
            z=g[c].dropna();row[c+'_rate']=float(z.astype(float).mean()) if len(z) else None;row[c+'_valid_datasets']=len(z);row[c+'_wilson_low'],row[c+'_wilson_high']=wilson(z.astype(float).sum(),len(z))
        summary.append(row)
    pd.DataFrame(summary).to_csv(RUN/'tables/simulation_summary.csv',index=False)
    write_json(RUN/'logs/simulation_runtime.json',{'seconds':time.perf_counter()-start})
if __name__=='__main__':main()
