from base import *
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import warnings

def fast_spearman(x):
    """Canonical rank arithmetic with process-wide single-thread BLAS, no per-call scan."""
    q=rankdata(x,axis=0);q-=q.mean(axis=0);norm=np.sqrt((q*q).sum(axis=0))
    if np.any(norm==0):raise ValueError('constant simulated probe')
    q/=norm;r=q.T@q;r=np.clip((r+r.T)/2,-1,1);np.fill_diagonal(r,1);return r

def truth(scale,nuisance):
    p=120;labels=np.repeat(np.arange(4),30);load=np.tile(np.linspace(.35,.85,30),4)*scale;L=np.zeros((p,4));L[np.arange(p),labels]=load
    g=np.linspace(.2,1.2,p).reshape(4,30);g[1::2]=g[1::2,::-1];g=g.ravel()*nuisance
    latent=L@L.T+np.eye(p);covariance=latent+np.outer(g,g)
    def corr(c):r=c/np.sqrt(np.outer(c.diagonal(),c.diagonal()));return 6/np.pi*np.arcsin(r/2)
    observed=corr(covariance);biological=corr(latent);hub=np.r_[27:30,57:60,87:90,117:120]
    minimum=float(np.linalg.eigvalsh(covariance).min());assert minimum>0
    if scale:
        for mod in range(4):
            ids=np.flatnonzero(labels==mod);strength=biological[np.ix_(ids,ids)].sum(axis=1)-1
            assert set(ids[top(strength,3)])==set(hub[labels[hub]==mod])
    return L,g,covariance,observed,biological,labels,hub,minimum

def matrices(r):
    w=np.abs(r).copy();np.fill_diagonal(w,0);u,v=np.triu_indices(len(r),1);t=np.quantile(w[u,v],.975);a=(w>=t).astype(float);np.fill_diagonal(a,0);soft=w**6
    # Monotone edge ranking is not a second independent success.
    q=np.quantile(soft[u,v],.975);assert np.array_equal(a[u,v]>0,soft[u,v]>=q)
    return a,soft,t
def modules(A):
    with warnings.catch_warnings(record=True) as logs:
        warnings.simplefilter('always');labels=SpectralClustering(n_clusters=4,affinity='precomputed',assign_labels='cluster_qr',random_state=SEED).fit_predict(A)
    return labels,len(logs)
def one(cell,rep,B=100,save_truth=False):
    n=cell['n'];scale=cell['scale'];nuisance=cell['nuisance'];L,g,C,pop,latent,labels,hub,eigen=truth(scale,nuisance);rng=rng_for(SEED,cell['id'],rep)
    def sample():return rng.normal(size=(n,4))@L.T+rng.normal(size=(n,1))*g+rng.normal(size=(n,120))
    x=sample();x2=sample();r=fast_spearman(x);r2=fast_spearman(x2);a,s,t=matrices(r);a2,s2,t2=matrices(r2)
    # Literal >= quantile rule at the all-zero null retains every off-diagonal pair.
    # The tie-degenerate population hard target is complete, not an invented empty graph.
    popa,pops,popt=matrices(pop)
    h=[];ss=[];hc=[];sc=[];hubdraw=[]
    for b in range(B):
        ix=rng_for(SEED,cell['id']+'_boot_'+str(rep),b).integers(0,n,n);rr=fast_spearman(x[ix]);w=np.abs(rr);np.fill_diagonal(w,0);ab=(w>=t).astype(float);np.fill_diagonal(ab,0)
        h.append(ab.sum(axis=1));ss.append((w**6).sum(axis=1));hc.append((ab*a).sum(axis=1));sc.append(((w**6)*a).sum(axis=1))
    h=np.array(h);ss=np.array(ss);hc=np.array(hc);sc=np.array(sc);S=np.flatnonzero(np.isfinite(cv(h))&np.isfinite(cv(ss)));SC=np.flatnonzero(np.isfinite(cv(hc))&np.isfinite(cv(sc)))
    out=[]
    for name,A,A2,PA,z,zc in [('hard',a,a2,popa,h,hc),('soft',s,s2,pops,ss,sc)]:
        strength=A.sum(axis=1);strength2=A2.sum(axis=1);pstrength=PA.sum(axis=1);part,warnings1=modules(A);part2,warnings2=modules(A2)
        out.append({'cell':cell['id'],'dataset':rep,'n':n,'signal_scale':scale,'nuisance':nuisance,'method':name,'B':B,'valid_B':B,'nodes':120,'paired_finite_CV_nodes':len(S),'mean_cv':cv(z)[S].mean(),'common_support_mean_cv':cv(zc)[SC].mean() if len(SC) else np.nan,'common_support_finite_nodes':len(SC),'mean_strength':z[:,S].mean(),'mean_sd':z[:,S].std(axis=0,ddof=1).mean(),'cv_prefix50_minus100':cv(z[:50])[S].mean()-cv(z)[S].mean(),'latent_hub_top12_recovery':len(set(top(strength,12))&set(hub))/12 if scale else np.nan,'latent_hub_rank_rho':rho(strength,(latent-np.eye(120)).sum(axis=1)) if scale else np.nan,'population_method_strength_rho':rho(strength,pstrength),'population_method_normalized_rmse':np.sqrt(np.mean((strength/119-pstrength/119)**2)),'module_ARI':adjusted_rand_score(labels,part) if scale else np.nan,'independent_module_ARI':adjusted_rand_score(part,part2),'independent_rank_rho':rho(strength,strength2),'independent_top12_overlap':len(set(top(strength,12))&set(top(strength2,12)))/12,'bootstrap_top12_overlap':np.mean([len(set(top(q,12))&set(top(strength,12)))/12 for q in z]),'spectral_warning_count':warnings1+warnings2,'graph_components':connected_components(A>0,directed=False)[0],'theta':t,'minimum_covariance_eigenvalue':eigen,'truth_status':'planted modules and verified within-module hubs' if scale else 'null: no true hubs/modules; recovery undefined','status':'completed'})
    if save_truth:np.savez_compressed(RUN/'models'/f"simulation_population_truth_{cell['id']}.npz",loading=L,nuisance_loading=g,covariance=C,population_observed_spearman=pop,population_latent_spearman=latent,module_labels=labels,latent_hub_nodes=hub if scale else np.array([],int),population_hard_adjacency=popa,population_soft_adjacency=pops)
    return out
def main():
    start=time.perf_counter();cells=[]
    for n in [60,240]:
        for scale in [.5,1.]:
            for nuisance in [0,.6]:cells.append({'id':f'n{n}_s{scale:g}_g{nuisance:g}','n':n,'scale':scale,'nuisance':nuisance})
        cells.append({'id':f'n{n}_null','n':n,'scale':0,'nuisance':0})
    freeze=RUN/'manifests/simulation_production_freeze.json'
    if not freeze.exists():
        pilots=[]
        for c in [cells[0],cells[8]]:
            t=time.perf_counter();out=one(c,100000);pilots.append({'cell':c['id'],'seconds':time.perf_counter()-t,'note':'pilot comparative outcomes not inspected; only runtime retained'})
        estimate=max(p['seconds'] for p in pilots)*10*60;R=60 if estimate<=2700 else 30
        write_json(freeze,{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pilot_runtime_only':pilots,'projected_seconds_R60':estimate,'datasets_per_cell':R,'B':100,'cells':cells,'reason':'runtime-based freeze before production outcomes','worst_case_normal_halfwidth_bounded_mean':1.96*.5/np.sqrt(R)})
        print('Production frozen:',R,'datasets/cell; projected seconds',round(estimate),flush=True)
    f=json.loads(freeze.read_text());R=f['datasets_per_cell'];allrows=[]
    for c in cells:
        folder=RUN/'models/simulation_checkpoints';folder.mkdir(exist_ok=True)
        for rep in range(R):
            p=folder/f"{c['id']}_{rep:03d}.json"
            if p.exists():rows=json.loads(p.read_text())
            else:
                try:rows=one(c,rep,save_truth=rep==0)
                except Exception as e:rows=[{'cell':c['id'],'dataset':rep,'n':c['n'],'signal_scale':c['scale'],'nuisance':c['nuisance'],'method':method,'status':'failed','error':repr(e)} for method in ['hard','soft']]
                write_json(p,rows)
            allrows.extend(rows)
        csv('simulation_results',allrows);print('Simulation',c['id'],'finished',flush=True)
    df=pd.DataFrame(allrows);summary=[];precision=[]
    metrics=['mean_cv','common_support_mean_cv','latent_hub_top12_recovery','latent_hub_rank_rho','module_ARI','independent_module_ARI','independent_rank_rho','independent_top12_overlap','bootstrap_top12_overlap','population_method_normalized_rmse']
    for (cell,method),g in df.groupby(['cell','method']):
        for metric in metrics:
            z=g[metric].dropna();se=z.std(ddof=1)/np.sqrt(len(z)) if len(z)>1 else np.nan
            summary.append({'cell':cell,'method':method,'endpoint':metric,'datasets':len(g),'valid_endpoint_datasets':len(z),'mean':z.mean(),'sd_between_datasets':z.std(ddof=1),'mean_MCSE':se,'mc_low':z.mean()-1.96*se,'mc_high':z.mean()+1.96*se,'interval':'normal MC approximation across independent simulated datasets; not biological population CI'})
        precision.append({'cell':cell,'method':method,'datasets':len(g),'valid_datasets':sum(g.status.eq('completed')),'max_abs_cv_prefix_change':g.cv_prefix50_minus100.abs().max(),'mean_abs_cv_prefix_change':g.cv_prefix50_minus100.abs().mean(),'B':100,'interpretation':'conditional bootstrap numerical diagnostic, separate from dataset MC'})
    csv('simulation_summary',summary);csv('simulation_precision',precision)
    paired=[]
    for cell,g in df.groupby('cell'):
        a=g[g.method.eq('hard')].set_index('dataset');b=g[g.method.eq('soft')].set_index('dataset')
        for metric in metrics:
            z=(b[metric]-a[metric]).dropna();se=z.std(ddof=1)/np.sqrt(len(z)) if len(z)>1 else np.nan
            paired.append({'cell':cell,'endpoint':metric,'valid_pairs':len(z),'soft_minus_hard':z.mean(),'MCSE':se,'mc_low':z.mean()-1.96*se,'mc_high':z.mean()+1.96*se})
    csv('simulation_paired_differences',paired);status('simulation',start,datasets=R*len(cells),method_rows=len(df),failed_rows=int(sum(df.status.ne('completed'))))
if __name__=='__main__':main()
