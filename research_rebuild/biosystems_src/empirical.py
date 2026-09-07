from base import *
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from external import read_external
import csv as csvlib,gzip

def inventory():
    rows=[]
    for acc,path,role in [(ACCS[0],ROOT/'GSE115513_series_matrix.txt.gz','tissue discovery'),(ACCS[1],ROOT/'GSE73002_series_matrix.txt.gz','serum cross-context internal assessment'),('GSE41655',BASE/'data/GSE41655_series_matrix.txt.gz','previously inspected exploratory external tissue')]:
        df,meta=cohort(acc) if acc in ACCS else read_external(path)[:2]
        if acc=='GSE41655':mask=meta.source_name_ch1.eq('Colorectal adenocarcinoma')&meta['pathologic grade'].eq('Adenocarcinoma');df=df.loc[:,mask];meta=meta.loc[mask]
        headers={}
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.startswith('!Sample_'):
                    v=next(csvlib.reader([line],delimiter='\t'));headers.setdefault(v[0],set()).update(v[1:])
                if line.startswith('!series_matrix_table_begin'):break
        processing=' | '.join(sorted(headers.get('!Sample_data_processing',[])))
        rows.append({'accession':acc,'source_url':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+acc,'role':role,'specimen':'serum' if acc==ACCS[1] else 'colorectal tissue','condition':'breast cancer' if acc==ACCS[1] else 'carcinoma / adenocarcinoma','platform':'; '.join(headers.get('!Sample_platform_id',[])),'assay':'miRNA microarray; processed total-RNA miRNA measurements','included_arrays':len(meta),'verified_people':meta.individual.nunique() if 'individual' in meta else np.nan,'repeat_handling':'one eligible array per recorded individual' if 'individual' in meta else 'person linkage unavailable; arrays assumed independent','batch_metadata':'no validated technical batch covariate; available characteristic names: '+', '.join(meta.columns),'detection_information':'processed zeros/censoring; probe-level detection-call matrix not supplied to this analysis','input_probes':len(df),'complete_variable_probes':int((np.isfinite(df).all(axis=1)&(df.max(axis=1)>df.min(axis=1))).sum()),'analysis_features':500 if acc in ACCS else 204,'mapping':'official platform mature identity; unique aliases only for cross-platform comparison','upstream_processing':processing,'input_path':str(path),'input_sha256':sha(path),'independence_status':'recorded individuals unique' if 'individual' in meta else 'assumption; not verified people','global_overlap':'different study submitter/source; no global linkage proof' if acc=='GSE41655' else 'no global person linkage across studies'})
    csv('cohort_inventory',rows)

def module_definitions():
    p=RUN/'tables/module_definitions.csv'
    if p.exists():return pd.read_csv(p)
    df,_=cohort(ACCS[0]);sel,_,_=select_features(df,500);r=spearman(sel.T.to_numpy());dist=1-np.abs(r);np.fill_diagonal(dist,0);dist=np.maximum((dist+dist.T)/2,0)
    labels=fcluster(linkage(squareform(dist,checks=False),method='average'),8,criterion='maxclust');data=pd.DataFrame({'node':range(500),'miRNA':sel.index,'module':labels})
    csv('module_definitions',data);np.savez_compressed(RUN/'models/discovery_network.npz',r=r,miRNA=sel.index.to_numpy(dtype=str),module=labels)
    write_json(RUN/'manifests/module_freeze.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'definition_sha256':sha(p),'rule':'discovery only; absSpearman; average linkage; maxclust8; all modules retained','external_outcomes_inspected_before_this_module_analysis':True,'new_module_outcomes_not_used_to_choose_rule':True})
    return data

def controls_hubs():
    start=time.perf_counter();rows=[];hubrows=[];noderows=[]
    for acc in ACCS:
        z=np.load(OLD/'models'/f'{acc}_unsigned_node_replicates.npz');base=pd.read_csv(OLD/'tables'/f'{acc}_unsigned_hub_priorities.csv');gcc=pd.read_csv(PREV/'tables'/f'{acc}_gcc_nodes.csv').node.to_numpy();S=np.flatnonzero(np.isfinite(cv(z['hard']))&np.isfinite(cv(z['soft'])))
        for method,key,refcol in [('hard','hard','baseline_degree'),('soft','soft','baseline_strength')]:
            x=z[key];orig_rank=rankdata(x,axis=1,method='average');ref=base[refcol].to_numpy();freq={k:np.zeros(500) for k in [10,20,50]}
            for b in range(len(x)):
                for k in [10,20,50]:
                    sel=top(x[b],k);freq[k][sel]+=1;hubrows.append({'cohort':acc,'method':method,'replicate':b,'k':k,'topk_overlap':len(set(sel)&set(top(ref,k)))/k,'rank_rho':rho(ref,x[b]),'status':'completed','universe':500,'tie_policy':'stable node order'})
            for i in range(500):noderows.append({'cohort':acc,'method':method,'node':i,'baseline_score':ref[i],'mean_rank':orig_rank[:,i].mean(),'rank_sd':orig_rank[:,i].std(ddof=1),**{f'top{k}_frequency':freq[k][i]/len(x) for k in freq}})
            for lam in [0,.25,.5,.9,.99]:
                shifted=(1-lam)*x+lam*.5*499
                formula=(1-lam)*x.std(axis=0,ddof=1)/((1-lam)*x.mean(axis=0)+lam*.5*499)
                defined=np.isfinite(formula)&np.isfinite(cv(shifted));err=np.max(np.abs(formula[defined]-cv(shifted)[defined]))
                ranks=rankdata(shifted,axis=1);viol=int(np.sum(np.any(ranks!=orig_rank,axis=1)));tolerant=int(np.sum(np.any(rankdata(np.round(shifted,12),axis=1)!=rankdata(np.round(x,12),axis=1),axis=1)))
                for focal,ids in [('fixed_original_finite_pair_set',S),('baseline_GCC',gcc)]:
                    rows.append({'cohort':acc,'method':method,'lambda':lam,'c':.5,'p':500,'focal':focal,'focal_nodes':len(ids),'B':len(x),'mean_strength':shifted[:,ids].mean(),'mean_sd':shifted[:,ids].std(axis=0,ddof=1).mean(),'mean_cv':cv(shifted)[ids].mean(),'max_formula_error':err,'exact_rank_changed_draws':viol,'rounded12_rank_changed_draws':tolerant,'mean_rank_concordance':np.mean([rho(x[b],shifted[b]) for b in range(len(x))]),'top20_changed_draws':int(np.sum([set(top(x[b],20))!=set(top(shifted[b],20)) for b in range(len(x))])),'support':'fixed full off-diagonal universe; diagonal zero','status':'completed'})
        print(acc,'controls and hubs complete',flush=True)
    csv('stability_control',rows);csv('hub_stability',hubrows);csv('node_rank_variability',noderows)
    df=pd.DataFrame(hubrows);csv('hub_stability_summary',df.groupby(['cohort','method','k'],as_index=False).agg(B=('replicate','count'),mean_overlap=('topk_overlap','mean'),q025_overlap=('topk_overlap',lambda x:x.quantile(.025)),q975_overlap=('topk_overlap',lambda x:x.quantile(.975)),mean_rank_rho=('rank_rho','mean')))
    # Original full/common support mean, SD and CV on exactly the same S.
    obs=[]
    for acc in ACCS:
        z=np.load(OLD/'models'/f'{acc}_unsigned_node_replicates.npz');gcc=pd.read_csv(PREV/'tables'/f'{acc}_gcc_nodes.csv').node.to_numpy()
        for support,keys in [('full',['hard','soft']),('common',['hard_common','soft_common'])]:
            for method,key in zip(['hard','soft'],keys):x=z[key][:,gcc];obs.append({'cohort':acc,'support':support,'method':method,'nodes':len(gcc),'B':1000,'mean_of_means':x.mean(),'mean_of_SD':x.std(axis=0,ddof=1).mean(),'mean_CV':cv(x).mean()})
    csv('original_mean_sd_cv',obs);status('controls_hubs',start)

def external_data():
    ext,meta,_=read_external(BASE/'data/GSE41655_series_matrix.txt.gz');mask=meta.source_name_ch1.eq('Colorectal adenocarcinoma')&meta['pathologic grade'].eq('Adenocarcinoma');m=pd.read_csv(BASE/'manifests/GSE41655_cross_platform_mapping.csv');freeze=json.loads((BASE/'manifests/external_analysis_freeze.json').read_text());ids=freeze['feature_ids'];m=m[(m.number_targets==1)&~m.duplicate_destination&m.mapped_v19.isin(ids)];v=ext.loc[m.external_probe,mask];v.index=m.mapped_v19;return v.loc[ids]

def assess_external(acc,v):
    defs=module_definitions();df,_=cohort(ACCS[0]);ids=v.index.tolist();rd=spearman(df.loc[ids].T.to_numpy());re=spearman(v.T.to_numpy());p=len(ids);u,w=np.triu_indices(p,1);td=np.quantile(np.abs(rd[u,w]),.975);te=np.quantile(np.abs(re[u,w]),.975)
    module=defs.set_index('miRNA').loc[ids,'module'].to_numpy();popsize=defs.module.value_counts();result=[];named=[];summaries=[];control=[]
    ad=np.abs(rd).copy();ae=np.abs(re).copy();np.fill_diagonal(ad,0);np.fill_diagonal(ae,0)
    for policy,t in [('A_frozen',td),('B_density_matched',te)]:
        hd=(ad>=td).sum(axis=1);he=(ae>=t).sum(axis=1);sd=(ad**6).sum(axis=1);se=(ae**6).sum(axis=1)
        summaries.append({'cohort':acc,'policy':policy,'arrays':v.shape[1],'nodes':p,'reference_theta':td,'external_theta':t,'edge_jaccard':np.sum((ad[u,w]>=td)&(ae[u,w]>=t))/np.sum((ad[u,w]>=td)|(ae[u,w]>=t)),'hard_rank_rho':rho(hd,he),'soft_rank_rho':rho(sd,se),**{f'{method}_top{k}_count':len(set(top(xx,k))&set(top(yy,k))) for method,xx,yy in [('hard',hd,he),('soft',sd,se)] for k in [10,20,50]},'uncertainty':'exploratory point estimates; no population confidence interval'})
        for i,id in enumerate(ids):named.append({'cohort':acc,'policy':policy,'miRNA':id,'module':int(module[i]),'discovery_hard':hd[i],'external_hard':he[i],'discovery_soft':sd[i],'external_soft':se[i],'discovery_hard_top20':i in top(hd,20),'external_hard_top20':i in top(he,20),'discovery_soft_top20':i in top(sd,20),'external_soft_top20':i in top(se,20)})
        for mod in sorted(popsize.index):
            ix=np.flatnonzero(module==mod);n=len(ix);coverage=n/popsize[mod];eligible=popsize[mod]>=10 and n>=10 and coverage>=.25
            row={'cohort':acc,'policy':policy,'module':int(mod),'discovery_size':int(popsize[mod]),'mapped_nodes':n,'coverage':coverage,'eligible':eligible,'status':'completed' if eligible else 'not_run','reason':'' if eligible else 'frozen size/coverage minimum unmet','identities':'; '.join(np.array(ids)[ix])}
            if eligible:
                D=ad[np.ix_(ix,ix)];E=ae[np.ix_(ix,ix)];tu,tv=np.triu_indices(n,1);dh=(D>=td).sum(axis=1);eh=(E>=t).sum(axis=1);ds=(D**6).sum(axis=1);es=(E**6).sum(axis=1)
                row.update(discovery_hard_density=np.mean(D[tu,tv]>=td),external_hard_density=np.mean(E[tu,tv]>=t),discovery_soft_mean_weight=np.mean(D[tu,tv]**6),external_soft_mean_weight=np.mean(E[tu,tv]**6),hard_connectivity_rho=rho(dh,eh),soft_connectivity_rho=rho(ds,es),hard_hub_top5_count=len(set(top(dh,5))&set(top(eh,5))),soft_hub_top5_count=len(set(top(ds,5))&set(top(es,5))),discovery_soft_hubs='; '.join(np.array(ids)[ix[top(ds,5)]]),external_soft_hubs='; '.join(np.array(ids)[ix[top(es,5)]]),pairs=n*(n-1)//2)
            result.append(row)
    # The same affine diagnostic on precisely this mapped external universe.
    for method,d,e in [('hard',(ad>=td).sum(axis=1),(ae>=te).sum(axis=1)),('soft',(ad**6).sum(axis=1),(ae**6).sum(axis=1))]:
        for lam in [0,.25,.5,.9,.99]:
            ds=(1-lam)*d+lam*.5*(p-1);es=(1-lam)*e+lam*.5*(p-1)
            control.append({'cohort':acc,'method':method,'lambda':lam,'p':p,'external_rank_rho':rho(ds,es),'original_rank_rho':rho(d,e),'top20_overlap_count':len(set(top(ds,20))&set(top(es,20))),'discovery_top20_invariant':set(top(ds,20))==set(top(d,20)),'external_top20_invariant':set(top(es,20))==set(top(e,20)),'interpretation':'fixed-universe affine control; density-matched hard reference; not a biological intervention'})
    np.savez_compressed(RUN/'models'/f'{acc}_mapped_networks.npz',discovery_r=rd,external_r=re,miRNA=np.array(ids),module=module)
    return result,named,summaries,control

def modules_external():
    start=time.perf_counter();defs=module_definitions();v=external_data();r,n,s,c=assess_external('GSE41655',v)
    csv('module_preservation',r);csv('named_hub_results',n);csv('external_summary',s);csv('external_affine_control',c);status('modules_existing_external',start,modules=len(defs.module.unique()),eligible_modules=sum(x['eligible'] for x in r)//2)

def graph_consequences():
    start=time.perf_counter();defs=module_definitions();old=pd.read_csv(PREV/'tables/edge_change_per_replicate.csv');rows=[];states=[]
    for acc in ACCS:
        ref=pd.read_csv(OLD/'tables'/f'{acc}_unsigned_d0.025_pair_universe.csv');u=ref.node_a.to_numpy();v=ref.node_b.to_numpy();base=ref.initially_retained.to_numpy();theta=float(pd.read_csv(PREV/'tables/verified_graph_summary.csv').set_index('cohort').loc[acc,'theta']);z=np.load(OLD/'models'/f'{acc}_unsigned_node_replicates.npz');hb=pd.read_csv(OLD/'tables'/f'{acc}_unsigned_hub_priorities.csv');membership=defs.module.to_numpy() if acc==ACCS[0] else None
        for ids,r in blocks(acc):
            a=np.abs(r)>=theta
            for b,state in zip(ids,a):
                D=int(np.count_nonzero(base&~state));G=int(np.count_nonzero(~base&state));M=int(state.sum());orig=old[old.cohort.eq(acc)&old.replicate.eq(b)].iloc[0];assert D==orig.disappearances and G==orig.appearances and M==3119-D+G
                states.append({'cohort':acc,'replicate':int(b),'reference_present':3119,'reference_absent':121631,'disappearances':D,'appearances':G,'unchanged_present':3119-D,'unchanged_absent':121631-G,'unevaluable_pairs':0,'M':M,'jaccard':(3119-D)/(3119+G),'density':M/124750,'components':orig.components,'gcc_nodes':orig.gcc_nodes,'isolates':orig.isolates})
                row={'cohort':acc,'replicate':int(b),'M':M,'D':D,'G':G,'hard_rank_rho':rho(hb.baseline_degree,z['hard'][b]),'soft_rank_rho':rho(hb.baseline_strength,z['soft'][b]),'hard_top20_overlap':len(set(top(hb.baseline_degree,20))&set(top(z['hard'][b],20)))/20,'soft_top20_overlap':len(set(top(hb.baseline_strength,20))&set(top(z['soft'][b],20)))/20}
                if membership is not None:
                    for mod in sorted(defs.module.unique()):
                        mask=(membership[u]==mod)&(membership[v]==mod);row[f'module{mod}_hard_density']=state[mask].mean() if mask.any() else np.nan
                rows.append(row)
                if b==0:np.savez_compressed(RUN/'models'/f'{acc}_display_draw0.npz',u=u,v=v,reference=base,bootstrap=state)
        print(acc,'graph states recounted',flush=True)
    csv('graph_state_accounting',states);csv('graph_consequences',rows)
    # Document the saved domain failure without recreating200 attempted matrices.
    d=pd.read_csv(PREV/'tables/degeneracy_audit.csv');r=pd.read_csv(PREV/'tables/degeneracy_per_replicate.csv')
    csv('preserved_failure_summary',[{'cohort':ACCS[0],'features':1000,'attempts':len(r),'complete_graph_valid':int(r.full_graph_valid.sum()),'probes_ever_constant':int((d.observed_constant_count>0).sum()),'observed_mean_constant':r.constant_probes.mean(),'predicted_mean_constant':d.predicted_constant_probability.sum(),'source':str(PREV/'tables/degeneracy_audit.csv'),'verification':'saved probe counts and per-attempt failure accounting reaggregated; no expensive rerun'}]);status('graph_recount',start,draws=len(states))

if __name__=='__main__':globals()[sys.argv[1]]()
