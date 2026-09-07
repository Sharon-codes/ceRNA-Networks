from common import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
import statsmodels.api as sm

LEDGER=[]
def verify(id,acc,reported,value,path,formula,explanation='',tol=1e-8):
    diff=abs(float(reported)-float(value));rel=diff/abs(float(reported)) if reported else (0 if diff==0 else np.nan)
    new=Path(path).is_relative_to(RUN)
    LEDGER.append({'result_id':id,'cohort':acc,'run_id':RUN.name,'source_run':RUN.name if new else OLD.name,'config_sha256':sha(RUN/'followup_analysis_amendment.md') if new else sha(BASE/'configs/primary.json'),'reported_value':reported,'recomputed_value':value,'absolute_discrepancy':diff,'relative_discrepancy':rel,'underlying_file':str(path),'sha256':sha(path),'computation':formula,'figure_table_link':str(path),'status':'verified' if diff<=tol*(1+abs(float(reported))) else 'discrepancy','explanation':explanation})
def flush_ledger():
    path=RUN/'tables/verified_results_ledger.csv'
    previous=pd.read_csv(path).to_dict('records') if path.exists() else []
    newids={(r['result_id'],r['cohort']) for r in LEDGER}
    csv('verified_results_ledger',[r for r in previous if (r['result_id'],r['cohort']) not in newids]+LEDGER)

def primary():
    start=time.perf_counter();changes=[];summaries=[];graphrows=[]
    for acc in ACCS:
        df,meta=cohort(acc);selection,_,_=select_features(df,500);x=selection.T.to_numpy();r=spearmanr(x,axis=0).statistic
        u,v=np.triu_indices(len(r),1);scores=np.abs(r[u,v]);theta=np.quantile(scores,.975,method='linear');base=scores>=theta
        a=np.zeros((len(r),len(r)),bool);a[u,v]=base;a|=a.T
        components,labels=connected_components(csr_matrix(a),directed=False);sizes=np.bincount(labels);gcc=np.flatnonzero(labels==np.argmax(sizes))
        values={'sample_count':len(meta),'nodes':len(r),'eligible_pairs':len(u),'full_edges':int(sum(base)),'full_density':base.mean(),'theta':theta,'components':components,'isolates':int(sum(a.sum(axis=0)==0)),'gcc_nodes':len(gcc),'gcc_edges':int(a[np.ix_(gcc,gcc)].sum()/2)}
        src=OLD/'tables'/f'{acc}_network_sensitivity.csv';old=pd.read_csv(src);row=old[old.analysis.eq(acc+'_unsigned_d0.025')].iloc[0]
        for k,value in values.items():verify(k,acc,row[k],value,ROOT/f'{acc}_series_matrix.txt.gz','Independent scipy Spearman; linear 97.5% quantile; scipy connected_components; full unordered pair universe')
        graphrows.append({'cohort':acc,**values,'all_input_probes':len(df),'complete_probes':int(np.isfinite(df).all(axis=1).sum()),'recorded_unique_people':meta.individual.nunique() if 'individual' in meta else np.nan})
        csv(acc+'_gcc_nodes',{'node':gcc})
        count=np.zeros(len(u),int);retained_boot=[];idsall=[];Mref=int(base.sum())
        for ids,boot in cache_blocks(acc+'_primary',x):
            state=np.abs(boot)>=theta;count+=state.sum(axis=0);retained_boot.append(np.abs(boot[:,base]));idsall.extend(ids)
            for b,bb in zip(ids,state):
                D=int(np.count_nonzero(base&~bb));G=int(np.count_nonzero(~base&bb));M=int(np.count_nonzero(bb));assert M==Mref-D+G
                ab=np.zeros(a.shape,bool);ab[u,v]=bb;ab|=ab.T
                nc,lab=connected_components(csr_matrix(ab),directed=False);sz=np.bincount(lab);gids=np.flatnonzero(lab==np.argmax(sz))
                changes.append({'cohort':acc,'replicate':int(b),'attempted_B':1000,'valid':True,'nodes':len(r),'pairs':len(u),'reference_edges':Mref,'retained_reference_edges':Mref-D,'disappearances':D,'appearances':G,'total_edges':M,'symmetric_difference':D+G,'jaccard':(Mref-D)/(Mref+G),'density':M/len(u),'components':nc,'gcc_nodes':len(gids),'gcc_edges':int(ab[np.ix_(gids,gids)].sum()/2),'isolates':int(np.count_nonzero(ab.sum(axis=0)==0)),'theta':theta,'identity_residual':M-(Mref-D+G)})
        assert idsall==list(range(1000));pp=count/1000
        verify('retained_dropout',acc,row.mean_retained_dropout,np.mean(1-pp[base]),cache_folder(acc+'_primary')/'identity.json','Mean over 3119 edges of 1 - present_count/1000')
        verify('absent_appearance',acc,row.mean_absent_appearance,np.mean(pp[~base]),cache_folder(acc+'_primary')/'identity.json','Mean over 121631 absent pairs of present_count/1000')
        pu=OLD/'tables'/f'{acc}_unsigned_d0.025_pair_universe.csv';saved=pd.read_csv(pu)
        verify('pair_probability_max_error',acc,0,np.max(np.abs(pp-saved.p_present)),pu,'Cached draw threshold counts versus saved all-pair probabilities')
        csv(acc+'_primary_edge_states',{'node_a':u,'node_b':v,'initially_retained':base,'attempted':1000,'present':count,'absent':1000-count,'unevaluable':0,'evaluable':1000,'fraction_unevaluable':0,'dropout_given_evaluable':np.where(base,1-pp,np.nan),'appearance_given_evaluable':np.where(~base,pp,np.nan)})
        c=pd.DataFrame([d for d in changes if d['cohort']==acc]);item={'cohort':acc,'B':1000,'reference_edges':int(sum(base)),'absent_pairs':int(sum(~base)),'retained_dropout':np.mean(1-pp[base]),'absent_appearance':np.mean(pp[~base]),'D_from_edge_probabilities':np.sum(1-pp[base]),'G_from_edge_probabilities':np.sum(pp[~base])}
        for col in ['disappearances','appearances','total_edges','symmetric_difference','jaccard','density','components','gcc_nodes','gcc_edges','isolates']:
            item.update({col+'_mean':c[col].mean(),col+'_resampling_q025':c[col].quantile(.025),col+'_resampling_q975':c[col].quantile(.975),col+'_mean_mcse':c[col].std(ddof=1)/np.sqrt(1000)})
        assert np.isclose(item['D_from_edge_probabilities'],item['disappearances_mean']) and np.isclose(item['G_from_edge_probabilities'],item['appearances_mean'])
        summaries.append(item)
        # Independent fits from saved design and freshly counted cached outcomes.
        label=acc+'_unsigned_d0.025';designs=np.load(OLD/'models'/f'{label}_designs.npz');coefsrc=OLD/'tables'/f'{label}_coefficients.csv';coefs=pd.read_csv(coefsrc);rb=np.concatenate(retained_boot);y=(rb<theta).astype(float);sd=rb.std(axis=0,ddof=1)
        for name in designs.files:
            keep=sd>=1e-10 if name.startswith('uncertainty') else np.ones(len(sd),bool);X=designs[name][keep];Y=y[:,keep];fit=sm.GLM(np.c_[Y.sum(axis=0),1000-Y.sum(axis=0)],X,family=sm.families.Binomial()).fit(maxiter=200)
            p=fit.predict(X);score=(Y-p)@X;bread=np.linalg.pinv(X.T@((p*(1-p))[:,None]*X));centered=score-score.mean(axis=0);cov=bread@(centered.T@centered/999)@bread/1000
            oldc=coefs[coefs.model.eq(name)].reset_index(drop=True)
            for j,beta in enumerate(fit.params):
                verify(name+'_coef'+str(j),acc,oldc.iloc[j].coefficient,beta,coefsrc,'Independent binomial GLM fit using saved design and re-counted bootstrap events','Conditional descriptive association; not population inference')
                verify(name+'_mcse'+str(j),acc,oldc.iloc[j].mc_se,np.sqrt(max(cov[j,j],0)),coefsrc,'Whole-draw score sandwich / B','MC only; SD/basis fixed, especially generated uncertainty predictor')
        print(acc,'primary independently verified',flush=True)
    csv('verified_graph_summary',graphrows);csv('edge_change_per_replicate',changes);csv('edge_change_summary',summaries)
    # Historical saved-replicate anchor; ddof=0 matches historical code.
    p=OLD/'historical/GSE115513/replicates.npz';z=np.load(p)
    h=z['hard'];s=z['soft'];hc=h.std(axis=0)/h.mean(axis=0);sc=s.std(axis=0)/s.mean(axis=0)
    for id,rep,val in [('historical_hard_cv',.25710309618001453,hc.mean()),('historical_soft_cv',.09294407612425502,sc.mean()),('historical_reduction',.6384949169994482,1-sc.mean()/hc.mean())]:verify(id,ACCS[0],rep,val,p,'Historical GCC node replicates; SD ddof=0; ratio of mean CVs')
    verify('manuscript_84_percent',ACCS[0],.84,1-sc.mean()/hc.mean(),p,'Historical saved-output reconstruction','Manuscript .218/.035 not reproduced; their arithmetic gives 83.94495%, which does not establish provenance')
    flush_ledger();status('primary_verification',start)

def statuses():
    start=time.perf_counter();rows=[];src=OLD/'tables/targeted_sensitivity_summary.csv';old=pd.read_csv(src)
    for _,r in old.iterrows():
        folder=cache_folder(r.analysis);completion=json.loads((folder/'completion.json').read_text());attempt=completion['requested'];n=0;ids=[];fail=[]
        for p in sorted(folder.glob('*.npz')):
            receipt=json.loads(p.with_suffix('.json').read_text());assert sha(p)==receipt['sha256'];z=np.load(p);n+=len(z['r']);ids+=z['replicate'].tolist();fail+=receipt['failures']
        assert n==completion['valid'] and len(set(ids))==n and n+len(fail)==attempt
        diag=OLD/'tables'/f'{r.analysis}_diagnostics.csv';valid_outputs=False
        if n>=20 and diag.exists():
            d=pd.read_csv(diag);valid_outputs=len(d)==6 and d.valid_B.eq(n).all() and d.converged.all() and np.isfinite(d.in_sample_logloss).all()
        state='failed' if n<20 else ('completed' if valid_outputs and n==attempt else 'partially_completed' if valid_outputs else 'unverified')
        logs=[]
        for p in (OLD/'logs').glob('status_*.json'):
            text=p.read_text();
            if r.analysis in text:logs.append(str(p))
        rows.append({'analysis':r.analysis,'previous_status':r.status,'status':state,'attempted_B':attempt,'valid_B':n,'failed_B':len(fail),'failure_reasons':'; '.join(sorted(set(f['error'] for f in fail))),'runtime_seconds_receipt':completion['runtime_seconds'],'runtime_caveat':'receipt may measure cache read on resume; stage logs preserve original elapsed time','output_validity':valid_outputs if n>=20 else 'complete-graph estimator domain failure','cache_receipt':str(folder/'completion.json'),'receipt_sha256':sha(folder/'completion.json'),'status_logs':'; '.join(logs),'evidence':'all chunk hashes, disjoint replicate IDs and failures, B agreement; six converged finite model diagnostics for completed settings'})
    csv('reconciled_stage_status',rows)
    figures=[]
    for f in json.loads((OLD/'figures/figure_manifest.json').read_text()):
        for s in f['source_tables']:
            p=Path(s['path']);figures.append({'figure':f['figure'],'source':str(p),'declared_sha256':s['sha256'],'actual_sha256':sha(p),'matches':sha(p)==s['sha256'],'command':f['plotting_command'],'run_id':f['run_id'],'config_sha256':f['config_sha256'],'input_manifest':str(RUN/'manifests/freeze.json'),'caption':f['caption'],'svg':f['svg'],'png':f['png'],'followup_action':'replace CV/heldout/external synthesis with new follow-up figures' if any(k in f['figure'] for k in ['hard_soft','heldout','external']) else 'retain verified existing figure'})
    csv('existing_figure_provenance_verification',figures)
    report('status_and_figure_audit','Serum missing status fields arise from the earlier sensitivity code path lacking the status key. Each new status is justified by checksummed chunks, attempted/valid/failed ID accounting and model diagnostics, not by CSV existence. The tissue 1000-feature setting remains failed for complete-graph estimators. Runtime receipts can be overwritten by cache reads; the preserved original stage logs are the runtime evidence.\n\nAll original figure source hashes were checked. The empirical density-sensitivity figure reads actual network sensitivity CSVs. The historical manually imposed enrichment formula remains only in preserved historical source/PDF evidence and is excluded from new empirical figures. Old CV intervals and bootstrap-test held-out panels are superseded by the explicitly new figure set; unchanged original panels remain available with their original run and captions.\n')
    status('status_figure_audit',start)

def degeneracy():
    start=time.perf_counter();acc=ACCS[0];df,meta=cohort(acc);sel,_,_=select_features(df,1000);x=sel.T.to_numpy();n,p=x.shape;u,v=np.triu_indices(p,1)
    theta=float(pd.read_csv(OLD/'tables/targeted_sensitivity_summary.csv').query("analysis=='GSE115513_features1000'").iloc[0].theta)
    base=np.abs(spearman(x)[u,v])>=theta;pred=[];uni=[];dom=[]
    for col in x.T:
        _,counts=np.unique(col,return_counts=True);freq=counts/n;pred.append(np.sum(freq**n));uni.append(len(counts));dom.append(max(freq))
    observed=np.zeros(p,int);present=np.zeros(len(u),int);validcounts=np.zeros(len(u),int);rep=[];const=[]
    identity=json.loads((cache_folder(acc+'_features1000')/'identity.json').read_text());assert identity['x']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    for b in range(200):
        idx=rng_for(SEED,acc+'_features1000',b).integers(0,n,n);r,valid=paircorr_partial(x[idx]);good=valid[u]&valid[v];state=(np.abs(r[u,v])>=theta)&good
        observed+=~valid;validcounts+=good;present+=state;const.append(~valid)
        rep.append({'replicate':b,'constant_probes':int((~valid).sum()),'evaluable_pairs':int(good.sum()),'unevaluable_pairs':int((~good).sum()),'full_graph_valid':bool(good.all()),'known_present':int(state.sum()),'known_absent':int((good&~state).sum()),'edge_count_completion_lower':int(state.sum()),'edge_count_completion_upper':int(state.sum()+(~good).sum()),'known_reference_disappearances':int((base&good&~state).sum()),'known_absent_appearances':int((~base&state).sum()),'unknown_reference_pairs':int((base&~good).sum()),'unknown_initially_absent_pairs':int((~base&~good).sum())})
        if (b+1)%50==0:print('degeneracy',b+1,'/200',flush=True)
    absent=validcounts-present;unevaluable=200-validcounts
    assert np.all(present+absent+unevaluable==200)
    with np.errstate(divide='ignore',invalid='ignore'):drop=absent/validcounts;appearance=present/validcounts
    csv('degeneracy_audit',{'node':np.arange(p),'probe_id':sel.index,'n':n,'bootstrap_m':n,'attempted_B':200,'unique_values':uni,'tie_fraction':1-np.array(uni)/n,'missing_fraction':np.mean(~np.isfinite(x),axis=0),'dominant_value_fraction':dom,'predicted_constant_probability':pred,'observed_constant_count':observed,'observed_constant_fraction':observed/200,'predicted_minus_observed':np.array(pred)-observed/200})
    csv('degeneracy_edge_states',{'node_a':u,'node_b':v,'probe_a':sel.index.to_numpy()[u],'probe_b':sel.index.to_numpy()[v],'initially_retained':base,'attempted':200,'present':present,'absent':absent,'unevaluable':unevaluable,'evaluable':validcounts,'fraction_unevaluable':unevaluable/200,'dropout_given_evaluable':np.where(base,drop,np.nan),'appearance_given_evaluable':np.where(~base,appearance,np.nan)})
    csv('degeneracy_per_replicate',rep);np.savez_compressed(RUN/'models/degenerate_probe_states.npz',constant=np.array(const),probe_id=sel.index.to_numpy(dtype=str))
    report('degeneracy_report',f'''# Degenerate empirical resamples

Reconstructed the original 200 streams, 750 independent recorded individuals, supplied-scale 1000-feature MAD selection; theta={theta:.16g}. {sum(observed>0)} probes became constant at least once. Predicted expected constant probes per draw is {sum(pred):.9g}; observed mean is {np.mean(np.array(const).sum(axis=1)):.9g}. The diagnostic sums empirical point-mass probabilities to power750, probe by probe. It does not assume independence between probes and is not used as a graph-failure probability.

Complete graph validity: {sum(d['full_graph_valid'] for d in rep)}/200. The old core.spearman raises on any constant selected column, causing rejection of the entire correlation matrix. This is more restrictive than pairwise evaluability. New per-pair counts retain finite correlations, with three explicit states; {sum(unevaluable>0)} of {len(u)} pairs were unevaluable in at least one attempt, including {sum(base&(unevaluable>0))} retained reference pairs. No undefined pair is assigned a zero correlation and no failed draw is replaced.

Conditional dropout/appearance divide only by that pair's evaluable attempts. These are not unconditional probabilities; pair-specific denominators prevent naive conversion of averaged conditional probabilities into a mean complete-graph count. Complete-graph CV, EBC and graph-change inference for this setting remain failed. Per-draw minimum/maximum edge counts are bounds over hypothetical binary completions of unknown pairs, not observed graphs or population confidence limits. No post-hoc probe filter was applied.
''')
    status('degeneracy',start,attempted=200,whole_graph_valid=sum(d['full_graph_valid'] for d in rep))

if __name__=='__main__':globals()[sys.argv[1]]()
