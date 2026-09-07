"""Frozen direct held-out outcome, train-only learning, paired edge scoring."""
from common import *
import statsmodels.api as sm
from scipy.stats import rankdata
from models import design

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);n=len(y);events=int(y.sum());non=n-events
    if not n:return dict(brier=np.nan,logloss=np.nan,auroc=np.nan,average_precision=np.nan)
    pc=np.clip(p,1e-12,1-1e-12)
    auc=(rankdata(p)[y==1].sum()-events*(events+1)/2)/(events*non) if events and non else np.nan
    # Threshold groups preserve tied-score AP semantics, unlike arbitrary within-tie ordering.
    order=np.argsort(-p,kind='stable');yp=y[order];ps=p[order];ends=np.r_[np.flatnonzero(np.diff(ps)!=0),n-1]
    tp=np.cumsum(yp)[ends];ap=np.sum(np.diff(np.r_[0,tp])/events*(tp/(ends+1))) if events and non else np.nan
    return {'brier':np.mean((p-y)**2),'logloss':-np.mean(y*np.log(pc)+(1-y)*np.log1p(-pc)),'auroc':auc,'average_precision':ap,'events':events,'nonevents':non,'prevalence':events/n,'mean_prediction':p.mean(),'calibration_bias':p.mean()-events/n,'clipped_count':int(np.sum(p!=pc)),'clipped_fraction':np.mean(p!=pc)}

def training_boot(x,g,acc,s,m):
    cache=RUN/'cache'/f'{acc}_split{s:02d}_train.npz'
    label=f'{acc}_followup_train_split{s:02d}'
    identity={'expression_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),'m':m,'B':250,'label':label,'seed':SEED,'theta':g['summary']['theta'],'edges_sha256':hashlib.sha256(g['edges'].tobytes()).hexdigest()}
    if cache.exists():
        receipt=json.loads(cache.with_suffix('.json').read_text());assert receipt['identity']==identity and receipt['sha256']==sha(cache)
        z=np.load(cache);return z['r'],z['invalid_nodes'],receipt
    start=time.perf_counter();u,v=g['edges'].T
    if s==0:
        blocks=list(cache_blocks(acc+'_split_reference',x));vals=np.concatenate([a[:,g['pair_indices']] for _,a in blocks]);assert len(vals)==250
        bad=np.zeros((250,x.shape[1]),bool);origin='verified original train cache'
    else:
        vals=[];bad=[]
        for b in range(250):
            idx=rng_for(SEED,label,b).integers(0,len(x),m)
            r,valid=paircorr_partial(x[idx]);vals.append(r[u,v]);bad.append(~valid)
        vals=np.array(vals);bad=np.array(bad);origin='new independent train bootstrap attempts'
    np.savez_compressed(cache,r=vals,invalid_nodes=bad)
    receipt={'identity':identity,'sha256':sha(cache),'origin':origin,'runtime_seconds':time.perf_counter()-start,'full_valid_B':int(np.sum(~bad.any(axis=1)))}
    write_json(cache.with_suffix('.json'),receipt)
    return vals,bad,receipt

def main():
    start=time.perf_counter();splits=json.loads((RUN/'manifests/splits.json').read_text());assert len(splits)==40
    allrows=[];pairs=[];calibration=[];audit=[];toprows=[]
    for acc in ACCS:
        df,meta=cohort(acc)
        for spec in [s for s in splits if s['cohort']==acc]:
            s=spec['split'];dest=RUN/'models'/f'{acc}_split{s:02d}.json'
            if dest.exists():
                receiptpath=dest.with_suffix('.receipt.json')
                if receiptpath.exists():
                    receipt=json.loads(receiptpath.read_text());assert receipt['json_sha256']==sha(dest) and receipt['npz_sha256']==sha(dest.with_suffix('.npz')) and receipt['split_manifest_sha256']==sha(RUN/'manifests/splits.json') and receipt['amendment_sha256']==sha(RUN/'followup_analysis_amendment.md')
                else:raise RuntimeError('Completed split lacks a validated receipt; inspect artifacts before reuse: '+str(dest))
                saved=json.loads(dest.read_text());allrows+=saved['results'];pairs+=saved['pairs'];calibration+=saved['calibration'];audit.append(saved['audit']);toprows.append(saved['hub_reproducibility']);continue
            t=time.perf_counter();a=spec['train_indices'];b=spec['test_indices']
            assert set(spec['train']).isdisjoint(spec['test']) and df.columns[a].tolist()==spec['train'] and df.columns[b].tolist()==spec['test']
            if 'individual' in meta:assert set(meta.iloc[a].individual).isdisjoint(meta.iloc[b].individual)
            selection,_,_=select_features(df.iloc[:,a],500);x=selection.T.to_numpy();g=graph_from(spearman(x));u,v=g['edges'].T
            boot,bad,receipt=training_boot(x,g,acc,s,len(b))
            evaluable=np.isfinite(boot);den=evaluable.sum(axis=0);drop=((np.abs(boot)<g['summary']['theta'])&evaluable).sum(axis=0)
            with np.errstate(invalid='ignore',divide='ignore'):sd=np.nanstd(np.abs(boot),axis=0,ddof=1)
            trainkeep=(den>=20)&(sd>=1e-10)&np.isfinite(sd)
            designs,scales=design(g['margin'][trainkeep],g['ebc'][trainkeep],sd[trainkeep])
            endog=np.c_[drop[trainkeep],den[trainkeep]-drop[trainkeep]]
            # All learning precedes access to held-out expression/outcomes.
            fits={};predictions={}
            for name,X in designs.items():
                fit=sm.GLM(endog,X,family=sm.families.Binomial()).fit(maxiter=200)
                fits[name]=fit;predictions[name]=fit.predict(X)
            val=df.loc[selection.index].iloc[:,b].T.to_numpy();rt,valid=paircorr_partial(val)
            candidate_valid=valid[u]&valid[v];keep=candidate_valid[trainkeep]
            y=(np.abs(rt[u[trainkeep],v[trainkeep]])<g['summary']['theta'])[keep].astype(int)
            common={'cohort':acc,'split':s,'outcome':'direct_heldout_retained_edge_disappearance','train_n':len(a),'test_n':len(b),'features':len(selection),'candidate_edges':len(u),'training_eligible_edges':int(sum(trainkeep)),'evaluable_edges':int(sum(keep)),'training_excluded_edges':int(sum(~trainkeep)),'heldout_unevaluable_edges':int(sum(~candidate_valid)),'heldout_valid_features':int(sum(valid)),'attempted_B':250,'full_valid_B':receipt['full_valid_B'],'edge_min_B':int(den[trainkeep].min()),'edge_max_B':int(den[trainkeep].max()),'theta':g['summary']['theta'],'split_source_sha256':sha(RUN/'manifests/splits.json'),'run_id':RUN.name,'config_sha256':sha(RUN/'followup_analysis_amendment.md'),'unit':spec['groups']}
            rows=[];cals=[];paired=[]
            for name,fit in fits.items():
                p=predictions[name][keep];score=metrics(y,p)
                rows.append({**common,'model':name,**score,'converged':bool(fit.converged),'status':'completed' if fit.converged else 'failed','failure_reason':'' if fit.converged else 'GLM did not converge'})
                bins=np.minimum((p*10).astype(int),9)
                for j in range(10):
                    z=bins==j;cals.append({'cohort':acc,'split':s,'model':name,'bin':j,'lower':j/10,'upper':(j+1)/10,'n':int(sum(z)),'mean_prediction':p[z].mean() if any(z) else np.nan,'event_fraction':y[z].mean() if any(z) else np.nan})
            scores={r['model']:r for r in rows}
            for name in ['linear_distance','spline_distance','uncertainty_distance']:
                n=scores[name];e=scores[name+'_ebc']
                paired.append({**common,'model_pair':name,'brier_without':n['brier'],'brier_with':e['brier'],'brier_without_minus_with':n['brier']-e['brier'],'logloss_without_minus_with':n['logloss']-e['logloss'],'calibration_bias_without':n['calibration_bias'],'calibration_bias_with':e['calibration_bias'],'both_converged':n['converged'] and e['converged'],'events':n['events'],'nonevents':n['nonevents']})
            # Descriptive hard/soft node reproducibility on shared evaluable held-out nodes.
            from scipy.stats import spearmanr
            ids=np.flatnonzero(valid);rtrain=g['r'][np.ix_(ids,ids)];rtest=rt[np.ix_(ids,ids)]
            def hubs(r):
                w=np.abs(r).copy();np.fill_diagonal(w,0);return (w>=g['summary']['theta']).sum(axis=1),(w**6).sum(axis=1)
            hd,ss=hubs(rtrain);ht,st=hubs(rtest);top=lambda z:set(np.argsort(-z,kind='stable')[:20])
            hub={'cohort':acc,'split':s,'nodes':len(ids),'hard_rank_spearman':spearmanr(hd,ht).statistic,'soft_rank_spearman':spearmanr(ss,st).statistic,'hard_top20_overlap':len(top(hd)&top(ht))/20,'soft_top20_overlap':len(top(ss)&top(st))/20,'scope':'exploratory shared evaluable nodes; frozen train theta'}
            trace={'cohort':acc,'split':s,'train_test_overlap':0,'group_overlap':0 if 'individual' in meta else None,'learning_inputs':'train only; fit before heldout access','features':selection.index.tolist(),'train_sample_sha256':hashlib.sha256(json.dumps(spec['train']).encode()).hexdigest(),'test_sample_sha256':hashlib.sha256(json.dumps(spec['test']).encode()).hexdigest(),'train_cache':str(RUN/'cache'/f'{acc}_split{s:02d}_train.npz'),'runtime_seconds':time.perf_counter()-t,'full_failed_train_draws':int(bad.any(axis=1).sum())}
            data={'results':rows,'pairs':paired,'calibration':cals,'audit':trace,'hub_reproducibility':hub,'fits':{n:{'coef':f.params,'converged':f.converged} for n,f in fits.items()}}
            np.savez_compressed(dest.with_suffix('.npz'),edge_nodes=g['edges'][trainkeep][keep],y=y,**{n:p[keep] for n,p in predictions.items()},**{'design_'+n:X for n,X in designs.items()},training_keep=trainkeep,evaluable_keep=keep,train_denominator=den,train_drop_count=drop)
            write_json(dest,data)
            write_json(dest.with_suffix('.receipt.json'),{'json_sha256':sha(dest),'npz_sha256':sha(dest.with_suffix('.npz')),'split_manifest_sha256':sha(RUN/'manifests/splits.json'),'amendment_sha256':sha(RUN/'followup_analysis_amendment.md')})
            allrows+=rows;pairs+=paired;calibration+=cals;audit.append(trace);toprows.append(hub)
            print(acc,'split',s,'completed;',round(time.perf_counter()-t,2),'seconds',flush=True)
    csv('heldout_repeated_split_results',allrows);p=csv('heldout_paired_comparisons',pairs);csv('heldout_calibration',calibration);csv('heldout_internal_hub_reproducibility',toprows)
    write_json(RUN/'manifests/heldout_leakage_audit.json',audit)
    summary=[]
    for (acc,model),g in pd.DataFrame(pairs).groupby(['cohort','model_pair']):
        d=g.brier_without_minus_with
        summary.append({'cohort':acc,'model_pair':model,'splits':len(g),'mean_brier_without':g.brier_without.mean(),'mean_brier_with':g.brier_with.mean(),'mean_difference':d.mean(),'median_difference':d.median(),'min_difference':d.min(),'max_difference':d.max(),'q25_difference':d.quantile(.25),'q75_difference':d.quantile(.75),'positive_splits':int(sum(d>0)),'negative_splits':int(sum(d<0)),'mean_logloss_difference':g.logloss_without_minus_with.mean(),'meaning':'descriptive overlapping splits; positive favors EBC; no population CI'})
    csv('heldout_paired_summary',summary);status('heldout',start,splits=len(audit),model_results=len(allrows),total_recorded_split_seconds=sum(a['runtime_seconds'] for a in audit))
if __name__=='__main__':main()
