from base import *
from scipy.special import expit

def score(y,p):
    pc=np.clip(p,1e-12,1-1e-12);return {'brier':np.mean((y-p)**2),'logloss':-np.mean(y*np.log(pc)+(1-y)*np.log1p(-pc)),'mean_prediction':np.mean(p),'prevalence':np.mean(y),'calibration_bias':np.mean(p)-np.mean(y),'clipped_fraction':np.mean(pc!=p)}
def main():
    start=time.perf_counter();rows=[];cal=[];evaluation=[];excluded=[];paired=[];reconciliation=[]
    old=pd.read_csv(PREV/'tables/heldout_repeated_split_results.csv')
    for acc in ACCS:
        df,meta=cohort(acc);splits=json.loads((PREV/'manifests/splits.json').read_text())
        for spec in [x for x in splits if x['cohort']==acc]:
            s=spec['split'];p=PREV/'models'/f'{acc}_split{s:02d}.npz';z=np.load(p);info=json.loads(p.with_suffix('.json').read_text());keep=z['evaluable_keep'];trainkeep=z['training_keep'];assert trainkeep.all();y=z['y'];den=z['train_denominator'];drop=z['train_drop_count'];intercept=drop[trainkeep].sum()/den[trainkeep].sum();models={'constant_0.5':np.full(len(keep),.5),'train_intercept':np.full(len(keep),intercept)}
            for name,fit in info['fits'].items():models[name]=expit(z['design_'+name]@np.array(fit['coef']))
            v=df.loc[info['audit']['features'],spec['test']];missing=~np.isfinite(v).all(axis=1);constant=(v.max(axis=1)<=v.min(axis=1))&~missing
            for name,fullpred in models.items():
                pred=fullpred[keep];sc=score(y,pred)
                if name=='constant_0.5':assert sc['brier']==.25
                if name in z.files:
                    err=np.max(np.abs(pred-z[name]));assert err<1e-10
                    prev=old[old.cohort.eq(acc)&old.split.eq(s)&old.model.eq(name)].iloc[0]
                    reconciliation.append({'cohort':acc,'split':s,'model':name,'quantity':'Brier','reported':prev.brier,'recomputed':sc['brier'],'absolute_difference':abs(sc['brier']-prev.brier),'verification':'saved coefficients/designs -> expit predictions -> direct saved outcome squared errors','source':str(p),'sha256':sha(p),'status':'reproduced_from_cache'})
                unk=fullpred[~keep];obs=((pred-y)**2).sum();lo=(obs+np.minimum(unk**2,(1-unk)**2).sum())/len(keep);hi=(obs+np.maximum(unk**2,(1-unk)**2).sum())/len(keep)
                rows.append({'cohort':acc,'split':s,'model':name,'edges':len(y),'candidate_edges':len(keep),'events':int(y.sum()),'nonevents':len(y)-int(y.sum()),'train_intercept':intercept,**sc,'full_universe_Brier_lower':lo,'full_universe_Brier_upper':hi,'bound_meaning':'sharp per-edge binary-label identification bounds given fixed predictions; not a population CI','status':'completed'})
                bins=np.minimum((pred*10).astype(int),9)
                for b in range(10):
                    ok=bins==b;cal.append({'cohort':acc,'split':s,'model':name,'bin':b,'n':int(ok.sum()),'mean_prediction':pred[ok].mean() if ok.any() else np.nan,'observed':y[ok].mean() if ok.any() else np.nan})
            evaluation.append({'cohort':acc,'split':s,'train_arrays':len(spec['train']),'test_arrays':len(spec['test']),'candidate_nodes':len(v),'evaluable_nodes':int((~missing&~constant).sum()),'missing_nodes':int(missing.sum()),'constant_nodes':int(constant.sum()),'candidate_edges':len(keep),'evaluable_edges':int(keep.sum()),'excluded_edges':int((~keep).sum()),'events':int(y.sum()),'nonevents':len(y)-int(y.sum()),'scope':'evaluable subset; no excluded-edge outcomes imputed'})
            X=z['design_linear_distance_ebc'];trainboot=np.load(PREV/'cache'/f'{acc}_split{s:02d}_train.npz')['r'];pdrop=drop/den;sd=np.std(np.abs(trainboot),axis=0,ddof=1)
            for group,mask in [('evaluable',keep),('excluded',~keep)]:
                excluded.append({'cohort':acc,'split':s,'group':group,'edges':int(mask.sum()),'mean_standardized_train_margin':X[mask,1].mean() if mask.any() else np.nan,'mean_standardized_train_EBC':X[mask,2].mean() if mask.any() else np.nan,'mean_train_dropout':pdrop[mask].mean() if mask.any() else np.nan,'mean_train_correlation_SD':sd[mask].mean() if mask.any() else np.nan,'interpretation':'observable training characteristics only; cannot recover excluded outcomes'})
        print(acc,'predictive baselines verified',flush=True)
    df=pd.DataFrame(rows);csv('predictive_baselines',df);csv('calibration',cal);csv('evaluability',evaluation);csv('excluded_edge_training_characteristics',excluded);csv('prediction_reconciliation',reconciliation)
    for (acc,s),g in df.groupby(['cohort','split']):
        g=g.set_index('model')
        for name in ['linear_distance','spline_distance','uncertainty_distance']:
            paired.append({'cohort':acc,'split':s,'comparison':name+' vs '+name+'_ebc','brier_without_minus_with':g.loc[name,'brier']-g.loc[name+'_ebc','brier'],'logloss_without_minus_with':g.loc[name,'logloss']-g.loc[name+'_ebc','logloss']})
    csv('predictive_paired_differences',paired);csv('predictive_summary',df.groupby(['cohort','model'],as_index=False).agg(mean_Brier=('brier','mean'),min_Brier=('brier','min'),max_Brier=('brier','max'),mean_logloss=('logloss','mean'),mean_prediction=('mean_prediction','mean'),prevalence=('prevalence','mean'),calibration_bias=('calibration_bias','mean'),clipping_fraction=('clipped_fraction','mean'),mean_full_Brier_lower=('full_universe_Brier_lower','mean'),mean_full_Brier_upper=('full_universe_Brier_upper','mean')))
    status('prediction_baselines',start,splits=40,model_rows=len(rows))
if __name__=='__main__':main()
