"""Production entry point. Saved tables are the only figure inputs."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import rankdata, spearmanr
from provenance import ROOT, BASE, RUN, sha, write_json, git
from core import cohort,select_features,spearman,graph_from,bootstrap,transformed,node_sums,cv,rng_for
from models import fit_all,logloss

CFG=json.loads((BASE/'configs/primary.json').read_text());CH=sha(BASE/'configs/primary.json')
def table(name,rows):
    path=RUN/'tables'/f'{name}.csv';tmp=path.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);os.replace(tmp,path)
def prepare(acc,k=500,scale='supplied',missing='complete',fraction=1):
    df,meta,raw=cohort(acc)
    if fraction<1:
        ids=rng_for(CFG['seed'],acc+'_subsample',int(fraction*100)).choice(len(meta),int(len(meta)*fraction),replace=False)
        df=df.iloc[:,np.sort(ids)];meta=meta.iloc[np.sort(ids)]
    selected,mad,frac=select_features(df,k,scale,missing)
    return df,meta,selected,selected.T.to_numpy(dtype=float)
def qc(acc):
    df,meta,selected,x=prepare(acc)
    finite=np.isfinite(df.to_numpy());complete=finite.all(axis=1);variable=df.max(axis=1)>df.min(axis=1)
    manifest=pd.DataFrame({'probe_id':df.index,'missing_fraction':1-finite.mean(axis=1),'zero_fraction':(df.to_numpy()==0).mean(axis=1),'unique_values':df.nunique(axis=1).values,'complete':complete,'variable':variable.values,'selected_primary':df.index.isin(selected.index)})
    manifest['tie_fraction']=1-manifest.unique_values/len(meta)
    manifest['exclusion_reason']=np.select([~manifest.complete,~manifest.variable,~manifest.selected_primary],['missing expression','constant','below MAD rank'],'selected')
    manifest.to_csv(BASE/'manifests'/f'{acc}_probes.csv',index=False)
    pd.DataFrame({'node':range(len(selected)),'probe_id':selected.index}).to_csv(BASE/'manifests'/f'{acc}_selected_probes.csv',index=False)
    samples=[]
    for id in df.columns:
        z=df[id].to_numpy();z=z[np.isfinite(z)]
        samples.append({'sample_id':id,'missing_fraction':1-len(z)/len(df),'minimum':min(z),'q25':np.quantile(z,.25),'median':np.median(z),'q75':np.quantile(z,.75),'maximum':max(z),'zero_fraction':np.mean(z==0)})
    table(acc+'_sample_qc',samples)
    r=spearman(x);u,v=np.triu_indices(len(selected),1)
    hist,edges=np.histogram(r[u,v],bins=np.linspace(-1,1,81));table(acc+'_correlation_hist',{'left':edges[:-1],'right':edges[1:],'count':hist})
    pca=np.linalg.svd((rankdata(x,axis=0)- (len(x)+1)/2)/np.std(rankdata(x,axis=0),axis=0),full_matrices=False)
    pc1=pca[0][:,0]*pca[1][0]
    s=pd.DataFrame(samples).set_index('sample_id');s['pc1']=pc1
    if acc=='GSE73002':s['stratum']=meta.title.str.split(':').str[0]
    else:s['stratum']=meta.organ
    s.to_csv(RUN/'tables'/f'{acc}_sample_pc1.csv')
    groups=[]
    for group,ids in s.groupby('stratum').groups.items():
        rows=[df.columns.get_loc(i) for i in ids]
        rr=spearman(x[rows]);groups.append({'stratum':group,'n':len(rows),'median_pair_correlation':np.median(rr[u,v]),'q95_pair_correlation':np.quantile(rr[u,v],.95)})
    table(acc+'_strata_qc',groups)
    return {'accession':acc,'selected_arrays':len(meta),'recorded_unique_subjects':meta.individual.nunique() if 'individual'in meta else None,'all_probes':len(df),'complete_probes':int(sum(complete)),'complete_variable_probes':int(sum(complete&variable)),'selected_probes':len(selected),'zero_fraction':float(np.mean(df.to_numpy()==0)),'missing_fraction':float(np.mean(~finite)),'duplicate_expression_samples':int(df.T.duplicated().sum()),'duplicate_expression_probes':int(df.duplicated().sum()),'pc1_variance_fraction':float(pca[1][0]**2/sum(pca[1]**2)),'pc1_vs_sample_median_spearman':float(spearmanr(pc1,s['median']).statistic),'median_pair_correlation':float(np.median(r[u,v]))}

def stability(acc,g,boot,powers=None):
    n=g['summary']['nodes'];B=len(boot);u,v=np.triu_indices(n,1)
    sign=g['summary']['sign'];score=transformed(boot,sign);theta=g['summary']['theta']
    base=transformed(g['r'],sign)[u,v];support=base>=theta
    hard=node_sums(score>=theta,n);hard_common=node_sums((score>=theta)*support,n)
    rows=[];nodes=[];biological=[]
    for beta in (powers or CFG['soft_powers']):
        weights=score**beta;soft=node_sums(weights,n);common=node_sums(weights*support,n)
        for mode,h,s in [('full',hard,soft),('common_support',hard_common,common)]:
            hcv,scv=cv(h),cv(s)
            for scope,ids in [('all_nodes',np.arange(n)),('gcc_focal',np.array(g['gcc']))]:
                valid=ids[np.isfinite(hcv[ids])&np.isfinite(scv[ids])]
                mh=float(np.mean(hcv[valid]));ms=float(np.mean(scv[valid]));reduction=1-ms/mh
                mc=[]
                for b in range(100):
                    draw=rng_for(CFG['seed'],f'{acc}_cv_mc',b).integers(0,B,B)
                    hc,sc=cv(h[draw][:,valid]),cv(s[draw][:,valid]);ok=np.isfinite(hc)&np.isfinite(sc)
                    mc.append(1-np.mean(sc[ok])/np.mean(hc[ok]))
                rows.append({'analysis':acc,'sign':sign,'beta':beta,'support':mode,'focal':scope,'valid_B':B,'nodes':len(ids),'valid_paired_nodes':len(valid),'undefined_nodes':len(ids)-len(valid),'mean_hard_cv':mh,'mean_soft_cv':ms,'median_hard_cv':np.median(hcv[valid]),'median_soft_cv':np.median(scv[valid]),'ratio_mean_reduction':reduction,'mean_nodewise_reduction':np.mean(1-scv[valid]/hcv[valid]),'reduction_mc_low':np.quantile(mc,.025),'reduction_mc_high':np.quantile(mc,.975),'interval_meaning':'complete-replicate-vector resampling; Monte Carlo only'})
            if beta==6:
                for i in range(n):nodes.append({'analysis':acc,'sign':sign,'support':mode,'node':i,'mean_hard':h[:,i].mean(),'mean_soft':s[:,i].mean(),'hard_cv':hcv[i],'soft_cv':scv[i]})
                if mode=='full':
                    np.savez_compressed(RUN/'models'/f'{acc}_{sign}_node_replicates.npz',hard=hard,soft=soft,hard_common=hard_common,soft_common=common)
                    bh=node_sums((base>=theta)[None,:],n)[0];bs=node_sums((base**6)[None,:],n)[0]
                    refh=set(np.argsort(-bh,kind='stable')[:20]);refs=set(np.argsort(-bs,kind='stable')[:20])
                    for b in range(B):
                        th=set(np.argsort(-h[b],kind='stable')[:20]);ts=set(np.argsort(-s[b],kind='stable')[:20])
                        biological.append({'replicate':b,'hard_top20_overlap':len(th&refh)/20,'soft_top20_overlap':len(ts&refs)/20,'hard_rank_spearman':spearmanr(bh,h[b]).statistic,'soft_rank_spearman':spearmanr(bs,s[b]).statistic})
                    freqh=np.zeros(n);freqs=np.zeros(n)
                    for b in range(B):freqh[np.argsort(-h[b],kind='stable')[:20]]+=1;freqs[np.argsort(-s[b],kind='stable')[:20]]+=1
                    table(f'{acc}_{sign}_hub_priorities',{'node':np.arange(n),'baseline_degree':bh,'baseline_strength':bs,'hard_top20_frequency':freqh/B,'soft_top20_frequency':freqs/B,'hard_tied_baseline':pd.Series(bh).duplicated(keep=False),'baseline_hard_top20':[i in refh for i in range(n)],'baseline_soft_top20':[i in refs for i in range(n)]})
    table(f'{acc}_{sign}_stability',rows);table(f'{acc}_{sign}_node_cv',nodes);table(f'{acc}_{sign}_hub_replicates',biological)
    return rows

def primary(acc):
    df,meta,selected,x=prepare(acc);r=spearman(x)
    boot,cache=bootstrap(x,CFG['bootstrap_replicates'],acc+'_primary',CFG['seed'],CH)
    summaries=[];coefs=[];u,v=np.triu_indices(len(r),1)
    for sign in ['unsigned','positive']:
        for density in CFG['densities']:
            g=graph_from(r,density,sign);label=f'{acc}_{sign}_d{density:g}'
            fitted=fit_all(g,boot,label)
            s=transformed(boot,sign);present=transformed(r,sign)[u,v]>=g['summary']['theta'];yp=(s>=g['summary']['theta']).mean(axis=0)
            unstable=fitted['y'].mean(axis=0)>.05
            summary={'analysis':label,'cohort':acc,'sample_count':len(x),'target_density':density,'valid_B':len(boot),**g['summary'],'mean_retained_dropout':float(np.mean(1-yp[present])),'mean_absent_appearance':float(np.mean(yp[~present])),'unstable_edges':int(sum(unstable)),'enrichment':float(np.mean(g['ebc'][unstable])/np.mean(g['ebc'][~unstable])) if any(unstable) and any(~unstable) else None}
            summaries.append(summary);coefs.extend(fitted['coefficients'])
            if density==CFG['primary_density']:
                table(label+'_pair_universe',{'node_a':u,'node_b':v,'initially_retained':present,'signed_margin':transformed(r,sign)[u,v]-g['summary']['theta'],'p_present':yp,'valid_B':len(boot)})
                stability(acc,g,boot)
                gg=graph_from(r,density,sign,scope='gcc');fg=fit_all(gg,boot,label+'_gcc');coefs.extend(fg['coefficients']);summaries.append({'analysis':label+'_gcc','cohort':acc,'sample_count':len(x),'target_density':density,'valid_B':len(boot),**gg['summary']})
        print(acc,sign,'density sweep complete',flush=True)
    table(acc+'_network_sensitivity',summaries);table(acc+'_all_coefficients',coefs)
    g=graph_from(r);y=(np.abs(boot[:,g['pair_indices']])<g['summary']['theta']);p=y.mean(axis=0);precision=[]
    for b in [100,250,500,1000]:
        pp=y[:b].mean(axis=0);precision.append({'B':b,'mean_mcse':np.mean(np.sqrt(pp*(1-pp)/b)),'max_mcse':np.max(np.sqrt(pp*(1-pp)/b)),'rms_difference_from_1000':np.sqrt(np.mean((pp-p)**2)),'interpretation':'numerical precision, not biological evidence'})
    table(acc+'_mc_precision',precision)
    return summaries

def sensitivity(acc):
    settings=[('features250',250,'supplied','complete',1),('features1000',1000,'supplied','complete',1),('sample50',500,'supplied','complete',.5),('sample75',500,'supplied','complete',.75)]
    if acc=='GSE115513':settings.append(('log1p_selection',500,'log1p','complete',1))
    else:settings.append(('median_imputation20',500,'supplied','median20',1))
    rows=[];coefs=[]
    base=prepare(acc)[2]
    for name,k,scale,missing,fraction in settings:
        df,meta,selected,x=prepare(acc,k,scale,missing,fraction);label=acc+'_'+name;r=spearman(x)
        g=graph_from(r);boot,_=bootstrap(x,CFG['sensitivity_replicates'],label,CFG['seed'],CH)
        model=fit_all(g,boot,label);coefs.extend(model['coefficients'])
        rows.append({'analysis':label,'samples':len(x),'requested_features':k,'selected_features':len(selected),'valid_B':len(boot),'feature_overlap_primary':len(set(selected.index)&set(base.index)),**g['summary']})
        if scale=='log1p':
            original=spearman(df.loc[selected.index].T.to_numpy());table(acc+'_rank_transform_check',[{'fixed_feature_max_abs_difference':np.max(np.abs(original-r)),'selected_overlap':len(set(selected.index)&set(base.index)),'selection_set_size':len(selected)}])
    table(acc+'_targeted_sensitivity',rows);table(acc+'_targeted_coefficients',coefs)
    return rows

def split_validation(acc):
    df,meta,_=cohort(acc);order=rng_for(CFG['seed'],acc+'_split',0).permutation(df.shape[1]);a,b=np.array_split(order,2)
    selection,_,_=select_features(df.iloc[:,a],500);ids=selection.index
    # Complete-case selection only uses reference half. Missing validation probes remain explicit exclusions.
    val=df.loc[ids].iloc[:,b];valid=val.notna().all(axis=1)
    if not valid.all():
        # No imputation; this changes the graph pair evaluation universe but never reference feature selection.
        raise ValueError(f'{int(sum(~valid))} reference-selected probes missing in validation; prespecified complete-case transfer blocked')
    x=selection.T.to_numpy();z=val.T.to_numpy();r=spearman(x);g=graph_from(r)
    train,_=bootstrap(x,CFG['split_replicates'],acc+'_split_reference',CFG['seed'],CH)
    test,_=bootstrap(z,CFG['split_replicates'],acc+'_split_validation',CFG['seed'],CH)
    fit=fit_all(g,train,acc+'_split_reference');y=(np.abs(test[:,g['pair_indices']])<g['summary']['theta']).astype(float)
    rows=[];losses={}
    for name,(model,design,keep,cov) in fit['objects'].items():
        pred=model.predict(design[keep]);loss=logloss(y[:,keep],pred);losses[name]=loss
        rows.append({'analysis':acc,'model':name,'reference_n':len(x),'validation_n':len(z),'valid_B':len(test),'validation_logloss':loss.mean(),'loss_mcse':loss.std(ddof=1)/np.sqrt(len(loss)),'meaning':'held-out arrays; intervals reflect bootstrap Monte Carlo error only'})
    contrasts=[]
    for name in ['linear_distance','spline_distance','uncertainty_distance']:
        change=losses[name+'_ebc']-losses[name];error=change.std(ddof=1)/np.sqrt(len(change))
        contrasts.append({'analysis':acc,'model':name,'ebc_minus_distance_logloss':change.mean(),'mc_low':change.mean()-1.96*error,'mc_high':change.mean()+1.96*error,'direction':'negative favors added EBC','uncertainty':'validation bootstrap MC only; reference-fit MC and finite-patient uncertainty not included'})
    table(acc+'_heldout_prediction',rows);table(acc+'_heldout_increment',contrasts)
    pd.DataFrame({'sample_id':df.columns,'split':['reference' if i in set(a) else 'validation' for i in range(df.shape[1])]}).to_csv(BASE/'manifests'/f'{acc}_split.csv',index=False)
    table(acc+'_split_features',{'probe_id':ids})
    return contrasts

def full_pipeline(acc):
    df,meta,_=cohort(acc);selected,_,_=select_features(df,500);g=graph_from(spearman(selected.T.to_numpy()));nodes=selected.index.to_numpy();u,v=g['edges'].T;base_edges=set(tuple(sorted((nodes[i],nodes[j]))) for i,j in zip(u,v));base_features=set(nodes)
    rows=[];start=time.perf_counter()
    for b in range(CFG['full_pipeline_replicates']):
        sample=rng_for(CFG['seed'],acc+'_full_pipeline',b).integers(0,len(meta),len(meta))
        x,_,_=select_features(df.iloc[:,sample],500);r=spearman(x.T.to_numpy());gg=graph_from(r);ids=x.index.to_numpy();edges=set(tuple(sorted((ids[i],ids[j]))) for i,j in gg['edges'])
        rows.append({'replicate':b,'selected_features':len(x),'feature_jaccard':len(set(ids)&base_features)/len(set(ids)|base_features),'edge_jaccard':len(edges&base_edges)/len(edges|base_edges),**gg['summary']})
        if (b+1)%20==0:print(acc,'full pipeline',b+1,'seconds',time.perf_counter()-start,flush=True)
    table(acc+'_full_pipeline',rows);return {'completed':len(rows),'seconds':time.perf_counter()-start}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['qc','primary','sensitivity','split','full']);parser.add_argument('--cohort',default='both');args=parser.parse_args()
    cohorts=['GSE115513','GSE73002'] if args.cohort=='both' else [args.cohort]
    status_file=RUN/'logs'/f'status_{args.stage}_{args.cohort}.json';states={};start=time.perf_counter()
    for acc in cohorts:
        states[acc]={'status':'running'};write_json(status_file,states)
        try:
            result={'qc':qc,'primary':primary,'sensitivity':sensitivity,'split':split_validation,'full':full_pipeline}[args.stage](acc)
            states[acc]={'status':'completed','result':result,'elapsed_seconds':time.perf_counter()-start}
        except Exception as e:
            traceback.print_exc();states[acc]={'status':'failed','error':repr(e),'traceback':traceback.format_exc()}
        write_json(status_file,states)
    print(json.dumps(states,default=str),flush=True)
    if any(x['status']=='failed' for x in states.values()):sys.exit(1)
if __name__=='__main__':main()
