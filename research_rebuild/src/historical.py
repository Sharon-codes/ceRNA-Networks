"""Isolated execution of historical numerical specification, not historical plotting."""
import json,time,traceback
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import median_abs_deviation
from provenance import BASE,ROOT,RUN,sha,write_json,read_geo
from core import spearman,graph_from,node_sums,cv,rng_for

def reproduce(acc,theta):
    t=time.perf_counter();df,meta,_=read_geo(ROOT/f'{acc}_series_matrix.txt.gz')
    mask=meta.tissue.str.contains('carcinoma',case=False) if acc=='GSE115513' else meta.diagnosis.str.contains('breast cancer',case=False)
    selected=df.loc[:,mask];selected=selected.loc[selected.isna().mean(axis=1)<=.2].dropna()
    if (selected.to_numpy()>50).any():selected=np.log2(selected+1)
    ids=pd.Series(median_abs_deviation(selected,axis=1),index=selected.index).nlargest(500).index
    x=selected.loc[ids].T.to_numpy();r=spearman(x);g=graph_from(r,sign='positive',theta=theta,scope='gcc')
    edges=g['edges'];rng=np.random.RandomState(42);B=1000;drop=np.zeros((B,len(edges)),dtype=np.uint8);hard=[];soft=[]
    for b in range(B):
        z=spearman(x[rng.choice(len(x),size=len(x),replace=True)])
        a=z>=theta;np.fill_diagonal(a,False);w=np.abs(z)**6;np.fill_diagonal(w,0)
        drop[b]=(z[edges[:,0],edges[:,1]]<theta);hard.append(a[g['gcc']].sum(axis=1));soft.append(w[g['gcc']].sum(axis=1))
        if (b+1)%100==0:print(acc,'HISTORICAL',b+1,'seconds',time.perf_counter()-t,flush=True)
    hard=np.array(hard);soft=np.array(soft);distance=np.abs(np.abs(r[edges[:,0],edges[:,1]])-theta)
    zd=(distance-distance.mean())/(distance.std()+1e-12);ze=(g['ebc']-g['ebc'].mean())/(g['ebc'].std()+1e-12)
    exog=pd.DataFrame({'const':1.,'z_Distance':zd,'z_EBC':ze});endog=np.c_[drop.sum(axis=0),B-drop.sum(axis=0)]
    fits=[];models={}
    for cov in ['nonrobust','HC0','Node_A']:
        kwargs={'cov_type':'cluster','cov_kwds':{'groups':edges[:,0]}} if cov=='Node_A' else {'cov_type':cov}
        model=sm.GLM(endog,exog,family=sm.families.Binomial()).fit(**kwargs);models[cov]=model
        for j,name in enumerate(exog):
            ci=model.conf_int().iloc[j]
            fits.append({'cohort':acc,'covariance':cov,'term':name,'coefficient':model.params.iloc[j],'se':model.bse.iloc[j],'z':model.tvalues.iloc[j],'p':model.pvalues.iloc[j],'OR':np.exp(model.params.iloc[j]),'OR_low':np.exp(ci.iloc[0]),'OR_high':np.exp(ci.iloc[1]),'deviance':model.deviance,'pearson_over_df':model.pearson_chi2/model.df_resid,'label':'HISTORICAL_REPRODUCTION; not valid population inference'})
    relabel=[]
    for b in range(100):
        order=rng_for(20260907,acc+'_historical_relabel',b).permutation(len(r));groups=np.min(order[edges],axis=1)
        m=sm.GLM(endog,exog,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':groups})
        relabel.append({'replicate':b,'ebc_coef':m.params.z_EBC,'ebc_se':m.bse.z_EBC,'ebc_p':m.pvalues.z_EBC,'label':'HISTORICAL one-endpoint clustering under node permutation'})
    out=RUN/'historical'/acc;out.mkdir(exist_ok=True)
    pd.DataFrame(fits).to_csv(out/'glm.csv',index=False);pd.DataFrame(relabel).to_csv(out/'node_order_sensitivity.csv',index=False)
    pd.DataFrame({'probe_id':ids}).to_csv(out/'selected_probes.csv',index=False)
    pd.DataFrame({'node_a':edges[:,0],'node_b':edges[:,1],'z_distance':zd,'z_ebc':ze,'drop_count':drop.sum(axis=0)}).to_csv(out/'edges.csv',index=False)
    np.savez_compressed(out/'replicates.npz',drop=drop,hard=hard,soft=soft)
    summary={'cohort':acc,'n':len(x),'B':B,**g['summary'],'hard_mean_cv':float(np.mean(cv(hard,ddof=0))),'soft_mean_cv':float(np.mean(cv(soft,ddof=0))),'reduction':float(1-np.mean(cv(soft,ddof=0))/np.mean(cv(hard,ddof=0))),'seconds':time.perf_counter()-t,'source_specification':'generate_final_sensitivity_analysis.py at 4a70722; RandomState(42), positive hard, unsigned soft, log selection, GCC focal, all neighbors','input_sha256':sha(ROOT/f'{acc}_series_matrix.txt.gz')}
    write_json(out/'summary.json',summary);return summary
if __name__=='__main__':
    results={}
    for acc,theta in [('GSE115513',.819),('GSE73002',.958)]:
        try:results[acc]={'status':'completed','result':reproduce(acc,theta)}
        except Exception as e:traceback.print_exc();results[acc]={'status':'failed','error':repr(e)}
        write_json(RUN/'logs/status_historical.json',results)
