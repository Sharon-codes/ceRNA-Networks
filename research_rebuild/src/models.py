"""Conditional descriptive binomial fits with replicate-score Monte Carlo covariance."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, xlogy
from scipy.stats import norm
from patsy import dmatrix, build_design_matrices
from provenance import RUN, write_json

def standardized(x):
    mean=float(np.mean(x));sd=float(np.std(x))
    if sd<=1e-12: raise ValueError('Predictor has no support')
    return (x-mean)/sd,{'mean':mean,'sd':sd}

def design(margin,ebc,se):
    zdist,ds=standardized(margin);ze,es=standardized(ebc)
    med=float(np.median(ebc[ebc>0]));logebc=np.log1p(ebc/med)
    zel,els=standardized(logebc);els['median_positive_ebc']=med
    zmargin=margin/np.maximum(se,1e-10)
    b=dmatrix('cr(m, df=4, constraints="center")',{'m':margin},return_type='dataframe')
    bu=dmatrix('cr(m, df=4, constraints="center")',{'m':zmargin},return_type='dataframe')
    models={'linear_distance':np.column_stack([np.ones(len(margin)),zdist]),'linear_distance_ebc':np.column_stack([np.ones(len(margin)),zdist,ze]),'spline_distance':np.asarray(b),'spline_distance_ebc':np.column_stack([b,zel]),'uncertainty_distance':np.asarray(bu),'uncertainty_distance_ebc':np.column_stack([bu,zel])}
    return models,{'distance':ds,'raw_ebc':es,'log_ebc':els,'uncertainty_floor':1e-10,'spline_distance_info':b.design_info,'spline_uncertainty_info':bu.design_info}

def fit_one(x,y):
    B=len(y);endog=np.column_stack([y.sum(axis=0),B-y.sum(axis=0)])
    model=sm.GLM(endog,x,family=sm.families.Binomial()).fit(maxiter=200)
    pred=model.predict(x)
    # M-estimator of a mean over independent complete patient-resample vectors.
    # This is simulation error, NOT finite-patient population uncertainty.
    scores=(y-pred)@x
    bread=np.linalg.pinv(x.T@((pred*(1-pred))[:,None]*x))
    cov=bread@np.atleast_2d(np.cov(scores,rowvar=False,ddof=1))@bread/B
    se=np.sqrt(np.maximum(np.diag(cov),0))
    return model,pred,se,cov

def logloss(y,p):
    p=np.clip(p,1e-12,1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log1p(-p)).mean(axis=1)

def fit_all(g,boot,label,save=True):
    idx=g['pair_indices'];s=np.abs(boot) if g['summary']['sign']=='unsigned' else boot
    y=(s[:,idx]<g['summary']['theta']).astype(float)
    se=s[:,idx].std(axis=0,ddof=1)
    designs,scales=design(g['margin'],g['ebc'],se)
    coefficients=[];diagnostics=[];predictions={};objects={};effects=[]
    for name,x in designs.items():
        keep=se>=1e-10 if name.startswith('uncertainty') else np.ones(len(se),bool)
        model,pred,mcse,cov=fit_one(x[keep],y[:,keep]);objects[name]=(model,x,keep,cov);predictions[name]=pred
        for j,b in enumerate(model.params):
            term='intercept' if j==0 else ('ebc' if name.endswith('_ebc') and j==len(model.params)-1 else f'distance_basis_{j}')
            coefficients.append({'analysis':label,'model':name,'term':term,'coefficient':b,'mc_se':mcse[j],'mc_ci_low':b-1.96*mcse[j],'mc_ci_high':b+1.96*mcse[j],'conditional_OR':np.exp(np.clip(b,-700,700)),'interval_meaning':'95% Monte Carlo normal approximation; NOT population CI','valid_B':len(y),'edges':int(sum(keep)),'converged':model.converged})
        diagnostics.append({'analysis':label,'model':name,'edges':sum(keep),'valid_B':len(y),'converged':model.converged,'deviance':model.deviance,'pearson_over_df':model.pearson_chi2/model.df_resid,'in_sample_logloss':logloss(y[:,keep],pred).mean(),'probability_rmse':np.sqrt(np.mean((pred-y[:,keep].mean(axis=0))**2)),'near_zero_sd_exclusions':int(sum(~keep))})
        if save:
            write_json(RUN/'models'/f'{label}_{name}.json',{'coefficients':model.params.tolist(),'mc_covariance':cov.tolist(),'design_matrix_file':f'{label}_designs.npz','converged':model.converged,'interval_meaning':'Monte Carlo only'})
    if save:
        np.savez_compressed(RUN/'models'/f'{label}_designs.npz',**designs)
        clean={k:v for k,v in scales.items() if not k.endswith('_info')}
        write_json(RUN/'models'/f'{label}_scaling.json',clean)
        pd.DataFrame(coefficients).to_csv(RUN/'tables'/f'{label}_coefficients.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(RUN/'tables'/f'{label}_diagnostics.csv',index=False)
        table=pd.DataFrame({'node_a':g['edges'][:,0],'node_b':g['edges'][:,1],'margin':g['margin'],'ebc':g['ebc'],'actual_bridge':g['bridge'],'correlation_sd':se,'drop_count':y.sum(axis=0),'valid_B':len(y),'p_drop':y.mean(axis=0),'mcse':np.sqrt(y.mean(axis=0)*(1-y.mean(axis=0))/len(y))})
        table['zero_event_mc_upper95']=np.where(table.drop_count==0,1-.05**(1/len(y)),np.nan)
        for name,pred in predictions.items():
            keep=objects[name][2];table.loc[keep,'pred_'+name]=pred
        table.to_csv(RUN/'tables'/f'{label}_edges.csv',index=False)
        bins=pd.qcut(table.margin,10,duplicates='drop')
        name='spline_distance_ebc';model,x,keep,cov=objects[name]
        for b,part in table.groupby(bins,observed=True):
            inds=part.index.to_numpy();q1,q3=np.quantile(x[inds,-1],[.25,.75]);base=x[inds].mean(axis=0)
            low=base.copy();high=base.copy();low[-1]=q1;high[-1]=q3
            plo=expit(low@model.params);phi=expit(high@model.params)
            grad=phi*(1-phi)*high-plo*(1-plo)*low
            err=np.sqrt(max(grad@cov@grad,0))
            effects.append({'analysis':label,'margin_mean':part.margin.mean(),'edges':len(part),'ebc_q25':part.ebc.quantile(.25),'ebc_q75':part.ebc.quantile(.75),'predicted_low':plo,'predicted_high':phi,'probability_contrast':phi-plo,'mc_ci_low':phi-plo-1.96*err,'mc_ci_high':phi-plo+1.96*err,'content':'MODEL-PREDICTED within observed distance bins; local interpolation'})
        pd.DataFrame(effects).to_csv(RUN/'tables'/f'{label}_effects.csv',index=False)
    return {'coefficients':coefficients,'diagnostics':diagnostics,'objects':objects,'y':y,'designs':designs,'scales':scales,'sd':se}

def predict_designs(margin,ebc,se,scales):
    # For validation, fixed reference edges retain their reference graph predictors.
    raise NotImplementedError('Use saved reference design for conditional validation; no refitting on held-out data')
