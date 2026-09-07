"""Figure-only rebuild. Read saved tables; never rerun a graph/bootstrap/model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from provenance import RUN,BASE,sha,write_json
T=RUN/'tables';F=RUN/'figures';R=RUN/'reports'
COLORS=['#176b91','#ba5337'];COHORTS=['GSE115513','GSE73002'];LABELS=['Colorectal tissue','Breast-cancer serum']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
records=[]
def read(name):return pd.read_csv(T/(name+'.csv'))
def save(fig,name,sources,caption):
    fig.tight_layout(pad=1.7);fig.savefig(F/(name+'.png'),dpi=300);fig.savefig(F/(name+'.svg'));plt.close(fig)
    records.append({'figure':name,'png':str(F/(name+'.png')),'svg':str(F/(name+'.svg')),'source_tables':[{'path':str(T/(s+'.csv')),'sha256':sha(T/(s+'.csv'))} for s in sources],'plotting_command':'python research_rebuild/src/figures.py','config_sha256':sha(BASE/'configs/primary.json'),'run_id':'20260907_rebuild_v1','caption':caption})

def main():
    q=read('cohort_summary');fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for i,(acc,label,color) in enumerate(zip(COHORTS,LABELS,COLORS)):
        row=q[q.accession.eq(acc)].iloc[0];axes[0].bar(i,row.selected_arrays,color=color);axes[0].text(i,row.selected_arrays+25,str(row.selected_arrays),ha='center')
        hist=read(acc+'_correlation_hist');axes[1].plot((hist.left+hist.right)/2,hist['count']/hist['count'].sum(),label=label,color=color)
        axes[2].bar(i-.15,row.complete_variable_probes,width=.3,color=color,alpha=.4);axes[2].bar(i+.15,row.selected_probes,width=.3,color=color)
    axes[0].set(xticks=[0,1],xticklabels=['Tissue','Serum'],ylabel='Selected arrays',ylim=(0,1450),title='EMPIRICAL cohort flow')
    axes[1].set(xlabel='Spearman correlation',ylabel='Fraction of unique pairs',title='Selected-probe correlations');axes[1].legend(frameon=False,fontsize=8)
    axes[2].set(xticks=[0,1],xticklabels=['Tissue','Serum'],ylabel='Probes',title='Complete variable / selected probes')
    save(fig,'fig01_qc',['cohort_summary']+[a+'_correlation_hist' for a in COHORTS],'EMPIRICAL. Exact GEO metadata selects 750 tissue and 1280 serum arrays. Correlation histograms use the 124750 unordered pairs among 500 selected probes. Pale bars show complete variable probes; solid bars show selected probes. Serum array independence is assumed. No intervals.')
    fig,axes=plt.subplots(2,2,figsize=(11,8));sources=[]
    for i,(a,label,c) in enumerate(zip(COHORTS,LABELS,COLORS)):
        name=a+'_unsigned_d0.025_edges';e=read(name);sources.append(name)
        axes[i,0].scatter(e.margin,e.p_drop,s=5,color=c,alpha=.25,rasterized=True)
        axes[i,1].scatter(e.margin/e.correlation_sd,e.p_drop,s=5,color=c,alpha=.25,rasterized=True)
        axes[i,0].set(xlabel='Retention margin |r| - theta',ylabel='Bootstrap dropout probability',title=label+' (EMPIRICAL)')
        axes[i,1].set(xlabel='Margin / bootstrap correlation SD',ylabel='Bootstrap dropout probability',xlim=(0,min(10,np.nanmax(e.margin/e.correlation_sd))),title='Uncertainty-scaled margin')
        for ax in axes[i]:ax.set_ylim(-.02,1.02)
    save(fig,'fig02_dropout',sources,'EMPIRICAL. Each point is a retained primary full-graph edge; dropout is its failure fraction over 1000 independent patient/array resamples, reranked within each draw. Right panels display uncertainty-scaled margin and restrict the displayed x range to <=10; complete values remain in source tables. SD is bootstrap Spearman-correlation variability; it is not a population interval for dropout. No fitted curve or invented trajectory.')
    e=read('primary_ebc_models');fig,axes=plt.subplots(1,2,figsize=(11,4.7));names=['linear_distance_ebc','spline_distance_ebc','uncertainty_distance_ebc'];labels=['Linear distance + raw EBC','Spline distance + log EBC','Scaled-margin spline + log EBC']
    for i,(a,label,c) in enumerate(zip(COHORTS,LABELS,COLORS)):
        z=e[e.analysis.eq(a+'_unsigned_d0.025')].set_index('model').loc[names]
        axes[i].errorbar(z.coefficient,np.arange(3),xerr=[z.coefficient-z.mc_ci_low,z.mc_ci_high-z.coefficient],fmt='o',color=c,capsize=4)
        axes[i].axvline(0,color='.5',ls='--',lw=1);axes[i].set(yticks=np.arange(3),yticklabels=labels,title=label,xlabel='Conditional coefficient (per predictor SD)')
    save(fig,'fig03_adjusted_ebc',['primary_ebc_models'],'MODEL-PREDICTED conditional associations. Primary unsigned, 2.5% graphs; 1000 valid resamples per cohort. Bars are 95% Monte Carlo normal intervals from whole-replicate scores, NOT population confidence intervals. Linear model uses raw-EBC SD; spline models use log1p(EBC/median-positive-EBC) SD. The uncertainty-scaled model holds estimated correlation SD/basis fixed and omits MC error in that predictor. Effect units differ across transformations and cohorts; no equivalence claim.')
    fig,axes=plt.subplots(2,2,figsize=(11,8));sources=[]
    for i,(a,label,c) in enumerate(zip(COHORTS,LABELS,COLORS)):
        net=read(a+'_network_sensitivity');co=read(a+'_all_coefficients');sources.extend([a+'_network_sensitivity',a+'_all_coefficients'])
        n=net[net.sign.eq('unsigned')&net.scope.eq('full')].sort_values('target_density')
        axes[i,0].plot(n.full_density*100,n.enrichment,'o-',color=c);axes[i,0].set(xlabel='Achieved full-graph density (%)',ylabel='Mean EBC unstable / stable',title=label+' (EMPIRICAL)')
        z=n.merge(co[co.model.eq('spline_distance_ebc')&co.term.eq('ebc')],on='analysis')
        axes[i,1].errorbar(z.full_density*100,z.coefficient,yerr=[z.coefficient-z.mc_ci_low,z.mc_ci_high-z.coefficient],fmt='o-',color=c,capsize=4)
        axes[i,1].axhline(0,ls='--',lw=1,color='.5');axes[i,1].set(xlabel='Achieved full-graph density (%)',ylabel='Adjusted log-EBC coefficient per SD',title='MODEL-PREDICTED adjustment')
    save(fig,'fig04_density_sensitivity',sources,'Left: EMPIRICAL enrichment, unstable defined as conditional dropout >0.05. Right: MODEL-PREDICTED spline-distance-adjusted log-EBC coefficient with 95% Monte Carlo intervals. Graphs, EBC and responses were computed at each of the four density targets using shared 1000 resamples within each cohort. Lines only connect computed settings; no formula-driven empirical trajectory. Positive-sign and GCC sensitivities are in source tables.')
    s=read('primary_stability');fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for i,(a,label,c) in enumerate(zip(COHORTS,LABELS,COLORS)):
        z=s[s.analysis.eq(a)&s.focal.eq('gcc_focal')].set_index('support').loc[['full','common_support']];x=np.arange(2)
        axes[i].bar(x-.18,z.mean_hard_cv,.36,color='#ba5337',label='Hard degree');axes[i].bar(x+.18,z.mean_soft_cv,.36,color='#176b91',label='Soft strength')
        axes[i].set(xticks=x,xticklabels=['Full networks','Common edge support'],ylabel='Mean node CV (SD / mean)',title=label+' (EMPIRICAL)');axes[i].legend(frameon=False)
    save(fig,'fig05_hard_soft',['primary_stability'],'EMPIRICAL. Same 1000 resamples, unsigned sign, beta=6, and fixed baseline GCC focal nodes; neighbors range over all 500 selected nodes. Full networks compare hard degree against all-pair soft strength; common support retains only baseline hard-edge pairs for both measures. CV uses sample SD/mean. Graph-isolate/undefined-CV accounting and all-node results are in the table. Lower CV is not evidence of biological accuracy; no population significance bars.')
    sim=read('simulation_summary');known=sim[sim.known_target.eq(True)].copy();fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    x=np.arange(len(known));axes[0].errorbar(x,known.mean_interval_coverage,yerr=1.96*known.mean_interval_coverage_dataset_mcse,fmt='o',capsize=4,color='#176b91');axes[0].axhline(.95,color='.5',ls='--');axes[0].set(xticks=x,xticklabels=known.scenario,ylim=(.75,1),ylabel='Mean pairwise interval coverage',title='SIMULATED Spearman coverage')
    axes[1].errorbar(x,known.null_reject_rate,yerr=[known.null_reject_rate-known.null_reject_wilson_low,known.null_reject_wilson_high-known.null_reject_rate],fmt='o',capsize=4,color='#ba5337');axes[1].axhline(.05,color='.5',ls='--');axes[1].set(xticks=x,xticklabels=known.scenario,ylim=(0,.25),ylabel='Rejection rate at true correlation zero',title='Pairwise correlation null; not EBC null')
    for ax in axes:ax.tick_params(axis='x',rotation=35)
    save(fig,'fig06_simulation_calibration',['simulation_summary'],'SIMULATED expression-level PSD latent-factor models. 200 independent datasets and 200 bootstrap draws per dataset per scenario. Left: percentile Spearman-correlation interval coverage against a known Gaussian target; bars are normal Monte Carlo uncertainty across independent dataset means. Right: null rejection for a prespecified independent-node pair with Wilson intervals across eligible datasets. Ties/outliers/batch scenarios have no validated Gaussian target and are excluded from coverage panels but retained in source results. This does NOT calibrate adjusted EBC population inference.')
    ext=read('external_biological_summary');fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    axes[0].bar(['Hard degree','Soft strength'],[ext.hard_top20_overlap.iloc[0],ext.soft_top20_overlap.iloc[0]],color=['#ba5337','#176b91']);axes[0].set(ylim=(0,1),ylabel='Fraction of discovery top 20 retained',title='EMPIRICAL external hub overlap')
    axes[1].bar(['Hard degree','Soft strength'],[ext.hard_rank_spearman.iloc[0],ext.soft_rank_spearman.iloc[0]],color=['#ba5337','#176b91']);axes[1].set(ylim=(0,1),ylabel='Spearman correlation of node rankings',title='External ranking agreement')
    save(fig,'fig07_external_biology',['external_biological_summary'],'EMPIRICAL cross-platform colorectal transfer. GSE115513 discovery restricted to 204 uniquely mapped discovery-selected mature miRNAs; GSE41655 includes 33 adenocarcinoma arrays. Cutoffs target 2.5% density independently on the fixed mapped node universe; unsigned weights use beta6. Top-20 ties are broken by fixed node order. No population intervals or claim of superior biological recovery. Patient identity is not globally linked; independent arrays are assumed.')
    pred=read('heldout_comparison')
    if len(pred):
        fig,axes=plt.subplots(1,2,figsize=(11,4.8))
        for i,(a,label,c) in enumerate(zip(COHORTS,LABELS,COLORS)):
            z=pred[pred.analysis.eq(a)];axes[i].errorbar(z.ebc_minus_distance_logloss,np.arange(len(z)),xerr=[z.ebc_minus_distance_logloss-z.mc_low,z.mc_high-z.ebc_minus_distance_logloss],fmt='o',color=c,capsize=4);axes[i].axvline(0,color='.5',ls='--');axes[i].set(yticks=np.arange(len(z)),yticklabels=z.model.str.replace('_',' '),xlabel='Added EBC minus distance-only log loss',title=label);axes[i].ticklabel_format(axis='x',style='sci',scilimits=(0,0));axes[i].xaxis.set_major_locator(MaxNLocator(4))
        save(fig,'fig08_heldout_prediction',['heldout_comparison'],'EMPIRICAL held-out evaluation of MODEL-PREDICTED dropout. Disjoint reference/validation halves, 250 bootstrap draws each; reference-selected features, reference EBC and cutoff frozen. Negative loss differences favor adding EBC. Intervals quantify validation bootstrap Monte Carlo error only; they omit finite-patient population uncertainty and reference-fit uncertainty. This is internal validation, not an independent-cohort effect estimate.')
    fig,axes=plt.subplots(2,2,figsize=(10,8));sources=[]
    for i,(a,label,c) in enumerate(zip(COHORTS,LABELS,COLORS)):
        e=read(a+'_unsigned_d0.025_edges');sources.append(a+'_unsigned_d0.025_edges');p=e.pred_uncertainty_distance_ebc
        axes[i,0].scatter(p,e.p_drop,s=6,alpha=.2,color=c,rasterized=True);axes[i,0].plot([0,.6],[0,.6],ls='--',color='.5');axes[i,0].set(xlabel='Fitted dropout probability',ylabel='Observed bootstrap fraction',title=label+' calibration')
        axes[i,1].scatter(e.margin,e.p_drop-p,s=6,alpha=.2,color=c,rasterized=True);axes[i,1].axhline(0,color='.5',ls='--');axes[i,1].set(xlabel='Retention margin',ylabel='Observed minus fitted probability',title='In-sample residuals')
    save(fig,'fig09_calibration_residuals',sources,'EMPIRICAL observed conditional fractions versus MODEL-PREDICTED probabilities from uncertainty-scaled-margin spline plus transformed EBC. Primary full unsigned graphs; 1000 resamples per cohort. Calibration and residuals are in-sample descriptive diagnostics; points are dependent edges and no independent-edge uncertainty bands are shown.')
    fig,axes=plt.subplots(1,2,figsize=(10,4.6));sources=[]
    for i,(a,label,c) in enumerate(zip(COHORTS,LABELS,COLORS)):
        name=a+'_unsigned_d0.025_effects';e=read(name);sources.append(name)
        axes[i].errorbar(e.margin_mean,e.probability_contrast,yerr=[e.probability_contrast-e.mc_ci_low,e.mc_ci_high-e.probability_contrast],fmt='o-',color=c,capsize=3);axes[i].axhline(0,color='.5',ls='--');axes[i].set(xlabel='Mean observed margin within bin',ylabel='Predicted dropout contrast (Q75 - Q25 EBC)',title=label);axes[i].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[i].xaxis.set_major_locator(MaxNLocator(5))
    save(fig,'fig10_local_ebc_contrasts',sources,'MODEL-PREDICTED probability contrasts from the distance-spline plus transformed EBC model. Each point compares within-distance-bin observed EBC quartiles at the mean bin spline basis, limiting interpolation to observed local predictor ranges. Bars are 95% Monte Carlo delta-method intervals; 1000 primary resamples. They are not causal effects, equivalence tests, or finite-patient population intervals.')
    write_json(F/'figure_manifest.json',records)
    (R/'figure_captions.md').write_text('\n\n'.join(f"## {r['figure']}\n\n{r['caption']}\n\nSource tables: "+', '.join(Path(s['path']).name for s in r['source_tables'])+f". Plot command: `{r['plotting_command']}`." for r in records),encoding='utf-8')
if __name__=='__main__':main()
