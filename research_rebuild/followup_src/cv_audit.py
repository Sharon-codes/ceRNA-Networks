from common import *
from audit import verify,flush_ledger

def reduction(h,s,S):
    hc=cv(h[:,S]);sc=cv(s[:,S])
    return 1-sc.mean()/hc.mean() if np.isfinite(hc).all() and np.isfinite(sc).all() and hc.mean()>0 else np.nan

def main():
    start=time.perf_counter();rows=[];attempts=[];prefix=[];sets=[];nodes=[];oldchecks=[];extension=[]
    for acc in ACCS:
        z=np.load(OLD/'models'/f'{acc}_unsigned_node_replicates.npz');gcc=pd.read_csv(RUN/'tables'/f'{acc}_gcc_nodes.csv').node.to_numpy();oldsrc=OLD/'tables'/f'{acc}_unsigned_stability.csv';old=pd.read_csv(oldsrc)
        need=False;specs=[]
        for support,h,s in [('full',z['hard'],z['soft']),('common_support',z['hard_common'],z['soft_common'])]:
            hc=cv(h);sc=cv(s)
            for node in range(len(hc)):nodes.append({'cohort':acc,'support':support,'node':node,'mean_degree':h[:,node].mean(),'hard_cv':hc[node],'soft_cv':sc[node],'in_gcc':node in gcc,'hard_positive_draws':int(np.sum(h[:,node]>0)),'degree_stratum':'zero' if h[:,node].mean()==0 else '<1' if h[:,node].mean()<1 else '1–5' if h[:,node].mean()<5 else '>=5'})
            for focal,universe in [('all_nodes',np.arange(h.shape[1])),('gcc_focal',gcc)]:
                S=universe[np.isfinite(hc[universe])&np.isfinite(sc[universe])];point=reduction(h,s,S);subset_name=f'{acc}_{support}_{focal}';specs.append((support,focal,S,point))
                sets.extend({'cohort':acc,'support':support,'focal':focal,'node':int(node),'in_finite_S':node in S} for node in universe)
                orig=old[old.beta.eq(6)&old.support.eq(support)&old.focal.eq(focal)].iloc[0]
                for name,val in [('mean_hard_cv',hc[S].mean()),('mean_soft_cv',sc[S].mean()),('ratio_mean_reduction',point)]:verify(subset_name+'_'+name,acc,orig[name],val,OLD/'models'/f'{acc}_unsigned_node_replicates.npz','Sample SD(ddof=1)/mean; common fixed finite paired S')
                vals=[];changed=[];biased=[]
                # Replay old streams exactly, including the problematic changing-S selection.
                for b in range(100):
                    ix=rng_for(SEED,acc+'_cv_mc',b).integers(0,len(h),len(h));hh=cv(h[ix][:,S]);ss=cv(s[ix][:,S]);ok=np.isfinite(hh)&np.isfinite(ss)
                    val=1-ss[ok].mean()/hh[ok].mean();biased.append(val);changed.append(int(sum(~ok)))
                    oldchecks.append({'cohort':acc,'support':support,'focal':focal,'interval_attempt':b,'removed_from_S':int(sum(~ok)),'changing_S_reduction':val,'fixed_S_reduction':1-ss.mean()/hh.mean() if ok.all() else np.nan})
                for name,val in [('reduction_mc_low',np.quantile(biased,.025)),('reduction_mc_high',np.quantile(biased,.975))]:verify(subset_name+'_old_'+name,acc,orig[name],val,oldsrc,'Replay 100 original joint-row resamples with changing S','Numerical reproduction does not validate changing estimand')
                for b in range(1000):
                    ix=rng_for(SEED,acc+'_followup_cv_mc',b).integers(0,len(h),len(h));hh=cv(h[ix][:,S]);ss=cv(s[ix][:,S]);ok=np.isfinite(hh)&np.isfinite(ss);valid=ok.all() and hh.mean()>0
                    val=1-ss.mean()/hh.mean() if valid else np.nan
                    vals.append(val);attempts.append({'cohort':acc,'support':support,'focal':focal,'interval_attempt':b,'fixed_S_size':len(S),'undefined_nodes':int(sum(~ok)),'valid':bool(valid),'reduction':val})
                vals=np.array(vals);valid=np.isfinite(vals);supported=valid.mean()>=.99
                nv=S[hc[S]>1e-12]
                row={'cohort':acc,'support':support,'focal':focal,'B':len(h),'original_nodes':len(universe),'finite_paired_nodes':len(S),'undefined_original_nodes':len(universe)-len(S),'mean_hard_cv':hc[S].mean(),'mean_soft_cv':sc[S].mean(),'ratio_mean_reduction':point,'mean_nodewise_reduction':np.mean(1-sc[nv]/hc[nv]),'nodewise_valid_nodes':len(nv),'interval_attempts':1000,'interval_valid':int(sum(valid)),'interval_failed':int(sum(~valid)),'interval_supported':supported,'mc_low':np.quantile(vals[valid],.025) if supported else np.nan,'mc_high':np.quantile(vals[valid],.975) if supported else np.nan,'valid_attempts_mean_minus_point_diagnostic':np.nanmean(vals)-point if any(valid) else np.nan,'old_interval_low':orig.reduction_mc_low,'old_interval_high':orig.reduction_mc_high,'old_changed_S_attempts':int(np.sum(np.array(changed)>0)),'old_mean_nodes_removed':np.mean(changed),'S_definition':'fixed paired finite nodes in original B1000; conditional on this selection','interval_meaning':'joint saved-row resampling; Monte Carlo only; unsupported if <99% valid'}
                for B in [100,250,500,1000]:prefix.append({'cohort':acc,'support':support,'focal':focal,'B':B,'fixed_S_size':len(S),'reduction':reduction(h[:B],s[:B],S),'undefined_hard_nodes':int(np.sum(~np.isfinite(cv(h[:B,S]))))})
                prefix500=reduction(h[:500],s[:500],S);row['prefix500_to1000_change']=point-prefix500
                if focal=='gcc_focal' and (not np.isfinite(prefix500) or abs(point-prefix500)>.005):need=True
                rows.append(row)
        if need:
            df,_=cohort(acc);sel,_,_=select_features(df,500);x=sel.T.to_numpy();r=spearman(x);u,v=np.triu_indices(500,1);theta=np.quantile(np.abs(r[u,v]),.975);base=np.abs(r[u,v])>=theta;hard=[];soft=[];hc=[];sc=[];validids=[]
            for b in range(200):
                ix=rng_for(SEED,acc+'_followup_cv_extension',b).integers(0,len(x),len(x));rr,ok=paircorr_partial(x[ix])
                if not ok.all():continue
                score=np.abs(rr[u,v]);hard.append(node_sums((score>=theta)[None,:],500)[0]);soft.append(node_sums((score**6)[None,:],500)[0]);hc.append(node_sums(((score>=theta)*base)[None,:],500)[0]);sc.append(node_sums(((score**6)*base)[None,:],500)[0]);validids.append(b)
            np.savez_compressed(RUN/'models'/f'{acc}_cv_extension.npz',hard=hard,soft=soft,hard_common=hc,soft_common=sc,replicate=validids)
            for support,focal,S,point in specs:
                ha,sa=(hard,soft) if support=='full' else (hc,sc);oh,os=(z['hard'],z['soft']) if support=='full' else (z['hard_common'],z['soft_common']);combined=reduction(np.concatenate([oh,ha]),np.concatenate([os,sa]),S)
                extension.append({'cohort':acc,'support':support,'focal':focal,'attempted_additional':200,'valid_additional':len(validids),'original_point':point,'combined_point':combined,'absolute_change':abs(combined-point),'precision_target':.005,'target_met':abs(combined-point)<=.005,'reason':'GCC prefix criterion exceeded; one frozen fresh-seed batch only'})
        else:extension.append({'cohort':acc,'attempted_additional':0,'reason':'Both GCC prefix500 vs1000 absolute changes <=.005; no fresh extension required'})
        print(acc,'CV audit complete; extension required',need,flush=True)
    csv('cv_fixed_S_summary',rows);csv('cv_interval_attempts',attempts);csv('cv_prefix_stability',prefix);csv('cv_fixed_node_sets',sets);csv('cv_node_diagnostics',nodes);csv('cv_old_interval_replay',oldchecks);csv('cv_extension_decision',extension);flush_ledger()
    report('cv_estimand_and_interval_audit','''# CV estimand and interval audit

CV uses sample SD (ddof=1) divided by mean, undefined for mean<=1e-12. Beta6 soft strength sums unsigned correlation weights. Full support includes every pair; common support fixes the reference hard edges for both methods. Focal GCC membership is frozen from the original graph. Every original focal identity is listed in cv_fixed_node_sets.csv; the reported ratio uses the same paired finite S in both means. S is defined from the original1000 saved draws, so both estimates and MC intervals condition on that empirical selection. The mean nodewise reduction also excludes zero hard-CV nodes and has a separate denominator.

The old implementation jointly resampled rows correctly but silently removed newly undefined nodes inside each resample. Replaying its original100 streams reproduces its endpoints. The serum all-node point lying outside its old interval is not itself an error: nonlinear bias can produce this. Here the estimator additionally changed S, disproportionately removing rare-degree nodes. The replay table quantifies those removals; fixed-S failed attempts and conditional-on-valid bias diagnostics separate this mechanism from the mathematical possibility of percentile bias. Intervals are never moved to contain the point.

New intervals resample whole draw rows jointly across both methods and all S nodes. A failed CV on any S member invalidates that interval attempt. Percentiles are unsupported if fewer than99% attempts are valid. All attempts remain saved; failed ones are not silently deleted to produce a reported interval. Nearly isolated-node effects are shown in mean-degree/CV plots and frozen strata zero,<1,1–5,>=5. Headline interpretation uses GCC-focal results, retaining all-node results and their failures.

Prefix checks and any fresh-seed extension follow the dated amendment's .005 absolute criterion. These are numerical diagnostics, not added biological sample size. No interval in this audit is a population confidence interval.
''')
    status('cv_audit',start)
if __name__=='__main__':main()
