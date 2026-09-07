from common import *
import platform,subprocess
def main():
    marker=RUN/'manifests/freeze.json'
    if marker.exists():print('Existing freeze preserved');return
    protocol='''# Follow-up analysis amendment — 2026-09-07

Run: 20260907_empirical_followup_v1. This is retrospective relative to all outcomes in 20260907_rebuild_v1, including the original held-out split and external comparison. It is not preregistration. The empirical scope is conditional stability, held-out prediction, and exploratory cross-platform reproducibility. Population EBC no-effect/equivalence inference remains unresolved and outside this pass.

Before new outcomes: use 20 deterministic half splits per cohort, split 0 exactly the original split. Splits 1–19 use rng_for(20260907, accession+'_followup_split', split). Tissue splits use recorded individuals (all unique); serum splits are array-level, with unknown patient linkage. Freeze all IDs now. Use 250 training bootstrap attempts with replacement, sample size equal to held-out n. Retain every failure; never redraw. Profile split 0 runtime, retain 20 unless an actual resource limit requires a documented amendment before inspecting further outcomes.

Train-only: complete variable probes, supplied-scale top-500 MAD stable selection, unsigned Spearman, 2.5% full-pair quantile linear cutoff, full-graph normalized EBC. Predict disappearance of train-retained edges in the direct held-out correlation matrix at frozen train cutoff. Train bootstrap estimates disappearance probabilities and correlation SD. Missing/constant held-out nodes make incident edges unevaluable. Training edges with fewer than 20 evaluable bootstrap draws or SD<1e-10 are excluded from the common training/evaluation universe for every model; record counts. Use per-edge binomial denominators when training draws are partial. Never learn predictors, knots, scaling or fitted parameters from held-out values. For split 0 reuse hash-verified training correlations when identity matches.

Models: existing centered natural cubic regression spline df=4 in raw distance, and the identical spline plus standardized log1p(EBC/median positive EBC); same pair for distance/bootstrap SD. Historical linear standardized distance ± standardized raw EBC is secondary. Fit binomial GLMs equally, maxiter=200, no tuning. Primary Brier and without-minus-with difference, positive favors EBC. Calibration: mean prediction vs event rate and fixed probability bins [0,.1,...,1], report all bins including empty. Secondary logloss clipped to [1e-12,1-1e-12], clipping count, AUROC and AP with ties handled in groups; undefined class-dependent metrics remain missing. Overlapping split distributions are descriptive, no naive t-test or population confidence interval.

CV: beta=6, sample SD ddof=1; full vs fixed reference support; all vs baseline GCC focal nodes. Fix S to paired finite-CV nodes in original 1000 draws. Resample 1000 entire saved draw rows jointly across nodes/methods for 1000 interval attempts using label accession+'_followup_cv_mc'. A draw undefined on any member of S is recorded as failed. Only show a percentile MC interval if at least 99% attempts are valid; otherwise mark unsupported. Also reproduce the old 100-attempt changing-S calculation for diagnosis. Prefixes 100,250,500,1000 on fixed S. Numerical criterion for headline GCC reductions: absolute change from prefix 500 to 1000 <= .005. If exceeded, execute one independent 200-draw extension (label accession+'_followup_cv_extension'), compare 1000 vs1200 with target .005; stop after that batch and report precision failure if unresolved. All-node failure due to vanishing degree is a domain limitation, not a reason for indefinite extra draws.

Degeneracy: reconstruct the original tissue 1000-feature 200 attempted streams exactly. Keep present/absent/unevaluable per pair with explicit conditional denominators. Empirical sum_v(n_v/n)^m is a probe diagnostic for independent row bootstrap only. Preserve 0/200 complete-graph validity if reproduced; add no feature filter. Graph completion edge-count lower/upper bounds are possible binary completions, not observed full graphs or confidence intervals.

Graph change: existing unsigned primary 1000 draws, fixed500 nodes, fixed theta. D,G,M,symmetric difference,Jaccard,density,components,GCC/isolates; verify per-draw identity and edge-probability/count averaging. Bootstrap percentile ranges describe the resampling distribution; SD/sqrt(B) describes MC precision only.

External: independently reconstruct GSE41655 exact33 arrays and unique mapped finite variable common identities. Keep original separate-density graph B. Add secondary A with discovery cutoff frozen on that same mapped universe. No new cohorts, no outcome-driven mapping, no k search; only k=20, stable node-order tie break. Report names and tie/coverage limitations. No population interval or biological truth claim.

Preserve old artifacts and claim history. Verify old inventory and cache checksums; every new table/figure has run/config/input provenance in manifests. Figures use only saved source tables. No nested production run, classifier, ceRNA mechanism, publishing or pushing.
'''
    (RUN/'followup_analysis_amendment.md').write_text(protocol,encoding='utf-8')
    report('heldout_protocol',protocol.split('CV:')[0]+'\nOriginal protocol audit: the old outcome was disappearance in 250 bootstraps of the held-out half, not direct held-out disappearance. Train feature selection, theta, EBC, scaling and fits used the training half. Held-out finite/nonconstant eligibility restricted evaluation, not fitting. The train bootstrap size and test bootstrap size equalled their half sizes. Its tight intervals quantify only validation-bootstrap Monte Carlo error and omit training-fit and population uncertainty. The new direct outcome changes the prediction target; old scores are retained, not pooled with new scores.\n')
    splits=[]
    for acc in ACCS:
        df,meta=cohort(acc)
        if 'individual' in meta:assert not meta.individual.duplicated().any()
        for s in range(20):
            label=acc+'_split' if s==0 else acc+'_followup_split'
            a,b=np.array_split(rng_for(SEED,label,s).permutation(df.shape[1]),2)
            splits.append({'cohort':acc,'split':s,'train':df.columns[a].tolist(),'test':df.columns[b].tolist(),'train_indices':a.tolist(),'test_indices':b.tolist(),'groups':'individual' if 'individual' in meta else 'array; patient unknown'})
    write_json(RUN/'manifests/splits.json',splits)
    inventory=json.loads((OLD/'output_inventory.json').read_text());checks=[]
    for row in inventory:
        p=OLD/row['path'];actual=sha(p) if p.exists() else None
        checks.append({**row,'actual_sha256':actual,'matches':actual==row['sha256']})
    csv('previous_inventory_verification',checks)
    write_json(marker,{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'commit':git('rev-parse','HEAD'),'git_status':git('status','--short'),'diff_sha':hashlib.sha256(git('diff','HEAD').encode()).hexdigest(),'python':sys.version,'platform':platform.platform(),'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),'amendment_sha256':sha(RUN/'followup_analysis_amendment.md'),'split_sha256':sha(RUN/'manifests/splits.json'),'primary_config_sha256':sha(BASE/'configs/primary.json'),'inputs':{str(p):sha(p) for p in [*(ROOT/f'{a}_series_matrix.txt.gz' for a in ACCS),BASE/'data/GSE41655_series_matrix.txt.gz']},'previous_inventory_matches':sum(r['matches'] for r in checks),'previous_inventory_entries':len(checks),'old_sources':{p.name:sha(p) for p in (BASE/'src').glob('*.py')}})
    print('Frozen',len(splits),'splits; inventory',sum(r['matches'] for r in checks),'/',len(checks),flush=True)
if __name__=='__main__':main()
