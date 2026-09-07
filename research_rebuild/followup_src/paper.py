from common import *

def read(name):return pd.read_csv(RUN/'tables'/f'{name}.csv')
def main():
    start=time.perf_counter();cv=read('cv_fixed_S_summary');edge=read('edge_change_summary');held=read('heldout_paired_summary');ext=read('external_transfer_verification');degen=read('degeneracy_audit');st=read('reconciled_stage_status');graph=read('verified_graph_summary');hubs=read('heldout_internal_hub_reproducibility');old= pd.read_csv(OLD/'tables/claim_verdicts.csv')
    integrated=[]
    for acc in ACCS:
        for _,c in cv[cv.cohort.eq(acc)&cv.focal.eq('gcc_focal')].iterrows():
            hr=hubs[hubs.cohort.eq(acc)]
            integrated.append({'cohort':acc,'domain':'conditional_bootstrap','support':c.support,'hard':c.mean_hard_cv,'soft':c.mean_soft_cv,'statistic':'mean CV; lower is less variable','reduction':c.ratio_mean_reduction,'node_coverage':c.finite_paired_nodes,'sample_coverage':int(graph[graph.cohort.eq(acc)].iloc[0].sample_count),'replicates_or_splits':1000,'uncertainty':'fixed-S whole-row Monte Carlo interval only','source':str(RUN/'tables/cv_fixed_S_summary.csv')})
        for metric in ['rank_spearman','top20_overlap']:
            integrated.append({'cohort':acc,'domain':'internal_direct_heldout_hubs','support':'shared evaluable selected identities; fixed train theta','hard':hr['hard_'+metric].mean(),'soft':hr['soft_'+metric].mean(),'statistic':'mean across20 overlapping splits of '+metric,'node_coverage':hr.nodes.mean(),'sample_coverage':375 if acc==ACCS[0] else 640,'replicates_or_splits':20,'uncertainty':'descriptive; shared node set conditions on heldout evaluability; patient linkage unknown in serum','source':str(RUN/'tables/heldout_internal_hub_reproducibility.csv')})
        for _,r in held[held.cohort.eq(acc)].iterrows():integrated.append({'cohort':acc,'domain':'internal_direct_heldout_edges','support':r.model_pair,'without_EBC':r.mean_brier_without,'with_EBC':r.mean_brier_with,'signed_gain':r.mean_difference,'statistic':'mean Brier; positive gain favors EBC','replicates_or_splits':20,'uncertainty':'overlapping splits descriptive; no population CI','source':str(RUN/'tables/heldout_paired_summary.csv')})
    for _,r in ext.iterrows():
        for metric in ['rank_spearman','top20_overlap']:integrated.append({'cohort':'GSE41655','domain':'external_'+r.analysis,'support':'common mapped nodes','hard':r['hard_'+metric],'soft':r['soft_'+metric],'statistic':metric,'node_coverage':204,'sample_coverage':33,'replicates_or_splits':1,'uncertainty':'exploratory point estimate; no population interval; 40.8% node coverage','source':str(RUN/'tables/external_transfer_verification.csv')})
    csv('integrated_stability_reproducibility',integrated)
    # Preserve every original field and verdict, adding explicit follow-up adjudication.
    old['prior_verdict']=old.resolution_status;old['revised_verdict']=old.resolution_status;old['scope']='historical reproduction';old['followup_evidence']='verified_results_ledger.csv; existing_figure_provenance_verification.csv';old['revision_reason']='Previous verdict retained; historical entry preserved.'
    updates={
      'C33':('weakened','conditional descriptive analysis','cv_fixed_S_summary.csv; external_transfer_verification.csv','Lower conditional strength CV is supported on specified support/focal sets; biological superiority and population significance remain unsupported.'),
      'C39':('weakened','held-out prediction','heldout_paired_summary.csv','Direct retained-edge prediction compares flexible margin models; it does not identify a causal explanation of topology.'),
      'C40':('weakened','held-out prediction; population inference','heldout_paired_summary.csv','Replace universal no-additional-information wording with observed signed Brier distributions. Predictive variation cannot establish population zero/equivalence.'),
      'C42':('unresolved','population inference','reports/proposed_paper_scope.md','Population EBC calibration remains outside scope, not an automatic blocker for this empirical review.'),
      'C45':('weakened','contribution/novelty','research_rebuild/references_and_novelty.md','Soft weighting and lower threshold sensitivity are established ideas. Contribution must be an auditable empirical comparison, not invention of weighting.'),
      'C47':('contradicted','conditional descriptive analysis','edge_change_per_replicate.csv; edge_change_summary.csv','Initially absent pairs do appear; dropout alone omits substantial whole-network change.')}
    for id,(verdict,scope,evidence,reason) in updates.items():
        mask=old.claim_id.eq(id);old.loc[mask,['revised_verdict','scope','followup_evidence','revision_reason']]=[verdict,scope,evidence,reason]
    old['followup_run']=RUN.name
    additions=[
      ('F01','Historical tissue reduction is84%.','contradicted','historical reproduction','verified_results_ledger.csv','Original84% claim removed; reproduced63.8495% historical anchor retained.'),
      ('F02','Support-specific GCC soft-strength CV reductions are reproducible.','supported','conditional descriptive analysis','cv_fixed_S_summary.csv','Finite paired fixed node sets, beta6 and support conventions explicitly defined.'),
      ('F03','Old all-node CV intervals estimate the same fixed-S quantity.','contradicted','conditional descriptive analysis','cv_old_interval_replay.csv; cv_interval_attempts.csv','Old implementation changed S. New fixed-S failures remain visible; unsupported intervals suppressed.'),
      ('F04','EBC supplies universally zero additional information.','unresolved','population inference','heldout_paired_summary.csv','No universal no-effect/equivalence inference; empirical paired scores only.'),
      ('F05','Soft weighting improves biological recovery.','unresolved','exploratory external transfer','external_transfer_verification.csv','Original density-matched hard point agreement exceeds soft; no gold standard or superiority inference.'),
      ('F06','Dropout alone characterizes whole-network instability.','contradicted','conditional descriptive analysis','edge_change_summary.csv','Appearances can outnumber disappearances under frozen thresholds.'),
      ('F07','The analysis is uniformly robust to feature/missingness/sample settings.','contradicted','conditional descriptive analysis','reconciled_stage_status.csv; degeneracy_audit.csv','1000-feature tissue full graphs fail all200 draws; selections and graph estimates vary across sensitivities.'),
      ('F08','External cohort validates transferable miRNA prioritization as biological truth.','weakened','exploratory external transfer','external_transfer_verification.csv; external_named_identities.csv','Only exploratory agreement on204 mapped identities in33 arrays; cutoff protocol and ties matter.'),
      ('F09','Twenty half-split direct outcomes permit a population t-test.','contradicted','held-out prediction','heldout_protocol.md','Overlapping splits are descriptive and do not supply20 independent population observations.')]
    extra=[]
    for id,claim,verdict,scope,ev,reason in additions:extra.append({'claim_id':id,'stated_value':claim,'prior_verdict':'not previously itemized','revised_verdict':verdict,'scope':scope,'followup_evidence':ev,'revision_reason':reason,'followup_run':RUN.name})
    updated=pd.concat([old,pd.DataFrame(extra)],ignore_index=True);csv('updated_claim_ledger',updated);csv('claim_change_history',updated[['claim_id','stated_value','prior_verdict','revised_verdict','scope','followup_evidence','revision_reason','followup_run']])
    ctable=cv[cv.focal.eq('gcc_focal')][['cohort','support','finite_paired_nodes','mean_hard_cv','mean_soft_cv','ratio_mean_reduction','mc_low','mc_high']].to_markdown(index=False,floatfmt='.8f')
    etable=edge[['cohort','reference_edges','absent_pairs','retained_dropout','absent_appearance','disappearances_mean','appearances_mean','total_edges_mean','jaccard_mean']].to_markdown(index=False,floatfmt='.8f')
    htable=held[['cohort','model_pair','mean_brier_without','mean_brier_with','mean_difference','min_difference','max_difference','positive_splits']].to_markdown(index=False,floatfmt='.9f')
    xtable=ext[['analysis','reference_theta','external_theta','edge_jaccard','hard_rank_spearman','soft_rank_spearman','hard_top20_count','soft_top20_count']].to_markdown(index=False,floatfmt='.8f')
    full_scores=read('heldout_repeated_split_results')
    metric_summary=full_scores.groupby(['cohort','model'],as_index=False)[['brier','logloss','prevalence','mean_prediction','calibration_bias','clipped_fraction','auroc','average_precision']].mean()
    csv('heldout_metric_summary',metric_summary)
    calibration_text=[]
    for acc in ACCS:
        rr=metric_summary[metric_summary.cohort.eq(acc)&metric_summary.model.eq('spline_distance')].iloc[0]
        ss=full_scores[full_scores.cohort.eq(acc)]
        calibration_text.append(f"{acc}: mean spline-distance prediction {rr.mean_prediction:.8f} versus event prevalence {rr.prevalence:.8f} (bias {rr.calibration_bias:.8f}); evaluable edges {ss.evaluable_edges.min()}–{ss.evaluable_edges.max()} of3119. Mean clipping fraction {rr.clipped_fraction:.8f} for spline-distance; every specification's clipping counts and losses are disclosed.")
    results=f'''# Working Results

All figures and results refer to completed follow-up {RUN.name}, reusing verified original1000-draw conditional caches. Full graph denominators are500 nodes,124750 unordered pairs,3119 retained edges and121631 initially absent pairs in each cohort. Discovery datasets contain750 tissue arrays from750 recorded individuals and1280 serum arrays with unknown person-level linkage. The original578 artifact hashes match. Independent raw correlation/graph computations and re-counted cached events reproduce primary values; saved-design binomial coefficient/MC-score checks reproduce descriptive fits.

## Support-specific conditional variability

{ctable}

Historical tissue mean CVs0.25710309618001453 and0.09294407612425502 give63.84949169994482% reduction under the historical pipeline. The manuscript0.218/0.035 pair and84% claim were not reproduced. Corrected estimates use different documented preprocessing and graph conventions. The two all-node full-support fixed-S intervals are unsupported when interval resamples make any selected node's CV undefined. Percentile intervals are not forced to contain point estimates. Complete failure accounting and prefix diagnostics are in cv_fixed_S_summary.csv and cv_prefix_stability.csv.

## Whole-network change under frozen thresholds

{etable}

All2000 primary draws satisfy M=Mref−D+G. Mean counts equal sums of per-edge probabilities with matched denominators. Fixed-threshold bootstraps do not preserve density. Graph-count percentile ranges describe the resampling distribution, while mean MCSE describes numerical precision; neither is presented as a population confidence interval.

## Direct held-out edge prediction

{htable}

Positive differences favor EBC. These are20 overlapping half splits per cohort, with the original split retained. The df4 flexible-distance and uncertainty-distance pairs are primary; linear-distance is secondary. Scores use identical evaluable reference edges across all six models within a split. Calibration, logloss with explicit clipping, AUROC, AP and class counts are retained in the complete tables. This is direct held-out disappearance, distinct from the old bootstrap-of-held-out-half outcome. Adding EBC slightly worsened mean Brier in both primary specifications in both cohorts. Individual split gains and losses varied; no consistent predictive improvement is demonstrated by this protocol. The secondary tissue linear mean gain is tiny and occurs in only6/20 splits. No population test, equivalence bound or universal zero-effect claim follows.

{' '.join(calibration_text)}

These systematic underpredictions, especially in serum, show poor probability transfer despite within-training bootstrap fitting. All40 training bootstrap sets had250 fully valid attempts and no training-edge exclusions. Tissue held-out nodes were all evaluable; serum retained418–438 of500 nodes per split. Clipping is a disclosed numerical scoring convention, not a repair for miscalibration.

## External agreement

{xtable}

All comparisons use33 GSE41655 adenocarcinoma arrays and204 common mature-miRNA identities (40.8% of discovery500). Original B is separately density matched. Secondary A transfers the discovery cutoff on this204-node universe. Soft strength is identical across A/B. The unfavorable B soft-versus-hard point comparison is retained. Named top20 intersections and all priorities are in external_named_identities.csv and external_transfer_verification.csv. Independent equivalent correlation arithmetic shifts nearly tied strength ranks by a tiny amount; canonical original arithmetic and the alternate result are both documented.

## Domain failures and sensitivity

Nine targeted sensitivity settings have verified complete outputs and one remains a documented complete-graph failure. In tissue1000 probes,121 became constant at least once; mean constant count28.055 agrees closely with the empirical diagnostic expectation27.93548. All200 complete-graph attempts fail, although pairwise information remains available. Three-state accounting covers499500 pairs, with113619 affected at least once and114 affected reference edges. No replacement draws or added probe filter were used.
'''
    report('working_results',results)
    report('working_research_question','''# Working research question

How do thresholded miRNA degree and continuous unsigned miRNA strength differ in conditional resampling variability, and how well do those priorities and train-derived edge-disappearance predictions reproduce in held-out arrays and an exploratory external cohort?

Secondary question: Does EBC improve direct held-out retained-edge disappearance prediction beyond matched flexible threshold-distance and uncertainty-distance models in these two cohorts? This is an empirical predictive question, not a population no-effect or causal claim.
''')
    report('working_methods_amendments',(RUN/'followup_analysis_amendment.md').read_text()+'\n\nAdditional descriptive integration: internal hard/soft hub agreement was computed on the shared evaluable selected node universe for each frozen split, using its training cutoff and stable top20 tie break. This is separate from train-only edge-model fitting and conditions on held-out evaluability. It was implemented before viewing these outcomes; it is a follow-up exploratory summary, not an original prespecified endpoint.\n')
    report('working_limitations','''# Working limitations

Conditional bootstrap variability conditions on observed cohort, selected nodes and frozen threshold. It is not population uncertainty or biological accuracy. Mean CV is sensitive to low expected degrees; fixed-S all-node interval failures expose this limitation. Common support and full support answer different questions. The1000-probe failure restricts the estimator domain and prevents a blanket robustness claim.

Split outcomes are dependent because samples and edges recur. No naive split t-test or SD/sqrt20 population interval is appropriate. Serum evaluations are array-level without patient linkage; tissue individuals are recorded and unique. Preprocessing performed by the data submitters cannot be retrospectively made train-only; our downstream selection, graph, cutoff, predictors and fitting are train-only. Deposited detection limits/zeros, missingness, batch structure and the dominant serum component remain plausible sources of instability. Train-bootstrap probabilities at held-out sample size target resamples of the empirical training distribution; a disjoint held-out sample is not exactly that distribution, so transfer miscalibration remains informative rather than impossible.

EBC coefficients and old/new CV MC intervals quantify numerical uncertainty conditional on empirical data. The existing uncertainty-model coefficient MC intervals hold estimated correlation SD and spline basis fixed; they omit this generated-predictor MC component. Repeated-split score variation includes many perturbations but is not a calibrated population interval. Population EBC no-effect/equivalence claims remain unresolved and are omitted.

External evidence is one small cross-platform cohort with204/500 mapped nodes and33 arrays, externally conditioned finite/nonconstant eligibility, unknown global patient overlap and no biological gold standard. Density matching and cutoff transfer are different designs; hard ranks and top20 sets depend on threshold and ties. Numerical near-ties modestly affect soft rank correlation, and stable identity-order top20 ties are arbitrary. External agreement validates limited empirical reproducibility, not biological recovery or ceRNA regulation. No classifier, pathway or mechanistic inference is attempted.

Lower CV from continuous weighting and sensitivity near a hard cutoff are established concepts. An auditable comparison across conditional, direct held-out and external targets may be useful, but scientific consistency does not establish novelty, journal fit, acceptance or publication readiness. Author/domain review and an explicit contribution argument are still needed.
''')
    report('proposed_paper_scope','''# Proposed empirical paper scope

Working title: **Conditional Stability and Empirical Reproducibility of Hard-Thresholded and Soft-Weighted miRNA Co-expression Networks**.

The strongest supported finding is the separation between conditional smoothness and empirical reproducibility: soft strength has lower specified bootstrap CV, while the density-matched external comparison does not show better soft agreement. Fixed-threshold graph change also includes substantial edge appearances, particularly in serum. Pair these observations with the direct held-out EBC comparison, preserving its cohort/specification-dependent score distribution.

Retain: reproducible historical correction; explicit full/common support and focal universes; fixed-S MC accounting; three-state degeneracy; joint D/G graph change;20 frozen direct held-out comparisons;33-array exploratory external A/B transfer on204 identities. Revise: “centrality has no additional information” into bounded empirical paired-score findings. Remove from conclusions:84% historical reproduction, population zero/equivalence, causal confounding, universal biological superiority, automatic bridge interpretation, and inferred ceRNA mechanism.

Population EBC calibration remains a blocker for population EBC no-effect/equivalence claims. It does not automatically block scientific review of this completed empirical scope. No further nested-bootstrap production run is required for the questions proposed here. A manuscript still needs author review, appropriate declarations, careful figure selection, and a persuasive contribution/novelty argument; reviewability is not a promise of publication.
''')
    verdict_counts=updated.revised_verdict.value_counts().to_dict()
    report('final_empirical_readiness_report',f'''# Final empirical readiness report

**Ready for manuscript-level scientific review within the narrowed empirical scope**, with the executed checks documented in logs/test_results.txt and final_manifest.json. This does not establish novelty, acceptance, or population-level EBC no-effect. The proposed scope explicitly omits those unsupported claims.

Errors addressed: missing serum sensitivity states reconciled from receipts/chunks/diagnostics; changing-S CV intervals replaced with fixed-S joint-row accounting and unsupported intervals marked; direct held-out outcomes added under frozen train-only20-split protocol; retained-edge dropout expanded to paired disappearance/appearance graph accounting; external density matching explicitly separated from frozen-cutoff transfer. A tiny rank discrepancy due to floating-point near-ties is disclosed with both calculations. Original data, outputs and failed settings remain preserved.

Exact results and score directions are in working_results.md and the source CSVs. EBC slightly worsened mean primary Brier in both cohorts, with mixed individual split directions; it did not provide a consistent improvement here. Substantial underprediction of held-out disappearance, especially in serum, is a stronger limitation than the very small EBC increments. Lower conditional CV did not translate into higher original density-matched external soft agreement. Original inventory:578/578 matching hashes. Targeted settings:9 completed and1 failed complete-graph domain. New held-out evaluation:40 splits,240 model rows,120 paired comparisons; all fixed ahead of these outcomes. The claim ledger retains47 historical entries and adds9 follow-up entries; revised totals: {verdict_counts}.

Remaining limitations: tissue1000-feature complete-graph failure; undefined nearly isolated-node CVs; repeated-split dependence; array-level serum identities; conditional MC intervals only; exploratory204-node/33-array external coverage; numerical and degree ties; no biological ground truth, causal interpretation, EBC equivalence or demonstrated novelty. These limit conclusions but are explicitly represented in the proposed empirical paper.

Review the actual signed Brier distribution in heldout_paired_summary.csv, not just the mean. Review lower CV alongside external hard/soft comparisons in integrated_stability_reproducibility.csv. Completion and per-stage runtimes are recorded in stage_runtime_summary.csv and logs/status_*.json. Commands and exact source/input hashes are in reproduction_resume.md and manifests. No background jobs are required after completion.
''')
    status('paper_synthesis',start)
if __name__=='__main__':main()
