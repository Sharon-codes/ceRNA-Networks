from base import *
import base64,html
from markdown_it import MarkdownIt

def read(n):return pd.read_csv(RUN/'tables'/f'{n}.csv')
def table(df,cols=None,fmt='.6f'):
    if cols is not None:df=df[cols]
    return df.to_markdown(index=False,floatfmt=fmt,missingval='undefined')
def fig(name,caption):return f'\n\n![{name}](figures/{name}.svg)\n\n**{name}.** {caption}\n\n'
def main():
    start=time.perf_counter();inv=read('cohort_inventory');ctrl=read('stability_control');hub=read('hub_stability_summary');sim=read('simulation_summary');paired=read('simulation_paired_differences');truth=read('simulation_truth_summary');ex=read('external_summary');mod=read('module_preservation');pred=read('predictive_summary');ev=read('evaluability');graph=read('graph_state_accounting');claims=read('updated_claim_ledger');fmanifest=read('figure_manifest');captions=fmanifest.set_index('figure').caption.to_dict();priorcv=read('fixed_S_interval_status');defs=read('module_definitions')
    gsum=graph.groupby('cohort',as_index=False).agg(B=('replicate','count'),D=('disappearances','mean'),G=('appearances','mean'),unchanged_present=('unchanged_present','mean'),unchanged_absent=('unchanged_absent','mean'),M=('M','mean'),Jaccard=('jaccard','mean'),components=('components','mean'),GCC_nodes=('gcc_nodes','mean'),isolates=('isolates','mean'))
    simwide=sim[sim.endpoint.isin(['mean_cv','latent_hub_top12_recovery','module_ARI','independent_top12_overlap'])].pivot(index=['cell','method'],columns='endpoint',values='mean').reset_index()
    control_show=ctrl[ctrl.focal.eq('baseline_GCC')&ctrl.method.eq('soft')&ctrl['lambda'].isin([0,.5,.99])]
    screening=read('cohort_screening');named=read('named_hub_results');intersections=[]
    for (acc,pol),g in named.groupby(['cohort','policy']):
        for method in ['hard','soft']:
            ids=g.loc[g['discovery_'+method+'_top20']&g['external_'+method+'_top20'],'miRNA'].tolist();intersections.append({'cohort':acc,'policy':pol,'method':method,'shared_top20':len(ids),'identities':'; '.join(ids)})
    ns=read('simulation_results');maxse=sim[sim.endpoint.isin(['latent_hub_top12_recovery','module_ARI','independent_top12_overlap'])].mean_MCSE.max();domains=read('simulation_domain_accounting');enums=ev.groupby('cohort',as_index=False).agg(splits=('split','count'),nodes_min=('evaluable_nodes','min'),nodes_max=('evaluable_nodes','max'),edges_min=('evaluable_edges','min'),edges_max=('evaluable_edges','max'),events_min=('events','min'),events_max=('events','max'))
    txts=[]
    txts.append(f'''# Stability is not biological validation: an integrated miRNA network evidence audit

**Run:** `{RUN.name}` · **Date:** 7 September 2026 · **Decision document, not a submitted manuscript.**

## 1. Executive decision

**The computations support a defensible empirical warning about using conditional variability as a proxy for hub or module recovery. They do not yet establish a distinct biological-organization contribution sufficient to justify a BioSystems submission.** The next action should be to evaluate the narrow contribution against the closest work and obtain one independently grounded biological target if this is to be positioned as biological insight. Do not expand bootstrap counts or add generic pathway enrichment simply to make a positive story.

Five judgments must be kept separate:

| Dimension | Evidence-based assessment |
|---|---|
| Computational correctness | Prior862 artifact hashes verified; current source/cached scoring and graph accounting checked; new analyses executed, with explicit implementation corrections and preserved failures. |
| Empirical interpretability | Strong after specifying support, focal nodes, thresholds, targets and missingness. Conditional stability, model transfer and biological recovery are distinct. |
| Novelty | Elementary affine control and known weighted-network ideas are not new methods. Recent BOONS and stability/module-preservation research substantially overlap the broader question. |
| Biological relevance | Two colorectal external cohorts and mature-miRNA identities provide relevant data, but modules are unbalanced association clusters without independent functional truth. |
| Publication readiness / BioSystems | Suitable for scientific methodological review; not currently a demonstrated distinct BioSystems biological contribution. Neither acceptance nor editorial timing is predictable. |

Decisive measured results:

* Tissue soft top20 conditional membership stability is0.95155 versus hard0.90930; **serum reverses that ordering: soft0.69925 versus hard0.73575**, despite much lower soft CV.
* A fixed positive background drives tissue GCC soft mean CV from0.09668775 to0.00012872 at lambda.99, and serum from0.03833912 to0.00040456, while preserving every top20 set in all1000 draws. This is a metric diagnostic, not a biological intervention.
* The bounded simulation completed300 independent datasets (10 cells ×30) and100 bootstrap attempts per dataset. It contains conditions where soft module recovery improves while soft CV is higher, and conditions where lower CV accompanies little hub improvement. Known-K4 and simple Gaussian factors limit generalization.
* Added EBC did not consistently improve held-out Brier. Serum spline Brier0.298412 exceeds the fixed.5 baseline's0.25, with predicted disappearance0.225680 versus observed0.490827. This transfer limitation is much larger than the EBC increment.
* GSE48267 adds61 primary tumors and204 mapped miRNAs. Its density-matched hard/soft rank agreement is{ex.query("cohort=='GSE48267' and policy=='B_density_matched'").iloc[0].hard_rank_rho:.6f}/{ex.query("cohort=='GSE48267' and policy=='B_density_matched'").iloc[0].soft_rank_rho:.6f}; top20 intersection is4/3. GSE41655's original13/12 comparison remains visible. Neither establishes soft biological superiority.

BioSystems requires a clear conceptual connection to biological organization. Its subscription route currently states no publication fee charged to authors, meeting the no-out-of-pocket publication-fee requirement if that route is chosen; optional OA is separate. Short communications require urgent findings and are not a remedy for limited data. A concise full empirical article is the potential format if the contribution becomes adequate. [Official scope](https://shop.elsevier.com/journals/biosystems/0303-2647), [author guide](https://www.sciencedirect.com/journal/biosystems/publish/guide-for-authors), [official fee policy](https://www.sciencedirect.com/journal/biosystems/about/insights).

## 2. Exact data and task definitions

The original PDF filename mentions ceRNA and pan-cancer detection, but its actual first-page title is **“Re-evaluating Topological Fragility in Hard-Thresholded miRNA Co-expression Networks: A Distance-Controlled Statistical Analysis.”** The10-page document was extracted and hashed again. Neither a validated cancer detector nor a demonstrated ceRNA mechanism is present in this audited analysis. A filename cannot supply that evidence.

{table(inv,['accession','role','specimen','platform','included_arrays','verified_people','analysis_features'],'.0f')}

The inventory counts analyzed disease/tissue strata, not accession-wide totals. GSE115513 includes carcinoma arrays only; recorded individual IDs are unique among750 selected arrays. GSE73002 includes breast-cancer serum arrays; unique arrays are not automatically verified people. GSE41655 uses33 adenocarcinomas and excludes adenomas/normals. GSE48267 uses61 tumor arrays and excludes their61 adjacent normals; site-qualified keys and the primary study support61 patients. Cross-study source differences support separate recruitment provenance but do not prove global non-overlap. [GSE115513](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115513), [GSE73002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73002), [GSE41655](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41655), [GSE48267](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48267), [61-patient primary report](https://pubmed.ncbi.nlm.nih.gov/24865442/).

The deposited processed scales are used as supplied. Discovery tissue is FFPE with deposited zeros and upstream QC/normalization. Serum has upstream handling of undetected measurements, without a usable complete detection-call matrix or validated batch/person covariates for this task. External GSE41655 used Agilent V3 and quantile normalization; GSE48267 used Agilent V3 with a signal floor and90th-percentile normalization, with site/preservation differences. No assay-incompatible hemolysis or detection threshold is introduced. Upstream normalization applied by submitters cannot be made train-only retrospectively; downstream learning is train-only.

For selected p nodes, unsigned hard adjacency is **Hᵢⱼ=1[|Spearman(Xᵢ,Xⱼ)|≥theta]** off diagonal, with diagonal zero. Theta is the linear empirical97.5th percentile of all p(p−1)/2 unordered pair magnitudes. Ties can exceed the target density. Soft adjacency is **Wᵢⱼ=|rhoᵢⱼ|⁶**, diagonal zero. Node degree/strength sums the appropriate row. Positive-only graphs and other powers are preserved sensitivities; the primary evidence here is unsigned beta6. EBC expands to **edge betweenness centrality**: normalized fractions of shortest paths crossing an edge, computed on the fixed full reference graph with unweighted paths. High EBC is not identical to graph-theoretic bridge status.

Full support sums every selected neighbor. Fixed common support restricts both methods to the reference retained edges. Baseline GCC focal nodes are209 tissue and183 serum; they are not all500 nodes. CV is sample SD(ddof1)/mean, undefined for near-zero mean. Reduction is1−mean(CVsoft over S)/mean(CVhard over the same S), not mean(1−CVsoft/CVhard). S is explicitly fixed to paired finite nodes in the original1000 draws; nodewise ratios require a further positive hard-CV denominator. Conditional bootstraps hold selection and theta fixed and rerank within each resample; repeated-split training repeats selection and graph construction within each training half.
''')
    txts.append(fig('fig01_workflow',captions['fig01_workflow']))
    txts.append(f'''## 3. Existing results and provenance reconciliation

The original578-artifact inventory and the284-artifact follow-up inventory match all862 hashes. The old run has `results_summary.json`; the follow-up instead has verified tables and `final_manifest.json`, so an absent follow-up summary JSON was not invented. This run supplies its own `results_summary.json`. Historical and corrected claims remain separate.

{table(priorcv[priorcv.focal.eq('gcc_focal')],['cohort','support','finite_paired_nodes','mean_hard_cv','mean_soft_cv','ratio_mean_reduction','mc_low','mc_high'],'.8f')}

Historical tissue means0.25710309618001453/0.09294407612425502 reproduce63.84949169994482%, not the manuscript84%. The manuscript0.218/0.035 pair has not been reconstructed. Corrected reductions above are new estimands and must not be substituted for historical reproduction. The all-node full-support point reductions0.93501645/0.98555688 use372/272 finite paired nodes; fixed-S interval failure counts960/1000 and964/1000 make those intervals unsupported. The old implementation changed S inside interval resamples; a percentile interval failing to contain its point is not by itself an error, but changing the estimator silently is.

Current verification levels are explicit: graph disappearances/appearances were re-counted from original pair-correlation caches; CV and hub/control quantities were independently reaggregated from node draw vectors; held-out scores were regenerated from saved coefficient/design predictions and direct outcome vectors; prior coefficient reproduction was checked through its preserved evidence ledger and hashes, not falsely described as another full fit. Numerical reconciliation retains457 prior checks and adds240 current prediction checks. Original external soft rho0.44550199 uses canonical arithmetic; an equivalent SciPy computation gave0.44547710 because nearly tied strengths reorder at floating precision. Both are preserved; the substantive comparison and top20 counts do not change. The historical imposed threshold-enrichment formula remains excluded from empirical evidence.

Evidence statuses: **verified_from_source** = rederived from source matrices/metadata; **reproduced_from_cache** = recomputed from saved draw/prediction objects; **reported_only** = available report without adequate reconstruction; **contradicted** = actual evidence disagrees; **blocked_missing_input** = required eligibility/input unavailable; **failed** = attempted computation or estimator failed; **not_run** = unexecuted analysis. Preserved estimator failures are never presented as successful complete-graph analyses.

## 4. Analytic control and empirical demonstration

For fixed p, c>0 and0≤lambda<1, adding the same constant to every off-diagonal adjacency entry gives

**S(lambda)=(1−lambda)S+lambda c(p−1).** Thus **mean(lambda)=(1−lambda)mean+lambda c(p−1)** and **SD(lambda)=(1−lambda)SD**. For positive mean,

**CV(lambda)=CV(0) × [(1−lambda)mean]/[(1−lambda)mean+lambda c(p−1)].**

It approaches zero as lambda approaches1. Every strength ordering and exact tie is unchanged for lambda<1 because this is a common positive affine transform. At lambda1 all nodes tie. This proof applies to fixed full support and a common node universe; it does not show invariance of paths or modules and does not apply unchanged to node-varying sparse supports. Lower CV is therefore insufficient evidence that ranking information improved. It does not prove that the original soft benefit is entirely an artifact.

The frozen diagnostic uses c=.5 and lambda0,.25,.5,.9,.99. Both hard and soft weights lie in[0,1]. All methods/lambdas and full-support finite/GCC focal subsets are in the source table. Representative soft results:

{table(control_show,['cohort','lambda','focal_nodes','mean_strength','mean_sd','mean_cv','exact_rank_changed_draws','top20_changed_draws'],'.8f')}

The original, unmodified full/common-support quantities on fixed GCC focal nodes are shown together below. Mean CV is the mean of nodewise ratios, so it need not equal mean SD divided by mean strength.

{table(read('original_mean_sd_cv'),fmt='.8f')}

External affine controls use each cohort's fixed 204-node universe and the density-matched hard comparison. The following results retain floating-point rank sensitivity, if any; top20 invariance is checked separately in discovery and external data.

{table(read('external_affine_control').query('`lambda` in [0, 0.5, 0.99]').drop(columns='interpretation'),fmt='.8f')}

Maximum observed CV formula discrepancy is{ctrl.max_formula_error.max():.3g}. Exact rank changes over all reported primary control cases total{int(ctrl.exact_rank_changed_draws.sum())}; top20 changes total{int(ctrl.top20_changed_draws.sum())}. Fixed external universes were checked separately, retaining any numerical rank sensitivity rather than rounding it away. Conditional hub stability is direct membership overlap with the corresponding original reference, not overlap between hard and soft methods.

{table(hub,['cohort','method','k','B','mean_overlap','q025_overlap','q975_overlap','mean_rank_rho'])}

The overlap quantiles describe the conditional resampling distribution, not population confidence limits. Ranks use average ties; top-k selection uses the stable original identity order. The serum soft method has higher overall rank concordance but lower top20 membership stability: small changes near the selection boundary matter even when most ranks remain similar.
''')
    txts.append(fig('fig02_control_hubs',captions['fig02_control_hubs'])+fig('fig02b_original_moments',captions['fig02b_original_moments']))
    txts.append(f'''## 5. Known-structure simulation and precision

For each condition, X=F Lᵀ+z gᵀ+epsilon, with independent standard-normal module factors F, shared nuisance z and unit-variance independent noise. L has four disjoint30-variable blocks, loadings linearly spaced.35–.85 times signal scale(.5 or1). g is0 or a fixed heterogeneous vector scaled by.6. Covariance LLᵀ+ggᵀ+I is positive definite. Population Gaussian Spearman correlations are6/pi × asin(Pearson/2). Explicit loading, covariance, observed/nuisance-free correlation and method-specific graph matrices are saved. The minimum covariance eigenvalue is{truth.minimum_eigenvalue.min():.8f}.

Three highest-loading variables per planted module define12 biological-factor hub targets independently of reconstruction method. Their ordering is checked in actual nuisance-free population association. Nuisance changes observed associations and can move apparent population hubs away from those targets. Hard degree and soft strength also have distinct population estimands; method-specific strength error is evaluated against its own population vector. In the no-signal/no-nuisance null, there are no true hubs/modules: their recovery is undefined. The literal hard quantile rule at all-zero correlations retains every pair because all tie at zero; this degenerate complete population target is disclosed, not replaced by an empty graph.

Both methods use the same unsigned correlations and fixed parameter opportunities; no tuning. Hard threshold is2.5% of the sample's full pair universe, soft power6. Thresholding the monotone soft adjacency at the same rank cutoff produces the same edge set apart from ties; the code verifies this identity and does not call it independent success. Spectral clustering uses knownK4 for both, with deterministic QR label assignment; disconnected hard-graph warnings and isolates are saved. This is an oracle-K benchmark, not a new community method. [Spectral clustering primary reference](https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html), [adjusted Rand index](https://doi.org/10.1007/BF01908075).

Runtime-only pilots initially projected3969.65 seconds for60 datasets/cell, so the declared fallback30 was frozen before production outcomes. Later a numerically exact implementation optimization removed redundant threadpool scanning; the count remained30. Before any null dataset checkpoint, an implementation error that gave the null an empty hard population graph was corrected to the literal all-tie rule. Existing nonnull checkpoints were preserved. All300 independent dataset pairs completed; each primary sample contributed100 bootstrap attempts, and a separate independent sample assessed reproducibility. No successful-draw replacement occurred.

{table(simwide,['cell','method','mean_cv','latent_hub_top12_recovery','module_ARI','independent_top12_overlap'])}

Cell names encode n, signal scale s and nuisance g. The numerical boundary is not a single universal winner. At n60,s1,g0, soft CV is higher (0.610333 versus0.592771) while module ARI is higher (0.861751 versus0.307170). At n240,s.5,g.6, soft CV is much lower (0.570354 versus1.551989), but hub recovery remains only0.083333 versus0.080556. Stronger signal/larger n aid module recovery in this generator, while nuisance can redirect apparent hubs. These findings separate module recovery from hub recovery and from conditional CV; they do not establish a biological mechanism.

Confidence bars quantify Monte Carlo precision across30 independent generated datasets. The maximum MCSE among displayed recovery/reproducibility endpoints is{maxse:.6f}. The planned worst-case bounded-mean95% normal half-width target.13 was not met after the budget fallback; R30 gives a conservative worst-case.178923. Actual endpoint errors are reported, so this modest benchmark is not called definitive. Bootstrap prefix50 versus100 discrepancies are a separate numerical diagnostic; finite-CV sets range from{int(domains.paired_finite_nodes_min.min())} to{int(domains.paired_finite_nodes_max.max())} of120 nodes, and undefined prefixes remain explicit. Common-support CV results are preserved as existing-method diagnostics, not a new method.

{table(read('simulation_precision'),['cell','method','datasets','valid_datasets','B','mean_abs_cv_prefix_change','max_abs_cv_prefix_change'])}
''')
    txts.append(fig('fig03_simulation_generator',captions['fig03_simulation_generator'])+fig('fig04_simulation_results',captions['fig04_simulation_results']))
    txts.append(f'''## 6. Biological hubs, fixed modules and external reproducibility

Discovery-only average-linkage clustering of1−|rho| on the selected500 tissue nodes was frozen at maxclust8 before new module preservation outcomes. Module sizes are **{', '.join('M'+str(k)+': '+str(v) for k,v in defs.module.value_counts().sort_index().items())}**. Only M3(91 nodes) and M6(384 nodes) satisfy external size≥10 and coverage≥25% with at least10 mapped nodes. All eight modules and failed coverage settings are retained. The largest two eligible discovery modules supply the named examples, not the best-preserving modules. The384-node cluster is especially broad; a partition label alone does not establish a functional unit.

Mapping preserves unique official mature-miRNA aliases, distinguishes known5p/3p and star names, excludes ambiguous/multiple destinations, and adds no invented arm labels to older unsuffixed names. Neither duplicate expression-driven aggregation nor outcome-driven node selection was used. Complete/nonconstant external eligibility conditions the compared universe. Both evaluated external cohorts yield the same204 discovery identities, but their specimens and assay preprocessing differ.

{table(ex,['cohort','policy','arrays','nodes','reference_theta','external_theta','edge_jaccard','hard_rank_rho','soft_rank_rho','hard_top20_count','soft_top20_count'])}

A transfers the discovery cutoff on the mapped204-node universe; it is not the original500-node cutoff. B independently matches2.5% density. Hard ranks refer to node degree, soft ranks to beta6 node strength. Soft ranks do not change between A/B because the soft adjacency has no hard cutoff. GSE41655 remains previously inspected/exploratory; GSE48267 was selected by metadata and mapping eligibility before its outcomes. No common biological gold standard underlies these external point comparisons.

Fixed-module density/connectivity metrics follow the distinction between preserving specific module properties and merely reclustering labels. They are descriptive, not WGCNA Zsummary or NetRep significance; no permutation null or universal preservation threshold is claimed. [Langfelder et al.](https://doi.org/10.1371/journal.pcbi.1001057), [Ritchie et al.](https://doi.org/10.1016/j.cels.2016.06.012).

{table(mod[mod.eligible],['cohort','policy','module','discovery_size','mapped_nodes','coverage','hard_connectivity_rho','soft_connectivity_rho','hard_hub_top5_count','soft_hub_top5_count'])}

{table(mod[mod.eligible],['cohort','policy','module','discovery_hard_density','external_hard_density','discovery_soft_mean_weight','external_soft_mean_weight'])}

M3 has negative soft connectivity agreement in the new cohort, while M6 has modest positive agreement. There is no consistently preserved method across module/endpoint/policy choices. Density, module size and coverage must accompany any apparent rank preservation. Actual shared whole-network top20 identities:

{table(pd.DataFrame(intersections),fmt='.0f')}

Named fixed-module examples (all eligible modules):

{table(mod[mod.eligible],['cohort','policy','module','discovery_soft_hubs','external_soft_hubs'])}

The large M6 includes miR-143-3p/miR-145-5p. Primary mouse lineage-specific experiments place a miR143/145 regenerative function in intestinal mesenchyme, making specimen composition a plausible context for bulk colorectal association. That paper does not independently validate this384-node module, identify its cell composition, or prove an arm-specific mechanism for these probes. Bulk expression can preserve composition-driven relationships while obscuring cell-intrinsic regulation. [Chivukula et al.,2014](https://doi.org/10.1016/j.cell.2014.03.055), [Farahbod and Pavlidis,2020](https://doi.org/10.1101/gr.256735.119).

No target/pathway enrichment was executed: this bounded package lacks an assembled independent versioned target/pathway matrix with measured-universe coverage and a justified empirical miRNA-set null. Consequently there are no enrichment p-values, significant pathways or functional-validation claims. Standard target-gene overrepresentation would risk selection/annotation bias; any future enrichment would require appropriate measured-miRNA sampling, coverage accounting and multiplicity. The literature context above is not that analysis. [Bleazard et al.,2015](https://doi.org/10.1093/bioinformatics/btv023).

The additional-cohort screen considered five candidates, all reported below. GSE29622 had65 untreated FFPE primary colon tumors but98 usable mapped identities, failing the frozen100-node gate. No network outcome was inspected and the gate was not relaxed. The next metadata-ranked candidate GSE48267 passed with61 tumors and204 identities. Its30 SBU FFPE and31 WU frozen cases combine site/preservation differences; pretreatment is unresolved. Those factors limit interpretation and were not “corrected” using external outcomes. [GSE29622 primary report](https://pubmed.ncbi.nlm.nih.gov/22362069/), [GSE48267 primary report](https://pubmed.ncbi.nlm.nih.gov/24865442/).

{table(screening,['accession','eligible_unique_tumors','platform','compartment','status','selection_reason'])}
''')
    txts.append(fig('fig07_external_modules',captions['fig07_external_modules']))
    txts.append(f'''## 7. Held-out prediction, baselines and intervals

The binary outcome is disappearance of a training-retained edge in direct held-out Spearman correlations at the frozen training cutoff. Training uses its own supplied-scale complete/variable MAD selection, graph, EBC, spline bases and scales;250 training bootstrap draws use the held-out sample size(375 tissue/640 serum). The fitted predictor remains the training reference edge's distance/EBC and correlation SD. The old outcome based on250 bootstraps of the test half is a different target and is not pooled with these scores.

Brier = mean over evaluable edges of(y−p)². Constant p=.5 gives exactly.25 on every split. The train-intercept baseline uses total training-bootstrap events divided by total training edge-trials before observing test eligibility/outcomes. It does not use test prevalence. Flexible raw-distance and uncertainty-distance splines each have df4 with and without standardized log-EBC; the linear raw-EBC model remains secondary. No new fitting/recalibration is performed on evaluation labels.

{table(pred,['cohort','model','mean_Brier','min_Brier','max_Brier','mean_logloss','mean_prediction','prevalence','clipping_fraction'],'.8f')}

Twenty split scores are dependent because arrays and edges overlap. Means and ranges are descriptive; no naive split t-test or SD/sqrt20 population interval is used. EBC gain is Brier_without−Brier_with, positive favoring EBC. Tissue spline/uncertainty mean gains are−0.000004823917/−0.000000448862; serum gains are−0.000090091030/−0.000037273790. These tiny increments do not establish equivalence or universal zero effect. AUROC/AP remain secondary in the preserved240-row scores; prevalence and eligibility affect their interpretation. No causal decomposition follows from the comparison.

Logloss clips predictions to[1e−12,1−1e−12], with clipping fractions shown above. Tissue uncertainty-model clipping near9.7% indicates extreme predicted probabilities, not missing outcomes. Calibration-in-the-large is mean prediction minus observed prevalence; the systematic negative bias in tissue and especially serum is substantial. The model can discriminate edges yet transfer probabilities poorly.

{table(enums,fmt='.0f')}

Training bootstrap events condition on selected high-correlation reference edges and empirical training samples. Direct disjoint held-out observations need not have the same selected-edge distribution, even with matched sample sizes. Feature/edge selection, correlation estimation and deposited preprocessing can contribute to a target mismatch; this is a scientific transport limitation rather than evidence of a coding leak or a newly identified causal confounder. All original train fits had250 complete attempts and no training-edge exclusions. Serum test exclusions arise from missing/constant probes; their outcomes stay unknown.

Excluded-edge training diagnostics compare standardized distance/EBC, train dropout and correlation SD without imputing missing test labels:

{table(read('excluded_edge_training_characteristics').groupby(['cohort','group'],as_index=False).mean(numeric_only=True).drop(columns=['split']),fmt='.6f')}

For unknown binary labels with fixed trained p, each missing edge contributes between min(p²,(1−p)²) and max(p²,(1−p)²) to Brier. Summing those with known losses yields valid full-universe identification bounds, not confidence intervals. Mean serum spline bounds are{pred.query("cohort=='GSE73002' and model=='spline_distance'").iloc[0].mean_full_Brier_lower:.8f}–{pred.query("cohort=='GSE73002' and model=='spline_distance'").iloc[0].mean_full_Brier_upper:.8f}; the observed subset score0.298412 must not be called a fully observed all-edge score.

Fixed-S CV intervals resample whole saved draw rows jointly across all nodes and both methods. They quantify numerical precision conditional on observed data, selected S and the empirical bootstrap distribution, not biological population uncertainty. Any resample undefined on an S member fails the whole interval attempt. The two all-node full-support intervals remain unsupported; all four GCC intervals had1000 valid interval attempts. No extra biological-data bootstrap extension was required by the previous frozen prefix tolerance.
''')
    txts.append(fig('fig05_prediction',captions['fig05_prediction'])+fig('fig05b_signed_ebc',captions['fig05b_signed_ebc']))
    txts.append(f'''## 8. Edge-state accounting, consequences and failures

Counts were re-counted directly from the1000 original complete primary correlation draws per cohort. Every primary draw has500 nodes,124750 unordered pairs,3119 retained reference edges and121631 initially absent pairs. Undefined primary pairs=0. Disappearance and appearance have separate denominators; unchanged present/absent states are also shown.

{table(gsum,['cohort','B','D','G','unchanged_present','unchanged_absent','M','Jaccard','components','GCC_nodes','isolates'])}

For every complete draw, **M=3119−D+G**. Mean rates are tissue D/3119=0.08307663 and G/121631=0.00259783, serum0.20895415 and0.01169830. Averaging per-edge probabilities and replicate counts agrees because all denominators match. Counts grow under fixed thresholds when appearances exceed disappearances; fixed-density resampling would be a different estimand. Conditional graph-count quantiles are resampling dispersion, not population confidence intervals.

{table(read('graph_consequence_summary'))}

Those within-draw correlations connect graph growth/change to actual ranks and top20 memberships, but are not causal effects. Fixed module densities for all eight tissue modules are saved per draw. The illustration below uses literal draw0 and a common deterministic layout, not a visually selected successful draw.

The1000-feature tissue setting remains **0/200 complete graphs**. Saved accounting shows121 probes became constant at least once; mean28.055 constant probes/draw versus empirical expectation27.93548. The independent-row diagnostic is sum_v(n_v/n)^750; it is not a whole-graph failure probability and is not applied to block resampling.113619 of499500 pairs, including114 retained reference pairs, were unevaluable at least once. Per-pair present/absent/unevaluable counts and conditional denominators remain in the preserved follow-up. No undefined correlation becomes zero, no failed draw is replaced, and no expensive degeneracy rerun was needed here.
''')
    txts.append(fig('fig06_graph_changes',captions['fig06_graph_changes']))
    literature=(RUN/'research/literature_and_journal_fit.md').read_text();lit_table=literature.split('## Closest-work table',1)[1]
    txts.append('## 9. Closest work and contribution assessment\n\n'+lit_table+'\n\n'+'''This targeted search used web search, publisher/official pages, PubMed/PMC, institutional primary manuscripts and bioRxiv on7 September2026; the exact queries, access levels and exclusions are in the appendix. It is not a systematic review. Recent2020–2026 searches found close2020–2025 work; no missing2026 match is treated as evidence of priority. Some PMC/ScienceDirect requests failed or returned access challenges; abstract-only sources are labeled and no restriction was bypassed.

The distinction among claims is decisive: weighted-network construction is an **already-known principle**; historical-number correction and conditional CV differences are **reproduced observations**; discordance across affine control, hub selection, module recovery and two external cohorts is a **potential empirical boundary**; a new functional or mechanistic statement about these biological modules remains **unverified**. First-ever uncertainty-to-hub-reproducibility novelty is **unsupported**, especially given the2025 BOONS preprint. The preprint is not represented as peer reviewed.

The official BioSystems scope requires conceptual biological-organization relevance. No close recent BioSystems article was verified in this bounded search; that absence proves nothing. The guide says short communications address urgent findings and are not a route for insufficient/preliminary data; no urgency is established here. Full original articles are allowed irrespective of length. Optional OA currently costsUSD3010 excluding taxes; subscription policy states no publication fee to authors. These are verified policies, not promises of future pricing or acceptance. [Scope](https://shop.elsevier.com/journals/biosystems/0303-2647), [guide](https://www.sciencedirect.com/journal/biosystems/publish/guide-for-authors), [Insights](https://www.sciencedirect.com/journal/biosystems/about/insights), [publication options](https://www.sciencedirect.com/journal/biosystems/publish/open-access-options).

## 10. Claims, manuscript scope and exact remaining blocker

**Neutral working title:** “Conditional Stability, Hub Recovery and External Reproducibility in miRNA Co-expression Networks.”

**Results-grounded abstract:** Conditional resampling variability is often used to assess network reliability, but its relationship to biological recovery depends on the statistic and target. We audited miRNA co-expression networks from750 colorectal tissue arrays and1280 breast-cancer serum arrays, preserving fixed support and missingness definitions. An affine-background control reduced strength CV while retaining hub rankings. Across300 independent120-variable simulated datasets, hard/soft variability, planted module recovery and hub recovery did not share a universal ordering. Soft top20 resampling stability exceeded hard in tissue(0.95155 versus0.90930) but was lower in serum(0.69925 versus0.73575), despite lower soft CV. Twenty overlapping half splits per cohort showed no consistent Brier improvement from edge betweenness centrality; serum predictions were poorly calibrated and worse on average than a constant.5 baseline. Exploratory tissue comparisons used33 and61 external tumors with204 mapped miRNAs per cohort and showed no consistent soft hub advantage. Fixed discovery modules were strongly unbalanced and had mixed external connectivity preservation. These results support evaluating variability alongside explicit recovery and transfer endpoints, while leaving functional module validity and population EBC equivalence unresolved.

**Concise contribution:** a reproducible empirical counterexample to treating lower strength CV as sufficient evidence of better hub identification, with explicit support, threshold, identity, null and transportability limits. This is a methodological inference boundary, not a newly validated cancer detector, ceRNA mechanism or biological network law.

**Potential four-figure article outline:**(1) affine diagnostic plus direct hub stability;(2) known-structure recovery versus variability;(3) honest held-out baselines/calibration and graph-change consequences;(4) two-cohort fixed-module/external comparison. Cohort flow, full parameter/claim tables, all failures and reproducibility detail can be supporting material.

**Decision and smallest missing decisive evidence:** do not submit as a claimed new biological-organization finding on this evidence alone. One independently defined, biologically interpretable miRNA module/hub target with adequate matched-assay external coverage and composition-aware interpretation would test whether this boundary changes a substantive biological conclusion. More bootstrap repetitions, post-hoc favorable k selection, or generic target enrichment would not resolve that gap. If that independent target is unavailable, retain this as a transparent methodological audit rather than continuing an unlimited rescue project. Population EBC calibration remains unnecessary for the limited descriptive conclusion, and remains unresolved for no-effect/equivalence claims.

The56 previous entries are preserved, with8 current claims added. “Supported” refers to the stated scope only; no researcher intent is inferred. Every proposed claim has a source, evidence level, limitation and permitted wording in the complete ledger below.
''')
    txts.append(table(claims,['claim_id','exact_claim','scope','verdict','evidence_level','limitation','permitted_wording']))
    txts.append('''

## 11. Reproducibility and evidence appendix

All analyses use the dated amendment and frozen configuration, which follow knowledge of earlier outcomes; this is not preregistration of the original study. The new dated directory preserves both prior runs and raw inputs. Current source/configuration/input/output hashes are in the final manifest. Numeric tables supplement the report; decisive values and figures are embedded above.

**Execution:** existing1000-draw conditional node/pair caches and40 held-out prediction checkpoints were reused after hash verification. New work comprised affine control/hub ranks, source-mapped fixed modules and one additional external cohort, honest predictive baselines/identification bounds,300 simulated datasets with30,000 bootstrap attempts, and direct graph/hub/module consequences. There is no nested population-EBC production. The two runtime pilots are separate from the300 production datasets. Each production dataset has an independently generated second sample.

**Failures and corrections:** GSE29622 remains blocked at98 mapped nodes; tissue1000-feature complete graphs remain failed0/200; two all-node full-support interval estimators remain unsupported; target enrichment and population EBC inference were not run. A null population-graph implementation error was corrected before any null checkpoint. Redundant per-call threadpool scanning was removed with exactly matching rank arithmetic; the original R30 freeze was retained. An older parser omitted organism/processing metadata fields; complete archived official sample metadata resolved those implementation errors without changing the selection rule. No failed scientific condition was hidden or replaced with a favorable result.

**Inference units:** identified persons where available, arrays otherwise; nodes/edges are dependent network measurements; bootstrap draws are conditional Monte Carlo units; overlapping splits are descriptive dependent assessments; independently generated simulation datasets support simulation Monte Carlo precision. None of these automatically provides calibrated biological population uncertainty.

**Verification:** eight targeted checks cover affine formulas and rank invariance, simulation population truth and rank arithmetic, predictive baselines and bounds, split separation and module mapping, graph accounting and preserved failures, simulation/interval attempt accounting, and self-contained figure/source integrity. The test log and browser render receipt accompany the package. Recorded simulation runtime is for the resumed invocation; it excludes the earlier stopped invocation and is not total computation time. Checkpoints preserve both phases.

**Supporting machine-readable package:** cohort inventory/screening; numerical reconciliation/evidence register; novelty and query tables; control/hub stability; population truth/simulation results/precision; module definitions/preservation/named hubs; predictive baselines/calibration/evaluability; graph accounting/consequences; updated claims; result summary; figure manifest; configuration, statuses, commands and hashes. The ZIP excludes raw expression and individual-level sample metadata/IDs; small derived outputs and source code are sufficient to inspect the logic, with public download manifests for reproducibility.
''')
    txts.append('\n\n### Cohort processing and assumptions\n\n'+table(inv,['accession','upstream_processing','repeat_handling','batch_metadata','detection_information','global_overlap']))
    txts.append('\n\n### All module eligibility states\n\n'+table(mod[['cohort','policy','module','discovery_size','mapped_nodes','coverage','status','reason']]))
    txts.append('\n\n### Simulation population targets\n\n'+table(truth.drop(columns=['source','sha256','latent_hubs'])))
    txts.append('\n\n### Paired simulation differences\n\nSoft-minus-hard,95% normal Monte Carlo approximation across independent simulated dataset pairs; bounds are not clipped to admissible endpoint ranges.\n\n'+table(paired[paired.endpoint.isin(['mean_cv','latent_hub_top12_recovery','module_ARI','independent_top12_overlap'])]))
    txts.append('\n\n### Explicitly missing or unexecuted inputs\n\n'+table(read('missing_input_inventory')))
    txts.append('\n\n### Figure sources and commands\n\n'+table(fmanifest[['figure','sources','command']]))
    for n in ['search_log','journal_policy_verification']:
        txts.append('\n\n### '+n.replace('_',' ').title()+'\n\n'+table(pd.read_csv(RUN/'research'/f'{n}.csv')))
    runtime=[]
    for p in sorted((RUN/'logs').glob('status_*.json')):
        z=json.loads(p.read_text());runtime.append({'stage':p.stem[7:],'status':z['status'],'seconds':z.get('seconds',np.nan),'detail':z.get('reason','')})
    csv('stage_runtime_summary',runtime);txts.append('\n\n### Completed stages, blocked stages and recorded runtimes\n\n'+table(pd.DataFrame(runtime)))
    commands='''$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONUTF8='1'
# The original freeze.py invocation is archived; do not rerun it in this completed directory.
# For a new independent run, choose a new RUN path in a copy of base.py and freeze there.
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\empirical.py inventory
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\empirical.py modules_external
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\empirical.py controls_hubs
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\empirical.py graph_consequences
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\prediction.py
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\additional_cohort.py
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\external48267.py
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\simulation.py
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\synthesis.py
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\plots.py
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\report.py
C:\\Python314\\python.exe research_rebuild\\biosystems_src\\checks.py
# Render with render_report.cjs, then run finalize.py after visual inspection.
'''
    txts.append('\n\n### Reproduction / resume commands\n\nRun only the affected stage; simulation checkpoints resume without replacing attempted outcomes. Original datasets and source annotations remain outside the ZIP.\n\n```powershell\n'+commands+'```\n')
    startmeta=json.loads((RUN/'manifests/start.json').read_text());txts.append('\n\n### Software and original configuration\n\n```text\n'+startmeta['python']+'\n'+startmeta['platform']+'\nBranch: '+startmeta['branch']+'\nCommit: '+startmeta['commit']+'\n'+startmeta['packages']+'```\n\n```json\n'+(RUN/'frozen_config.json').read_text()+'\n```\n')
    markdown='\n'.join(txts)
    # Readability only: normalize common number-adjacent prose without altering identifiers/paths.
    import re
    markdown=re.sub(r'\b(all|All|of|on|from|in|with|and|per|at|than|only|the|uses|across|tissue|serum|Tissue|Serum|original|prior|Prior|these|These|lambda|beta|defined|known|cohort|count|rule|total|contributes|over|for|datasets|completed)(?=\d)',r'\1 ',markdown)
    markdown=re.sub(r'\b(is|soft|hard|versus|to|The|Each|another|includes|excludes|exactly|both|retaining|using|fixed|known|used|below|above|approximately|at|lambda|scale|power|USD|costsUSD|only)(?=[\d.]\d)',r'\1 ',markdown)
    txt('FINAL_REPORT.md',markdown)
    engine=MarkdownIt('default',{'html':True});body=engine.render(markdown)
    body=re.sub(r'<p>(<img [^>]+>)</p>\s*<p>(<strong>fig[^<]+</strong>.*?)</p>',r'<figure>\1<figcaption>\2</figcaption></figure>',body,flags=re.S)
    for f in fmanifest.itertuples():
        svg=(RUN/'figures'/f'{f.figure}.svg').read_bytes();body=body.replace('src="figures/'+f.figure+'.svg"','src="data:image/svg+xml;base64,'+base64.b64encode(svg).decode()+'"')
    css='''body{font:16px/1.55 system-ui,Segoe UI,sans-serif;color:#20303b;margin:0;background:#edf1f3}main{max-width:1220px;margin:24px auto;background:white;padding:40px 48px;box-shadow:0 3px 20px #0001}h1{font-size:34px;line-height:1.15;color:#19475f}h2{margin-top:48px;border-top:3px solid #35758c;padding-top:15px;color:#19475f}h3{margin-top:30px}table{width:100%;border-collapse:collapse;font-size:12px;margin:22px 0;table-layout:auto}th,td{padding:8px;border-bottom:1px solid #d7e1e5;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#e4eff4}tr:nth-child(even){background:#f5f8fa}img{width:100%;height:auto;margin:18px 0}a{color:#1d678e}pre{background:#eef3f5;padding:15px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}code{font-size:.9em}p,li{max-width:1150px}.top{padding:12px;background:#e4eff4} @media print{@page{size:A4 landscape;margin:13mm}body{background:white;font-size:10pt}main{margin:0;padding:0;max-width:none;box-shadow:none}h1{font-size:24pt}h2{break-before:page;font-size:17pt}h3{break-after:avoid}table{font-size:7.5pt}th,td{padding:4px}thead{display:table-header-group}tr{break-inside:avoid}img{max-height:165mm;object-fit:contain;break-inside:avoid}pre{font-size:7pt}a{color:#185977;text-decoration:none}}'''
    css+='figure{margin:24px 0}figcaption{font-size:14px;line-height:1.45}@media print{h2{break-before:auto;break-after:avoid;margin-top:25px}h1{break-after:avoid}figure{break-inside:avoid;margin:12px 0}figure img{max-height:120mm;margin:5px 0}figcaption{font-size:8.5pt;line-height:1.3}}'
    txt('FINAL_REPORT.html','<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>miRNA stability and reproducibility — evidence decision</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
    txt('reproduction_README.md','# Reproduction and resume\n\n'+commands+'\nThe frozen simulation count is30 per cell after runtime pilot. No unfinished scientific jobs. Original files and research source downloads are separate; use public URL/hash manifests. Run only required stages. ZIP excludes individual sample metadata and raw expression. FINAL_REPORT.html embeds every figure; FINAL_REPORT.md is its readable text counterpart. final_manifest.json excludes its own hash, and the external ZIP is hashed in a separate receipt to avoid circularity.\n')
    write_json(RUN/'results_summary.json',{'run_id':RUN.name,'decision':'methodological boundary supported; distinct BioSystems biological contribution unestablished; defer submission as biological insight','cohorts':inv.to_dict('records'),'hub_stability':hub.to_dict('records'),'external':ex.to_dict('records'),'prediction':pred.to_dict('records'),'simulation_datasets':300,'simulation_method_rows':600,'simulation_conditions':10,'simulation_B':100,'prior_hashes':862,'graph_summary':gsum.to_dict('records'),'claim_verdicts':claims.verdict.value_counts().to_dict(),'not_run':read('missing_input_inventory').to_dict('records')})
    status('report_generation',start)
if __name__=='__main__':main()
