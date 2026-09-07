"""Build evidence reports from saved results; never execute analysis or invent missing results."""
import json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from provenance import BASE,ROOT,RUN,sha,write_json,git

T=RUN/'tables';R=RUN/'reports'
def read(name):
    p=T/(name+'.csv');return pd.read_csv(p) if p.exists() else pd.DataFrame()
def md(df):return df.to_markdown(index=False,floatfmt='.6g') if len(df) else 'Not available; see stage status.'
def put(name,text):
    (R/name).write_text(text,encoding='utf-8')
def main():
    qc=json.loads((RUN/'logs/status_qc_both.json').read_text());q=pd.DataFrame([v['result'] for v in qc.values()]);q.to_csv(T/'cohort_summary.csv',index=False)
    networks=pd.concat([read(a+'_network_sensitivity') for a in ['GSE115513','GSE73002']],ignore_index=True)
    primary=networks[networks.sign.eq('unsigned')&networks.scope.eq('full')&networks.target_density.eq(.025)]
    coefficients=pd.concat([read(a+'_unsigned_d0.025_coefficients') for a in ['GSE115513','GSE73002']],ignore_index=True)
    ec=coefficients[coefficients.term.eq('ebc')]
    st=pd.concat([read(a+'_unsigned_stability') for a in ['GSE115513','GSE73002']],ignore_index=True);st=st[st.beta.eq(6)]
    prediction=pd.concat([read(a+'_heldout_increment') for a in ['GSE115513','GSE73002']],ignore_index=True)
    simulation=read('simulation_summary');external=read('external_biological_summary')
    hist={a:json.loads((RUN/'historical'/a/'summary.json').read_text()) for a in ['GSE115513','GSE73002']}
    hf=pd.concat([pd.read_csv(RUN/'historical'/a/'glm.csv') for a in hist],ignore_index=True)
    order=pd.concat([pd.read_csv(RUN/'historical'/a/'node_order_sensitivity.csv').assign(cohort=a) for a in hist],ignore_index=True)
    order_summary=order.groupby('cohort').agg(ebc_p_min=('ebc_p','min'),ebc_p_max=('ebc_p','max'),ebc_se_min=('ebc_se','min'),ebc_se_max=('ebc_se','max')).reset_index();order_summary.to_csv(T/'historical_node_order_summary.csv',index=False)
    primary.to_csv(T/'primary_networks.csv',index=False);ec.to_csv(T/'primary_ebc_models.csv',index=False);st.to_csv(T/'primary_stability.csv',index=False);prediction.to_csv(T/'heldout_comparison.csv',index=False)
    sensitivity=pd.concat([read(a+'_targeted_sensitivity') for a in hist],ignore_index=True);sensitivity.to_csv(T/'targeted_sensitivity_summary.csv',index=False)
    full=pd.concat([read(a+'_full_pipeline').assign(cohort=a) for a in hist],ignore_index=True)
    if len(full):full.groupby('cohort').mean(numeric_only=True).reset_index().to_csv(T/'full_pipeline_summary.csv',index=False)
    claims=[]
    def claim(id,loc,stated,source,actual,verdict,severity,reason):
        claims.append({'claim_id':id,'manuscript_location':loc,'stated_value':stated,'source_script_output':source,'run_provenance':'20260907_rebuild_v1; see production_start.json, stage provenance and output_inventory.json','independently_recomputed_value':actual,'discrepancy':reason,'severity':severity,'resolution_status':verdict})
    claim('C01','Abstract/2.1','750 colorectal carcinoma samples','manifests/GSE115513_sample_selection.csv',750,'supported','none','750 arrays and 750 distinct recorded individual IDs')
    claim('C02','Abstract/2.1','1280 breast-cancer serum samples','manifests/GSE73002_sample_selection.csv',1280,'supported','moderate','1280 arrays; no explicit patient identifier; independent persons cannot be fully established')
    claim('C03','2.2','uncensored data','manifests/*_metadata.json','tissue includes deposited zeros; serum undetected values replaced upstream','weakened','high','Rebuild adds no censoring, but deposited measurements are not uncensored latent expression')
    claim('C04','2.2','drop probes missing >20%','generate_final_sensitivity_analysis.py:64-65','complete-case after <=20% filter; serum 1956/2540 complete','contradicted','high','Manuscript omits additional complete-case deletion')
    claim('C05','2.2','supplied scale preserved','generate_final_sensitivity_analysis.py:67-68','tissue log2(x+1) selected by max>50 rule','contradicted','high','Undocumented scale-dependent feature selection')
    claim('C06','2.2','Mean Absolute Deviation','generate_final_sensitivity_analysis.py:70','scipy median_abs_deviation','contradicted','moderate','Median, not mean, absolute deviation')
    claim('C07','2.2','top 500 probes','manifests/*_selected_probes.csv','500 in both primary cohorts','supported','none','Identity manifests saved; sensitivity uses actual usable counts')
    claim('C08','2.3/2.4','unsigned adjacency I(abs(r)>=theta)','generate_final_sensitivity_analysis.py:96','historical positive-only hard edges; corrected unsigned','contradicted','high','Methods and implementation differ; evaluated both compatible conventions')
    claim('C09','2.3','about 2.5% global density','historical/*/summary.json',f"tissue {hist['GSE115513']['full_density']}; serum {hist['GSE73002']['full_density']}",'supported','none','Approximate global density; historical sweep labels full edge count as GCC count')
    claim('C10','2.3','GCC candidate retained edges','historical/*/summary.json','historical GCC restriction reproduced; corrected primary full fixed-node graph','supported','none','Both universes explicitly separated')
    claim('C11','2.4','1000 bootstrap resamples','historical/*/replicates.npz; cache/*/completion.json','1000 valid in each main cohort','supported','none','Historical and corrected production executed separately')
    claim('C12','2.4/3.1','threshold sweep empirical continuous enrichment','generate_fuck_it_figures.py: ratio_th; manuscript_figure_provenance.csv','Figure 4 exact pixel match to formula-generated fig3_threshold_sweep.png','contradicted','critical','Imposed trajectory is not a measured sweep; separate run_statistical_redesign_pipeline.py recomputed 50 draws/threshold but is not this embedded image')
    claim('C13','2.5','one-endpoint clustering conservative','historical_node_order_summary.csv',order_summary.to_json(orient='records'),'unresolved','high','Node_A omits cross-endpoint and disjoint-edge patient-induced dependence; conservatism not established')
    claim('C14','2.5','shared edges cause marginal binomial overdispersion','statistical_validation.md','Cross-edge dependence does not itself change each edge marginal Binomial(B,p)','contradicted','moderate','Pearson dispersion can reflect misspecified conditional mean; marginal dependence and cross-edge dependence are distinct')
    claim('C15','2.6','same focal GCC nodes','historical/*/replicates.npz','same historical GCC focal nodes; sums over all 500 neighbors','supported','moderate','Neighbor universe was underspecified')
    claim('C16','2.6','same sign hard degree and soft strength','generate_final_sensitivity_analysis.py','positive hard degree vs unsigned abs(r)^6','contradicted','high','Corrected comparisons use identical signs and explicitly fixed common support')
    claim('C17','2.6','beta=6 selected a priori','historical source history','beta6 in code; prior scientific justification beyond historical assertion unavailable','unresolved','low','Rebuild specifies beta6 prospectively and tests 1,2,4,6,8,12')
    for a,id,loc in [('GSE115513','C18','3.1'),('GSE73002','C19','3.3')]:
        h=hist[a];claim(id,loc,f"theta {h['theta']}; GCC {h['gcc_nodes']} nodes / {h['gcc_edges']} edges",f'historical/{a}/summary.json',f"{h['theta']}; {h['gcc_nodes']}; {h['gcc_edges']}",'supported','none','Historical specification only; corrected primary differs where preprocessing/density rule changes')
    for term,stated,id in [('const','-6.6887; SE .013; z -519.39','C20'),('z_Distance','-5.0545; SE .036; z -139.73','C21'),('z_EBC','.0018; SE .004; z .379; p .704','C22')]:
        x=hf[hf.cohort.eq('GSE115513')&hf.covariance.eq('Node_A')&hf.term.eq(term)].iloc[0]
        claim(id,'Table 1',stated,'historical/GSE115513/glm.csv',f"beta {x.coefficient}; SE {x.se}; z {x.z}; p {x.p}; OR CI [{x.OR_low},{x.OR_high}]",'contradicted' if term=='const' else 'supported','high' if term=='const' else 'moderate','Intercept uses nominal SE/z and nominal OR CI despite clustered table title' if term=='const' else 'Historical numbers reproduced; z_EBC SE rounds to .005 rather than .004; not population-valid inference')
    claim('C23','3.2','Deviance 7196.34; Pearson/df 2.27','historical/GSE115513/glm.csv','7196.34167684; 2.27164796','supported','none','Diagnostics reproduced; do not prove covariance validity')
    d=hf[hf.cohort.eq('GSE115513')&hf.covariance.eq('Node_A')&hf.term.eq('z_Distance')].iloc[0]
    claim('C24','Abstract/3.2','99.36% lower odds per distance SD','historical/GSE115513/glm.csv',100*(1-d.OR),'supported','moderate','Conditional historical logit association; neither risk reduction nor causal effect; cohort-specific SD')
    claim('C25','Abstract/3.2','EBC p=.704','historical/GSE115513/glm.csv',float(hf[hf.cohort.eq('GSE115513')&hf.covariance.eq('Node_A')&hf.term.eq('z_EBC')].p.iloc[0]),'supported','high','Reproduces chosen covariance estimator, does not establish negligible biological effect')
    claim('C26','Abstract/3.3','serum EBC p=.094; beta=-.0037','historical/GSE73002/glm.csv',hf[hf.cohort.eq('GSE73002')&hf.covariance.eq('Node_A')&hf.term.eq('z_EBC')][['coefficient','p','OR','OR_low','OR_high']].to_json(orient='records'),'supported','high','Chosen estimator reproduces; generalization claim remains conditional')
    claim('C27','3.3','serum distance beta=-1.6110; OR .199','historical/GSE73002/glm.csv','beta=-1.61095998; OR=.19969582','supported','low','OR rounds to .200 at three decimal places')
    claim('C28','3.2/Figure 2','curves completely overlap; zero partial effect','generate_figure5_partial_effects.py; historical/GSE115513/glm.csv','nonzero coefficient; z_EBC=-1 can imply impossible negative raw EBC','contradicted','high','Visual overlap is resolution-dependent; corrected effects stay within observed local predictor support')
    claim('C29','3.4','hard mean CV .218','historical/GSE115513/summary.json',hist['GSE115513']['hard_mean_cv'],'contradicted','high','Reproduced .25710; no matching saved .218 computation found')
    claim('C30','3.4','soft mean CV .035','historical/GSE115513/summary.json',hist['GSE115513']['soft_mean_cv'],'contradicted','high','Reproduced .09294; no matching saved .035 computation found')
    claim('C31','Abstract/3.4/Figure 5 caption','84.0% reduction','historical/GSE115513/summary.json',100*hist['GSE115513']['reduction'],'contradicted','critical','Executed historical reduction 63.8495%; stated CV pair implies 83.94495%, not evidence for that pair')
    claim('C32','Figure 5 image','hard .257; soft .093','manuscript_figure_provenance.csv; historical/GSE115513/summary.json','hard .25710310; soft .09294408','supported','none','Embedded figure values reproduced')
    claim('C33','Figure 5 title','significantly improves topological stability','primary_stability.csv; external_biological_summary.csv','CV reductions depend on sign, support, power and focal-node universe','weakened','high','No valid population significance test in historical analysis; lower strength CV does not establish biological recovery')
    claim('C34','3.1 figure reference','sweep Fig.3','manuscript full text/pages','actual sweep Fig.4','contradicted','moderate','Corrected captions numbered consistently')
    claim('C35','3.2 figure reference','partial effects Fig.5','manuscript full text/pages','actual partial effects Fig.2','contradicted','moderate','Corrected captions numbered consistently')
    claim('C36','3.3 figure reference','forest Fig.2','manuscript full text/pages','actual forest Fig.3','contradicted','moderate','Corrected captions numbered consistently')
    claim('C37','3.4 figure reference','weighted comparison Fig.4','manuscript full text/pages','actual weighted comparison Fig.5','contradicted','moderate','Corrected captions numbered consistently')
    claim('C38','Figure 4 left axis','GCC edge count','generate_fuck_it_figures.py edge_counts','full-graph count used','contradicted','moderate','Rebuilt tables distinguish both denominators')
    claim('C39','Discussion/Conclusion','threshold proximity dominant explanation','primary_ebc_models.csv; heldout_comparison.csv','conditional margin association reproduced; flexible and uncertainty models compared','weakened','high','No causal decomposition; no population-calibrated EBC comparison')
    claim('C40','Discussion/Conclusion','centrality contributes little independent information','primary_ebc_models.csv; heldout_comparison.csv','conditional effects vary with cohort/specification; finite-patient uncertainty unresolved','weakened','high','Non-significance cannot establish negligible effect; compare held-out losses and intervals')
    claim('C41','Discussion','high EBC equals bridge/bottleneck','*_edges.csv actual_bridge field','EBC and graph-theoretic bridge status computed separately','weakened','moderate','High EBC is not identical to an actual bridge')
    claim('C42','Discussion','topological nulls insufficient because geometry destroyed','analysis_plan.md; statistical_validation.md','no exchangeable conditional EBC null validated','unresolved','moderate','Logical concern plausible; specific null calibration not demonstrated in manuscript')
    claim('C43','Conclusion','node clustering accounts for structural dependence','historical_node_order_summary.csv','one-endpoint sensitivity does not cover all dependence','contradicted','high','Corrected MC covariance resamples full replicate vectors; biological inference remains unresolved')
    claim('C44','3.3','serum is cross-context, not biological replication','GEO records','different disease, compartment and platform','supported','none','Separate external tissue evaluation added')
    claim('C45','Introduction/Discussion','weighted-network robustness novelty','references_and_novelty.md','weighted coexpression and network stability precede this study','weakened','moderate','Potential novelty is a validated distance/uncertainty/EBC analysis, not soft weighting itself')
    claim('C46','Declarations','funding, competing interests, ethics and original AI-use declaration','original manuscript','not independently audited','unresolved','none','Do not infer intent; corrected working sections were generated with AI assistance and need author review')
    claim('C47','2.3','only initially present edges can be unstable','*_pair_universe.csv','initially absent edges can appear; dropout is a restricted estimand','weakened','moderate','Appearance probabilities now reported separately')
    pd.DataFrame(claims).to_csv(T/'claim_verdicts.csv',index=False)

    put('audit_and_discrepancies.md',f'''# Audit and discrepancies

Supplied manuscript identified by contents, SHA-256 `4c67ccfccf182a85d96c7c93e81758df5c8357a4018fc978842e9a7f5ed22931`, 10 pages. All pages extracted and rendered. Original preserved at its supplied Downloads path. No TeX source found in the checkout. Initial local and remote HEAD were `4a70722cf5eb5e1a8b0bfc03ff97ca03ac8f63c1`; branch `codex/research-rebuild-20260907` isolates new tracked work. Existing untracked files were recorded and preserved.

The embedded figures match all five repository PNGs exactly after decoding their pixels. See `tables/manuscript_figure_provenance.csv`. In particular, page 8 Figure 4 matches `fuck it/fig3_threshold_sweep.png`; `generate_fuck_it_figures.py` imposes its enrichment via `target_enrichment_ratio * (1 + .15 * (.819 - th))`. Its edge-count axis also labels full counts as GCC counts. A distinct implementation in `run_statistical_redesign_pipeline.py` actually resamples 50 times at each threshold; that does not establish provenance for the formula-generated figure. Replacement sweeps genuinely recompute graphs, EBC, dropout, enrichment and adjusted models for each prespecified density.

Historical tissue CV={hist['GSE115513']['hard_mean_cv']:.9f} and {hist['GSE115513']['soft_mean_cv']:.9f}, reduction={100*hist['GSE115513']['reduction']:.5f}%. These reproduce the image, not Results 3.4's .218/.035 or the 84% caption. The provenance of the latter pair remains unresolved. No fabrication motive is inferred.

Table 1 mixes the nominal intercept SE/z/interval with clustered distance and EBC inference. The clustered intercept is SE=.04207247, z=-158.9806; the displayed .013/-519.39 comes from nominal covariance. EBC's exact clustered SE=.00481462 rounds to .005. The historical serum distance OR=.19969582 rounds to .200. Small formatting differences are distinct from the substantive covariance mismatch.

Confirmed methods discrepancies: median rather than mean absolute deviation; complete-case deletion beyond the stated 20% filter; max>50-triggered tissue log2(x+1); positive hard edges versus unsigned soft weights; GCC focal nodes with all-500-node neighbors; undefined correlations replaced by zero in historical routines; and unsupported zero-effect annotation/negative-EBC predictor grid. Historical functions were not imported because they execute analyses and overwrite outputs at module top level.

Historical covariance and ordering sensitivity:

{md(order_summary)}

Every substantive numerical result, figure and major inferential claim is covered by `tables/claim_verdicts.csv` ({len(claims)} entries). Supporting source line hits are in `tables/source_audit_locations.csv`. Administrative author declarations and completeness of the literature's support remain explicitly outside independent verification.
''')
    put('data_and_preprocessing.md',f'''# Data and preprocessing

Both original local matrices were re-downloaded from official NCBI FTP and matched byte-for-byte. Exact URLs, retrieval dates and SHA-256 hashes: `manifests/input_provenance.json`. Matrix metadata sample IDs and expression columns match exactly and are unique. All inclusion/exclusion and selected-node identities are in `manifests/`.

{md(q)}

GSE115513 uses Agilent GPL18402 (miRBase v19 in the study), total gene signals scaled using sample 75th percentiles. Deposited zeros remain zeros; the complete matrix has 66.36% zeros. There are 822 duplicate probe vectors, dominated by all-zero probes; constant probes are excluded before ranking. All 750 selected recorded individual IDs are distinct; matched normal/adenoma specimens elsewhere in the accession are excluded. Available organ, site, age, sex and stage metadata are preserved. The word 'rep' in an array title is not treated as proof of a technical replicate when the individual field identifies distinct subjects.

GSE73002 uses Toray GPL18941 v20. Depositor preprocessing subtracts negative-control background for detected signals, substitutes undetected signals with an array-specific minimum-minus-0.1 log2 value, then quantile-normalizes. Values already on the log2 scale include valid negatives. No further log, truncation or imputation in the primary analysis. Exactly 584 probes have some missingness and are removed by complete-case filtering, leaving 1956. There are no duplicated entire sample or probe expression vectors. Titles provide TR_B (74) versus TS_B (1206) study-set labels; these are not verified laboratory batches or patient IDs. Patient independence is assumed, not established by unique array IDs.

Primary MAD is median absolute deviation on the supplied scale, removing nonconstant/complete-case failures before selecting 500. All selected miRNAs/probes remain in graph construction including isolates. The log1p tissue feature sensitivity distinguishes rank invariance on fixed probes from changed selection. The serum median-imputation sensitivity is separately labeled exploratory and learned once on its reference cohort; it does not override primary complete-case analysis.

Rank-PCA and metadata-stratum correlation tables are diagnostic. Serum PC1 explains about 82.4% of selected rank variance and nearly tracks the sample median; this is compatible with a global shared signal, detection-floor/normalization patterns or biology, and cannot identify their causes without raw flags and richer laboratory metadata. No automatic batch adjustment was applied. Stratum analyses and summary distributions are saved rather than interpreted as causal corrections.

Official references: [Slattery et al. original colorectal study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4766359/), [GSE73002 study record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73002). The GEO processed measurements do not support a claim of fully uncensored underlying biology.
''')
    put('statistical_validation.md',f'''# Statistical validation

The primary model estimates the conditional bootstrap dropout surface on fixed observed edges. A single edge's count over independent resamples has a binomial Monte Carlo interpretation. Shared resamples correlate outcomes across edges; that does not itself make the edge's marginal count overdispersed. No independent-edge likelihood standard errors are used as biological uncertainty.

Corrected coefficient intervals use the sandwich of complete-replicate score vectors, divided by B, with the binomial mean-model Hessian as bread. They quantify variability in numerical estimation of the conditional bootstrap functional. They preserve covariance between all edges, including disjoint edges, arising from a common resample. They do not quantify finite-cohort population error or graph-selection error. Patient-sampling uncertainty is conceptually separate from Monte Carlo precision. Effect tables label local interpolation and are not causal effects or equivalence tests. Raw and log-transformed EBC coefficients use different units; cohort-specific SD effects are not common raw-unit effects.

Primary conditional EBC results:

{md(ec[['analysis','model','coefficient','mc_se','mc_ci_low','mc_ci_high','edges','valid_B']])}

The uncertainty model uses empirical bootstrap SD of the transformed Spearman correlations, not a Pearson Fisher formula. Near-zero SD probes/edges are explicitly excluded from the uncertainty model; no undefined correlation becomes zero. Its reported score intervals hold that estimated SD and the derived spline basis fixed, so they omit additional Monte Carlo error in estimating the uncertainty predictor. The ordinary distance-spline plus EBC intervals do not have that generated-predictor limitation. Spline bases, design matrices, coefficient covariance and scaling constants are saved. Calibration/residual plots compare saved observations with model predictions. Deviance is a descriptive misspecification diagnostic, not evidence of independent biological edges.

Held-out reference/validation patient-array halves:

{md(prediction)}

Reference-half selection, cutoff and EBC predictors are frozen. Validation has its own independent bootstrap draws; unavailable validation probes are excluded only from the evaluated reference-pair universe. The saved intervals currently describe validation Monte Carlo error, not uncertainty in fitting the reference model or between patient populations. Random edge splitting was not used. This is limited internal validation. Serum reference-model transfer is poor in absolute terms: distance-only linear validation log loss is {read('GSE73002_heldout_prediction').set_index('model').loc['linear_distance','validation_logloss']:.5f} and the uncertainty-spline score is {read('GSE73002_heldout_prediction').set_index('model').loc['uncertainty_distance','validation_logloss']:.5f}. EBC worsens each tested serum validation model. Good in-sample conditional calibration therefore does not establish generalization to held-out samples.

Expression-level simulations: eight scenarios, 200 independent datasets each and 200 resamples per dataset. Latent-factor covariance is PSD by construction; normal cases have exact Spearman targets. Ties, outliers and batch mixtures change the population target, so coverage against an unmodified Gaussian target is deliberately not reported for them. Missingness is independent in its declared simulation. Known zero/alternative targets concern **pairwise Spearman correlations**, not an adjusted EBC effect. Pairwise correlation null rejection and coverage must not be relabeled as EBC false-positive rate or power.

{md(simulation)}

The 20x100 nested pilot per cohort reselects features and cutoff; it profiles full-estimator variability but supplies no automatically valid population CI. Its changing-edge estimand is a procedure-dependent standardized coefficient. An expression-level null with a known zero adjusted-EBC target and a verified nonzero alternative has not been established, so population EBC bias, coverage, false-positive rate, power and equivalence remain **unresolved**. A larger nested run would not fix this calibration gap. See `nested_pilot.csv` for all completed/failed runs and comparison of linearized versus exact Monte Carlo refits.

The 1000-feature tissue setting can fail because rare selected probes become constant in bootstrap samples; failures are counted instead of zero-filled. All requested settings, including unsuccessful ones, are retained. Tests cover alignment/subjects, feature/missingness policy, rank reranking, graph signs/ties/denominator, frozen cutoff, independent seeds, undefined correlations, dropout/appearance, CV arithmetic and covariance relabeling invariance. Execution reports and final checks provide actual test results.
''')
    put('results_and_limitations.md',f'''# Corrected results and limitations

These are executed empirical results from `20260907_rebuild_v1`, with historical reproduction isolated under `historical/`. They support review of the conditional stability analysis; the manuscript's stronger population-level story is not yet validated.

## Cohorts and primary graphs

{md(primary[['cohort','sample_count','nodes','full_edges','theta','components','isolates','gcc_nodes','gcc_edges','mean_retained_dropout','mean_absent_appearance']])}

All primary graphs use unsigned Spearman, 500 MAD-selected probes on the supplied scale, 2.5% full-pair density and 1000 valid conditional draws. Pair appearance uses the same fixed all-pair universe. Positive-only sensitivities are separate; any numerical agreement occurs because these stringent retained-edge cutoffs have no qualifying negative correlations, not because sign conventions are conceptually identical.

## Distance and EBC

{md(ec[['analysis','model','coefficient','mc_ci_low','mc_ci_high']])}

Distance dominates the broad dropout gradient, while EBC estimates depend on mean-model specification and context. These are descriptive conditional associations with Monte Carlo intervals, not biological no-effect tests. No equivalence bound was justified. Held-out predictive comparisons appear in `heldout_comparison.csv`; all density/feature/missingness results are retained, including reversals and failures.

## Hard versus soft

{md(st[['analysis','support','focal','nodes','valid_paired_nodes','undefined_nodes','mean_hard_cv','mean_soft_cv','ratio_mean_reduction','reduction_mc_low','reduction_mc_high']])}

CV=sample SD/mean. All 500 focal identities are retained in replicate files; undefined zero-mean CVs are disclosed, not converted to zero. 'All nodes' means are over the explicitly reported finite paired-CV subset and can be dominated by nearly isolated nodes. Fixed baseline GCC focal summaries are more directly comparable to historical figure summaries, but still use corrected preprocessing/graph conventions. Full soft strength contributes all selected neighbors; common support only contributes baseline hard-retained neighbors. A ratio of mean CVs differs from the mean of nodewise ratios. A lower CV is not biological accuracy.

Historical tissue means {hist['GSE115513']['hard_mean_cv']:.8f}/{hist['GSE115513']['soft_mean_cv']:.8f} reproduce **{100*hist['GSE115513']['reduction']:.4f}%**, not 84%. The manuscript's .218/.035 pair was not reproduced. Corrected percentages are newly computed estimates and must not be substituted to claim retrospective reproduction.

## Sensitivities and biology

{md(sensitivity)}

{md(external)}

External transfer uses 33 GSE41655 adenocarcinoma arrays and 204 mature-miRNA mappings to a discovery-selected common node set. Unique aliases come from official platform annotation; no speculative 3p/5p renaming. Hard degree top-20 overlap is 13/20 and soft strength overlap is 12/20; this small cross-platform cohort does not demonstrate superior soft-weighted biological recovery. Patient independence and lack of overlap are supported by different study provenance but cannot be proven from global patient identifiers. Named hub tables give computed identities and selection frequencies. No targets/pathways or ceRNA mechanism were inferred from miRNA coexpression alone.

Limitations: processed values reflect upstream detection/normalization decisions; no full raw detection-flag reconstruction; incomplete serum patient/batch metadata; limited external sample size and mapping coverage; conditional fits and Monte Carlo intervals are not population inference; one internal split; exploratory nested pilot with unvalidated EBC target; percentile Spearman calibration varies by scenario. The report exposes these limits rather than calling the original zero-effect, causal-confounding or biological-superiority claims established.
''')
    put('corrected_working_sections.md',f'''# Replacement working Methods, Results and limitations

AI-assisted working sections for scientific review; not a submission-ready manuscript. Original PDF and declarations remain unchanged. Figure captions live in `figure_captions.md` and are generated from saved result sources.

## Methods
Official GSE115513 and GSE73002 series matrices were independently downloaded and checksum-verified. Exact metadata fields selected 750 carcinoma arrays with unique recorded subjects and 1280 breast-cancer serum arrays without an explicit patient identifier. Complete, nonconstant probes were ranked by median absolute deviation on their deposited processed scale; the first 500 were retained. Spearman correlation was recomputed after reranking every nonparametric sample resample. The unsigned graph used the off-diagonal 97.5th percentile as a fixed cutoff; all selected nodes and unique unordered pairs were accounted for. Conditional disappearance among retained edges and appearance among initially absent pairs were evaluated using 1000 independent resamples per cohort.

Distance-only linear and natural-spline binomial mean models were compared with EBC-augmented models and models using empirical bootstrap correlation SD to standardize the margin. The model response was the vector of bootstrap dropout counts, while uncertainty was based on complete-replicate vectors and labeled Monte Carlo uncertainty. Disjoint cohort halves supported limited internal evaluation with reference features and graph predictors fixed. Hard degree and compatible abs(r)^beta strength used identical resamples, sign and focal nodes, both with full and fixed common edge support. Named hub reproducibility was assessed internally and across 204 uniquely mapped miRNAs in 33 independent-study GSE41655 adenocarcinoma arrays.

## Results
{md(primary[['cohort','sample_count','nodes','full_edges','theta','gcc_nodes','gcc_edges']])}

{md(ec[['analysis','model','coefficient','mc_ci_low','mc_ci_high']])}

{md(st[st.focal.eq('gcc_focal')][['analysis','support','mean_hard_cv','mean_soft_cv','ratio_mean_reduction']])}

The historical tissue figure was independently reproduced at a 63.8495% CV reduction; its caption's 84% claim and text's different CV pair were not reproduced. Historical tissue and serum p-values were numerically reproduced under their specified one-endpoint covariance estimator but do not establish population-level negligible effects. External top-20 overlap was 65% for hard degree and 60% for soft strength on the common mapped node set.

## Limitations and interpretation
Threshold proximity explains much of the conditional dropout gradient. These results neither establish a causal EBC effect nor demonstrate equivalence to zero. Estimator calibration at the finite-patient level remains unresolved, and stronger biological recovery claims require additional validation. Lower weighted-strength CV is a variability result, not proof of biological accuracy or a ceRNA mechanism. Breast serum is cross-context evaluation, not colorectal biological replication. All prespecified sensitivity failures and simulation limits must accompany any subsequent manuscript revision.
''')
    status={p.stem:json.loads(p.read_text()) for p in (RUN/'logs').glob('status_*.json')}
    if 'status_sensitivity_GSE115513' in status:
        status['status_sensitivity_both']['GSE115513']=status['status_sensitivity_GSE115513']['GSE115513']
        status['superseded_receipt']='Original failed sensitivity tissue receipt preserved on disk; latest continuation used here. The 1000-feature setting remains explicitly failed.'
    summary={'run_id':'20260907_rebuild_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code_commit':git('rev-parse','HEAD'),'config_sha256':sha(BASE/'configs/primary.json'),'cohorts':q.to_dict('records'),'primary_networks':primary.to_dict('records'),'conditional_ebc':ec.to_dict('records'),'hard_soft':st.to_dict('records'),'heldout_comparison':prediction.to_dict('records'),'historical':hist,'external':external.to_dict('records'),'simulations':simulation.to_dict('records'),'claim_verdict_counts':pd.DataFrame(claims).resolution_status.value_counts().to_dict(),'stage_status':status,'readiness':'not yet scientifically reviewable as a fully corrected confirmatory study','reason':'Population-level EBC calibration is unresolved; conditional/descriptive evidence package is available for review.'}
    write_json(RUN/'results_summary.json',summary)
    put('final_readiness_report.md',f'''# Final readiness decision

**Not yet scientifically reviewable as a complete confirmatory study.** The executed conditional/descriptive rebuild and provenance audit are reviewable now. This is not a claim of publication readiness.

The pivotal unresolved requirement is population-level EBC calibration: no expression-level DGP with a verified zero adjusted-EBC target and a demonstrably nonzero alternative has been established. The nested pilot and correlation-coverage simulations cannot substitute for that. Accordingly, no biological EBC p-values, equivalence claims or validated population intervals are asserted.

Other limits are missing serum patient/batch identifiers and raw detection flags, one modest external cohort with incomplete mapping, and failures of the 1000-feature tissue conditional setting where rare probes become constant. Every attempted stage, runtime, failure and output appears in `logs/status_*.json`, `tables/targeted_sensitivity_summary.csv`, `tables/nested_pilot.csv` and `results_summary.json`. See `reproducibility_README.md` for exact execution/resume commands. Original files and pre-fix results were preserved. Nothing was pushed, published, submitted, or communicated externally.

Verdicts: {pd.DataFrame(claims).resolution_status.value_counts().to_dict()} across {len(claims)} ledger entries. These classify claims, not researcher intent. Source verification, numerical reproduction, corrected conditional evidence and unresolved population inference are explicitly distinct.

Unfinished calibration cannot be solved by merely running the pilot longer. Runnable continuation: `python research_rebuild/src/nested_pilot.py` profiles the declared estimator; a new target-validation method must be specified in an amendment and tested on fresh expression simulations before using an outer distribution as an inferential CI. Large brute-force nested production was not launched without that justification.
''')
if __name__=='__main__':main()
