from base import *
import shutil
def main():
    start=time.perf_counter()
    truth=[]
    for p in sorted((RUN/'models').glob('simulation_population_truth_*.npz')):
        z=np.load(p);hub=z['latent_hub_nodes'];C=z['covariance'];latent=z['population_latent_spearman'];obs=z['population_observed_spearman'];a=z['population_hard_adjacency'];s=z['population_soft_adjacency'];null=len(hub)==0
        truth.append({'cell':p.stem.removeprefix('simulation_population_truth_'),'p':len(C),'minimum_eigenvalue':np.linalg.eigvalsh(C).min(),'latent_hub_count':len(hub),'latent_hubs':' '.join(map(str,hub)),'observed_population_hard_edges':a.sum()/2,'observed_population_soft_mean_weight':s[np.triu_indices(120,1)].mean(),'population_hard_hub_recovery':len(set(top(a.sum(axis=1),12))&set(hub))/12 if not null else np.nan,'population_soft_hub_recovery':len(set(top(s.sum(axis=1),12))&set(hub))/12 if not null else np.nan,'latent_vs_observed_strength_rho':rho((latent-np.eye(120)).sum(axis=1),(obs-np.eye(120)).sum(axis=1)) if not null else np.nan,'null_note':'no latent hubs/modules; literal hard >=zero-quantile target is complete due all ties' if null else 'hub truth from verified nuisance-free module loadings','source':str(p),'sha256':sha(p)})
    csv('simulation_truth_summary',truth)
    old=pd.read_csv(PREV/'tables/updated_claim_ledger.csv');old['previous_verdict']=old.revised_verdict;old['exact_claim']=old.stated_value;old['verdict']=old.revised_verdict;old['evidence_level']='previous audit retained';old['limitation']=old.revision_reason;old['permitted_wording']=old.revision_reason;old['current_source']=old.followup_evidence;old['new_run']=RUN.name
    additions=[
      ('B01','On fixed full support, affine background can reduce strength CV without changing exact ranks.','supported','mathematical diagnostic and conditional arrays','analytic derivation +1000-draw checks','stability_control.csv','lambda<1, c>0, fixed p; finite precision and ties disclosed; no path/module invariance','Lower strength CV alone is insufficient evidence of added hub-ranking information.'),
      ('B02','Lower soft CV guarantees more stable top-k selection.','contradicted','conditional empirical arrays','saved-draw hub selections','hub_stability_summary.csv','k and rank universes fixed; no biological population claim','Serum soft top20 stability was lower than hard despite lower soft CV.'),
      ('B03','Soft representation improves planted module recovery in every biological network.','weakened','specified independent Gaussian simulations','known-structure benchmark','simulation_summary.csv','K4 known; simple factors; R30; density and nuisance conventions restricted','Soft spectral modules improved recovery in these simulated signal conditions; this is not universal biological accuracy.'),
      ('B04','Soft weighting has consistently better external hub agreement.','contradicted','two exploratory colorectal tissue cohorts','source-mapped direct external comparisons','external_summary.csv','cohort/platform composition confounding; point estimates only','No consistent external soft hub advantage was observed under the frozen policies.'),
      ('B05','The serum dropout model transfers as a well-calibrated probability model.','contradicted','20 overlapping serum array half splits','saved predictions and honest baselines','predictive_summary.csv','evaluation subset and target mismatch; no independent split CI','Serum model Brier exceeded0.25 and predictions substantially underestimated heldout disappearance.'),
      ('B06','The discovered modules are established biological functional units.','unresolved','discovery tissue association modules','fixed-membership exploratory connectivity','module_preservation.csv','one module has384 nodes; no independent pathway truth or bias-aware enrichment','The two evaluable modules are association-defined examples with mixed external connectivity agreement.'),
      ('B07','This evidence establishes a distinct BioSystems biological-organization contribution.','unresolved','contribution and journal-fit assessment','closest primary literature and official scope','novelty_comparison.csv','existing principles and close recent work; no new mechanism demonstrated','The evidence supports a methodological boundary; distinct biological insight remains unestablished.'),
      ('B08','GSE29622 meets the frozen external node-coverage requirement.','contradicted','metadata-selected candidate eligibility','official alias mapping','GSE29622_mapping_audit.csv','98 usable identities below100; no outcome-based threshold relaxation','GSE29622 was screened but not analyzed as external network evidence; next-ranked GSE48267 was used.')]
    new=[]
    for id,claim,verdict,scope,level,src,lim,word in additions:new.append({'claim_id':id,'exact_claim':claim,'scope':scope,'verdict':verdict,'previous_verdict':'new','evidence_level':level,'current_source':src,'limitation':lim,'permitted_wording':word,'new_run':RUN.name})
    ledger=pd.concat([old,pd.DataFrame(new)],ignore_index=True);csv('updated_claim_ledger',ledger)
    # Compact prior reconciliation explicitly describes the level of verification.
    prior=pd.read_csv(PREV/'tables/verified_results_ledger.csv');prior['current_verification']='prior source/cached reconstruction; artifact hashes independently rechecked in this run';prior['verification_status']=np.where(prior.status.eq('verified'),'reproduced_from_cache','contradicted');csv('numerical_reconciliation',prior)
    for name in ['novelty_comparison','cohort_screening']:
        shutil.copyfile(RUN/'research'/f'{name}.csv',RUN/'tables'/f'{name}.csv')
    screening=pd.read_csv(RUN/'tables/cohort_screening.csv');screening.loc[screening.accession.eq('GSE48267'),'status']='analyzed_after_mapping_gate_204_nodes';csv('cohort_screening',screening)
    consequence=pd.read_csv(RUN/'tables/graph_consequences.csv');summ=[]
    for acc,g in consequence.groupby('cohort'):
        for outcome in ['hard_rank_rho','soft_rank_rho','hard_top20_overlap','soft_top20_overlap']:
            summ.append({'cohort':acc,'endpoint':outcome,'draws':len(g),'mean':g[outcome].mean(),'rho_with_total_edges':rho(g.M,g[outcome]),'rho_with_disappearances':rho(g.D,g[outcome]),'rho_with_appearances':rho(g.G,g[outcome]),'interpretation':'conditional within-draw association; not causal/population inference'})
    csv('graph_consequence_summary',summ)
    intervals=pd.read_csv(PREV/'tables/cv_fixed_S_summary.csv');attempt=pd.read_csv(PREV/'tables/cv_interval_attempts.csv');assert (intervals.interval_failed+intervals.interval_valid==1000).all()
    for (acc,support,focal),g in attempt.groupby(['cohort','support','focal']):
        oldrow=intervals[intervals.cohort.eq(acc)&intervals.support.eq(support)&intervals.focal.eq(focal)].iloc[0];assert g.valid.sum()==oldrow.interval_valid and len(g)==1000
    csv('fixed_S_interval_status',intervals)
    sim=pd.read_csv(RUN/'tables/simulation_results.csv');precision=pd.read_csv(RUN/'tables/simulation_precision.csv');definition=[]
    for (cell,method),g in sim.groupby(['cell','method']):definition.append({'cell':cell,'method':method,'prefix_defined_datasets':int(g.cv_prefix50_minus100.notna().sum()),'datasets':len(g),'paired_finite_nodes_min':g.paired_finite_CV_nodes.min(),'paired_finite_nodes_max':g.paired_finite_CV_nodes.max()})
    csv('simulation_domain_accounting',definition)
    missing=[{'analysis':'miRNA target/pathway enrichment','status':'not_run','missing_input':'No assembled independent versioned target/pathway matrix and measured-universe coverage model in this bounded study.','consequence':'No enrichment p-values or functionally validated module claim. Primary functional context is explicitly separate.','next_requirement':'Needed only if functional module claims are pursued; use empirical measured-miRNA-set null, annotation coverage and multiplicity.'},{'analysis':'population EBC equivalence','status':'not_run','missing_input':'No justified equivalence bounds or calibrated biological sampling inference.','consequence':'No zero-effect/equivalence claim; outside scope.','next_requirement':'Not needed for present descriptive boundary.'},{'analysis':'GSE29622 external network','status':'blocked_missing_input','missing_input':'98 rather than >=100 unique complete variable mapped discovery identities','consequence':'No network outcomes compared; fixed fallback GSE48267 used.','next_requirement':'No relaxation or further cohort hunt in this run.'}]
    csv('missing_input_inventory',missing)
    status('synthesis',start)
if __name__=='__main__':main()
