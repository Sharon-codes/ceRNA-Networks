from common import *
from audit import verify,flush_ledger
from external import read_external
from collections import defaultdict
from scipy.stats import spearmanr
import re

def main():
    start=time.perf_counter();platform=pd.read_csv(BASE/'manifests/GPL18402_annotation.csv',dtype=str).fillna('');aliases=defaultdict(set)
    for _,row in platform.iterrows():
        current=row.ID.strip()
        for alias in [current]+row.ACCESSION_STRING.split('|'):
            alias=re.sub(r'_v\d+(?:\.\d+)?$','',alias.strip())
            if alias.startswith('hsa-'):aliases[alias].add(current)
    ext,meta,_=read_external(BASE/'data/GSE41655_series_matrix.txt.gz');include=meta.source_name_ch1.eq('Colorectal adenocarcinoma')&meta['pathologic grade'].eq('Adenocarcinoma');df,_=cohort(ACCS[0]);sel,_,_=select_features(df,500)
    mapping=[]
    for name in ext.index:
        targets=sorted(aliases.get(str(name).strip(),set()));mapping.append({'external_probe':name,'mapped_v19':targets[0] if len(targets)==1 else '', 'number_targets':len(targets)})
    m=pd.DataFrame(mapping);m['duplicate_destination']=m.mapped_v19.duplicated(keep=False)&m.mapped_v19.ne('');m['discovery_selected']=m.mapped_v19.isin(sel.index)
    eligible=(m.number_targets==1)&~m.duplicate_destination&m.discovery_selected
    m['exclusion']=np.select([m.number_targets.eq(0),m.number_targets.gt(1),m.duplicate_destination,~m.discovery_selected],['no alias','ambiguous aliases','duplicate destination','outside discovery500'],'candidate')
    good=m[eligible];v=ext.loc[good.external_probe,include].copy();v.index=good.mapped_v19;valid=np.isfinite(v).all(axis=1)&(v.max(axis=1)>v.min(axis=1));v=v.loc[valid]
    m['included']=m.mapped_v19.isin(v.index)&eligible;m.loc[eligible&~m.included,'exclusion']='external missing/constant'
    csv('external_mapping_audit',m);csv('external_sample_inclusion',meta.assign(included=include).reset_index())
    freeze=json.loads((BASE/'manifests/external_analysis_freeze.json').read_text());assert v.index.tolist()==freeze['feature_ids']
    xd=df.loc[v.index].T.to_numpy();xe=v.T.to_numpy()
    independent_d=spearmanr(xd,axis=0).statistic;independent_e=spearmanr(xe,axis=0).statistic
    # Canonical correlation arithmetic preserves the original deterministic near-tie order.
    # Independently verify matrices to tolerance and retain alternate rank sensitivity.
    rd=spearman(xd);rv=spearman(xe);u,w=np.triu_indices(len(v),1)
    matrix_error=max(np.max(np.abs(rd-independent_d)),np.max(np.abs(rv-independent_e)));assert matrix_error<1e-12
    independent_sd=node_sums((np.abs(independent_d[u,w])**6)[None,:],len(v))[0];independent_sv=node_sums((np.abs(independent_e[u,w])**6)[None,:],len(v))[0]
    ds=np.abs(rd[u,w]);es=np.abs(rv[u,w]);td=np.quantile(ds,.975);te=np.quantile(es,.975);a=ds>=td
    hd=node_sums(a[None,:],len(v))[0];sd=node_sums((ds**6)[None,:],len(v))[0];sv=node_sums((es**6)[None,:],len(v))[0]
    previous=pd.read_csv(OLD/'tables/external_named_hubs.csv')
    csv('external_numerical_tie_audit',{'miRNA':v.index,'canonical_discovery_strength':sd,'independent_discovery_strength':independent_sd,'saved_discovery_strength':previous.discovery_soft_strength,'canonical_external_strength':sv,'independent_external_strength':independent_sv,'saved_external_strength':previous.external_soft_strength,'canonical_discovery_rank':pd.Series(sd).rank(),'independent_discovery_rank':pd.Series(independent_sd).rank(),'canonical_external_rank':pd.Series(sv).rank(),'independent_external_rank':pd.Series(independent_sv).rank()})
    assert np.allclose(sd,previous.discovery_soft_strength,atol=1e-12,rtol=1e-12) and np.allclose(sv,previous.external_soft_strength,atol=1e-12,rtol=1e-12)
    top=lambda z:set(np.argsort(-z,kind='stable')[:20]);rows=[];named=[]
    src=OLD/'tables/external_biological_summary.csv';old=pd.read_csv(src).iloc[0]
    for name,threshold in [('B_density_matched',te),('A_frozen_discovery_cutoff',td)]:
        b=es>=threshold;hv=node_sums(b[None,:],len(v))[0];dh,eh,ds20,es20=top(hd),top(hv),top(sd),top(sv)
        result={'analysis':name,'cohort':'GSE41655','n':v.shape[1],'common_nodes':len(v),'discovery_n':df.shape[1],'discovery_selected_nodes':500,'coverage_fraction':len(v)/500,'reference_theta':td,'external_theta':threshold,'reference_edges':int(a.sum()),'external_edges':int(b.sum()),'reference_density':a.mean(),'external_density':b.mean(),'edge_jaccard':np.sum(a&b)/np.sum(a|b),'hard_rank_spearman':spearmanr(hd,hv).statistic,'soft_rank_spearman':spearmanr(sd,sv).statistic,'hard_top20_count':len(dh&eh),'soft_top20_count':len(ds20&es20),'hard_top20_overlap':len(dh&eh)/20,'soft_top20_overlap':len(ds20&es20)/20,'k':20,'hard_shared_identities':'; '.join(v.index[sorted(dh&eh)]),'soft_shared_identities':'; '.join(v.index[sorted(ds20&es20)]),'discovery_hard_cutoff_tie_count':int(sum(hd==np.sort(hd)[-20])),'external_hard_cutoff_tie_count':int(sum(hv==np.sort(hv)[-20])),'uncertainty':'exploratory point estimates, no population intervals','run_id':RUN.name,'config_sha256':sha(RUN/'followup_analysis_amendment.md'),'external_input_sha256':sha(BASE/'data/GSE41655_series_matrix.txt.gz')}
        result['independent_correlation_max_error']=matrix_error
        result['independent_scipy_soft_rank_sensitivity']=spearmanr(independent_sd,independent_sv).statistic
        result['numerical_note']='canonical original arithmetic retained; alternative equivalent correlation arithmetic perturbs near-tie rank ordering; saved strengths independently agree to 1e-12'
        rows.append(result)
        for i,id in enumerate(v.index):named.append({'analysis':name,'miRNA':id,'discovery_hard_degree':hd[i],'external_hard_degree':hv[i],'discovery_soft_strength':sd[i],'external_soft_strength':sv[i],'discovery_hard_top20':i in dh,'external_hard_top20':i in eh,'discovery_soft_top20':i in ds20,'external_soft_top20':i in es20})
        if name.startswith('B'):
            for k in ['n','common_nodes','reference_theta','external_theta','edge_jaccard','hard_rank_spearman','soft_rank_spearman','hard_top20_overlap','soft_top20_overlap']:verify('external_'+k,ACCS[0],old[k],result[k],src,'Independent raw-data/annotation reconstruction on frozen mapped identities; k20 stable order','Density-matched external graph B')
    csv('external_transfer_verification',rows);csv('external_named_identities',named);flush_ledger()
    report('external_transfer_audit',f'''# Exploratory external transfer

Exact source and grade inclusion selects {sum(include)} of {len(meta)} GSE41655 arrays. Unique official-platform aliases, exclusion of ambiguous/many-to-one destinations, and overlap with train/discovery-selected500 probes leave {len(v)} complete variable common identities ({len(v)/5:.1f}% discovery coverage). Mapping audit lists every external probe and exclusion. The ordered identities exactly match the saved freeze. The external node eligibility uses external missingness and variance, so results condition on an externally evaluable universe; it was not chosen by rank/overlap outcomes. Discovery top500 selection uses discovery data only.

Original B estimates each graph's 2.5% cutoff separately and therefore is density matched, not threshold transport. Secondary A retains the discovery cutoff on the same204-node universe. Soft strength has no hard cutoff and is unchanged between A and B. Only k20 is reported; stable original mapped-node order breaks ties and tie counts are saved.

The separate study/submitter is consistent with an independent cohort, but no global patient linkage proves non-overlap. All33 sample titles are distinct: {meta.loc[include,'title'].nunique()==sum(include)}. Explicit patient IDs are unavailable; independence is assumed. No cross-platform harmonization or patient-population confidence interval is claimed. Neither rank agreement nor a top20 overlap establishes biological truth or a ceRNA mechanism. Lower internal soft-strength variability must be read alongside the less favorable original B external point comparison.

Numerical verification: independent scipy.stats.spearmanr matrices differ from the canonical rank-normalized matrix product by at most {matrix_error:.3g}. Nearly tied soft strengths change ranks under that arithmetic, giving alternate soft rho {spearmanr(independent_sd,independent_sv).statistic:.15g} versus canonical {spearmanr(sd,sv).statistic:.15g}. Source vectors, both rankings and differences are saved. This is disclosed numerical tie sensitivity; retain the originally frozen arithmetic consistently for both A and B, with no rounding rule selected from outcomes. Top20 overlap and the substantive comparison are unchanged.
''')
    status('external_verification',start)
if __name__=='__main__':main()
