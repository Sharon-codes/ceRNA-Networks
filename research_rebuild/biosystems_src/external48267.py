from base import *
from empirical import assess_external
from external import read_external
from collections import defaultdict
import re
def main():
    start=time.perf_counter();acc='GSE48267';path=RUN/'data'/f'{acc}_series_matrix.txt.gz';df,meta,_=read_external(path);gpl=pd.read_csv(RUN/'research/GPL10850_annotation.csv',dtype=str).fillna('');oldgpl=pd.read_csv(BASE/'manifests/GPL18402_annotation.csv',dtype=str).fillna('');aliases=defaultdict(set)
    for _,r in oldgpl.iterrows():
        for a in [r.ID]+r.ACCESSION_STRING.split('|'):
            a=re.sub(r'_v\d+(?:\.\d+)?$','',a.strip())
            if a.startswith('hsa-'):aliases[a].add(r.ID.strip())
    # The older numerical parser intentionally retains only selected metadata fields.
    # Use the independently archived complete GEO metadata for the organism check.
    full=pd.read_csv(RUN/'research/GSE48267_sample_metadata.csv',dtype=str).set_index('geo_accession').loc[df.columns]
    assert full.tissue.tolist()==meta.tissue.tolist()
    meta=full
    mask=meta.tissue.eq('Colon Cancer Tumor')&meta.platform_id.eq('GPL10850')&full.organism_ch1.eq('Homo sapiens');assert mask.sum()==61
    keys=meta['data source']+':'+meta.title.str.extract(r'_patient(\d+)$',expand=False);assert keys[mask].nunique()==61
    discovery=pd.read_csv(BASE/'manifests/GSE115513_selected_probes.csv').probe_id.tolist();annot=gpl.set_index('ID').miRNA_ID;rows=[]
    for id in df.index:
        name=annot.get(id,'');targets=sorted(aliases.get(name,set()));rows.append({'probe':id,'platform_mature_name':name,'alias_targets':len(targets),'mapped_mature':targets[0] if len(targets)==1 else ''})
    m=pd.DataFrame(rows);m['duplicate_destination']=m.mapped_mature.duplicated(keep=False)&m.mapped_mature.ne('');m['discovery_selected']=m.mapped_mature.isin(discovery);eligible=m.alias_targets.eq(1)&~m.duplicate_destination&m.discovery_selected
    v=df.loc[m.loc[eligible,'probe'],mask].copy();v.index=m.loc[eligible,'mapped_mature'];valid=np.isfinite(v).all(axis=1)&(v.max(axis=1)>v.min(axis=1));v=v.loc[valid]
    m['included']=m.mapped_mature.isin(v.index)&eligible;m['reason']=np.select([m.alias_targets.eq(0),m.alias_targets.gt(1),m.duplicate_destination,~m.discovery_selected,~m.included],['no alias','ambiguous','duplicate destination','outside discovery500','missing/constant'],'included');csv('GSE48267_mapping_audit',m)
    write_json(RUN/'manifests/GSE48267_analysis_freeze.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'n':61,'mapped':len(v),'ids':v.index.tolist(),'input_sha256':sha(path),'eligibility':'metadata-qualified tumors; no graph outcomes used','policy':'same A/B and discovery modules as configuration','mapping_sha256':sha(RUN/'tables/GSE48267_mapping_audit.csv')})
    if len(v)<100:status('additional_external_GSE48267',start,'blocked',reason='mapping minimum unmet',mapped=len(v));return
    r,n,s,c=assess_external(acc,v)
    for name,items in [('module_preservation',r),('named_hub_results',n),('external_summary',s),('external_affine_control',c)]:
        old=pd.read_csv(RUN/'tables'/f'{name}.csv');csv(name,pd.concat([old[old.cohort.ne(acc)],pd.DataFrame(items)],ignore_index=True))
    inventory=pd.read_csv(RUN/'tables/cohort_inventory.csv');row={'accession':acc,'source_url':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+acc,'role':'new metadata-selected exploratory external tissue after GSE29622 mapping failure','specimen':'primary colon tumors;30 FFPE SBU,31 frozen WU','condition':'primary colon cancer; pretreatment unverified','platform':'GPL10850','assay':'Agilent Human miRNA V3 microarray','included_arrays':61,'verified_people':61,'repeat_handling':'one tumor per site-qualified patient key;61 paired normals excluded','batch_metadata':'site/preservation/race are coupled; no correction fitted on these outcomes','detection_information':'depositor signal floor1; no complete compatible detection calls used','input_probes':len(df),'complete_variable_probes':int((np.isfinite(df.loc[:,mask]).all(axis=1)&(df.loc[:,mask].max(axis=1)>df.loc[:,mask].min(axis=1))).sum()),'analysis_features':len(v),'mapping':'unique official mature aliases; no 5p/3p guessing or many-to-one aggregation','upstream_processing':meta.data_processing.iloc[0],'input_path':str(path),'input_sha256':sha(path),'independence_status':'61 unique site-qualified patient keys, consistent with original primary publication PMID24865442','global_overlap':'SBU/WU versus Utah/Kaiser; global linkage unavailable'}
    csv('cohort_inventory',pd.concat([inventory[inventory.accession.ne(acc)],pd.DataFrame([row])],ignore_index=True));status('additional_external_GSE48267',start,arrays=61,mapped_nodes=len(v))
    print(acc,'executed:',len(v),'mapped nodes,61 tumors',flush=True)
if __name__=='__main__':main()
