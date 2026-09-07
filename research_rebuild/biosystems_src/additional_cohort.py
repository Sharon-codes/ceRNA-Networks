from base import *
from empirical import assess_external
from external import read_external
from collections import defaultdict
import re
def main():
    start=time.perf_counter();acc='GSE29622';path=RUN/'data'/f'{acc}_series_matrix.txt.gz';df,meta,_=read_external(path);gpl=pd.read_csv(RUN/'research/GPL11162_annotation.csv',dtype=str).fillna('');oldgpl=pd.read_csv(BASE/'manifests/GPL18402_annotation.csv',dtype=str).fillna('');aliases=defaultdict(set)
    for _,r in oldgpl.iterrows():
        for a in [r.ID]+r.ACCESSION_STRING.split('|'):
            a=re.sub(r'_v\d+(?:\.\d+)?$','',a.strip())
            if a.startswith('hsa-'):aliases[a].add(r.ID.strip())
    discovery=pd.read_csv(BASE/'manifests/GSE115513_selected_probes.csv').probe_id.tolist();annotation=gpl.set_index('ID').miRNA_ID;rows=[]
    for id in df.index:
        name=annotation.get(id,'');targets=sorted(aliases.get(name,set()));rows.append({'probe':id,'platform_mature_name':name,'n_alias_targets':len(targets),'mapped_mature':targets[0] if len(targets)==1 else ''})
    m=pd.DataFrame(rows);m['duplicate_destination']=m.mapped_mature.duplicated(keep=False)&m.mapped_mature.ne('');m['in_discovery']=m.mapped_mature.isin(discovery);eligible=(m.n_alias_targets==1)&~m.duplicate_destination&m.in_discovery
    # All65 carcinoma samples are metadata eligible; no sample selection from graph outcomes.
    assert df.shape[1]==65 and meta['preoperative chemo'].eq('N').all()
    v=df.loc[m.loc[eligible,'probe']].copy();v.index=m.loc[eligible,'mapped_mature'];valid=np.isfinite(v).all(axis=1)&(v.max(axis=1)>v.min(axis=1));v=v.loc[valid]
    m['included']=m.mapped_mature.isin(v.index)&eligible;m['reason']=np.select([m.n_alias_targets.eq(0),m.n_alias_targets.gt(1),m.duplicate_destination,~m.in_discovery,~m.included],['no unique official alias','ambiguous alias','duplicate destination','not in discovery500','missing or constant external values'],'included');csv('GSE29622_mapping_audit',m)
    write_json(RUN/'manifests/GSE29622_analysis_freeze.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'accession':acc,'n':65,'mapped_nodes':len(v),'ids':v.index.tolist(),'input_sha256':sha(path),'policy':'A discovery cutoff and B density matched, no outcome tuning; same fixed discovery modules','mature_identity':'unique GPL11162 mature names through official GPL18402 aliases; 5p/3p and star distinctions not guessed','upstream_detection':'depositor Ct>32 deemed undetectable, normalized to RNU44 and transformed to expression; no new detection repair','status':'eligible' if len(v)>=100 else 'blocked_missing_input'})
    if len(v)<100:status('additional_external',start,'blocked',reason='fewer than100 unique complete variable mapped discovery nodes',mapped=len(v));return
    r,n,s,c=assess_external(acc,v)
    for name,items in [('module_preservation',r),('named_hub_results',n),('external_summary',s),('external_affine_control',c)]:
        old=pd.read_csv(RUN/'tables'/f'{name}.csv');csv(name,pd.concat([old[old.cohort.ne(acc)],pd.DataFrame(items)],ignore_index=True))
    inventory=pd.read_csv(RUN/'tables/cohort_inventory.csv');row={'accession':acc,'source_url':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+acc,'role':'new metadata-selected exploratory external tissue','specimen':'FFPE primary colon adenocarcinoma','condition':'primary colon cancer, no preoperative chemotherapy; stages I–IV','platform':'GPL11162','assay':'TaqMan Human miRNA v2 RT-qPCR','included_arrays':65,'verified_people':np.nan,'repeat_handling':'65 unique sample titles;63 cross-links to GSE17536; no global patient linkage proof','batch_metadata':'no explicit usable technical batch; FFPE and RT-qPCR differ from discovery arrays','detection_information':'depositor Ct>32 rule; RNU44 normalization; no assay-incompatible new thresholds','input_probes':len(df),'complete_variable_probes':int((np.isfinite(df).all(axis=1)&(df.max(axis=1)>df.min(axis=1))).sum()),'analysis_features':len(v),'mapping':'unique official mature aliases; duplicate/ambiguous destinations excluded','upstream_processing':meta.data_processing.iloc[0],'input_path':str(path),'input_sha256':sha(path),'independence_status':'unique specimens; identified-person count not independently verified','global_overlap':'Moffitt source vs Utah/Kaiser discovery; global linkage unavailable'};csv('cohort_inventory',pd.concat([inventory[inventory.accession.ne(acc)],pd.DataFrame([row])],ignore_index=True));status('additional_external',start,arrays=65,mapped_nodes=len(v))
if __name__=='__main__':main()
