"""Traceable hub prioritization and external mature-miRNA mapping."""
import json,re,time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from core import cohort,select_features,spearman,graph_from,bootstrap,node_sums,cv,rng_for
from external import read_external
from provenance import BASE,RUN,sha,write_json

def main():
    gpl19=pd.read_csv(BASE/'manifests/GPL18402_annotation.csv',dtype=str).fillna('')
    gpl20=pd.read_csv(BASE/'manifests/GPL18941_annotation.csv',dtype=str).fillna('')
    audit=[]
    for acc,gpl,col in [('GSE115513',gpl19,'miRNA_ID'),('GSE73002',gpl20,'miRNA_ID_LIST')]:
        p=pd.read_csv(BASE/'manifests'/f'{acc}_selected_probes.csv');gpl['ID']=gpl.ID.str.strip();gpl[col]=gpl[col].str.strip()
        mapping=gpl.groupby('ID')[col].agg(lambda s:sorted(set(s)))
        p['miRNA']=p.probe_id.map(lambda k:mapping.get(k,[''])[0] if len(mapping.get(k,[]))==1 else '')
        p['mapping_status']=np.where(p.miRNA.str.startswith('hsa-') & ~p.miRNA.str.contains('//|,'),'unique_platform_mapping','ambiguous_or_missing')
        p['duplicate_miRNA_mapping']=p.miRNA.duplicated(keep=False)&p.miRNA.ne('')
        p.to_csv(BASE/'manifests'/f'{acc}_named_probe_mapping.csv',index=False)
        for sign in ['unsigned','positive']:
            path=RUN/'tables'/f'{acc}_{sign}_hub_priorities.csv'
            if path.exists():
                hubs=pd.read_csv(path).merge(p,on='node',validate='one_to_one')
                hubs.to_csv(RUN/'tables'/f'{acc}_{sign}_named_hubs.csv',index=False)
        audit.append({'cohort':acc,'selected':len(p),'unique_named':sum(p.mapping_status.eq('unique_platform_mapping')),'duplicate_mapped_nodes':sum(p.duplicate_miRNA_mapping),'annotation':'platform version, no unsupported modernization'})
    pd.DataFrame(audit).to_csv(RUN/'tables/annotation_summary.csv',index=False)

    # Map v12-style mature names to v19 using alias strings from the official v19 platform.
    aliases=defaultdict(set)
    for _,row in gpl19.iterrows():
        current=row['ID'].strip()
        tokens=[current]+row['ACCESSION_STRING'].split('|')
        for token in tokens:
            token=re.sub(r'_v\d+(?:\.\d+)?$','',token.strip())
            if token.startswith('hsa-'):aliases[token].add(current)
    ext,meta,raw=read_external(BASE/'data/GSE41655_series_matrix.txt.gz')
    include=meta.source_name_ch1.eq('Colorectal adenocarcinoma') & meta['pathologic grade'].eq('Adenocarcinoma')
    meta['included']=include;meta['reason']=np.where(include,'exact adenocarcinoma source and grade','other lesion or normal');meta.to_csv(BASE/'manifests/GSE41655_selection.csv')
    mapping=[]
    for probe in ext.index:
        name=str(probe).strip();targets=sorted(aliases.get(name,set()))
        mapping.append({'external_probe':probe,'alias':name,'mapped_v19':targets[0] if len(targets)==1 else '', 'number_targets':len(targets)})
    m=pd.DataFrame(mapping);m['duplicate_destination']=m.mapped_v19.duplicated(keep=False)&m.mapped_v19.ne('');m.to_csv(BASE/'manifests/GSE41655_cross_platform_mapping.csv',index=False)
    discovery,_,_=cohort('GSE115513');selection,_,_=select_features(discovery,500)
    good=m[(m.number_targets==1)&~m.duplicate_destination&m.mapped_v19.isin(selection.index)]
    v=ext.loc[good.external_probe,include].copy();v.index=good.mapped_v19
    finite=np.isfinite(v).all(axis=1);variable=v.max(axis=1)>v.min(axis=1);v=v.loc[finite&variable]
    # Outcomes are first computed only after the mapping/eligibility manifests are saved.
    freeze={'accession':'GSE41655','selected_n':v.shape[1],'mapped_discovery_features':len(v),'independent_patient_status':'33 unique carcinoma sample titles; no explicit patient identifier, independence assumed','mapping':'unique official GPL18402 ACCESSION_STRING aliases; ambiguous and duplicate destinations excluded','patient_overlap':'separate submitter/cohort from Utah/Kaiser discovery; explicit global patient linkage unavailable','input_sha256':sha(BASE/'data/GSE41655_series_matrix.txt.gz'),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'feature_ids':v.index.tolist(),'transfer':'2.5% density estimated per cohort; unsigned, beta6; common mapped discovery-selected nodes; no external outcome selection'}
    write_json(BASE/'manifests/external_analysis_freeze.json',freeze)
    if len(v)<100 or v.shape[1]<30:
        write_json(RUN/'logs/status_biology.json',{'status':'blocked','reason':'Prespecified mapping/sample minimum failed',**freeze});return
    d=discovery.loc[v.index];rd=spearman(d.T.to_numpy());rv=spearman(v.T.to_numpy());gd=graph_from(rd);gv=graph_from(rv);u,w=np.triu_indices(len(v),1)
    a=np.abs(rd[u,w])>=gd['summary']['theta'];b=np.abs(rv[u,w])>=gv['summary']['theta']
    hd=node_sums(a[None,:],len(v))[0];hv=node_sums(b[None,:],len(v))[0];sd=node_sums((np.abs(rd[u,w])**6)[None,:],len(v))[0];sv=node_sums((np.abs(rv[u,w])**6)[None,:],len(v))[0]
    top=lambda z:set(np.argsort(-z,kind='stable')[:20])
    summary={'content':'EMPIRICAL external transfer, exploratory','accession':'GSE41655','n':v.shape[1],'common_nodes':len(v),'reference_theta':gd['summary']['theta'],'external_theta':gv['summary']['theta'],'edge_jaccard':np.sum(a&b)/np.sum(a|b),'hard_rank_spearman':spearmanr(hd,hv).statistic,'soft_rank_spearman':spearmanr(sd,sv).statistic,'hard_top20_overlap':len(top(hd)&top(hv))/20,'soft_top20_overlap':len(top(sd)&top(sv))/20,'population_CI':'not estimated','patient_identity':'assumed independent external arrays'}
    pd.DataFrame([summary]).to_csv(RUN/'tables/external_biological_summary.csv',index=False)
    pd.DataFrame({'miRNA':v.index,'discovery_hard_degree':hd,'external_hard_degree':hv,'discovery_soft_strength':sd,'external_soft_strength':sv,'discovery_hard_top20':[i in top(hd) for i in range(len(v))],'external_hard_top20':[i in top(hv) for i in range(len(v))],'discovery_soft_top20':[i in top(sd) for i in range(len(v))],'external_soft_top20':[i in top(sv) for i in range(len(v))]}).to_csv(RUN/'tables/external_named_hubs.csv',index=False)
    write_json(RUN/'logs/status_biology.json',{'status':'completed','result':summary,'mapping_audit':audit})
if __name__=='__main__':main()
