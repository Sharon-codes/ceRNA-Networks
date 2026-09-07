"""Outcome-blind accession/annotation retrieval and external compatibility assessment."""
import csv,gzip,io,json,time,urllib.request
from pathlib import Path
import pandas as pd
from provenance import BASE,RUN,sha,read_geo,write_json

def read_external(dest):
    with gzip.open(dest,'rt',encoding='utf-8') as f:text=f.read()
    header,body=text.split('!series_matrix_table_begin\n',1)
    raw=[r for r in csv.reader(io.StringIO(header),delimiter='\t') if r]
    ids=next(r[1:] for r in raw if r[0]=='!Sample_geo_accession')
    sample=[{} for _ in ids]
    for row in raw:
        if row[0]=='!Sample_characteristics_ch1':
            for i,value in enumerate(row[1:]):
                if not value:continue
                if ': ' not in value:raise ValueError('Unstructured characteristic: '+value)
                k,v=value.split(': ',1);k=k.lower()
                if k in sample[i] and sample[i][k]!=v:raise ValueError('Conflicting characteristic')
                sample[i][k]=v
        elif row[0] in ['!Sample_title','!Sample_source_name_ch1','!Sample_platform_id']:
            for i,value in enumerate(row[1:]):sample[i][row[0][8:]]=value
    df=pd.read_csv(io.StringIO(body.split('!series_matrix_table_end')[0]),sep='\t',index_col=0)
    if list(df.columns)!=ids:raise ValueError('Alignment failure')
    meta=pd.DataFrame(sample,index=ids);meta.index.name='sample_id'
    return df,meta,raw

def fetch(url,dest):
    if not dest.exists():
        with urllib.request.urlopen(url,timeout=60) as r:content=r.read()
        dest.write_bytes(content)
    return {'url':url,'path':str(dest),'sha256':sha(dest),'retrieval_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def main():
    records=[]
    for acc in ['GSE39845','GSE41655','GSE68377','GSE128446']:
        name=acc+'_series_matrix.txt.gz';url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/matrix/{name}';dest=BASE/'data'/name
        try:
            rec=fetch(url,dest);df,meta,raw=read_external(dest);meta.to_csv(BASE/'manifests'/f'{acc}_external_samples.csv')
            rec.update(shape=df.shape,fields={c:meta[c].value_counts().to_dict() for c in meta},processing=[r[1] for r in raw if r[0]=='!Sample_data_processing'],series=[r for r in raw if r[0].startswith('!Series_') and r[0] not in ['!Series_sample_id']])
            print(acc,df.shape,rec['fields'],flush=True)
        except Exception as e:rec={'accession':acc,'url':url,'error':repr(e)}
        rec['accession']=acc;records.append(rec);write_json(BASE/'manifests/external_search.json',records)
    for platform in ['GPL18402','GPL18941','GPL11487','GPL14767']:
        url=f'https://ftp.ncbi.nlm.nih.gov/geo/platforms/{platform[:-3]}nnn/{platform}/soft/{platform}_family.soft.gz';dest=BASE/'data'/f'{platform}_family.soft.gz'
        try:
            rec=fetch(url,dest)
            with gzip.open(dest,'rt',encoding='utf-8') as f:text=f.read()
            body=text.split('!platform_table_begin\n')[1].split('!platform_table_end')[0]
            pd.read_csv(io.StringIO(body),sep='\t',dtype=str).to_csv(BASE/'manifests'/f'{platform}_annotation.csv',index=False)
            rec['platform']=platform
        except Exception as e:rec={'platform':platform,'error':repr(e),'url':url}
        records.append(rec);write_json(BASE/'manifests/external_search.json',records)
if __name__=='__main__':main()
