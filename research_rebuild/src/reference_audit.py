"""Crossref bibliographic checks; contextual relevance remains explicitly assessed."""
import concurrent.futures,json,time,urllib.request,urllib.parse
from difflib import SequenceMatcher
import pandas as pd
from provenance import BASE,RUN,write_json

REFERENCES=[
('Albert 2002','Statistical mechanics of complex networks'),
('Barrett 2013','NCBI GEO: archive for functional genomics data sets--update'),
('Basso 2005','Reverse engineering of regulatory networks in human B cells'),
('Blondel 2008','Fast unfolding of communities in large networks'),
('De Smet 2010','Advantages and limitations of current network inference methods'),
('Girvan 2002','Community structure in social and biological networks'),
('Kipf 2017','Semi-supervised classification with graph convolutional networks'),
('Kolaczyk 2009','Statistical Analysis of Network Data: Methods and Models'),
('Langfelder 2008','WGCNA: an R package for weighted correlation network analysis'),
('Little Rubin 2002','Statistical Analysis with Missing Data'),
('Lu 2005','MicroRNA expression profiles classify human cancers'),
('Marbach 2012','Wisdom of crowds for robust gene network inference'),
('McCullagh Nelder 1989','Generalized Linear Models'),
('Troyanskaya 2001','Missing value estimation methods for DNA microarrays'),
('Volinia 2006','A microRNA expression signature of human solid tumors defines cancer gene targets')]
def query(item):
    label,title=item;url='https://api.crossref.org/works?rows=3&query.bibliographic='+urllib.parse.quote(label+' '+title)
    row={'manuscript_reference':label,'stated_title':title,'query_url':url}
    try:
        with urllib.request.urlopen(url,timeout=30) as response:data=json.load(response)
        write_json(BASE/'manifests'/('reference_'+label.replace(' ','_')+'.json'),data)
        hits=data['message']['items'];hit=max(hits,key=lambda h:SequenceMatcher(None,title.lower(),' '.join(h.get('title',[])).lower()).ratio())
        found=' '.join(hit.get('title',[]));score=SequenceMatcher(None,title.lower(),found.lower()).ratio()
        row.update(matched_title=found,doi=hit.get('DOI'),url=hit.get('URL'),title_similarity=score,published=hit.get('published',{}).get('date-parts'),status='bibliographic title match; verify edition/year separately' if score>.85 else 'unresolved automatic match; manual verification required')
    except Exception as e:row.update(status='access failed',error=repr(e))
    return row
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(query,REFERENCES))
    pd.DataFrame(rows).to_csv(RUN/'tables/reference_audit.csv',index=False)
