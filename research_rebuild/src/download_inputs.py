"""Download official matrices only when absent; never replace local user data."""
from pathlib import Path
import urllib.request,os
from provenance import ROOT,BASE,sha,write_json
if __name__=='__main__':
    for d in ['tables','figures','models','logs','reports','historical','cache']:
        from provenance import RUN
        (RUN/d).mkdir(parents=True,exist_ok=True)
    (BASE/'data').mkdir(parents=True,exist_ok=True)
    for acc in ['GSE115513','GSE73002']:
        name=acc+'_series_matrix.txt.gz';path=ROOT/name
        if path.exists():print('Preserved',path);continue
        url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/matrix/{name}'
        with urllib.request.urlopen(url,timeout=120) as r,open(str(path)+'.download','wb') as f:
            while block:=r.read(2**20):f.write(block)
        os.replace(str(path)+'.download',path);print('Downloaded',acc,sha(path))
