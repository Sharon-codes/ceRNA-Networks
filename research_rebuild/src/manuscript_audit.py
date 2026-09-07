"""Identify supplied PDF by contents and prove embedded figure provenance."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import pymupdf
from PIL import Image
from provenance import BASE,ROOT,RUN,sha,write_json

MANUSCRIPT=Path(r'C:\Users\Samsunh\Downloads\Integrating_ceRNA_Network_Topology_with_Expression_Features_Improves_Pan_Cancer_Detection_in_Liquid_Biopsy.pdf')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manuscript',type=Path,default=MANUSCRIPT);args=parser.parse_args();manuscript=args.manuscript
    doc=pymupdf.open(manuscript);out=RUN/'historical/supplied_manuscript';out.mkdir(exist_ok=True)
    text='\n\n'.join(f'PAGE {i+1}\n'+p.get_text() for i,p in enumerate(doc));(out/'complete_text.txt').write_text(text,encoding='utf-8')
    assert 'Re-evaluating Topological Fragility' in text
    known={hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest():p for p in (ROOT/'fuck it').glob('*.png')}
    rows=[]
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25)).save(out/f'page_{i+1}.png')
        for z in page.get_images():
            pix=pymupdf.Pixmap(doc,z[0]);h=hashlib.sha256(pix.samples).hexdigest();match=known.get(h)
            rows.append({'page':i+1,'xref':z[0],'width':pix.width,'height':pix.height,'decoded_pixels_sha256':h,'exact_matching_repository_image':str(match) if match else None,'image_sha256':sha(match) if match else None})
    pd.DataFrame(rows).to_csv(RUN/'tables/manuscript_figure_provenance.csv',index=False)
    write_json(BASE/'manifests/supplied_manuscript.json',{'path':str(manuscript),'sha256':sha(manuscript),'pages':len(doc),'identity':'Re-evaluating Topological Fragility in Hard-Thresholded miRNA Co-expression Networks: A Distance-Controlled Statistical Analysis','figures':rows,'source_tex':'No .tex file found in repository'})
if __name__=='__main__':main()
