from base import *
import pymupdf
from PIL import Image,ImageDraw

doc=pymupdf.open(RUN/'FINAL_REPORT.pdf');thumbs=[];text_checks=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=pymupdf.Matrix(.45,.45));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
    canvas=Image.new('RGB',(400,305),'#dadfe3');canvas.paste(im,(10,20));ImageDraw.Draw(canvas).text((10,4),f'Page {i+1}',fill='black');thumbs.append(canvas)
    for block in page.get_text('dict')['blocks']:
        if block['type']!=0:continue
        for line in block['lines']:
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<0 or y0<0 or x1>page.rect.width+1 or y1>page.rect.height+1:text_checks.append({'page':i+1,'bbox':span['bbox'],'text':span['text']})
out=Image.new('RGB',(1600,305*((len(thumbs)+3)//4)),'#dadfe3')
for i,im in enumerate(thumbs):out.paste(im,((i%4)*400,(i//4)*305))
out.save(RUN/'figures/pdf_contact_sheet.png')
write_json(RUN/'logs/pdf_layout_check.json',{'pages':len(doc),'off_page_text':text_checks})
print('PDF pages:',len(doc),'off-page text spans:',len(text_checks))
for i in [0,5,10,20]:
    if i<len(doc):doc[i].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(RUN/'figures'/f'pdf_spot_page{i+1}.png')
