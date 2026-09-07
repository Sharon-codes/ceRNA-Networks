"""Profile full-estimator outer/inner bootstrap; pilot distributions are not population CIs."""
import json,time,traceback
import numpy as np
import pandas as pd
from core import *
from models import fit_all,fit_one
from provenance import BASE,RUN,write_json
CFG=json.loads((BASE/'configs/primary.json').read_text())
def main():
    rows=[];start=time.perf_counter()
    for acc in ['GSE115513','GSE73002']:
        df,meta,_=cohort(acc)
        for b in range(CFG['outer_pilot_replicates']):
            t=time.perf_counter();rng=rng_for(CFG['seed'],acc+'_outer_pilot',b)
            outer=df.iloc[:,rng.integers(0,len(meta),len(meta))]
            try:
                s,_,_=select_features(outer,500);x=s.T.to_numpy();r=spearman(x);g=graph_from(r);u,v=np.triu_indices(len(r),1);inner=[];fail=0
                for k in range(CFG['outer_inner_replicates']):
                    try:inner.append(spearman(x[rng.integers(0,len(x),len(x))])[u,v])
                    except ValueError:fail+=1
                fit=fit_all(g,np.array(inner),acc+'_pilot',save=False)
                e=[q for q in fit['coefficients'] if q['model']=='spline_distance_ebc' and q['term']=='ebc'][0]
                row={'cohort':acc,'outer':b,'status':'completed','n':len(x),'features':len(s),'edges':len(g['edges']),'theta':g['summary']['theta'],'inner_valid':len(inner),'inner_failed':fail,'ebc_coefficient':e['coefficient'],'ebc_mcse':e['mc_se'],'seconds':time.perf_counter()-t,'interpretation':'full-estimator pilot distribution; not calibrated population confidence interval'}
                # First outer run also compares score linearization with exact vector refits.
                if b==0:
                    model,design,keep,cov=fit['objects']['spline_distance_ebc'];ys=fit['y'][:,keep];params=[]
                    for j in range(50):
                        rr=rng_for(CFG['seed'],acc+'_vector_refit',j).integers(0,len(ys),len(ys));m,_,_,_=fit_one(design[keep],ys[rr]);params.append(m.params[-1])
                    row['exact_vector_refit_sd']=float(np.std(params,ddof=1))
                rows.append(row)
            except Exception as e:rows.append({'cohort':acc,'outer':b,'status':'failed','error':repr(e),'seconds':time.perf_counter()-t});traceback.print_exc()
            print(acc,'outer pilot',b+1,'total seconds',time.perf_counter()-start,flush=True)
            pd.DataFrame(rows).to_csv(RUN/'tables/nested_pilot.csv',index=False)
    write_json(RUN/'logs/status_nested_pilot.json',{'status':'completed','outer_attempts':len(rows),'successful':sum(r['status']=='completed' for r in rows),'seconds':time.perf_counter()-start,'population_inference':'UNRESOLVED: EBC target calibration not established'})
if __name__=='__main__':main()
