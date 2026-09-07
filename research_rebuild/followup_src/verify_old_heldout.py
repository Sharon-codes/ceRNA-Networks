from common import *
from audit import verify,flush_ledger
def main():
    start=time.perf_counter();rows=[]
    for acc in ACCS:
        df,_=cohort(acc);order=rng_for(SEED,acc+'_split',0).permutation(df.shape[1]);a,b=np.array_split(order,2);selection,_,_=select_features(df.iloc[:,a],500);val=df.loc[selection.index].iloc[:,b];valid=np.isfinite(val).all(axis=1)&(val.max(axis=1)>val.min(axis=1));x=val.loc[valid].T.to_numpy()
        edges=pd.read_csv(OLD/'tables'/f'{acc}_split_reference_edges.csv');u=edges.node_a.to_numpy();v=edges.node_b.to_numpy();eligible=valid.to_numpy()[u]&valid.to_numpy()[v];new=np.full(len(valid),-1);new[np.flatnonzero(valid)]=np.arange(sum(valid));iu,iv=np.triu_indices(sum(valid),1);lookup={(i,j):k for k,(i,j) in enumerate(zip(iu,iv))};idx=np.array([lookup[(new[i],new[j])] for i,j in zip(u[eligible],v[eligible])]);theta=np.abs(spearman(selection.T.to_numpy())[u,v])-edges.margin.to_numpy();assert np.ptp(theta)<1e-12;theta=theta[0]
        outcomes=[]
        for ids,boot in cache_blocks(acc+'_split_validation',x):outcomes.append((np.abs(boot[:,idx])<theta).astype(float))
        y=np.concatenate(outcomes);losses={};src=OLD/'tables'/f'{acc}_heldout_prediction.csv';old=pd.read_csv(src)
        for _,r in old.iterrows():
            p=edges.loc[eligible,'pred_'+r.model].to_numpy();ok=np.isfinite(p);pc=np.clip(p[ok],1e-12,1-1e-12);loss=-(y[:,ok]*np.log(pc)+(1-y[:,ok])*np.log1p(-pc)).mean(axis=1);losses[r.model]=loss
            verify('old_heldout_'+r.model,acc,r.validation_logloss,loss.mean(),src,'Re-count old heldout-half bootstrap cache; saved train prediction vectors; clip1e-12','Old bootstrap-test outcome, not new direct heldout outcome')
        src=OLD/'tables'/f'{acc}_heldout_increment.csv';old=pd.read_csv(src)
        for _,r in old.iterrows():
            diff=losses[r.model+'_ebc']-losses[r.model];se=diff.std(ddof=1)/np.sqrt(len(diff))
            for field,value in [('ebc_minus_distance_logloss',diff.mean()),('mc_low',diff.mean()-1.96*se),('mc_high',diff.mean()+1.96*se)]:verify('old_'+r.model+'_'+field,acc,r[field],value,src,'Paired whole test-bootstrap vectors; normal MC mean interval','Validation MC only, omits fitting MC and population uncertainty')
        rows.append({'cohort':acc,'train_n':len(a),'test_n':len(b),'candidate_edges':len(edges),'evaluable_edges':int(eligible.sum()),'test_valid_nodes':int(valid.sum()),'valid_test_B':len(y),'old_outcome':'bootstrap-of-heldout-half disappearance','new_outcome':'direct heldout disappearance'})
    csv('old_heldout_protocol_verification',rows)
    nested=pd.read_csv(OLD/'tables/nested_pilot.csv');csv('nested_pilot_preservation_audit',nested.groupby('cohort').agg(rows=('outer','count'),completed=('status',lambda s:sum(s=='completed')),recorded_runtime_seconds=('seconds','sum')).reset_index())
    # New primary scores independently reconstructed from saved direct outcome vectors.
    scores=pd.read_csv(RUN/'tables/heldout_repeated_split_results.csv')
    for (acc,s),group in scores.groupby(['cohort','split']):
        path=RUN/'models'/f'{acc}_split{s:02d}.npz';z=np.load(path);y=z['y']
        for _,row in group.iterrows():
            verify(f'new_split{s:02d}_{row.model}_brier',acc,row.brier,np.dot(z[row.model]-y,z[row.model]-y)/len(y),path,'Independent dot-product squared error on saved direct heldout outcomes','New follow-up outcome; paired edges; no population CI')
    flush_ledger();status('old_heldout_verification',start)
if __name__=='__main__':main()
