from base import *
from simulation import fast_spearman,matrices
t=time.perf_counter();x=rng_for(SEED,'implementation_check',0).normal(size=(240,120));r=spearman(x);s=fast_spearman(x)
assert np.array_equal(r,s)
a,w,theta=matrices(np.eye(120));assert a.sum()==120*119 and w.sum()==0 and theta==0
write_json(RUN/'logs/simulation_implementation_amendment.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'verified','description':'Before any null checkpoint was generated, corrected null hard population target to literal >=quantile(0), a complete graph with all ties. Empty graph was an implementation error. Preserved all previously completed nonnull checkpoints. Removed redundant per-call threadpool library scan under already single-thread process; canonical rank arithmetic exactly equal on independent diagnostic matrix. R30 and B100 remain frozen, no outcome-based count change. Interrupted in-progress dataset deterministically resumes.','max_correlation_difference':float(np.max(np.abs(r-s))),'existing_completed_checkpoints':len(list((RUN/'models/simulation_checkpoints').glob('*.json'))),'existing_null_checkpoints':len(list((RUN/'models/simulation_checkpoints').glob('*null*'))),'seconds':time.perf_counter()-t})
print('Canonical arithmetic exact; null all-tie graph correctly complete')
