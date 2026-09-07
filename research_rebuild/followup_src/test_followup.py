"""Targeted numerical/domain/leakage checks for the empirical follow-up."""
import unittest
from common import *
from heldout import metrics
from cv_audit import reduction

class ScientificRisks(unittest.TestCase):
    def test_partial_correlations_preserve_undefined(self):
        x=np.array([[0,1,4],[0,2,3],[0,3,2],[0,4,1]],float)
        r,valid=paircorr_partial(x)
        self.assertEqual(valid.tolist(),[False,True,True]);self.assertTrue(np.isnan(r[0]).all());self.assertAlmostEqual(r[1,2],-1)
        present=np.isfinite(r)&(np.abs(r)>=.5);absent=np.isfinite(r)&~present;unknown=~np.isfinite(r)
        self.assertTrue(np.all(present.astype(int)+absent+unknown==1))
    def test_joint_cv_rows_and_fixed_domain(self):
        h=np.array([[0,1],[0,2],[1,3],[3,6]],float);s=2*h
        for ix in [[0,1,2,3],[0,0,3,3],[2,2,3,3]]:
            self.assertAlmostEqual(reduction(h[ix],s[ix],np.array([0,1])),0)
        self.assertTrue(np.isnan(reduction(h[[0,0,1,1]],s[[0,0,1,1]],np.array([0,1]))))
    def test_cv_sd_convention(self):
        h=np.array([[1.],[2.],[3.]])
        self.assertAlmostEqual(cv(h)[0],.5)
    def test_known_scoring_and_ties(self):
        r=metrics([0,1],[.5,.5]);self.assertAlmostEqual(r['brier'],.25);self.assertAlmostEqual(r['auroc'],.5);self.assertAlmostEqual(r['average_precision'],.5)
        r=metrics([0,1],[0,1]);self.assertEqual(r['clipped_count'],2);self.assertAlmostEqual(r['brier'],0)
        self.assertTrue(np.isnan(metrics([0,0],[.2,.2])['auroc']))
    def test_frozen_split_leakage(self):
        splits=json.loads((RUN/'manifests/splits.json').read_text());self.assertEqual(len(splits),40)
        for acc in ACCS:
            df,meta=cohort(acc);old=pd.read_csv(BASE/'manifests'/f'{acc}_split.csv')
            for spec in [s for s in splits if s['cohort']==acc]:
                self.assertFalse(set(spec['train'])&set(spec['test']));self.assertEqual(set(spec['train'])|set(spec['test']),set(df.columns))
                if 'individual' in meta:self.assertFalse(set(meta.loc[spec['train'],'individual'])&set(meta.loc[spec['test'],'individual']))
                if spec['split']==0:self.assertEqual(set(spec['train']),set(old.loc[old.split.eq('reference'),'sample_id']))
    def test_actual_paired_scoring(self):
        scores=pd.read_csv(RUN/'tables/heldout_repeated_split_results.csv');paired=pd.read_csv(RUN/'tables/heldout_paired_comparisons.csv');self.assertEqual(len(scores),240);self.assertEqual(len(paired),120)
        for (acc,s),g in scores.groupby(['cohort','split']):
            self.assertEqual(g.evaluable_edges.nunique(),1);self.assertTrue(g.converged.all())
            z=np.load(RUN/'models'/f'{acc}_split{s:02d}.npz');y=z['y']
            for _,r in g.iterrows():
                self.assertTrue(np.isfinite(z[r.model]).all());self.assertAlmostEqual(metrics(y,z[r.model])['brier'],r.brier,places=12)
            for _,r in paired[paired.cohort.eq(acc)&paired.split.eq(s)].iterrows():
                independent=np.mean((z[r.model_pair]-y)**2-(z[r.model_pair+'_ebc']-y)**2)
                self.assertAlmostEqual(independent,r.brier_without_minus_with,places=12)
    def test_actual_edge_count_and_degeneracy_accounting(self):
        c=pd.read_csv(RUN/'tables/edge_change_per_replicate.csv');self.assertEqual(len(c),2000)
        self.assertTrue((c.total_edges==c.reference_edges-c.disappearances+c.appearances).all())
        for chunk in pd.read_csv(RUN/'tables/degeneracy_edge_states.csv',chunksize=100000):
            self.assertTrue((chunk.present+chunk.absent+chunk.unevaluable==chunk.attempted).all());self.assertTrue((chunk.evaluable==chunk.present+chunk.absent).all())
        r=pd.read_csv(RUN/'tables/reconciled_stage_status.csv');self.assertEqual(len(r),10);self.assertEqual(sum(r.status.eq('completed')),9);self.assertEqual(sum(r.status.eq('failed')),1);self.assertTrue((r.valid_B+r.failed_B==r.attempted_B).all())
    def test_existing_results_independently_verified(self):
        r=pd.read_csv(RUN/'tables/verified_results_ledger.csv');bad=r[r.status.ne('verified')];self.assertEqual(set(bad.result_id),{'manuscript_84_percent'})
        r=pd.read_csv(RUN/'tables/cv_fixed_S_summary.csv');self.assertTrue(np.allclose(r.ratio_mean_reduction,1-r.mean_soft_cv/r.mean_hard_cv));self.assertTrue((r.interval_valid+r.interval_failed==r.interval_attempts).all())

if __name__=='__main__':unittest.main(verbosity=2)
