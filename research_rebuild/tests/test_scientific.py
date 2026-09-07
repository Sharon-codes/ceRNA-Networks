import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from core import *
from models import fit_one

class ScientificTests(unittest.TestCase):
    def test_rerank_with_ties(self):
        x=np.array([[1,3,5],[2,3,1],[2,1,4],[3,4,2],[7,2,3]],float)
        rows=[0,0,2,3,3,4]
        np.testing.assert_allclose(spearman(x[rows]),spearmanr(x[rows],axis=0).statistic,atol=1e-14)
        self.assertGreater(np.max(abs(spearman(x[rows])-np.corrcoef(rankdata(x,axis=0)[rows],rowvar=False))),.001)
    def test_invalid_not_zero(self):
        with self.assertRaises(ValueError):spearman(np.ones((8,3)))
        with self.assertRaises(ValueError):spearman(np.array([[1,np.nan],[2,3]]))
    def test_feature_missing_policy(self):
        x=pd.DataFrame([[1,2,3,4,5],[1,1,1,1,1],[1,2,np.nan,4,9],[0,0,0,2,3]],index=list('abcd'))
        s,m,f=select_features(x,500)
        self.assertEqual(s.index.tolist(),['a','d'])
        s,_,_=select_features(x,500,missing='median20');self.assertIn('c',s.index)
        self.assertTrue(np.isfinite(s).all().all())
    def test_sign_density_ties(self):
        r=np.array([[1,-.9,.5],[-.9,1,.5],[.5,.5,1]])
        g=graph_from(r,density=1/3);self.assertEqual(g['summary']['full_edges'],1)
        self.assertTrue(g['adjacency'][0,1]);self.assertFalse(np.diag(g['adjacency']).any())
        p=graph_from(r,sign='positive',theta=.5);self.assertEqual(p['summary']['full_edges'],2)
        self.assertEqual(p['summary']['eligible_pairs'],3)
        self.assertEqual(p['summary']['cutoff_ties'],2)
        self.assertTrue(all(i<j for i,j in p['edges']))
    def test_frozen_cutoff(self):
        r=np.array([[1,.9,.6],[.9,1,.5],[.6,.5,1]])
        g=graph_from(r,density=1/3);b=r.copy();b[0,1]=b[1,0]=.55
        fixed=graph_from(b,theta=g['summary']['theta']);estimated=graph_from(b,density=1/3)
        self.assertEqual(fixed['summary']['full_edges'],0);self.assertEqual(estimated['summary']['full_edges'],1)
    def test_seeds(self):
        np.testing.assert_array_equal(rng_for(4,'x',8).integers(0,99,100),rng_for(4,'x',8).integers(0,99,100))
        self.assertFalse(np.array_equal(rng_for(4,'x',8).random(10),rng_for(4,'x',9).random(10)))
    def test_cv_support_and_percent(self):
        a=np.array([[0,1,3],[0,2,3],[0,3,3.]])
        z=cv(a);self.assertTrue(np.isnan(z[0]));self.assertEqual(z[1],.5);self.assertEqual(z[2],0)
        self.assertAlmostEqual(1-.035/.218,.8394495412844036)
        self.assertAlmostEqual(1-.093/.257,.6381322957198443)
        np.testing.assert_array_equal(node_sums(np.array([[1,2,3]]),3),[[3,4,5]])
    def test_dropout_appearance(self):
        b=np.array([[.8,.3],[.2,.7],[.6,.4]])
        self.assertEqual(np.sum(b[:,0]<.5),1);self.assertEqual(np.sum(b[:,1]>=.5),1)
    def test_mc_covariance_relabel_invariant(self):
        rng=np.random.default_rng(4);x=np.column_stack([np.ones(20),np.linspace(-1,1,20)])
        y=(rng.random((100,20))<.3).astype(float)
        m,p,s,c=fit_one(x,y);order=rng.permutation(20);m2,p2,s2,c2=fit_one(x[order],y[:,order])
        np.testing.assert_allclose(c,c2,rtol=1e-10,atol=1e-12)
    def test_actual_alignment_subjects(self):
        for a in ['GSE115513','GSE73002']:
            d,m,_=cohort(a);self.assertEqual(list(d.columns),list(m.index));self.assertTrue(m.index.is_unique)
            if 'individual'in m:self.assertTrue(m.individual.is_unique)
    def test_simulated_batch_spearman(self):
        from simulations import batch_spearman
        rng=np.random.default_rng(4);x=np.round(rng.normal(size=(30,8)),1);ids=rng.integers(0,30,(10,30))
        r,bad=batch_spearman(x,ids);self.assertEqual(bad,0)
        for i,row in enumerate(ids):np.testing.assert_allclose(r[i],spearmanr(x[row],axis=0).statistic,atol=1e-14)
    def test_repeated_subject_rejected(self):
        from unittest.mock import patch
        d=pd.DataFrame([[1,2],[3,4]],columns=['a','b'])
        m=pd.DataFrame({'tissue':['Carcinoma','Carcinoma'],'individual':['same','same']},index=['a','b'])
        with patch('core.read_geo',return_value=(d,m,[])):
            with self.assertRaises(ValueError):cohort('GSE115513')
        # Restore the real deterministic selection manifest after this numerical fixture.
        cohort('GSE115513')
    def test_cache_resume_and_invalidation(self):
        x=np.arange(40).reshape(10,4)+np.random.default_rng(1).normal(size=(10,4))
        b,path=bootstrap(x,4,'test_SIMULATED_cache_fixture',17,'test_config')
        c,path2=bootstrap(x,4,'test_SIMULATED_cache_fixture',17,'test_config')
        np.testing.assert_array_equal(b,c);self.assertEqual(path,path2)
        f=path/'00000.npz';f.write_bytes(b'intentionally corrupted unit-test cache')
        d,_=bootstrap(x,4,'test_SIMULATED_cache_fixture',17,'test_config');np.testing.assert_array_equal(b,d)
        _,different=bootstrap(x,4,'test_SIMULATED_cache_fixture',17,'different_config');self.assertNotEqual(path,different)
    def test_saved_cv_and_figures(self):
        import json
        for acc in ['GSE115513','GSE73002']:
            for sign in ['unsigned','positive']:
                p=RUN/'models'/f'{acc}_{sign}_node_replicates.npz'
                if not p.exists():continue
                z=np.load(p);tab=pd.read_csv(RUN/'tables'/f'{acc}_{sign}_node_cv.csv')
                for support,hs,ss in [('full','hard','soft'),('common_support','hard_common','soft_common')]:
                    rows=tab[tab.support.eq(support)].sort_values('node')
                    np.testing.assert_allclose(rows.hard_cv,cv(z[hs]),equal_nan=True,atol=1e-12)
                    np.testing.assert_allclose(rows.soft_cv,cv(z[ss]),equal_nan=True,atol=1e-12)
        path=RUN/'figures/figure_manifest.json'
        if path.exists():
            for fig in json.loads(path.read_text()):
                for source in fig['source_tables']:self.assertEqual(sha(source['path']),source['sha256'])
                self.assertTrue(Path(fig['png']).exists());self.assertTrue(Path(fig['svg']).exists())
    def test_simulation_empirical_separation(self):
        path=RUN/'tables/simulation_summary.csv'
        if path.exists():
            d=pd.read_csv(path);self.assertTrue(d.content.eq('SIMULATED').all());self.assertTrue(d.datasets.eq(200).all())
            self.assertTrue(d.coverage_target.str.contains('NOT adjusted EBC').all())

if __name__=='__main__':unittest.main()
