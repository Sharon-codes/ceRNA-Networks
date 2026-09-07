from base import *
import unittest
from simulation import matrices,fast_spearman

class ScientificChecks(unittest.TestCase):
    def test_affine_adjacency_strength_and_CV(self):
        rng=rng_for(SEED,'test_affine',0);A=rng.uniform(size=(40,9,9));A=(A+A.transpose(0,2,1))/2
        for b in A:np.fill_diagonal(b,0)
        S=A.sum(axis=2)
        for lam in [0,.25,.5,.9,.99]:
            B=(1-lam)*A+lam*.5
            for b in B:np.fill_diagonal(b,0)
            strength=B.sum(axis=2);formula=(1-lam)*S+lam*.5*8
            self.assertTrue(np.allclose(strength,formula,atol=1e-12));self.assertTrue(np.array_equal(rankdata(strength,axis=1),rankdata(S,axis=1)))
            target=(1-lam)*S.std(axis=0,ddof=1)/((1-lam)*S.mean(axis=0)+lam*.5*8);self.assertTrue(np.allclose(cv(strength),target))
        c=pd.read_csv(RUN/'tables/stability_control.csv');self.assertLess(c.max_formula_error.max(),1e-10);self.assertEqual(c.top20_changed_draws.sum(),0)
    def test_population_generator_targets(self):
        for p in (RUN/'models').glob('simulation_population_truth_*.npz'):
            z=np.load(p);C=z['loading']@z['loading'].T+np.outer(z['nuisance_loading'],z['nuisance_loading'])+np.eye(120)
            self.assertTrue(np.allclose(C,z['covariance']));self.assertGreater(np.linalg.eigvalsh(C).min(),.999999)
            for module in range(4):
                if len(z['latent_hub_nodes']):
                    nodes=np.flatnonzero(z['module_labels']==module);r=z['population_latent_spearman'][np.ix_(nodes,nodes)].sum(axis=1)-1
                    self.assertEqual(set(nodes[top(r,3)]),set(z['latent_hub_nodes'][z['module_labels'][z['latent_hub_nodes']]==module]))
            if 'null' in p.name:self.assertEqual(z['population_hard_adjacency'].sum(),120*119)
    def test_monotone_edges_and_rank_arithmetic(self):
        x=rng_for(SEED,'test_r',0).normal(size=(100,120));self.assertTrue(np.array_equal(spearman(x),fast_spearman(x)));a,s,t=matrices(spearman(x));u,v=np.triu_indices(120,1);self.assertTrue(np.array_equal(a[u,v]>0,s[u,v]>=np.quantile(s[u,v],.975)))
    def test_predictive_baselines_and_bounds(self):
        p=pd.read_csv(RUN/'tables/predictive_baselines.csv');self.assertEqual(len(p),320);self.assertTrue(p[p.model.eq('constant_0.5')].brier.eq(.25).all())
        self.assertTrue((p.full_universe_Brier_lower<=p.full_universe_Brier_upper+1e-12).all());self.assertTrue((p.full_universe_Brier_lower>=0).all());self.assertTrue((p.full_universe_Brier_upper<=1).all())
        for (acc,s),g in p.groupby(['cohort','split']):
            self.assertEqual(g.edges.nunique(),1);z=np.load(PREV/'models'/f'{acc}_split{s:02d}.npz');learned=z['train_drop_count'].sum()/z['train_denominator'].sum();self.assertAlmostEqual(g.train_intercept.iloc[0],learned)
            y=z['y'];known=z['spline_distance'];self.assertAlmostEqual(g[g.model.eq('spline_distance')].brier.iloc[0],((y-known)**2).mean())
        r=pd.read_csv(RUN/'tables/prediction_reconciliation.csv');self.assertLess(r.absolute_difference.max(),1e-10)
    def test_split_leakage_and_module_partition(self):
        splits=json.loads((PREV/'manifests/splits.json').read_text());self.assertEqual(len(splits),40)
        for s in splits:self.assertFalse(set(s['train'])&set(s['test']))
        d=pd.read_csv(RUN/'tables/module_definitions.csv');self.assertEqual(len(d),500);self.assertEqual(d.miRNA.nunique(),500);self.assertEqual(d.module.nunique(),8)
        for cohortid in ['GSE41655','GSE48267']:
            n=pd.read_csv(RUN/'tables/named_hub_results.csv');n=n[n.cohort.eq(cohortid)&n.policy.eq('A_frozen')];self.assertEqual(len(n),204);self.assertEqual(n.miRNA.nunique(),204)
            expected=d.set_index('miRNA').loc[n.miRNA,'module'].to_numpy();self.assertTrue(np.array_equal(expected,n.module.to_numpy()))
    def test_graph_accounting_and_failures(self):
        g=pd.read_csv(RUN/'tables/graph_state_accounting.csv');self.assertEqual(len(g),2000);self.assertTrue((g.M==3119-g.disappearances+g.appearances).all());self.assertTrue((g.disappearances+g.unchanged_present==3119).all());self.assertTrue((g.appearances+g.unchanged_absent==121631).all());self.assertTrue(g.unevaluable_pairs.eq(0).all())
        f=pd.read_csv(RUN/'tables/preserved_failure_summary.csv').iloc[0];self.assertEqual(f.complete_graph_valid,0);self.assertEqual(f.attempts,200)
    def test_simulation_and_interval_accounting(self):
        s=pd.read_csv(RUN/'tables/simulation_results.csv');self.assertEqual(len(s),600);self.assertTrue(s.status.eq('completed').all());self.assertEqual(s.groupby(['cell','method']).size().unique().tolist(),[30]);self.assertTrue(s.valid_B.eq(100).all())
        self.assertTrue(s[s.signal_scale.eq(0)].module_ARI.isna().all());self.assertTrue(s[s.signal_scale.gt(0)].module_ARI.notna().all())
        i=pd.read_csv(RUN/'tables/fixed_S_interval_status.csv');self.assertTrue((i.interval_valid+i.interval_failed==1000).all());self.assertTrue(i[~i.interval_supported].mc_low.isna().all())
    def test_report_embedding_and_figure_sources(self):
        text=(RUN/'FINAL_REPORT.html').read_text();self.assertEqual(text.count('src="data:image/svg+xml;base64,'),9);self.assertNotIn('src="http',text);self.assertNotIn('src="figures/',text)
        for r in pd.read_csv(RUN/'tables/figure_manifest.csv').itertuples():
            for s,h in json.loads(r.source_hashes).items():self.assertEqual(sha(RUN/'tables'/f'{s}.csv'),h)

if __name__=='__main__':unittest.main(verbosity=2)
