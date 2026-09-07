from base import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.collections import LineCollection
BLUE='#246a92';ORANGE='#d17032';GREEN='#2c8064';RED='#b4464e';NAMES={ACCS[0]:'Tissue',ACCS[1]:'Serum'}
manifest=[]
def read(n):return pd.read_csv(RUN/'tables'/f'{n}.csv')
def save(fig,name,sources,caption):
    for ext in ['svg','png','pdf']:fig.savefig(RUN/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);manifest.append({'figure':name,'caption':caption,'sources':'; '.join(sources),'source_hashes':json.dumps({s:sha(RUN/'tables'/f'{s}.csv') for s in sources}),'config_sha256':sha(RUN/'frozen_config.json'),'script':str(Path(__file__)),'command':'C:\\Python314\\python.exe research_rebuild/biosystems_src/plots.py','png':str(RUN/'figures'/f'{name}.png'),'svg':str(RUN/'figures'/f'{name}.svg')})
def box(ax,x,y,w,h,s,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#859099'));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=10,wrap=True)
def arrow(ax,x,y,x2,y2):ax.add_patch(FancyArrowPatch((x,y),(x2,y2),arrowstyle='-|>',mutation_scale=12,color='#526471',lw=1.5))
def main():
    start=time.perf_counter();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.02,.72,.29,.22,'GSE115513\n750 carcinoma arrays / 750 IDs\nGSE73002\n1,280 serum arrays; people unknown','#e4eff5')
    box(ax,.37,.72,.27,.22,'Supplied processed scale\nComplete, variable → top500 MAD\nUnsigned Spearman; 2.5% cutoff\nNo causal arrows implied','#f0f2f2');arrow(ax,.31,.83,.37,.83)
    box(ax,.71,.72,.27,.22,'Conditional analysis\n1,000 existing full-row draws\nFixed features and cutoff\nFull / common edge support','#e7efe8');arrow(ax,.64,.83,.71,.83)
    box(ax,.02,.36,.29,.22,'Internal assessment\n20 fixed half splits per cohort\nTrain-only features / graph / fit\nDirect held-out disappearance','#f7eee2');arrow(ax,.17,.72,.17,.58)
    box(ax,.37,.36,.27,.22,'Frozen tissue modules\nAverage linkage; 8 modules\n2 meet external size / coverage\nNo functional truth assigned','#e7efe8');arrow(ax,.5,.72,.5,.58)
    box(ax,.71,.36,.27,.22,'Hub and graph endpoints\nDegree / strength; k10,20,50\nDisappearances + appearances\nAffine control; fixed p','#e7efe8');arrow(ax,.84,.72,.84,.58)
    box(ax,.19,.03,.62,.21,'Exploratory external tissue comparisons\nGSE41655: 33 arrays × 204 mapped miRNAs\nGSE48267: 61 tumors × 204 mapped miRNAs (site / preservation confounded)\nA: frozen discovery cutoff; B: separately matched density\nGSE29622 excluded: 98 mapped < frozen 100-node requirement','#eee8f5');arrow(ax,.5,.36,.5,.24)
    fig.suptitle('Study roles and actual data flow',fontsize=15,fontweight='bold')
    save(fig,'fig01_workflow',['cohort_inventory'],'Schematic data flow, not a causal diagram. Discovery selection and module learning use tissue discovery only; split-specific predictors use training halves. Conditional analysis reuses1000 verified draws on500 nodes. External universes are204 mature identities,40.8% of discovery500; GSE48267 has61 site-qualified patients but mixed preservation. Serum person linkage and cross-study global overlap remain unresolved.')
    ctrl=read('stability_control');hub=read('hub_stability_summary');obs=read('original_mean_sd_cv');fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,acc in enumerate(ACCS):
        g=ctrl[ctrl.cohort.eq(acc)&ctrl.focal.eq('baseline_GCC')]
        for method,col in [('hard',BLUE),('soft',ORANGE)]:
            z=g[g.method.eq(method)];axs[j,0].plot(z['lambda'],z.mean_cv,'o-',color=col,label=method);axs[j,1].plot(z['lambda'],z.mean_sd,'o-',color=col,label='SD '+method)
        axs[j,0].set_yscale('log');axs[j,0].set_ylabel(NAMES[acc]+' mean CV');axs[j,0].set_xlabel('λ (constant background)');axs[j,0].legend();axs[j,1].set_xlabel('λ');axs[j,1].set_ylabel('Mean node SD');axs[j,1].legend()
        h=hub[hub.cohort.eq(acc)]
        for method,col in [('hard',BLUE),('soft',ORANGE)]:z=h[h.method.eq(method)];axs[j,2].plot(z.k,z.mean_overlap,'o-',label=method,color=col)
        axs[j,2].set_ylim(0,1.03);axs[j,2].set_xticks([10,20,50]);axs[j,2].set_xlabel('Prespecified k');axs[j,2].set_ylabel('Mean bootstrap top-k overlap');axs[j,2].legend()
    fig.suptitle('Conditional variability, affine control and actual hub-selection stability',fontsize=14)
    save(fig,'fig02_control_hubs',['stability_control','hub_stability_summary','original_mean_sd_cv'],'Measured conditional results. Left/middle: c=.5, lambda0,.25,.5,.9,.99 applied on full500-node support, with209 tissue/183 serum fixed GCC focal nodes for aggregation. Mean and SD formula table accompanies this figure. All original top20 sets remain unchanged under the affine control. Right: original full-universe hard/soft top-k membership versus the reference over1000 joint draws, k10,20,50 fixed before outcomes; no population confidence bars. Lower serum soft CV coexists with lower top20 selection stability.')
    fig,axs=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for j,acc in enumerate(ACCS):
        for a,field,title in zip(axs[j],['mean_of_means','mean_of_SD','mean_CV'],['Mean node strength','Mean node SD','Mean node CV']):
            for k,(method,col) in enumerate([('hard',BLUE),('soft',ORANGE)]):
                z=obs[obs.cohort.eq(acc)&obs.method.eq(method)].set_index('support').reindex(['full','common']);a.bar(np.arange(2)+(k-.5)*.34,z[field],width=.32,color=col,label=method)
            a.set_xticks([0,1],['Full support','Common support']);a.set_title(NAMES[acc]+' • '+title);a.legend(fontsize=8)
    save(fig,'fig02b_original_moments',['original_mean_sd_cv'],'Original conditional moments before the affine intervention. Full and fixed reference-edge common support use the same fixed GCC focal nodes (209 tissue;183 serum), with1000 draws per cohort. Nodewise means, sample SDs and CVs are averaged separately; their aggregate ratio is not the mean CV. Hard degree and beta6 soft strength use different weight scales. Panels show descriptive conditional quantities, without biological population confidence bars.')
    fig,ax=plt.subplots(figsize=(12,4.5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    for x,s,c in [(.02,'PLANTED\n4 independent module factors\n30 variables/module\nheterogeneous loadings\n3 high-loading hubs/module','#e7efe8'),(.36,'OBSERVED EXPRESSION\nX = F Lᵀ + z gᵀ + ε\nCov(X) = LLᵀ + ggᵀ + I\nSample sizes60 / 240\nNuisance g0 / 0.6','#e4eff5'),(.70,'INFERRED / EVALUATED\nHard 2.5%; soft |ρ|⁶\n100 bootstrap attempts\nKnown-K4 spectral modules\nTrue hub/module recovery','#f7eee2')]:box(ax,x,.3,.28,.52,s,c)
    arrow(ax,.30,.56,.36,.56);arrow(ax,.64,.56,.70,.56);ax.text(.5,.1,'SIMULATED • 30 independent datasets × 10 frozen conditions • independent second sample per dataset\nNull: no planted hubs/modules; truth-recovery metrics undefined. Covariance positive definite by construction.',ha='center',va='center',fontsize=10)
    save(fig,'fig03_simulation_generator',['simulation_truth_summary'],'Simulation schematic, not biological regulation. Four Gaussian module factors with heterogeneous loadings, optional shared nuisance factor and independent unit-variance noise yield positive-definite covariance. Hub targets are verified from nuisance-free population association; observed population association and hard/soft population graph targets are retained separately. Module reconstruction uses knownK4, an oracle simplification.30 independent datasets per each of10 conditions follow runtime-only pilot freeze; all100 attempts retained.')
    sim=read('simulation_summary');order=list(dict.fromkeys(read('simulation_results').cell));fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for a,metric,title in zip(axs.flat,['mean_cv','latent_hub_top12_recovery','module_ARI','independent_top12_overlap'],['Mean conditional CV','Planted hub top-12 recovery','Planted module adjusted Rand index','Independent-sample top-12 overlap']):
        for method,col,shift in [('hard',BLUE,-.13),('soft',ORANGE,.13)]:
            g=sim[sim.method.eq(method)&sim.endpoint.eq(metric)].set_index('cell').reindex(order);a.errorbar(np.arange(len(order))+shift,g['mean'],yerr=1.96*g.mean_MCSE,fmt='o',ms=4,color=col,label=method,capsize=2)
        a.set_xticks(range(len(order)),[c.replace('_','\n') for c in order],fontsize=7);a.set_title(title);a.legend(fontsize=8);a.grid(axis='y',alpha=.2)
    fig.suptitle('SIMULATED: variability and recovery answer different questions',fontsize=14)
    save(fig,'fig04_simulation_results',['simulation_results','simulation_summary'],'Simulation means with95% normal Monte Carlo intervals across30 independent datasets per cell; not biological population intervals. All cells shown (s=loading scale; g=nuisance amplitude; n=sample size). CV uses paired finite node sets with counts saved. Hub target comprises12 verified latent high-loading nodes; module target is the planted four-group partition. Null recovery is undefined and therefore absent, not zero. Hard/soft pair rankings at a matched quantile coincide, but their strengths and spectral module reconstructions need not. Bootstrap prefix precision is separately reported.')
    pred=read('predictive_baselines');ps=read('predictive_summary');cal=read('calibration');fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');mods=['constant_0.5','train_intercept','spline_distance','spline_distance_ebc','uncertainty_distance','uncertainty_distance_ebc'];labels=['0.5','Train\nintercept','Spline','Spline\n+EBC','Uncert.','Uncert.\n+EBC']
    for j,acc in enumerate(ACCS):
        a=axs[0,j]
        for i,m in enumerate(mods):g=pred[pred.cohort.eq(acc)&pred.model.eq(m)];a.scatter(i+np.linspace(-.16,.16,len(g)),g.brier,s=11,alpha=.55,color=BLUE);a.hlines(g.brier.mean(),i-.24,i+.24,color=ORANGE,lw=3)
        a.set_xticks(range(len(mods)),labels,fontsize=8);a.set_ylabel('Binary Brier score');a.set_title(NAMES[acc]+' • 20 dependent half splits');a.axhline(.25,ls=':',color='gray')
        a=axs[1,j];a.plot([0,1],[0,1],ls=':',color='gray')
        for m,col in [('spline_distance',BLUE),('spline_distance_ebc',ORANGE),('train_intercept',GREEN)]:
            c=cal[cal.cohort.eq(acc)&cal.model.eq(m)&cal.n.gt(0)];pts=[]
            for _,g in c.groupby('bin'):pts.append((np.average(g.mean_prediction,weights=g.n),np.average(g.observed,weights=g.n)))
            pts=np.array(pts);a.plot(pts[:,0],pts[:,1],'o-',label=m.replace('_distance','').replace('_',' '),color=col,ms=4)
        a.set_xlim(0,1);a.set_ylim(0,1);a.set_xlabel('Predicted disappearance');a.set_ylabel('Observed disappearance');a.legend(fontsize=8)
    save(fig,'fig05_prediction',['predictive_baselines','predictive_summary','calibration'],'Direct held-out disappearance of train-retained edges at frozen training cutoff. Scores use identical evaluable edges:3119 tissue,2278–2747 serum per split. Constant.5 Brier=.25 invariant; train-intercept probability uses training bootstrap events only. Points are all20 overlapping splits; orange marks are means, without population intervals. Calibration bins pool repeated edge evaluations descriptively. EBC means edge betweenness centrality. No test-label recalibration; logloss clipping and full-universe identification bounds are tabulated separately.')
    diff=read('predictive_paired_differences');fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for a,acc in zip(axs,ACCS):
        for family,col in [('linear',GREEN),('spline',BLUE),('uncertainty',ORANGE)]:
            d=diff[diff.cohort.eq(acc)&diff.comparison.str.startswith(family)].sort_values('split');a.plot(d.split,d.brier_without_minus_with,'o-',ms=4,lw=.8,color=col,label=family)
        a.axhline(0,color='black',lw=.8);a.set_title(NAMES[acc]);a.set_xlabel('Fixed split ID');a.set_ylabel('Brier without EBC − Brier with EBC');a.ticklabel_format(axis='y',style='sci',scilimits=(0,0));a.legend(fontsize=9);a.set_xticks([0,5,10,15,19])
    save(fig,'fig05b_signed_ebc',['predictive_paired_differences'],'Signed paired Brier differences for all20 fixed overlapping splits per cohort and all three fitted distance families. Positive values favor adding EBC; negative values favor the model without EBC. Each difference uses identical held-out evaluable edges within its split. These dependent split points are descriptive, without population confidence intervals or a no-effect/equivalence claim. Separate vertical scales preserve the measured magnitudes; no favorable split is omitted.')
    graph=read('graph_state_accounting');fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,acc in enumerate(ACCS):
        g=graph[graph.cohort.eq(acc)];axs[0,j].scatter(g.disappearances,g.appearances,s=6,alpha=.35,color=BLUE);axs[0,j].set_title(NAMES[acc]+f' • mean M={g.M.mean():.3f}');axs[0,j].set_xlabel('Disappearance count (of 3,119 reference edges)');axs[0,j].set_ylabel('Appearance count (of 121,631 absent pairs)')
    z=np.load(RUN/'models/GSE115513_display_draw0.npz');defs=read('module_definitions');order=np.argsort(defs.module,kind='stable');angle=np.empty(500);angle[order]=np.linspace(0,2*np.pi,500,endpoint=False);pos=np.c_[np.cos(angle),np.sin(angle)];pairs=np.c_[z['u'],z['v']];base=z['reference'];draw=z['bootstrap']
    for a,mask,title in [(axs[1,0],base,'Tissue reference • fixed500-node layout'),(axs[1,1],draw,'Tissue bootstrap draw0 • same layout')]:
        a.set_aspect('equal');a.axis('off');a.set_title(title)
        if title.startswith('Tissue reference'):a.add_collection(LineCollection(pos[pairs[mask]],colors='#88969d',linewidths=.25,alpha=.12))
        else:
            for m,col,label in [(base&draw,'#88969d','unchanged'),(base&~draw,RED,'disappeared'),(~base&draw,GREEN,'appeared')]:a.add_collection(LineCollection(pos[pairs[m]],colors=col,linewidths=.35,alpha=.15 if label=='unchanged' else .5,label=label))
            a.legend(fontsize=8,loc='upper left')
        a.scatter(pos[:,0],pos[:,1],c=defs.module,cmap='tab10',s=5);a.set_xlim(-1.15,1.15);a.set_ylim(-1.15,1.15)
    save(fig,'fig06_graph_changes',['graph_state_accounting','module_definitions'],'Graph accounting on1000 complete500-node draws per cohort. Every draw satisfies M=3119−D+G; no density retuning. Lower panels are schematic renderings of actual tissue edges, replicate0 chosen before outcomes, same deterministic module-ordered circular layout and500 node identities. Red edges are missing in the displayed draw, green newly present; dense unchanged edges are faint. Layout implies no biological geometry. Undefined primary pairs=0; failed1000-feature setting is separate.')
    ex=read('external_summary');mod=read('module_preservation');fig,grid=plt.subplots(2,2,figsize=(12,8),layout='constrained');axs=grid.flat;x=np.arange(len(ex));lab=[r.cohort+'\n'+('A frozen' if r.policy=='A_frozen' else 'B density') for r in ex.itertuples()]
    for a,field,title in [(axs[0],'rank_rho','External node-rank agreement'),(axs[1],'top20_count','External top-20 intersection')]:
        a.bar(x-.18,ex['hard_'+field],.35,color=BLUE,label='hard');a.bar(x+.18,ex['soft_'+field],.35,color=ORANGE,label='soft');a.set_xticks(x,lab,fontsize=8);a.set_title(title);a.legend(fontsize=8)
    a=axs[2];m=mod[mod.eligible].reset_index(drop=True);xx=np.arange(len(m));a.scatter(xx-.12,m.hard_connectivity_rho,color=BLUE,label='hard');a.scatter(xx+.12,m.soft_connectivity_rho,color=ORANGE,label='soft');a.axhline(0,color='gray',lw=.6);a.set_xticks(xx,[r.cohort[-3:]+' '+r.policy[0]+' M'+str(r.module) for r in m.itertuples()],rotation=60,fontsize=8);a.set_title('Fixed-module connectivity ranks');a.set_ylabel('Spearman correlation');a.legend(fontsize=8)
    a=axs[3]
    for i,acc in enumerate(ACCS):
        h=hub[hub.cohort.eq(acc)&hub.k.eq(20)].set_index('method');a.bar(i-.18,h.loc['hard','mean_overlap'],.35,color=BLUE,label='hard' if i==0 else None);a.bar(i+.18,h.loc['soft','mean_overlap'],.35,color=ORANGE,label='soft' if i==0 else None)
    a.set_xticks([0,1],['Tissue: 500 nodes','Serum: 500 nodes']);a.set_ylim(0,1);a.set_ylabel('Mean conditional top-20 overlap fraction');a.set_title('Internal selection stability • different universe');a.legend(fontsize=8)
    save(fig,'fig07_external_modules',['external_summary','module_preservation','hub_stability_summary'],'Both external cohorts and both policies shown: GSE41655 n33, GSE48267 n61; each204 unique mapped mature miRNAs on the same discovery-selected identities. A transfers the discovery cutoff; B matches2.5% density independently. Degree and beta6 strength ranks are different endpoints. Only two discovery modules satisfy predeclared size/coverage minima: M3 n91→34 and M6 n384→155. Within-module connectivity comparisons use fixed memberships and all eligible modules. Bottom right repeats internal top20 membership fractions over1000 draws on each500-node universe; this differs from external intersection counts and204-node universes. These are descriptive point estimates, not preservation significance or biological validation.')
    csv('figure_manifest',manifest);write_json(RUN/'figures/figure_manifest.json',manifest);status('figures',start,figures=len(manifest))
if __name__=='__main__':main()
