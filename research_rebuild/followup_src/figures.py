from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

NAMES={'GSE115513':'Tissue','GSE73002':'Serum'}
BLUE='#246a92';ORANGE='#d17032';PURPLE='#815ba6';GREEN='#33836b'
def read(name):return pd.read_csv(RUN/'tables'/f'{name}.csv')
MANIFEST=[]
def save(fig,name,sources,caption):
    fig.suptitle(caption.split('.')[0],fontsize=14,fontweight='bold')
    for ext in ['svg','png']:fig.savefig(RUN/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    MANIFEST.append({'figure':name,'run_id':RUN.name,'config_sha256':sha(RUN/'followup_analysis_amendment.md'),'input_hashes_manifest':str(RUN/'manifests/freeze.json'),'source_tables':[{'path':str(RUN/'tables'/f'{s}.csv'),'sha256':sha(RUN/'tables'/f'{s}.csv')} for s in sources],'plotting_command':'C:\\Python314\\python.exe research_rebuild/followup_src/figures.py','plot_source_sha256':sha(Path(__file__)),'caption':caption,'png':str(RUN/'figures'/f'{name}.png'),'svg':str(RUN/'figures'/f'{name}.svg')})

def main():
    start=time.perf_counter();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    cv=read('cv_fixed_S_summary');nodes=read('cv_node_diagnostics');held=read('heldout_paired_comparisons');cal=read('heldout_calibration');ext=read('external_transfer_verification');edge=read('edge_change_per_replicate');deg=read('degeneracy_audit')
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for j,acc in enumerate(ACCS):
        g=cv[cv.cohort.eq(acc)&cv.focal.eq('gcc_focal')].set_index('support').loc[['full','common_support']];a=axs[0,j];x=np.arange(2)
        a.bar(x-.18,g.mean_hard_cv,.35,label='Hard degree',color=BLUE);a.bar(x+.18,g.mean_soft_cv,.35,label='Soft strength β=6',color=ORANGE);a.set_xticks(x,['Full support','Common support']);a.set_ylabel('Mean sample CV');a.set_title(f'{NAMES[acc]} • fixed GCC focal nodes (n={int(g.iloc[0].finite_paired_nodes)})');a.legend(fontsize=8)
        for i,r in enumerate(g.itertuples()):a.text(i,max(r.mean_hard_cv,r.mean_soft_cv)*1.06,f'{100*r.ratio_mean_reduction:.2f}% reduction',ha='center',fontsize=8)
        a.set_ylim(0,g.mean_hard_cv.max()*1.24)
        g=nodes[nodes.cohort.eq(acc)&nodes.support.eq('full')];a=axs[1,j];valid=g.mean_degree>0
        a.scatter(g.loc[valid,'mean_degree'],g.loc[valid,'hard_cv'],s=13,alpha=.5,color=BLUE,label='Hard degree');a.scatter(g.loc[valid,'mean_degree'],g.loc[valid,'soft_cv'],s=13,alpha=.5,color=ORANGE,label='Soft strength');a.set_xscale('log');a.set_yscale('log');a.set_xlabel('Mean hard degree across 1,000 draws');a.set_ylabel('Node CV (log scale)');a.set_title(f'Full support • {sum(~valid)} zero-mean nodes excluded from axes');a.axvline(1,color='gray',lw=.8,ls=':')
    save(fig,'fig01_conditional_variability',['cv_fixed_S_summary','cv_node_diagnostics'],'Conditional variability depends on support and degree. Top: mean CV over identical fixed baseline-GCC focal nodes, sample SD ddof1, beta6, full and common edge support. Bottom: all original nodes with positive mean hard degree, retaining nearly isolated nodes; zero-mean hard CVs are undefined and excluded only from logarithmic axes. No population intervals or biological accuracy inference.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for a,acc in zip(axs,ACCS):
        g=cv[cv.cohort.eq(acc)].reset_index(drop=True)
        for i,r in g.iterrows():
            a.plot(r.ratio_mean_reduction,i,'o',color=BLUE)
            if r.interval_supported:a.hlines(i,r.mc_low,r.mc_high,color=BLUE,lw=2)
            else:a.text(r.ratio_mean_reduction-.008,i+.2,f'Interval unsupported: {r.interval_failed}/1000 failed',ha='right',fontsize=8,color='#ae3737')
        a.set_yticks(range(4),['Full • all finite nodes','Full • GCC focal','Common • all finite nodes','Common • GCC focal']);a.set_ylim(-.5,3.5);a.invert_yaxis();a.set_xlabel('1 − mean soft CV / mean hard CV');a.xaxis.set_major_formatter(PercentFormatter(1));a.set_title(NAMES[acc])
    save(fig,'fig02_fixed_set_intervals',['cv_fixed_S_summary'],'Fixed node sets expose undefined interval resamples. Points use paired finite S defined from the original1000 draws. Bars are percentile Monte Carlo ranges from1000 joint whole-row interval attempts, conditional on S; not population confidence intervals. Intervals with fewer than99% valid attempts are unsupported. Failed attempts are shown; endpoints are not adjusted to contain points.')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');models=['linear_distance','spline_distance','uncertainty_distance'];labels=['Linear\nsecondary','Spline df4','Uncertainty\nspline df4']
    for j,acc in enumerate(ACCS):
        a=axs[0,j]
        for i,model in enumerate(models):
            g=held[held.cohort.eq(acc)&held.model_pair.eq(model)].sort_values('split');a.scatter(i+np.linspace(-.16,.16,len(g)),g.brier_without_minus_with,s=24,alpha=.8,color=[PURPLE,BLUE,GREEN][i]);a.hlines(g.brier_without_minus_with.mean(),i-.23,i+.23,color='black',lw=2)
        a.axhline(0,color='gray',lw=1);a.set_xticks(range(3),labels);a.set_ylabel('Brier without EBC − with EBC');a.set_title(NAMES[acc]+' • positive favors EBC');a.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
        a=axs[1,j];a.plot([0,1],[0,1],ls=':',color='gray');c=cal[cal.cohort.eq(acc)]
        for model,color in [('spline_distance',BLUE),('spline_distance_ebc',ORANGE),('uncertainty_distance',GREEN),('uncertainty_distance_ebc',PURPLE)]:
            rows=[]
            for _,g in c[c.model.eq(model)&c.n.gt(0)].groupby('bin'):rows.append([np.average(g.mean_prediction,weights=g.n),np.average(g.event_fraction,weights=g.n)])
            rows=np.array(rows);a.plot(rows[:,0],rows[:,1],'o-',ms=3,color=color,label=model.replace('_distance','').replace('_',' '))
        a.set_xlim(0,1);a.set_ylim(0,1);a.set_xlabel('Mean predicted disappearance');a.set_ylabel('Observed disappearance fraction');a.legend(fontsize=8);a.set_title('Calibration bins • descriptive repeated edge evaluations')
    save(fig,'fig03_direct_heldout_prediction',['heldout_paired_comparisons','heldout_calibration'],'Direct held-out prediction across frozen half splits. Top:20 overlapping splits per cohort, identical paired evaluable edges, Brier gains positive for EBC; black marks are arithmetic means, not confidence limits. Bottom: fixed probability bins pooled by their evaluation counts across overlapping splits; repeated arrays/edges are not independent observations. Models learn selection, graph, cutoff, EBC, scaling, SD and fits from training only. Tissue is individual-level; serum is array-level with patient linkage unknown.')
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for j,acc in enumerate(ACCS):
        g=edge[edge.cohort.eq(acc)];a=axs[0,j];a.scatter(g.disappearances,g.appearances,s=7,alpha=.35,color=BLUE);lim=max(g.disappearances.max(),g.appearances.max());a.plot([0,lim],[0,lim],ls=':',color='gray');a.set_xlabel('Reference-edge disappearances D');a.set_ylabel('Initially absent-edge appearances G');a.set_title(NAMES[acc]+' • paired bootstrap draws')
        a=axs[1,j];a.hist(g.total_edges,bins=35,color=ORANGE,alpha=.8);a.axvline(3119,color=BLUE,lw=2,label='Reference: 3,119');a.set_xlabel('Total edges M = 3,119 − D + G');a.set_ylabel('Draw count');a.legend();a.set_title(f'Mean M={g.total_edges.mean():,.3f}; mean Jaccard={g.jaccard.mean():.4f}')
    save(fig,'fig04_edge_change',['edge_change_per_replicate'],'Appearances are a substantial part of whole-network change. Each point is one of1000 fully evaluable fixed-threshold draws per cohort on500 nodes and124750 pairs. Histograms are resampling distributions, not population confidence intervals. No per-draw cutoff retuning. Every draw satisfies the displayed identity.')
    fig,axs=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for a,metric,title in zip(axs[:2],['rank_spearman','top20_overlap'],['Node rank agreement','Top-20 overlap']):
        x=np.arange(2);a.bar(x-.18,ext['hard_'+metric],.35,color=BLUE,label='Hard');a.bar(x+.18,ext['soft_'+metric],.35,color=ORANGE,label='Soft β=6');a.set_xticks(x,['B: density\nmatched','A: frozen\ncutoff']);a.set_ylim(0,1);a.set_ylabel(title);a.legend()
        if metric=='top20_overlap':
            for i,r in ext.iterrows():
                a.text(i-.18,r.hard_top20_overlap+.03,f'{r.hard_top20_count}/20',ha='center',fontsize=8);a.text(i+.18,r.soft_top20_overlap+.03,f'{r.soft_top20_count}/20',ha='center',fontsize=8)
    a=axs[2];a.bar(['Mapped','Not in common\nevaluable universe'],[204,296],color=[GREEN,'#cbd0d3']);a.set_ylabel('Discovery selected identities');a.set_ylim(0,350);a.set_title('33 external arrays • 204/500 nodes')
    save(fig,'fig05_external_reproducibility',['external_transfer_verification'],'Lower conditional variability does not establish better external agreement. GSE41655 exploratory comparison:33 adenocarcinoma arrays,204 mapped complete variable mature-miRNA identities. B estimates2.5% cutoffs separately; A freezes discovery cutoff on the same common universe. Beta6 strength is unchanged between A/B. Top20 uses stable identity-order tie breaking; canonical arithmetic and numerical near-tie sensitivity are documented. Points/bars have no population intervals and are not biological ground truth.')
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained');a=axs[0];a.scatter(deg.predicted_constant_probability,deg.observed_constant_fraction,s=15,alpha=.65,color=PURPLE);a.plot([0,1],[0,1],ls=':',color='gray');a.set_xlabel('Empirical predicted constant probability');a.set_ylabel('Observed fraction in 200 attempts');a.set_title('1,000 tissue probes • 121 degenerate at least once')
    reps=read('degeneracy_per_replicate');a=axs[1];a.hist(reps.constant_probes,bins=np.arange(reps.constant_probes.min()-.5,reps.constant_probes.max()+1.5),color=PURPLE);a.set_xlabel('Constant probes per attempted draw');a.set_ylabel('Draw count');a.set_title('0/200 complete graphs • no replacement draws')
    save(fig,'fig06_degeneracy',['degeneracy_audit','degeneracy_per_replicate'],'Rare empirical probes can invalidate complete-graph estimators. Prediction uses sum_v(n_v/n)^750 for independent row bootstrap from750 recorded unique individuals. Probe dependence is preserved by joint resampling; this formula is not a graph-failure probability. Undefined incident pairs remain unevaluable, not absent. Per-edge conditional denominators and graph-completion count bounds are separate source tables.')
    write_json(RUN/'figures/figure_manifest.json',MANIFEST);report('working_figure_captions','\n\n'.join(f"## Figure {i+1}: {f['figure']}\n\n{f['caption']}\n\nSources: "+', '.join(s['path'] for s in f['source_tables']) for i,f in enumerate(MANIFEST)))
    # Contact sheet is an inspection aid, not a scientific source figure.
    from PIL import Image,ImageOps,ImageDraw
    thumbs=[]
    for f in MANIFEST:
        im=Image.open(f['png']).convert('RGB');im.thumbnail((900,620));canvas=Image.new('RGB',(920,660),'white');canvas.paste(im,((920-im.width)//2,20));ImageDraw.Draw(canvas).text((10,640),f['figure'],fill='black');thumbs.append(canvas)
    sheet=Image.new('RGB',(1840,1980),'#dddddd')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*920,(i//2)*660))
    sheet.save(RUN/'figures/inspection_montage.png');status('figures',start,count=len(MANIFEST))
if __name__=='__main__':main()
