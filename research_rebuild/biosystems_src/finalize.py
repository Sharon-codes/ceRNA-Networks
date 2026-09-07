"""Finalize after numerical checks, browser rendering and human visual inspection."""
from base import *
import zipfile

def main():
    tests=(RUN/'logs/test_results.txt').read_text()
    assert 'Ran 8 tests' in tests and tests.rstrip().endswith('OK')
    render=json.loads((RUN/'logs/render_check.json').read_text())
    assert len(render['images'])==9 and all(i['loaded'] and i['width'] for i in render['images']) and not render['horizontalOverflow']
    qa=json.loads((RUN/'logs/visual_inspection.json').read_text());assert qa['status']=='passed'
    previous=pd.read_csv(RUN/'tables/previous_artifact_inventory.csv');checks=[]
    for r in previous.itertuples():
        actual=sha(Path(r.path));checks.append({'path':r.path,'expected_sha256':r.expected_sha256,'final_sha256':actual,'unchanged':actual==r.expected_sha256})
    assert len(checks)==862 and all(c['unchanged'] for c in checks)
    start=json.loads((RUN/'manifests/start.json').read_text());sourcechecks=[]
    for rel,h in start['prior_source_hashes'].items():
        p=BASE/rel;actual=sha(p);sourcechecks.append({'path':str(p),'expected_sha256':h,'final_sha256':actual,'unchanged':actual==h})
    assert all(c['unchanged'] for c in sourcechecks)
    inputchecks=[]
    for r in pd.read_csv(RUN/'tables/cohort_inventory.csv').itertuples():
        if isinstance(r.input_path,str) and Path(r.input_path).is_file():
            h=sha(Path(r.input_path));inputchecks.append({'path':r.input_path,'expected_sha256':r.input_sha256,'final_sha256':h,'unchanged':h==r.input_sha256})
    assert inputchecks and all(c['unchanged'] for c in inputchecks)
    csv('final_preservation_verification',checks+sourcechecks+inputchecks)
    write_json(RUN/'logs/final_verification.json',{'status':'passed','prior_artifacts_unchanged':len(checks),'prior_source_files_unchanged':len(sourcechecks),'cohort_inputs_unchanged':len(inputchecks),'targeted_tests':8,'embedded_figures':9,'visual_inspection':qa,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    # This register records exact table rows and their provenance, not a new inferential analysis.
    groups={
      'stability_control':('empirical.py','reproduced_from_cache','B01'),
      'original_mean_sd_cv':('empirical.py','reproduced_from_cache','B01'),
      'external_affine_control':('empirical.py; external48267.py','verified_from_source','B01'),
      'hub_stability_summary':('empirical.py','reproduced_from_cache','B02'),
      'simulation_summary':('simulation.py','new_simulation_executed','B03'),
      'simulation_paired_differences':('simulation.py','new_simulation_executed','B03'),
      'simulation_precision':('simulation.py','new_simulation_executed','B03'),
      'external_summary':('empirical.py; external48267.py','verified_from_source','B04'),
      'module_preservation':('empirical.py; external48267.py','verified_from_source','B06'),
      'predictive_summary':('prediction.py','reproduced_from_cache','B05'),
      'predictive_paired_differences':('prediction.py','reproduced_from_cache','B05'),
      'evaluability':('prediction.py','reproduced_from_cache','B05'),
      'graph_consequence_summary':('synthesis.py','reproduced_from_cache','C01-C56; direct graph accounting'),
      'fixed_S_interval_status':('synthesis.py','preserved_cache_and_accounting_verified','C01-C56; see claim ledger'),
      'updated_claim_ledger':('synthesis.py','per_row_verdict_and_evidence_level','all claims'),
      'novelty_comparison':('research/write_research.py','primary_source_review','B07'),
      'missing_input_inventory':('synthesis.py','explicitly_missing_or_not_run','B08')}
    rows=[]
    for name,(scripts,level,claim) in groups.items():
        p=RUN/'tables'/f'{name}.csv';df=pd.read_csv(p);hashes={}
        for s in scripts.split('; '):
            sp=RUN/s if s.startswith('research/') else BASE/'biosystems_src'/s
            if sp.exists():hashes[s]=sha(sp)
        for j,row in enumerate(df.to_dict('records'),2):
            rows.append({'evidence_id':f'{name}:row{j}','related_claims':claim,'output_table':str(p.relative_to(RUN)),'csv_line':j,'output_sha256':sha(p),'generating_scripts':scripts,'script_sha256':json.dumps(hashes),'config_sha256':sha(RUN/'frozen_config.json'),'verification_level':level,'exact_row_values_units_in_column_names':json.dumps(row,default=str),'upstream_provenance':'final_manifest.json input/source inventory; previous_artifact_inventory.csv for preserved cache chain; source script states exact input paths'})
    csv('evidence_register',rows)
    # Public inputs remain separate from distributable derived evidence.
    inputs=[]
    for p in sorted(ROOT.glob('GSE*_series_matrix.txt.gz')):
        if p.name.startswith(('GSE115513_','GSE73002_')):inputs.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'role':'original public processed expression; not bundled'})
    for p in sorted((PREV/'data').glob('*')):
        if p.is_file():inputs.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'role':'prior external source; not bundled'})
    files=[]
    for p in sorted(RUN.rglob('*')):
        if p.is_file() and p.name!='final_manifest.json':files.append({'path':str(p.relative_to(RUN)),'sha256':sha(p),'bytes':p.stat().st_size})
    sources=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for folder in ['biosystems_src','src','followup_src'] for p in sorted((BASE/folder).glob('*')) if p.is_file() and p.suffix in ['.py','.cjs']]
    write_json(RUN/'final_manifest.json',{'run_id':RUN.name,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'git_commit':git('rev-parse','HEAD'),'git_branch':git('branch','--show-current'),'self_hash_excluded':True,'artifact_inventory':files,'source_inventory':sources,'original_input_inventory':inputs,'previous_artifacts_verified':862,'evidence_register_rows':len(rows),'package_policy':'No raw expression or individual-level sample metadata/IDs; public source download manifests and derived biological feature identities included.'})
    archive=RUN.parent/(RUN.name+'_evidence.zip')
    safe_research={'cohort_screen_protocol.md','cohort_screening_initial_preserved.csv','cohort_screening.csv','external_fallback_selection_freeze.json','journal_policy_verification.csv','literature_and_journal_fit.md','novelty_comparison.csv','research_manifest.json','search_log.csv','sources_and_selection.md','write_research.py','GPL10850_annotation.csv','GPL11162_annotation.csv'}
    selected=[]
    for p in RUN.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(RUN);parts=rel.parts
        include=len(parts)==1 or parts[0] in ['tables','logs'] or (parts[0]=='figures' and (p.name.startswith('fig0') or p.name=='figure_manifest.json')) or (parts[0]=='research' and p.name in safe_research) or (parts[0]=='manifests' and p.name in ['start.json','module_freeze.json','simulation_production_freeze.json','GSE29622_analysis_freeze.json','GSE48267_analysis_freeze.json']) or (parts[0]=='data' and p.suffix=='.json') or (parts[0]=='models' and p.name.startswith('simulation_population_truth_'))
        if include:selected.append(p)
    selected.extend(ROOT/s['path'] for s in sources)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(selected)):z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any('_sample_metadata' in n or '_sample_selection' in n or n.endswith('.gz') for n in z.namelist())
        count=len(z.namelist())
    for r in files:assert sha(RUN/r['path'])==r['sha256']
    receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'members':count,'final_manifest_sha256':sha(RUN/'final_manifest.json'),'integrity':'ZIP CRC and final manifest hashes verified','reports':[str(RUN/f'FINAL_REPORT.{x}') for x in ['html','pdf','md']]}
    write_json(archive.with_suffix('.receipt.json'),receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
