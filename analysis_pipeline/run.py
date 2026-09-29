"""Run numerical workflows with explicit dependencies and a recorded command log."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time
REPO=Path(__file__).resolve().parents[1];PIPE=REPO/'analysis_pipeline'
STAGES={
 'core':['core/src/phase0_metadata.py','core/src/run_decomposition.py'],
 'controls':['core/src/gate1_controls.py','core/src/sampling_sensitivity.py','core/src/roi_sensitivity.py','core/src/additional_controls.py'],
 'replication':['cross_specimen/scripts/prepare_metadata.py','cross_specimen/scripts/run_replication.py'],
 'references':['cross_specimen/scripts/run_references.py'],
 'coarsening':['cross_specimen/scripts/run_coarsening.py'],
 'spatial':['cross_specimen/scripts/run_spatial.py','cross_specimen/scripts/run_spatial_reference.py'],
 'model':['core/src/model_compatibility.py'],
 'archived':['core/src/run_archived_diagnostics.py'],
 'probes':['core/src/calibrate_probes.py --phase development','core/src/calibrate_probes.py --phase confirmation'],
 'plot':['plot_results.py'],
 'verify':['validate_results.py'],
}
DEFAULT_STAGES=['core','controls','replication','references','coarsening','spatial','verify','plot']
REQUIRED={
 'core':['outputs/processed_typed_core/edges.parquet','outputs/processed_typed_core/nodes.parquet','analysis_pipeline/core/data_external/raw_typed_metadata.parquet'],
 'controls':['analysis_pipeline/core/results/development/source_type_body_inputs.parquet','analysis_pipeline/core/audit/typed_body_metadata.csv.gz'],
 'replication':['analysis_pipeline/core/results/development/universe_decomposition.csv','analysis_pipeline/cross_specimen/data_external/female_proofread_connections.parquet','analysis_pipeline/cross_specimen/data_external/female_neuron_table.csv','analysis_pipeline/cross_specimen/data_external/female_column_assignment.csv.gz','analysis_pipeline/cross_specimen/data_external/Nern2025/Sup_Table_7_MatchingCellTypes_final.xlsx'],
 'references':['analysis_pipeline/cross_specimen/results/male_matched_right_inputs.parquet','analysis_pipeline/cross_specimen/results/female_matched_right_inputs.parquet'],
 'coarsening':['analysis_pipeline/cross_specimen/data_external/male_native_right_edges.parquet','analysis_pipeline/cross_specimen/data_external/female_native_right_edges.parquet'],
 'spatial':['analysis_pipeline/cross_specimen/results/matched_primary_alphabets.json','analysis_pipeline/cross_specimen/results/female_matched_left_lr_inputs.parquet'],
 'model':['analysis_pipeline/core/data_external/flyvis_nodes.parquet','analysis_pipeline/core/data_external/flyvis_edges.parquet','analysis_pipeline/core/data_external/external_input_manifest.json'],
 'archived':['analysis_pipeline/diagnostics/data/spatial_metadata.csv','analysis_pipeline/diagnostics/data/edge_sample_00.csv.gz']+[f'analysis_pipeline/diagnostics/data/motion_edges_{i}.csv.gz' for i in range(5)],
 'probes':['analysis_pipeline/diagnostics/data/spatial_metadata.csv']+[f'analysis_pipeline/diagnostics/data/motion_edges_{i}.csv.gz' for i in range(5)],
 'plot':['analysis_pipeline/core/results/development/universe_decomposition.csv'],
 'verify':['analysis_pipeline/core/results/development/universe_decomposition.csv'],
}

def missing_inputs(stage,root=REPO):return [p for p in REQUIRED.get(stage,[]) if not (root/p).is_file()]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','test',*STAGES],default='test')
    p.add_argument('--dry-run',action='store_true',help='Print the selected commands without opening inputs or executing stages')
    a=p.parse_args();stages=DEFAULT_STAGES if a.stage=='all' else [a.stage]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    records=[]
    for stage in stages:
        commands=[[sys.executable,'-m','pytest','-q']] if stage=='test' else [[sys.executable,str(PIPE/f.split()[0]),*f.split()[1:]] for f in STAGES[stage]]
        if not a.dry_run:
            missing=missing_inputs(stage)
            if missing:p.exit(2,'Missing inputs for '+stage+':\n  '+'\n  '.join(missing)+'\nUse prepare_inputs.py and the documented preceding stages.\n')
        for command in commands:
            print(stage+': '+' '.join(command),flush=True)
            if a.dry_run:continue
            t=time.monotonic();result=subprocess.run(command,cwd=REPO,env=env)
            records.append({'stage':stage,'command':command,'returncode':result.returncode,'seconds':time.monotonic()-t})
            log=PIPE/'runs/latest.json';log.parent.mkdir(parents=True,exist_ok=True);log.write_text(json.dumps({'run_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':records},indent=2)+'\n')
            if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
