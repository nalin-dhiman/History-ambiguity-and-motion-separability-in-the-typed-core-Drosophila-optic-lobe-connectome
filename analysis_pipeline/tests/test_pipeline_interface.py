from pathlib import Path
import importlib.util,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pipeline_runner',ROOT/'analysis_pipeline/run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def test_default_stages_have_scripts_and_explicit_input_dependencies():
    for stage in runner.DEFAULT_STAGES:
        assert stage in runner.REQUIRED
        for command in runner.STAGES[stage]:assert (ROOT/'analysis_pipeline'/command.split()[0]).is_file()
    assert runner.DEFAULT_STAGES.index('core')<runner.DEFAULT_STAGES.index('replication')<runner.DEFAULT_STAGES.index('spatial')

def test_preflight_names_missing_data_without_running_analysis(tmp_path):
    missing=runner.missing_inputs('core',tmp_path)
    assert set(missing)==set(runner.REQUIRED['core'])
    f=tmp_path/missing[0];f.parent.mkdir(parents=True);f.touch()
    assert f.relative_to(tmp_path).as_posix() not in runner.missing_inputs('core',tmp_path)

def test_dry_run_works_from_an_unrelated_directory(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'analysis_pipeline/run.py'),'--stage','all','--dry-run'],cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0 and 'run_spatial_reference.py' in result.stdout and 'calibrate_probes.py' not in result.stdout

def test_numeric_import_does_not_copy_unrelated_documents(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('prepare_numeric',ROOT/'analysis_pipeline/prepare_inputs.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=tmp_path/'source';repo=tmp_path/'destination';base=source/'reproducibility';graph=base/'outputs/processed_typed_core';graph.mkdir(parents=True)
    for name in ['edges.parquet','nodes.parquet']:(graph/name).write_bytes(b'numerical-input-fixture')
    (source/'private_document.tex').write_text('must stay local')
    monkeypatch.setattr(m,'REPO',repo);monkeypatch.setattr(m,'PIPE',repo/'analysis_pipeline');monkeypatch.setattr(m,'CORE',repo/'analysis_pipeline/core');monkeypatch.setattr(m,'CROSS',repo/'analysis_pipeline/cross_specimen')
    records=[];m.import_numeric_archive(source,records)
    assert len(records)==2 and not list(repo.rglob('*.tex'))
