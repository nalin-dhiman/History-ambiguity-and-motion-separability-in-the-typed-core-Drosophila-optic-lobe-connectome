"""Stage only numerical inputs in ignored runtime directories."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
REPO=Path(__file__).resolve().parents[1];PIPE=REPO/'analysis_pipeline'
CORE=PIPE/'core';CROSS=PIPE/'cross_specimen'

def copy_input(source,dest,records):
    if not source.is_file():raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=dest.resolve():shutil.copy2(source,dest)
    h=hashlib.sha256()
    with dest.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    records.append({'file':str(dest.relative_to(REPO)),'sha256':h.hexdigest(),'bytes':dest.stat().st_size})

def import_numeric_archive(source,records):
    candidates=[source/'reproducibility',source/'extended_submission/reproducibility',source]
    base=next((p for p in candidates if (p/'outputs/processed_typed_core/edges.parquet').is_file()),None)
    if base is None:raise FileNotFoundError('No processed typed-core input found in the specified directory')
    for name in ['edges.parquet','nodes.parquet']:copy_input(base/'outputs/processed_typed_core'/name,REPO/'outputs/processed_typed_core'/name,records)
    for name in ['raw_typed_metadata.parquet','flyvis_nodes.parquet','flyvis_edges.parquet','external_input_manifest.json']:
        p=base/'pnas_extension/data_external'/name
        if p.exists():copy_input(p,CORE/'data_external'/name,records)
    legacy=base/'revised_submission/analysis/data'
    for name in ['spatial_metadata.csv','edge_sample_00.csv.gz']+[f'motion_edges_{i}.csv.gz' for i in range(5)]:
        p=legacy/name
        if p.exists():copy_input(p,PIPE/'diagnostics/data'/name,records)
    candidates=[source/'cross_specimen_extension',source/'extended_submission/cross_specimen_extension']
    ext=next((p for p in candidates if (p/'data_external/female_proofread_connections.parquet').exists()),None)
    if ext:
        for name in ['female_proofread_connections.parquet','female_neuron_table.csv','female_column_assignment.csv.gz','female_annotation_LICENSE','female_data_README.md','male_coordinate_systems.md']:
            p=ext/'data_external'/name
            if p.exists():copy_input(p,CROSS/'data_external'/name,records)
        copy_input(ext/'data_external/Nern2025/Sup_Table_7_MatchingCellTypes_final.xlsx',CROSS/'data_external/Nern2025/Sup_Table_7_MatchingCellTypes_final.xlsx',records)

def project_female(folder,records):
    import numpy as np,pandas as pd
    path=folder/'proofread_connections_783.feather'
    cols=['pre_pt_root_id','post_pt_root_id','neuropil','syn_count']
    data=pd.read_feather(path,columns=cols).rename(columns=dict(zip(cols,['src','dst','neuropil','weight'])))
    ids=np.load(folder/'proofread_root_ids_783.npy',allow_pickle=False)
    for col in ['src','dst','weight']:
        if not pd.api.types.is_integer_dtype(data[col]):raise ValueError(f'{col} must use exact integer values')
    if not (data.src.isin(ids)&data.dst.isin(ids)).all():raise ValueError('Connections contain endpoints outside the supplied proofread IDs')
    if not (data.weight>0).all():raise ValueError('Connections require positive counts')
    if data.duplicated(['src','dst','neuropil']).any():raise ValueError('Duplicate pair/neuropil records')
    out=CROSS/'data_external/female_proofread_connections.parquet';out.parent.mkdir(parents=True,exist_ok=True);data.to_parquet(out,index=False)
    copy_input(out,out,records);copy_input(folder/'proofread_root_ids_783.npy',CROSS/'data_external/female_proofread_root_ids.npy',records)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--numeric-archive',type=Path,help='Import a local numerical archive using an explicit input-file allowlist')
    p.add_argument('--male-export',type=Path,help='Raw optic-lobe neuPrint table directory')
    p.add_argument('--female-dir',type=Path,help='Directory with proofread_connections_783.feather and proofread_root_ids_783.npy')
    a=p.parse_args()
    if not any([a.numeric_archive,a.male_export,a.female_dir]):p.error('provide at least one input source')
    if a.numeric_archive and a.male_export:p.error('choose an archive or raw male export, not both')
    records=[]
    for target in [CORE/'data_external',CORE/'provenance',CROSS/'data_external',CROSS/'provenance',CROSS/'results']:target.mkdir(parents=True,exist_ok=True)
    if a.numeric_archive:import_numeric_archive(a.numeric_archive.resolve(),records)
    if a.male_export:
        import pandas as pd
        folder=a.male_export.resolve();subprocess.run([sys.executable,str(REPO/'scripts/01_prepare_optic_lobe.py'),'--data-dir',str(folder),'--typed-core','--out-dir',str(REPO/'outputs/processed_typed_core')],cwd=REPO,check=True)
        node=pd.read_parquet(REPO/'outputs/processed_typed_core/nodes.parquet')
        columns=['bodyId:long','type:string','instance:string','status:string','statusLabel:string','assignedOlHex1:float','assignedOlHex2:float','somaLocation:point{srid:9157}']
        raw=pd.read_feather(folder/'Neuprint_Neurons.feather',columns=columns);raw=raw[raw['bodyId:long'].isin(node.bodyId)]
        out=CORE/'data_external/raw_typed_metadata.parquet';raw.to_parquet(out,index=False);copy_input(out,out,records)
        for name in ['nodes.parquet','edges.parquet']:
            out=REPO/'outputs/processed_typed_core'/name;copy_input(out,out,records)
    if a.female_dir:project_female(a.female_dir.resolve(),records)
    (CORE/'provenance/staged_input_hashes.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Staged {len(records)} numerical inputs. All destination data directories are git-ignored.')
if __name__=='__main__':main()
