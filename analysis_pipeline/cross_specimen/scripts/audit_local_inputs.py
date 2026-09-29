"""Provenance and independent synapse-to-connection audit for supplied FlyWire files."""
from pathlib import Path
import hashlib,json,time,argparse
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.ipc as ipc
ROOT=Path(__file__).resolve().parents[1];WS=ROOT.parent
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--female-dir',type=Path,default=WS.parent/'female_fruitfly');args=parser.parse_args();rawdir=args.female_dir
 t0=time.time();files=[]
 for p in sorted(rawdir.iterdir()):
  if p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
   files.append(dict(file=p.name,bytes=p.stat().st_size,sha256=h.hexdigest()));print('HASH',p.name,'seconds',round(time.time()-t0,1),flush=True)
 (ROOT/'provenance/local_input_hashes.json').write_text(json.dumps(files,indent=2))
 e=pd.read_parquet(ROOT/'data_external/female_proofread_connections.parquet');ids=pd.Index(np.load(rawdir/'proofread_root_ids_783.npy').astype(np.int64));regions=pd.Index(sorted(e.neuropil.unique()));N=len(ids);K=len(regions)
 key=(ids.get_indexer(e.src)*N+ids.get_indexer(e.dst))*K+regions.get_indexer(e.neuropil);lookup=pd.Index(key);assert lookup.is_unique;counts=np.zeros(len(e),np.int64);total=proofread=autapses=unknown_roi=missing_keys=0;minscore=10**9;maxscore=-1;missing_records=[];unassigned_labels=0
 with pa.memory_map(str(rawdir/'flywire_synapses_783.feather'),'r') as mm:
  reader=ipc.open_file(mm)
  for bi in range(reader.num_record_batches):
   b=reader.get_batch(bi);total+=len(b);pre=b.column(b.schema.get_field_index('pre_pt_root_id')).to_numpy();post=b.column(b.schema.get_field_index('post_pt_root_id')).to_numpy();si=ids.get_indexer(pre);di=ids.get_indexer(post);sel=(si>=0)&(di>=0);proofread+=int(sel.sum());autapses+=int((sel&(si==di)).sum());sel&=si!=di
   if sel.any():
    roi=np.asarray(b.column(b.schema.get_field_index('neuropil')).to_pylist(),dtype=object)[sel];unassigned=(pd.isna(roi)|(roi=='None'));unassigned_labels+=int(unassigned.sum());roi[unassigned]='UNASGD';ri=regions.get_indexer(roi);unknown_roi+=int((ri<0).sum());kk=(si[sel]*N+di[sel])*K+ri;ix=lookup.get_indexer(kk);valid=(ix>=0)&(ri>=0);missing_keys+=int((~valid).sum());np.add.at(counts,ix[valid],1)
    if np.any(~valid):missing_records.extend(zip(pre[sel][~valid].tolist(),post[sel][~valid].tolist(),roi[~valid].tolist()))
    scores=b.column(b.schema.get_field_index('cleft_score')).to_numpy()[sel];minscore=min(minscore,int(scores.min()));maxscore=max(maxscore,int(scores.max()))
   if bi%300==0:print('synapse batches',bi,'/',reader.num_record_batches,'elapsed',round(time.time()-t0,1),flush=True)
 diff=counts-e.weight.to_numpy();report={'raw_synapse_rows':total,'both_proofread_rows':proofread,'excluded_autapse_synapses':autapses,'nonself_proofread_rows':proofread-autapses,'connection_table_contacts':int(e.weight.sum()),'null_region_labels_normalized_to_UNASGD':unassigned_labels,'unknown_region_rows':unknown_roi,'missing_connection_key_rows':missing_keys,'mismatching_pair_neuropil_counts':int((diff!=0).sum()),'maximum_absolute_count_difference':int(abs(diff).max()),'proofread_nonself_min_cleft_score':minscore,'proofread_nonself_max_cleft_score':maxscore,'elapsed_seconds':time.time()-t0,'exact_reconstruction':bool((diff==0).all() and missing_keys==0 and unknown_roi==0)}
 (ROOT/'provenance/female_synapse_aggregation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 # Record mismatches if any; do not alter supplied scientific inputs to make them agree.
 e.loc[diff!=0].assign(raw_synapse_count=counts[diff!=0],difference=diff[diff!=0]).to_parquet(ROOT/'provenance/synapse_connection_disagreements.parquet',index=False)

 if missing_records:
  missing=pd.DataFrame(missing_records,columns=['src','dst','neuropil']).value_counts().reset_index(name='raw_synapse_count');missing.to_csv(ROOT/'provenance/raw_synapses_absent_from_connections.csv',index=False)
