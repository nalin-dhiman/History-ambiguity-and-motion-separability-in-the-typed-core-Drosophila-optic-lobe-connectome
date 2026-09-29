from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
root=Path.cwd();out=root/'analysis_pipeline/core/audit';out.mkdir(parents=True,exist_ok=True)
cols=['bodyId:long','type:string','instance:string','status:string','statusLabel:string','assignedOlHex1:float','assignedOlHex2:float','somaLocation:point{srid:9157}']
export=root/'analysis_pipeline/core/data_external/raw_typed_metadata.parquet'
r=pd.read_parquet(export,columns=cols).rename(columns={'bodyId:long':'bodyId'})
n=pd.read_parquet(root/'outputs/processed_typed_core/nodes.parquet');r=n[['bodyId','type','instance']].merge(r,on='bodyId',how='left',validate='one_to_one')
report={'source_export':'Neuprint_Neurons.feather', 'bundled_input':str(export.relative_to(root)),'typed_bodies':len(r),'raw_metadata_matched':int(r['type:string'].notna().sum()),'type_disagreements':int((r.type!=r['type:string']).sum()),'fields':{},'fragment_marker_pattern':'fragment|unclear|uncertain','hex_joint_present':int(r[['assignedOlHex1:float','assignedOlHex2:float']].notna().all(1).sum()),'hex_one_only':int(r[['assignedOlHex1:float','assignedOlHex2:float']].notna().sum(1).eq(1).sum())}
for c in ['type','instance','status:string','statusLabel:string']:
 report['fields'][c]={'missing':int(r[c].isna().sum()),'marker_rows':int(r[c].fillna('').str.contains('fragment|unclear|uncertain',case=False).sum())}
 if c.startswith('status'): report['fields'][c]['counts']=r[c].fillna('<missing>').value_counts().to_dict()
r.to_csv(out/'typed_body_metadata.csv.gz',index=False);(out/'metadata_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
