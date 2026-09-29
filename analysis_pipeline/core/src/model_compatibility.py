from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from decomposition import decompose
root=Path.cwd();base=root/'analysis_pipeline/core';out=base/'results/development';out.mkdir(parents=True,exist_ok=True)
external=base/'data_external'
n=pd.read_parquet(external/'flyvis_nodes.parquet');e=pd.read_parquet(external/'flyvis_edges.parquet');origin=json.loads((external/'external_input_manifest.json').read_text())
specpath=origin['spec_file'];cache=origin['cache']
assert (e.src_type.to_numpy()==n.set_index('index').loc[e.src,'type'].to_numpy()).all();assert (e.dst_type.to_numpy()==n.set_index('index').loc[e.dst,'type'].to_numpy()).all();assert (e.weight>=0).all()
# Weight = unsigned anatomical synapse estimate. Physiological signs are separately audited.
sums=[]
for radius in [15,10,5,0]:
 ids=set(n.loc[np.maximum.reduce([np.abs(n.u),np.abs(n.v),np.abs(n.u+n.v)])<=radius,'index']);bb,tt,g=decompose(e[e.dst.isin(ids)]);tt.to_csv(out/f'flyvis_types_radius_{radius}.csv',index=False);sums.append(dict(target_radius=radius,**g))
pd.DataFrame(sums).to_csv(out/'flyvis_decomposition.csv',index=False)
emp=pd.read_parquet(root/'outputs/processed_typed_core/nodes.parquet');modeltypes=set(n.type);etype=set(emp.type);union=sorted(etype|modeltypes);rows=[]
for t in union:
 rows.append(dict(label=t,empirical_present=t in etype,model_present=t in modeltypes,mapping_status='exact_name_candidate_not_identity_validation' if t in etype&modeltypes else ('model_only' if t in modeltypes else 'empirical_only'),empirical_bodies=int((emp.type==t).sum()),model_nodes=int((n.type==t).sum())))
pd.DataFrame(rows).to_csv(base/'literature/flyvis_type_coverage.csv',index=False)
meta={'spec_file':str(specpath),'spec_sha256':origin['spec_sha256'],'cache':str(cache),'cache_meta':origin['cache_meta'],'nodes':len(n),'edges':len(e),'model_types':len(modeltypes),'empirical_labels':len(etype),'exact_name_candidates':len(etype&modeltypes),'empirical_absent':len(etype-modeltypes),'model_only':sorted(modeltypes-etype),'sign_counts':e.sign.value_counts().to_dict(),'readout':'unsigned anatomical count decomposition; no checkpoint simulation or functional transfer performed','boundary':'finite hexagonal graph; radius subsets retain complete incoming lists from full model','inference':'name intersection is coverage only, not a validated cross-atlas identity mapping'}
(out/'flyvis_compatibility.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2));print(pd.DataFrame(sums).to_string(index=False))
