from pathlib import Path
import json,time
import numpy as np,pandas as pd
from decomposition import decompose
root=Path.cwd();base=root/'analysis_pipeline/core';out=base/'results/development';cfg=json.loads((base/'configs/sampling_protocol.json').read_text());e=pd.read_parquet(root/'outputs/processed_typed_core/edges.parquet',columns=['src','dst','weight','src_type','dst_type']);ids=e.dst.unique();rng=np.random.default_rng(cfg['seed']);records=[];t0=time.time()
for frac in cfg['fractions']:
 for draw in range(cfg['draws']):
  edge_idx=rng.choice(len(e),round(frac*len(e)),replace=False);bid=rng.choice(ids,round(frac*len(ids)),replace=False)
  for method,sub in [('uniform_edge_rows',e.iloc[edge_idx]),('uniform_target_bodies_complete_inputs',e[e.dst.isin(bid)])]:
   _,_,m=decompose(sub);records.append(dict(fraction=frac,draw=draw,method=method,edge_count=len(sub),**m))
 pd.DataFrame(records).to_csv(out/'sampling_sensitivity.csv',index=False);print(frac,'seconds',time.time()-t0,flush=True)
