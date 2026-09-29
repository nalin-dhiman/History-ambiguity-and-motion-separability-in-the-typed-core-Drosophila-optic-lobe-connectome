from pathlib import Path
import pandas as pd
from decomposition import decompose
root=Path.cwd();out=root/'analysis_pipeline/core/results/development';e=pd.read_parquet(root/'outputs/processed_typed_core/edges.parquet',columns=['src','dst','src_type','dst_type','weight','edge_major_roi','src_major_roi','dst_major_roi']);rows=[]
for method in ['edge_dominant','source_major','target_major','strict_consensus']:
 for roi in ['ME(R)','LO(R)','LOP(R)']:
  mask={'edge_dominant':e.edge_major_roi.eq(roi),'source_major':e.src_major_roi.eq(roi),'target_major':e.dst_major_roi.eq(roi),'strict_consensus':e.edge_major_roi.eq(roi)&e.src_major_roi.eq(roi)&e.dst_major_roi.eq(roi)}[method]
  _,t,g=decompose(e[mask]);rows.append(dict(method=method,roi=roi,edge_count=int(mask.sum()),**g));t.to_csv(out/f'roi_types_{method}_{roi[:roi.index("(")]}.csv',index=False)
pd.DataFrame(rows).to_csv(out/'roi_sensitivity.csv',index=False);print(pd.DataFrame(rows)[['method','roi','A','C','R','weight']].to_string(index=False))
