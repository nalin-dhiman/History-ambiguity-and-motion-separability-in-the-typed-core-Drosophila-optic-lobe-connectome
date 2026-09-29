"""Reproduce the existing 200k-edge and five archived 80-step diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from decomposition import decompose
root=Path.cwd();base=root/'analysis_pipeline/core';legacy=root/'analysis_pipeline/diagnostics'
sys.path.insert(0,str(legacy))
from diagnostic_helpers.synthetic_tasks import generate_task_data
from diagnostic_helpers.plasticity_metrics import split_train_test,train_linear_probe
node=pd.read_csv(legacy/'data/spatial_metadata.csv').set_index('bodyId')
e=pd.read_csv(legacy/'data/edge_sample_00.csv.gz')
e['src_type']=e.src.map(node.type);e['dst_type']=e.dst.map(node.type)
_,_,v=decompose(e)
out=base/'results/development';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame([v]).to_csv(out/'archived_sample_decomposition.csv',index=False)
print('Archived 200k-edge decomposition:',{k:v[k] for k in ['A','C','R']})
values=[]
for sample in range(5):
    e=pd.read_csv(legacy/f'data/motion_edges_{sample}.csv.gz')
    src=pd.unique(e.src).tolist();dst=pd.unique(e.dst).tolist()
    # Preserve the archived per-edge normalization and float32 CSR construction.
    denom=e.groupby('dst').weight.transform('sum').replace(0,np.nan)
    weights=(e.weight/denom).fillna(0).to_numpy(dtype=np.float32)
    ri=e.src.map(pd.Series(np.arange(len(src)),index=src)).to_numpy()
    ci=e.dst.map(pd.Series(np.arange(len(dst)),index=dst)).to_numpy()
    W=sparse.coo_matrix((weights,(ri,ci)),shape=(len(src),len(dst))).tocsr()
    g=SimpleNamespace(source_nodes=src,target_nodes=dst,source_types=node.loc[src,'type'].tolist())
    seed=10000+1000*sample
    task=generate_task_data(g,None,'motion_direction',2000,.25,seed)
    tr,te,yt,ye=split_train_test(task['X_source'],task['y'],seed)
    result=train_linear_probe(np.asarray(tr@W,dtype=np.float32),yt,np.asarray(te@W,dtype=np.float32),ye,seed=seed,epochs=80,lr=.03)
    values.append(result['accuracy'])
print('Five archived static-task accuracies:',values,'mean:',np.mean(values))

pd.DataFrame({'sample_id':range(5),'accuracy':values}).to_csv(out/'archived_probe_scores.csv',index=False)
