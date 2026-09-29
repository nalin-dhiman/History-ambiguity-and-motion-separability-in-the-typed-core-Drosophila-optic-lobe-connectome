from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np,pandas as pd
from decomposition import decompose,matrix_metrics
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args();root=a.root;out=root/'analysis_pipeline/core/results/development';out.mkdir(parents=True,exist_ok=True)
t0=time.time();e=pd.read_parquet(root/'outputs/processed_typed_core/edges.parquet',columns=['src','dst','weight','src_type','dst_type','edge_major_roi']);n=pd.read_parquet(root/'outputs/processed_typed_core/nodes.parquet');summ=[]
b,t,g=decompose(e);b.to_parquet(out/'body_decomposition.parquet',index=False);t.to_csv(out/'type_decomposition.csv',index=False);summ.append(dict(universe='full',**g));print(summ[-1],flush=True)
e.groupby(['dst_type','dst','src_type'],observed=True).weight.sum().rename('weight').reset_index().to_parquet(out/'source_type_body_inputs.parquet',index=False)
# Recompute each compartment using its complete retained input universe.
for roi in ['ME(R)','LO(R)','LOP(R)']:
 bb,tt,gg=decompose(e[e.edge_major_roi==roi]);bb.to_parquet(out/f'body_{roi[:roi.index("(")]}.parquet',index=False);tt.to_csv(out/f'type_{roi[:roi.index("(")]}.csv',index=False);summ.append(dict(universe=roi,**gg));print(summ[-1],flush=True)
for threshold in [3,5,10]:
 _,tt,gg=decompose(e[e.weight>=threshold]);tt.to_csv(out/f'type_threshold_{threshold}.csv',index=False);summ.append(dict(universe=f'edge_weight_ge_{threshold}',**gg));print(summ[-1],flush=True)
flag=n.type.str.contains('fragment|unclear|uncertain',case=False,regex=True,na=False);ids=set(n.loc[flag,'bodyId']);_,tt,gg=decompose(e[~e.src.isin(ids)&~e.dst.isin(ids)]);tt.to_csv(out/'type_clear_labels.csv',index=False);summ.append(dict(universe='exclude_unclear_label_endpoints',**gg));print(summ[-1],flush=True)
# Equal-body measure: normalize EACH target body to total mass 1, then recompute
# source priors and target-type weighting under that new measure.
den=e.groupby('dst',observed=True).weight.sum();eq=e.copy();eq['weight']=eq.weight/eq.dst.map(den);_,tt,gg=decompose(eq);tt.to_csv(out/'type_equal_body.csv',index=False);summ.append(dict(universe='equal_body_measure',**gg));print(summ[-1],flush=True)
pd.DataFrame(summ).to_csv(out/'universe_decomposition.csv',index=False)
# Independent sparse calculation on all source-type/body aggregated entries.
x=pd.read_parquet(out/'source_type_body_inputs.parquet');ind=[]
from scipy.sparse import coo_matrix
for typ,part in x.groupby('dst_type',observed=True):
 s,st=pd.factorize(part.src_type);j,jt=pd.factorize(part.dst);m=coo_matrix((part.weight,(s,j)),shape=(len(st),len(jt))).tocsr();r=matrix_metrics(m);ind.append(dict(dst_type=typ,**r))
c=pd.DataFrame(ind).merge(t,on='dst_type',suffixes=('_sparse','_groupby'))
err=max((c[k+'_sparse']-c[k+'_groupby']).abs().max() for k in ['A','C','R']);assert err<1e-10
stats={'elapsed_seconds':time.time()-t0,'full_edges':len(e),'full_bodies':len(n),'zero_input_bodies':len(n)-len(b),'independent_sparse_max_error':float(err),'unclear_label_bodies':int(flag.sum()),'input_hashes':{str(f):hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['outputs/processed_typed_core/nodes.parquet','outputs/processed_typed_core/edges.parquet']}}
(out/'decomposition_run.json').write_text(json.dumps(stats,indent=2));print(stats,flush=True)
