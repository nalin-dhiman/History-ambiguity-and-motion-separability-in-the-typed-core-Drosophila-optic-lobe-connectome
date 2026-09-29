from pathlib import Path
import json,time
import numpy as np,pandas as pd
from scipy.stats import random_table,spearmanr
from decomposition import matrix_metrics,decompose
root=Path.cwd();out=root/'analysis_pipeline/core/results/development';cfg=json.loads((root/'analysis_pipeline/core/configs/gate1.json').read_text());t0=time.time()
x=pd.read_parquet(out/'source_type_body_inputs.parquet');types=pd.read_csv(out/'type_decomposition.csv');support=types[(types.bodies>=20)&(types.weight>=1000)];rng=np.random.default_rng(cfg['fixed_margin_null']['seed']);rows=[];draws=[];rare=[]
for idx,typ in enumerate(support.dst_type):
 q=x[x.dst_type==typ];m=q.pivot_table(index='src_type',columns='dst',values='weight',aggfunc='sum',fill_value=0).to_numpy(dtype=np.int64);obs=matrix_metrics(m);r=m.sum(1);c=m.sum(0);vv=[]
 for k in range(cfg['fixed_margin_null']['draws']):
  z=random_table.rvs(r,c,method='patefield',random_state=rng)
  assert np.array_equal(z.sum(1),r) and np.array_equal(z.sum(0),c)
  v=matrix_metrics(z)['R'];vv.append(v);draws.append(dict(dst_type=typ,draw=k,R=v))
 rows.append(dict(dst_type=typ,**obs,null_R_mean=np.mean(vv),null_R_sd=np.std(vv,ddof=1),null_R_q025=np.quantile(vv,.025),null_R_q975=np.quantile(vv,.975),R_excess=obs['R']-np.mean(vv),bodies=len(c),weight=c.sum(),source_types=len(r)))
 if idx%25==0:print('fixed margins',idx,len(support),'seconds',round(time.time()-t0,2),flush=True)
pd.DataFrame(rows).to_csv(out/'fixed_margin_summary.csv',index=False);pd.DataFrame(draws).to_csv(out/'fixed_margin_draws.csv.gz',index=False)
rng=np.random.default_rng(cfg['input_count_matching']['seed']);n=cfg['input_count_matching']['contacts_per_body']
for typ in support.dst_type:
 q=x[x.dst_type==typ];m=q.pivot_table(index='src_type',columns='dst',values='weight',aggfunc='sum',fill_value=0).to_numpy(dtype=np.int64);m=m[:,m.sum(0)>=n]
 if m.shape[1]<20:continue
 base=matrix_metrics(m)
 for k in range(cfg['input_count_matching']['draws']):
  z=np.column_stack([rng.multivariate_hypergeometric(m[:,j],n) for j in range(m.shape[1])]);v=matrix_metrics(z)
  rare.append(dict(dst_type=typ,draw=k,eligible_bodies=m.shape[1],full_eligible_A=base['A'],full_eligible_C=base['C'],full_eligible_R=base['R'],**v))
pd.DataFrame(rare).to_csv(out/'count_matched_draws.csv.gz',index=False)
r=pd.DataFrame(rare);s=r.groupby('dst_type').agg(A_mean=('A','mean'),C_mean=('C','mean'),R_mean=('R','mean'),R_q025=('R',lambda x:x.quantile(.025)),R_q975=('R',lambda x:x.quantile(.975)),full_eligible_R=('full_eligible_R','first'),eligible_bodies=('eligible_bodies','first')).reset_index();s.to_csv(out/'count_matched_summary.csv',index=False)
# Annotation restrictions change the input universe; report both types and totals.
e=pd.read_parquet(root/'outputs/processed_typed_core/edges.parquet',columns=['src','dst','weight','src_type','dst_type']);meta=pd.read_csv(root/'analysis_pipeline/core/audit/typed_body_metadata.csv.gz');summ=[]
for col,value in [('status:string','Traced'),('statusLabel:string','Roughly traced')]:
 ids=set(meta.loc[meta[col]==value,'bodyId']);_,tt,gg=decompose(e[e.src.isin(ids)&e.dst.isin(ids)]);label='traced' if col=='status:string' else 'roughly_traced';tt.to_csv(out/f'type_{label}.csv',index=False);summ.append(dict(universe=label,**gg))
pd.DataFrame(summ).to_csv(out/'status_decomposition.csv',index=False)
(out/'gate1_runtime.json').write_text(json.dumps({'elapsed_seconds':time.time()-t0,'supported_types':len(support),'null_draws':len(draws),'rare_types':s.dst_type.nunique(),'rare_draws':len(rare),'count_matched_rank_correlation':float(spearmanr(s.R_mean,s.full_eligible_R).statistic)},indent=2));print((out/'gate1_runtime.json').read_text(),flush=True)
