"""Exact-margin and fixed-count controls in the recorded extension plan."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time
import numpy as np,pandas as pd
from scipy.stats import random_table,spearmanr
from scipy.special import xlogy
from decomposition import matrix_metrics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';CFG=json.loads((ROOT/'config.json').read_text());L2=np.log(2.)
def seed(name):return int.from_bytes(hashlib.sha256((str(CFG['base_seed'])+name).encode()).digest()[:8],'little')
def calcR(z):
 z=np.asarray(z,dtype=np.float64);r=z.sum(1);c=z.sum(0);W=c.sum();return float(np.log2(W)+(xlogy(z,z).sum()-xlogy(r,r).sum()-xlogy(c,c).sum())/(W*L2))
def job(arg):
 name,typ,m,rare=arg;m=np.asarray(m,dtype=np.int64);rng=np.random.default_rng(seed(name+typ));obs=matrix_metrics(m);r=m.sum(1);c=m.sum(0);vals=[]
 for k in range(CFG['fixed_margin_draws']):
  z=random_table.rvs(r,c,method='patefield',random_state=rng);assert np.array_equal(z.sum(1),r) and np.array_equal(z.sum(0),c);v=calcR(z)
  if k==0:assert np.isclose(v,matrix_metrics(z)['R'],atol=1e-11)
  vals.append(v)
 result=dict(universe=name,dst_type=typ,**obs,bodies=len(c),weight=int(c.sum()),source_types=len(r),null_R_mean=float(np.mean(vals)),null_R_sd=float(np.std(vals,ddof=1)),null_R_q025=float(np.quantile(vals,.025)),null_R_q975=float(np.quantile(vals,.975)),R_excess=float(obs['R']-np.mean(vals)))
 draws=[dict(universe=name,dst_type=typ,draw=k,R=v) for k,v in enumerate(vals)];rd=[]
 if rare:
  m=m[:,m.sum(0)>=CFG['contacts_per_body']]
  if m.shape[1]>=CFG['support_bodies']:
   rng=np.random.default_rng(seed(name+typ+'rarefaction'));base=matrix_metrics(m)
   for k in range(CFG['count_matched_draws']):
    z=np.column_stack([rng.multivariate_hypergeometric(m[:,j],CFG['contacts_per_body']) for j in range(m.shape[1])]);v=matrix_metrics(z);rd.append(dict(universe=name,dst_type=typ,draw=k,eligible_bodies=m.shape[1],full_eligible_R=base['R'],**v))
 return result,draws,rd
if __name__=='__main__':
 t0=time.time()
 for name in ['male_matched_right','female_matched_right','female_native_right']:
  typ=pd.read_csv(OUT/(name+'_types.csv'));support=set(typ[(typ.bodies>=20)&(typ.weight>=1000)].dst_type);x=pd.read_parquet(OUT/(name+'_inputs.parquet'));jobs=[]
  for t,q in x[x.dst_type.isin(support)].groupby('dst_type',sort=True):
   m=q.pivot_table(index='src_type',columns='dst',values='weight',aggfunc='sum',fill_value=0).to_numpy(dtype=np.int64);jobs.append((name,t,m,name!='female_native_right'))
  summaries=[];draws=[];rare=[]
  with ProcessPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(job,j) for j in jobs]
   for i,f in enumerate(as_completed(futures)):
    a,b,c=f.result();summaries.append(a);draws.extend(b);rare.extend(c)
    if i%25==0:print(name,i+1,'/',len(jobs),'elapsed',round(time.time()-t0,1),flush=True)
  s=pd.DataFrame(summaries).sort_values('dst_type');s.to_csv(OUT/(name+'_fixed_margin_summary.csv'),index=False);pd.DataFrame(draws).sort_values(['dst_type','draw']).to_csv(OUT/(name+'_fixed_margin_draws.csv.gz'),index=False)
  if rare:
   rr=pd.DataFrame(rare).sort_values(['dst_type','draw']);rr.to_csv(OUT/(name+'_count_matched_draws.csv.gz'),index=False);agg=rr.groupby('dst_type').agg(A_mean=('A','mean'),C_mean=('C','mean'),R_mean=('R','mean'),R_sd=('R','std'),full_eligible_R=('full_eligible_R','first'),eligible_bodies=('eligible_bodies','first')).reset_index();agg['R_fraction']=agg.R_mean/agg.A_mean;agg.to_csv(OUT/(name+'_count_matched_summary.csv'),index=False)
  print('DONE',name,'supported',len(s),'above upper',int((s.R>s.null_R_q975).sum()),'elapsed',round(time.time()-t0,1),flush=True)
 comparisons=[]
 for label,suffix,metric in [('reference_excess','_fixed_margin_summary.csv','R_excess'),('count_matched','_count_matched_summary.csv','R_mean'),('count_matched_fraction','_count_matched_summary.csv','R_fraction')]:
  a=pd.read_csv(OUT/('male_matched_right'+suffix));b=pd.read_csv(OUT/('female_matched_right'+suffix));p=a.merge(b,on='dst_type',suffixes=('_male','_female'));p.to_csv(OUT/(label+'_paired_types.csv'),index=False);x=p[metric+'_male'];y=p[metric+'_female'];comparisons.append(dict(comparison=label,metric=metric,types=len(p),spearman=float(spearmanr(x,y).statistic),mean_absolute_difference=float(abs(y-x).mean()),median_absolute_difference=float(abs(y-x).median()),mean_female_minus_male=float((y-x).mean())))
 pd.DataFrame(comparisons).to_csv(OUT/'controlled_type_agreement.csv',index=False);print(pd.DataFrame(comparisons).to_string(index=False),flush=True)
