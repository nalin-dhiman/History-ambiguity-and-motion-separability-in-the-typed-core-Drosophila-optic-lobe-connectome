"""Conditional allocation within the predeclared target-type/spatial blocks."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import numpy as np,pandas as pd
from scipy.stats import random_table,spearmanr
from decomposition import matrix_metrics
from run_references import calcR,seed
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';CFG=json.loads((ROOT/'config.json').read_text())
def one_job(arg):
 dataset,typ,blocks=arg;W=sum(m.sum() for m in blocks);observed=sum(m.sum()/W*matrix_metrics(m)['R'] for m in blocks);vv=np.zeros(CFG['fixed_margin_draws']);rng=np.random.default_rng(seed('spatial'+dataset+typ))
 for m in blocks:
  mass=m.sum();r=m.sum(1);c=m.sum(0)
  if len(c)==1:continue
  for k in range(len(vv)):
   z=random_table.rvs(r,c,method='patefield',random_state=rng);assert np.array_equal(z.sum(1),r) and np.array_equal(z.sum(0),c);vv[k]+=mass/W*calcR(z)
 return dict(dataset=dataset,dst_type=typ,weight=int(W),bins=len(blocks),observed_residual=float(observed),reference_mean=float(vv.mean()),reference_q025=float(np.quantile(vv,.025)),reference_q975=float(np.quantile(vv,.975)),residual_excess=float(observed-vv.mean())),[dict(dataset=dataset,dst_type=typ,draw=k,R_residual=float(v)) for k,v in enumerate(vv)]
if __name__=='__main__':
 # These products must exist first, preserving the planned order of evaluation.
 assert (OUT/'spatial_prediction_global.csv').exists();coverage=pd.read_csv(OUT/'spatial_coverage.csv');jobs=[]
 for dataset in ['male_right','female_right','female_left']:
  q=pd.read_parquet(OUT/(dataset+'_spatial_inputs.parquet'));c=pd.read_csv(OUT/(dataset+'_spatial_coordinates.csv'));q=q.merge(c.rename(columns={'body_id':'dst'}),on='dst',validate='many_to_one');orig=coverage[coverage.dataset==dataset].iloc[0];q['bx']=np.floor((q.x-orig.origin_x)/CFG['spatial_primary_width']).astype(int);q['by']=np.floor((q.y-orig.origin_y)/CFG['spatial_primary_width']).astype(int)
  for typ,g in q.groupby('dst_type'):
   blocks=[v.pivot_table(index='src_type',columns='dst',values='weight',aggfunc='sum',fill_value=0).to_numpy(dtype=np.int64) for _,v in g.groupby(['bx','by'])];jobs.append((dataset,typ,blocks))
 summaries=[];draws=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,f in enumerate(as_completed([pool.submit(one_job,j) for j in jobs])):
   a,b=f.result();summaries.append(a);draws.extend(b)
   if i%10==0:print('completed',i+1,'/',len(jobs),flush=True)
 s=pd.DataFrame(summaries).sort_values(['dataset','dst_type']);s.to_csv(OUT/'spatial_reference_summary.csv',index=False);pd.DataFrame(draws).sort_values(['dataset','dst_type','draw']).to_csv(OUT/'spatial_reference_draws.csv.gz',index=False)
 exact=pd.read_csv(OUT/'spatial_type_decomposition.csv');exact=exact[(exact.width==CFG['spatial_primary_width'])&(exact['shift']==0)];z=s.merge(exact[['dataset','dst_type','R_residual']],on=['dataset','dst_type'],validate='one_to_one');assert np.allclose(z.observed_residual,z.R_residual,atol=1e-11)
 glob=[]
 for d,g in s.groupby('dataset'):glob.append(dict(dataset=d,types=len(g),above_reference_q975=int((g.observed_residual>g.reference_q975).sum()),weight=int(g.weight.sum()),**{col:float(np.average(g[col],weights=g.weight)) for col in ['observed_residual','reference_mean','residual_excess']}))
 pd.DataFrame(glob).to_csv(OUT/'spatial_reference_global.csv',index=False)
 a=s[s.dataset=='male_right'];b=s[s.dataset=='female_right'];p=a.merge(b,on='dst_type',suffixes=('_male','_female'));p.to_csv(OUT/'spatial_reference_paired_types.csv',index=False);rec={'types':len(p),'excess_spearman':float(spearmanr(p.residual_excess_male,p.residual_excess_female).statistic),'excess_MAE_bits':float(abs(p.residual_excess_male-p.residual_excess_female).mean())};(OUT/'spatial_reference_agreement.json').write_text(json.dumps(rec,indent=2));print(pd.DataFrame(glob).to_string(index=False),flush=True);print(rec,flush=True)
