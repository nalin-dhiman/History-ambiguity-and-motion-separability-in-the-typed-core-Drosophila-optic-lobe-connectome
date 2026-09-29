"""Fixed-resolution conditional information and spatially blocked prediction."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from decomposition import matrix_metrics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';DATA=ROOT/'data_external';CFG=json.loads((ROOT/'config.json').read_text())
if len(CFG['spatial_shift'])!=2 or CFG['spatial_shift'][0]!=CFG['spatial_shift'][1]:raise ValueError('The shifted-grid output format requires equal shifts on both axes')
def partition(m,groups):
 groups=np.asarray(groups);levels,inverse=np.unique(groups,return_inverse=True);obs=matrix_metrics(m);W=m.sum();pooled=np.column_stack([m[:,inverse==i].sum(1) for i in range(len(levels))]);sp=matrix_metrics(pooled)['R'];res=0.;singleton_mass=0;singleton_bins=0
 for i in range(len(levels)):
  q=m[:,inverse==i];res+=q.sum()/W*matrix_metrics(q)['R']
  if q.shape[1]==1:singleton_bins+=1;singleton_mass+=q.sum()
 assert np.isclose(obs['R'],sp+res,atol=1e-11)
 return dict(**obs,R_spatial=sp,R_residual=res,spatial_share_R=sp/obs['R'] if obs['R']>1e-12 else np.nan,bins=len(levels),singleton_bins=singleton_bins,singleton_contact_fraction=singleton_mass/W,identity_error=obs['R']-sp-res)
def probabilities(train_counts,train_xy,test_xy,kernel_width=5,prior_contacts=100,pseudocount=.5):
 totals=train_counts.sum(0);base=(totals+pseudocount)/(totals.sum()+pseudocount*len(totals));delta=test_xy[:,None,:]-train_xy[None,:,:];kernel=np.exp(-np.sum(delta**2,axis=2)/(2*kernel_width**2));n=kernel@train_counts;p=(n+prior_contacts*base)/(n.sum(1,keepdims=True)+prior_contacts)
 assert np.allclose(p.sum(1),1) and np.all(p>0) and np.isclose(base.sum(),1);return base,p

def predict(m,xy,bodies,bounds,dataset,typ):
 counts=m.T;folds=[];bod=[];assigned=np.zeros(len(xy),dtype=int)
 for f,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
  test=(xy[:,0]>=lo)&(xy[:,0]<hi);train=(xy[:,0]<lo-CFG['predictive_buffer_steps'])|(xy[:,0]>=hi+CFG['predictive_buffer_steps']);assigned+=test
  assert not np.any(train&test)
  if train.sum()<50 or test.sum()<10:
   folds.append(dict(dataset=dataset,dst_type=typ,fold=f,status='insufficient_coverage',train_bodies=int(train.sum()),test_bodies=int(test.sum())));continue
  base,p=probabilities(counts[train],xy[train],xy[test],CFG['predictive_bandwidth'],CFG['predictive_prior_contacts'],CFG['predictive_pseudocount'])
  testcounts=counts[test];weight=testcounts.sum(1);l0=-(testcounts*np.log2(base)).sum(1);l1=-(testcounts*np.log2(p)).sum(1);gain=(l0-l1)/weight
  folds.append(dict(dataset=dataset,dst_type=typ,fold=f,status='evaluated',train_bodies=int(train.sum()),test_bodies=int(test.sum()),test_contacts=int(weight.sum()),baseline_ce_bits=float(l0.sum()/weight.sum()),spatial_ce_bits=float(l1.sum()/weight.sum()),gain_bits=float((l0-l1).sum()/weight.sum()),train_body_sha256=hashlib.sha256(bodies[train].astype(np.int64).tobytes()).hexdigest(),test_body_sha256=hashlib.sha256(bodies[test].astype(np.int64).tobytes()).hexdigest()))
  for j,idx in enumerate(np.flatnonzero(test)):bod.append(dict(dataset=dataset,dst_type=typ,fold=f,dst=int(bodies[idx]),contacts=int(weight[j]),x=float(xy[idx,0]),y=float(xy[idx,1]),baseline_ce_bits=float(l0[j]/weight[j]),spatial_ce_bits=float(l1[j]/weight[j]),gain_bits=float(gain[j])))
 assert np.all(assigned==1)
 return folds,bod
if __name__=='__main__':
 common=json.loads((OUT/'common_spatial_types.json').read_text());alph=json.loads((OUT/'matched_primary_alphabets.json').read_text());sources=alph['source_types'];mapping=pd.read_csv(OUT/'one_to_one_mapping.csv');fm=dict(zip(mapping.female_type,mapping.OL_type));rows=[];cv=[];bodycv=[];coverage=[]
 for dataset,graph,side in [('male_right','male_matched_right','right'),('female_right','female_matched_right','right'),('female_left','female_matched_left_lr','left')]:
  meta=pd.read_parquet(DATA/('male_metadata.parquet' if dataset.startswith('male') else 'female_metadata.parquet'));meta=meta[(meta.hemisphere==side)&meta.coordinate_valid].copy();meta['canonical_type']=meta.type if dataset.startswith('male') else meta.type.map(fm)
  x0=float(meta.x.min());y0=float(meta.y.min());xmax=float(meta.x.max());bounds=np.linspace(x0,xmax+1,CFG['predictive_folds']+1)
  x=pd.read_parquet(OUT/(graph+'_inputs.parquet'));selected=x[x.dst_type.isin(common)&x.dst.isin(meta.body_id)].copy();selected.to_parquet(OUT/(dataset+'_spatial_inputs.parquet'),index=False)
  coords=meta[meta.body_id.isin(selected.dst)].sort_values('body_id');coords[['body_id','canonical_type','x','y']].to_csv(OUT/(dataset+'_spatial_coordinates.csv'),index=False)
  for typ in common:
   q=selected[selected.dst_type==typ];ids=np.sort(q.dst.unique());m=q.pivot_table(index='src_type',columns='dst',values='weight',aggfunc='sum',fill_value=0).reindex(index=sources,columns=ids,fill_value=0).to_numpy(dtype=np.int64)
   assert m.shape[1]>=CFG['spatial_support_bodies'] and m.sum()>=CFG['spatial_support_contacts'];md=coords.set_index('body_id').loc[ids];assert (md.canonical_type==typ).all();xy=md[['x','y']].to_numpy();allq=x[x.dst_type==typ]
   coverage.append(dict(dataset=dataset,dst_type=typ,all_input_bodies=allq.dst.nunique(),coordinate_bodies=len(ids),all_contacts=int(allq.weight.sum()),coordinate_contacts=int(m.sum()),body_coverage=len(ids)/allq.dst.nunique(),contact_coverage=m.sum()/allq.weight.sum(),origin_x=x0,origin_y=y0))
   for width,shift in [(CFG['spatial_primary_width'],0)]+[(w,0) for w in CFG['spatial_other_widths']]+[(CFG['spatial_primary_width'],CFG['spatial_shift'][0])]:
    bx=np.floor((xy[:,0]-x0-shift)/width).astype(int);by=np.floor((xy[:,1]-y0-shift)/width).astype(int);groups=np.array([str(a)+':'+str(b) for a,b in zip(bx,by)]);v=partition(m,groups);rows.append(dict(dataset=dataset,dst_type=typ,width=width,shift=shift,weight=int(m.sum()),bodies=len(ids),**v))
   a,b=predict(m,xy,ids,bounds,dataset,typ);cv.extend(a);bodycv.extend(b)
  print(dataset,'spatial types',len(common),'bodies',selected.dst.nunique(),'contacts',int(selected.weight.sum()),flush=True)
 result=pd.DataFrame(rows);result.to_csv(OUT/'spatial_type_decomposition.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'spatial_coverage.csv',index=False);c=pd.DataFrame(cv);c.to_csv(OUT/'spatial_prediction_folds.csv',index=False);bc=pd.DataFrame(bodycv);bc.to_parquet(OUT/'spatial_prediction_bodies.parquet',index=False)
 globalrows=[]
 for (dataset,width,shift),g in result.groupby(['dataset','width','shift']):
  r={v:float(np.average(g[v],weights=g.weight)) for v in ['A','C','R','R_spatial','R_residual','singleton_contact_fraction']};r.update(dataset=dataset,width=int(width),shift=int(shift),weight=int(g.weight.sum()),bodies=int(g.bodies.sum()),types=len(g),C_share=100*r['C']/r['A'],spatial_share_R=100*r['R_spatial']/r['R'],residual_share_R=100*r['R_residual']/r['R']);globalrows.append(r)
 pd.DataFrame(globalrows).to_csv(OUT/'spatial_global.csv',index=False)
 pred=[]
 for (dataset,typ),g in bc.groupby(['dataset','dst_type']):pred.append(dict(dataset=dataset,dst_type=typ,bodies=len(g),contacts=int(g.contacts.sum()),**{v:float(np.average(g[v],weights=g.contacts)) for v in ['baseline_ce_bits','spatial_ce_bits','gain_bits']}))
 p=pd.DataFrame(pred);p.to_csv(OUT/'spatial_prediction_types.csv',index=False)
 pg=[]
 for dataset,g in p.groupby('dataset'):pg.append(dict(dataset=dataset,types=len(g),positive_gain_types=int((g.gain_bits>0).sum()),bodies=int(g.bodies.sum()),contacts=int(g.contacts.sum()),**{v:float(np.average(g[v],weights=g.contacts)) for v in ['baseline_ce_bits','spatial_ce_bits','gain_bits']}))
 pd.DataFrame(pg).to_csv(OUT/'spatial_prediction_global.csv',index=False)
 comparisons=[]
 for a,b,label in [('male_right','female_right','male_female_spatial'),('female_right','female_left','female_right_left_spatial')]:
  aa=result[(result.dataset==a)&(result.width==CFG['spatial_primary_width'])&(result['shift']==0)];bb=result[(result.dataset==b)&(result.width==CFG['spatial_primary_width'])&(result['shift']==0)];pair=aa.merge(bb,on='dst_type',suffixes=('_a','_b'));pair.to_csv(OUT/(label+'_paired_types.csv'),index=False)
  for metric in ['R','R_spatial','R_residual','spatial_share_R']:
   xx=pair[metric+'_a'];yy=pair[metric+'_b'];comparisons.append(dict(comparison=label,metric=metric,types=len(pair),spearman=float(spearmanr(xx,yy).statistic),mean_absolute_difference=float(abs(xx-yy).mean()),mean_b_minus_a=float((yy-xx).mean())))
 pd.DataFrame(comparisons).to_csv(OUT/'spatial_type_agreement.csv',index=False)
 primary=pd.DataFrame(globalrows);print(primary[(primary.width==CFG['spatial_primary_width'])&(primary['shift']==0)].to_string(index=False),flush=True);print(pd.DataFrame(pg).to_string(index=False),flush=True);print(pd.DataFrame(comparisons).to_string(index=False),flush=True)
