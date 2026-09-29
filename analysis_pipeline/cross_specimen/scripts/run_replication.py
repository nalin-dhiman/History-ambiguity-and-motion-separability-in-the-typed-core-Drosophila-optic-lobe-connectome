"""Declared empirical replication. Reads local frozen connectomes and pinned metadata."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from scipy import sparse
from decomposition import decompose,matrix_metrics
ROOT=Path(__file__).resolve().parents[1];WS=ROOT.parent;DATA=ROOT/'data_external';OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
lock=json.loads((ROOT/'config.json').read_text())
BASE=WS.parent
CORE=WS/'core'
rows=[];checks=[]
def annotate(edges,meta,label='type'):
 ids=meta.set_index('body_id')[label];e=edges[edges.src.isin(ids.index)&edges.dst.isin(ids.index)].copy();e=e[e.src!=e.dst]
 e=e.groupby(['src','dst'],sort=True,observed=True).weight.sum().reset_index();e['src_type']=e.src.map(ids);e['dst_type']=e.dst.map(ids)
 assert not e.isna().any().any() and (e.weight>0).all() and (e.weight==np.floor(e.weight)).all()
 return e

def shared_roles(a,b):
 for _ in range(100):
  s=set(a.src_type)&set(b.src_type);d=set(a.dst_type)&set(b.dst_type);an=a[a.src_type.isin(s)&a.dst_type.isin(d)].copy();bn=b[b.src_type.isin(s)&b.dst_type.isin(d)].copy()
  if len(an)==len(a) and len(bn)==len(b):
   assert set(an.src_type)==set(bn.src_type) and set(an.dst_type)==set(bn.dst_type);return an,bn,sorted(s),sorted(d)
  a,b=an,bn
 raise AssertionError('No common-role convergence')

def save(e,name,keep_edges=True):
 t=time.time();b,typ,g=decompose(e);g.update(universe=name,edges=len(e),C_share=100*g['C']/g['A'],R_share=100*g['R']/g['A']);rows.append(g)
 assert np.isclose(g['weight'],e.weight.sum(),atol=1e-8,rtol=1e-12) and np.isclose(g['A'],g['C']+g['R'],atol=1e-12)
 b.to_parquet(OUT/(name+'_bodies.parquet'),index=False);typ.to_csv(OUT/(name+'_types.csv'),index=False)
 x=e.groupby(['dst_type','dst','src_type'],observed=True,sort=True).weight.sum().reset_index();x.to_parquet(OUT/(name+'_inputs.parquet'),index=False)
 # Independent sparse calculations for every positive-input type.
 maxerr=0.
 for tname,q in x.groupby('dst_type',observed=True):
  si,_=pd.factorize(q.src_type,sort=True);ji,_=pd.factorize(q.dst,sort=True);m=sparse.coo_matrix((q.weight.to_numpy(),(si,ji))).tocsr();v=matrix_metrics(m);v2=typ[typ.dst_type==tname].iloc[0]
  maxerr=max(maxerr,max(abs(v[k]-v2[k]) for k in ['A','C','R']))
 assert maxerr<1e-10
 checks.append({'universe':name,'independent_sparse_max_error':maxerr,'mass_exact':True,'type_rows':len(typ)})
 if keep_edges:e.to_parquet(DATA/(name+'_edges.parquet'),index=False)
 print(name,{k:g[k] for k in ['A','C','R','C_share','weight','target_bodies','source_types','target_types']},'seconds',round(time.time()-t,1),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'replication_global.csv',index=False)
 return typ

def compare(ta,tb,name):
 a=ta[(ta.bodies>=lock['support_bodies'])&(ta.weight>=lock['support_contacts'])];b=tb[(tb.bodies>=lock['support_bodies'])&(tb.weight>=lock['support_contacts'])];p=a.merge(b,on='dst_type',suffixes=('_a','_b'),validate='one_to_one');p.to_csv(OUT/(name+'_paired_types.csv'),index=False)
 rec=[]
 for metric in ['R','R_fraction','A','C']:
  x=p[metric+'_a'];y=p[metric+'_b'];ok=x.notna()&y.notna();x=x[ok];y=y[ok];rec.append(dict(comparison=name,metric=metric,types=len(x),spearman=float(spearmanr(x,y).statistic),mean_absolute_difference=float(np.mean(abs(y-x))),median_absolute_difference=float(np.median(abs(y-x))),mean_b_minus_a=float(np.mean(y-x))))
 return rec

if __name__=='__main__':
 male=pd.read_parquet(DATA/'male_metadata.parquet');female=pd.read_parquet(DATA/'female_metadata.parquet');mapping=pd.read_csv(OUT/'one_to_one_mapping.csv');fm=dict(zip(mapping.female_type,mapping.OL_type));mm=dict(zip(mapping.OL_type,mapping.OL_type))
 me=pd.read_parquet(BASE/'outputs/processed_typed_core/edges.parquet',columns=['src','dst','weight']);fe=pd.read_parquet(DATA/'female_proofread_connections.parquet',columns=['src','dst','weight'])
 raw=pd.read_csv(CORE/'results/development/universe_decomposition.csv').query("universe=='full'").iloc[0].to_dict();raw['universe']='male_original_audited';raw['C_share']=100*raw['C']/raw['A'];raw['R_share']=100*raw['R']/raw['A'];rows.append(raw)
 graphs={};types={}
 for name,ed,md in [('male_native_right',me,male[male.hemisphere=='right']),('female_native_visual_pooled',fe,female[female.visual]),('female_native_right',fe,female[female.hemisphere=='right']),('female_native_left',fe,female[female.hemisphere=='left'])]:
  e=annotate(ed,md);types[name]=save(e,name,keep_edges=name!='female_native_visual_pooled')
  if name!='female_native_visual_pooled':graphs[name]=e
 del me,fe
 for label in ['male','female']:
  meta=(male if label=='male' else female).copy();meta['canonical_type']=meta.type.map(mm if label=='male' else fm);meta=meta[meta.canonical_type.notna()&(meta.hemisphere=='right')]
  graphs[label+'_matched_right']=annotate(graphs[label+'_native_right'],meta,'canonical_type')
 a,b,ss,dd=shared_roles(graphs['male_matched_right'],graphs['female_matched_right']);graphs['male_matched_right']=a;graphs['female_matched_right']=b
 (OUT/'matched_primary_alphabets.json').write_text(json.dumps({'source_types':ss,'target_types':dd},indent=2))
 agreements=[]
 for name in ['male_matched_right','female_matched_right']:
  e=graphs[name];types[name]=save(e,name)
  for thr in lock['edge_thresholds']:save(e[e.weight>=thr],name+'_threshold_'+str(thr),False)
  eq=e.copy();tot=eq.groupby('dst',observed=True).weight.transform('sum');eq['weight']=eq.weight/tot;save(eq,name+'_equal_body',False)
 agreements+=compare(types['male_matched_right'],types['female_matched_right'],'male_female_primary')
 leftmeta=female[(female.hemisphere=='left')&female.type.isin(fm)].copy();leftmeta['canonical_type']=leftmeta.type.map(fm)
 left=annotate(graphs['female_native_left'],leftmeta,'canonical_type');right=graphs['female_matched_right'];left,right,s,d=shared_roles(left,right)
 types['female_matched_left_lr']=save(left,'female_matched_left_lr');types['female_matched_right_lr']=save(right,'female_matched_right_lr')
 (OUT/'matched_hemisphere_alphabets.json').write_text(json.dumps({'source_types':s,'target_types':d},indent=2));agreements+=compare(types['female_matched_right_lr'],types['female_matched_left_lr'],'female_right_left')
 # Equal-body total weight is body count, so ordinary contact support above is inappropriate: use primary-supported labels instead.
 primary=pd.read_csv(OUT/'male_female_primary_paired_types.csv').dst_type
 aa=pd.read_csv(OUT/'male_matched_right_equal_body_types.csv');bb=pd.read_csv(OUT/'female_matched_right_equal_body_types.csv');p=aa[aa.dst_type.isin(primary)].merge(bb[bb.dst_type.isin(primary)],on='dst_type',suffixes=('_a','_b'));p.to_csv(OUT/'male_female_equal_body_paired_types.csv',index=False)
 agreements=[x for x in agreements if x['comparison']!='male_female_equal_body']
 for metric in ['R','R_fraction']:
  x=p[metric+'_a'];y=p[metric+'_b'];agreements.append(dict(comparison='male_female_equal_body',metric=metric,types=len(x),spearman=float(spearmanr(x,y).statistic),mean_absolute_difference=float(np.mean(abs(y-x))),median_absolute_difference=float(np.median(abs(y-x))),mean_b_minus_a=float(np.mean(y-x))))
 pd.DataFrame(agreements).to_csv(OUT/'type_agreement.csv',index=False);(ROOT/'provenance/replication_validation.json').write_text(json.dumps(checks,indent=2));print(pd.DataFrame(agreements).to_string(index=False),flush=True)
