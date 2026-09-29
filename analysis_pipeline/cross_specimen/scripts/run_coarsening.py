"""Conservative sensitivity using fully resolved published matching components."""
from pathlib import Path
import json,re
import pandas as pd,numpy as np
from decomposition import decompose
from run_replication import shared_roles
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_external';OUT=ROOT/'results'
if __name__=='__main__':
 m=pd.read_parquet(DATA/'male_metadata.parquet');f=pd.read_parquet(DATA/'female_metadata.parquet');m=m[m.hemisphere=='right'];f=f[f.hemisphere=='right'];km=set(m.type);kf=set(f.type);table=pd.read_csv(DATA/'Nern2025_published_matching.csv');parent={};bad=set();audit=[]
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(x,y):parent[find(y)]=find(x)
 def resolve(text,known):
  if pd.isna(text):return []
  text=str(text).strip()
  if text in known:return [text]
  toks=[x.strip() for x in re.split(r'[+,;]',text)]
  return toks if len(toks)>1 and all(x in known for x in toks) else []
 for idx,row in table.iterrows():
  text=row.Matsliah_type if pd.notna(row.Matsliah_type) else row.Schlegel_type;ms=resolve(row.OL_type,km);fs=resolve(text,kf)
  ok=bool(ms and fs and row['matched as']!='unmatched')
  if ok:
   nodes=['m:'+v for v in ms]+['f:'+v for v in fs]
   for n in nodes:union(nodes[0],n)
  else:
   # Recognized literal members of an unresolved mapping invalidate any linked component.
   for prefix,value,known in [('m:',row.OL_type,km),('f:',text,kf)]:
    if pd.isna(value):continue
    for typ in known:
     if re.search(r'(?<![A-Za-z0-9_-])'+re.escape(typ)+r'(?![A-Za-z0-9_-])',str(value)):bad.add(prefix+typ)
  audit.append(dict(row=idx,male_text=row.OL_type,female_text=text,match_class=row['matched as'],parsed_complete=ok))
 components={}
 for n in parent:components.setdefault(find(n),[]).append(n)
 mm={};fm={};groups=[]
 for nodes in components.values():
  males=sorted(x[2:] for x in nodes if x.startswith('m:'));females=sorted(x[2:] for x in nodes if x.startswith('f:'));include=not bool(set(nodes)&bad);label=' / '.join(males)
  groups.append(dict(group_label=label,male_types=';'.join(males),female_types=';'.join(females),n_male_types=len(males),n_female_types=len(females),included=include))
  if include:
   mm.update({x:label for x in males});fm.update({x:label for x in females})
 pd.DataFrame(audit).to_csv(OUT/'coarsening_mapping_rows.csv',index=False);pd.DataFrame(groups).to_csv(OUT/'coarsening_components.csv',index=False)
 graphs=[]
 for name,mp in [('male_native_right',mm),('female_native_right',fm)]:
  e=pd.read_parquet(DATA/(name+'_edges.parquet'));e['src_type']=e.src_type.map(mp);e['dst_type']=e.dst_type.map(mp);e=e.dropna(subset=['src_type','dst_type']);graphs.append(e)
 a,b,ss,dd=shared_roles(*graphs);rows=[]
 for name,e in [('male_coarsened_right',a),('female_coarsened_right',b)]:
  bod,typ,g=decompose(e);g.update(universe=name,edges=len(e),C_share=100*g['C']/g['A'],R_share=100*g['R']/g['A']);rows.append(g);typ.to_csv(OUT/(name+'_types.csv'),index=False)
 pd.DataFrame(rows).to_csv(OUT/'coarsened_global.csv',index=False)
 (OUT/'coarsened_alphabets.json').write_text(json.dumps({'source_types':ss,'target_types':dd},indent=2))
 print(pd.DataFrame(rows).to_string(index=False));print('Fully included components',sum(x['included'] for x in groups),'complex included',sum(x['included'] and (x['n_male_types']>1 or x['n_female_types']>1) for x in groups))
