from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,shutil,datetime
root=Path(__file__).resolve().parents[1];workspace=root.parent
cfg=json.loads((root/'config.json').read_text())
BASE=workspace/'core'
(root/'results').mkdir(parents=True,exist_ok=True)
(root/'provenance').mkdir(parents=True,exist_ok=True)
male=pd.read_parquet(BASE/'data_external/raw_typed_metadata.parquet').rename(columns={'bodyId:long':'body_id','type:string':'type','instance:string':'instance','assignedOlHex1:float':'x','assignedOlHex2:float':'y'})
female=pd.read_csv(root/'data_external/female_neuron_table.csv').rename(columns={'cell ID':'body_id','resolved type':'type'})
assert male.body_id.is_unique and female.body_id.is_unique
female['hemisphere']='other'
for side in ['left','right']:
 idx=female.category.isin([side+' intrinsic',side+' boundary'])|((female.category=='bilateral')&(female['% syns '+side+' lobe']>female['% syns '+('right' if side=='left' else 'left')+' lobe']))
 female.loc[idx,'hemisphere']=side
female['visual']=female.category!='excluded'
male['hemisphere']=np.where(male.instance.str.endswith('_R',na=False),'right',np.where(male.instance.str.endswith('_L',na=False),'left','other'))
path=next((root/'data_external/Nern2025').glob('*.xlsx'));mapping=pd.read_excel(path);mapping.to_csv(root/'data_external/Nern2025_published_matching.csv',index=False)
# The published matching classification uses Matsliah labels when present; do not fall back to a coarser Schlegel label in that case.
mapping['female_type']=mapping.Matsliah_type.where(mapping.Matsliah_type.notna(),mapping.Schlegel_type).str.strip()
knownm=set(male.loc[male.hemisphere=='right','type']);knownf=set(female.loc[female.hemisphere=='right','type'].dropna())
mapping['reason']='eligible'
mapping.loc[mapping['matched as']!='1-to-1','reason']='not_one_to_one'
mapping.loc[(mapping.reason=='eligible')&~mapping.OL_type.isin(knownm),'reason']='male_label_absent'
mapping.loc[(mapping.reason=='eligible')&~mapping.female_type.isin(knownf),'reason']='female_exact_label_absent'
cand=mapping[mapping.reason=='eligible'];dup_m=set(cand.loc[cand.OL_type.duplicated(False),'OL_type']);dup_f=set(cand.loc[cand.female_type.duplicated(False),'female_type']);mapping.loc[(mapping.reason=='eligible')&(mapping.OL_type.isin(dup_m)|mapping.female_type.isin(dup_f)),'reason']='non_unique_correspondence'
matched=mapping[mapping.reason=='eligible'].copy();assert matched.OL_type.is_unique and matched.female_type.is_unique
mapping.to_csv(root/'results/type_mapping_audit.csv',index=False);matched[['OL_type','female_type','group','matched as']].to_csv(root/'results/one_to_one_mapping.csv',index=False)
# Coordinate metadata are joined by exact int64 body/root IDs. No coordinates are inferred from new connectivity fits.
coord=pd.read_csv(root/'data_external/female_column_assignment.csv.gz')[['root_id','hemisphere','type','p','q','column_id']].rename(columns={'root_id':'body_id','type':'coordinate_type','p':'x','q':'y','hemisphere':'coordinate_hemisphere'})
coord=coord[['body_id','coordinate_type','coordinate_hemisphere','x','y','column_id']]
female=female.merge(coord,on='body_id',how='left',validate='one_to_one');female['coordinate_valid']=female.x.notna()&female.y.notna()&(female.type==female.coordinate_type)&(female.hemisphere==female.coordinate_hemisphere)
male['coordinate_valid']=male.x.notna()&male.y.notna()&(male.hemisphere=='right')
male.to_parquet(root/'data_external/male_metadata.parquet',index=False);female.to_parquet(root/'data_external/female_metadata.parquet',index=False)
fm=dict(zip(matched.female_type,matched.OL_type));female['canonical_type']=female.type.map(fm)
rows=[]
for dataset,df in [('male_right',male[male.hemisphere=='right']),('female_right',female[female.hemisphere=='right']),('female_left',female[female.hemisphere=='left'])]:
 for typ,g in df.groupby('type'):
  rows.append(dict(dataset=dataset,type=typ,canonical_type=typ if dataset.startswith('male') else fm.get(typ),bodies=len(g),valid_coordinates=int(g.coordinate_valid.sum()),coordinate_fraction=float(g.coordinate_valid.mean())))
coverage=pd.DataFrame(rows);coverage.to_csv(root/'results/coordinate_coverage.csv',index=False)
common=set(coverage[(coverage.dataset=='male_right')&(coverage.valid_coordinates>=cfg['spatial_support_bodies'])].canonical_type)&set(coverage[(coverage.dataset=='female_right')&(coverage.valid_coordinates>=cfg['spatial_support_bodies'])].canonical_type)
(root/'results/common_spatial_types.json').write_text(json.dumps(sorted(common),indent=2))
# Copy the authoritative mathematical implementation, with its source hash recorded.
src=BASE/'src/decomposition.py';shutil.copy2(src,root/'scripts/decomposition.py')
report={'mapping_source':'Nern2025 published Supplementary Table 7','matching_rows':len(mapping),'exclusion_counts':mapping.reason.value_counts().to_dict(),'one_to_one_types_metadata':len(matched),'female_neuron_ids':len(female),'female_visual_categories':female.groupby(['category','hemisphere']).size().to_dict() if False else female.category.value_counts().to_dict(),'female_coordinate_rows':len(coord),'female_valid_coordinates':int(female.coordinate_valid.sum()),'female_coordinate_type_mismatches':int((female.coordinate_type.notna()&(female.coordinate_type!=female.type)).sum()),'male_coordinates':int(male.coordinate_valid.sum()),'common_spatial_types':sorted(common),'copied_decomposition_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'metadata_audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'provenance/metadata_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
