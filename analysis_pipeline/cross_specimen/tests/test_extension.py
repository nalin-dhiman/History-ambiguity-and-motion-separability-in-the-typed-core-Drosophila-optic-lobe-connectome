from pathlib import Path
import sys
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_spatial import partition,probabilities
from run_replication import annotate,shared_roles
from run_references import calcR
from decomposition import matrix_metrics

def test_spatial_partition_exact_mixed_and_segregated():
 m=np.array([[8,8,2,2],[2,2,8,8]])
 p=partition(m,['left','left','right','right'])
 assert p['R_spatial']>0 and abs(p['R_residual'])<1e-12
 p=partition(m,['a','b','a','b'])
 assert abs(p['R_spatial'])<1e-12 and p['R_residual']>0

def test_unique_coordinate_trap_is_exposed():
 m=np.array([[7,2,1],[1,6,7]])
 p=partition(m,[1,2,3]);assert abs(p['R_residual'])<1e-12 and p['singleton_contact_fraction']==1
 p=partition(m,[0,0,0]);assert abs(p['R_spatial'])<1e-12 and np.isclose(p['R_residual'],p['R'])

def test_female_neuropil_rows_aggregate_before_threshold_and_no_float_ids():
 ids=[720575940596125868,720575940597856265]
 meta=pd.DataFrame({'body_id':ids,'type':['s','d']})
 edges=pd.DataFrame({'src':[ids[0],ids[0],ids[1]],'dst':[ids[1],ids[1],ids[1]],'weight':[2,2,9]})
 x=annotate(edges,meta);assert len(x)==1 and x.iloc[0].weight==4 and x.iloc[0].src==ids[0] and x.iloc[0].dst==ids[1]
 assert len(x[x.weight>=3])==1

def test_common_role_pruning_removes_cascading_unmatched_edges():
 a=pd.DataFrame({'src_type':['a','b','c'],'dst_type':['d','e','f']})
 b=pd.DataFrame({'src_type':['a','b'],'dst_type':['d','f']})
 aa,bb,s,d=shared_roles(a,b);assert s==['a'] and d==['d'] and len(aa)==len(bb)==1

def test_fast_reference_entropy_agrees_with_independent_sparse_form():
 for m in [np.array([[0,9],[6,1]]),np.array([[2,0,8],[0,4,1],[3,0,0]]),np.array([[1,2,3]])]:assert np.isclose(calcR(m),matrix_metrics(m)['R'],atol=1e-12)

def test_prediction_is_training_only_and_handles_unseen_source_types():
 counts=np.array([[9,1,0],[8,2,0],[1,9,0],[2,8,0]])
 xy=np.array([[0,0],[0,1],[10,0],[10,1]],float);target=np.array([[0,0],[10,0]],float)
 base,p=probabilities(counts,xy,target)
 assert p[0,0]>p[1,0] and np.all(p[:,2]>0) and np.allclose(p.sum(1),1)
 # Held-out source counts are not an input to the fit: changing outcomes cannot change predictions.
 heldout=np.array([[9,0,1],[0,9,1]]);heldout[:]=100
 base2,p2=probabilities(counts,xy,target);assert np.array_equal(base,base2) and np.array_equal(p,p2)
