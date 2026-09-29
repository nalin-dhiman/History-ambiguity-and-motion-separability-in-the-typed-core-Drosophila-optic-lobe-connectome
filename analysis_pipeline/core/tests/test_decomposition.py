import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from decomposition import decompose,matrix_metrics

def edges(m,types=None):
 m=np.asarray(m);r,c=np.nonzero(m);return pd.DataFrame({'src_type':['s'+str(i) for i in r],'dst':c,'dst_type':['d' if types is None else types[j] for j in c],'weight':m[r,c]})
def test_segregated_and_mixed():
 for m,want in [([[1,0],[0,1]],(1,0,1)),([[.5,.5],[.5,.5]],(1,1,0))]:
  _,t,g=decompose(edges(m));assert np.allclose([g[k] for k in ['A','C','R']],want)
def test_singleton_identical_zero_one_source():
 for m in [[[2],[3]],[[1,2],[3,6]],[[0,1,2],[0,3,6]],[[1,2,0]]]:
  r=matrix_metrics(m);assert abs(r['R'])<1e-12
 assert np.isnan(matrix_metrics(np.zeros((2,2)))['A'])
def test_scale_labels_disconnected():
 m=np.array([[2,0,1],[3,1,0],[0,2,4]])
 for k in [1,13,.001]:
  r=matrix_metrics(m*k);assert np.allclose(list(r.values()),list(matrix_metrics(m[::-1,::-1]).values()))
 e=edges(m,['a','b','b']);_,t,g=decompose(e)
 assert abs(g['A']-g['C']-g['R'])<1e-12
 assert t.loc[t.dst_type=='a','R'].abs().max()<1e-12
 for _,row in t.iterrows():
  d=e[e.dst_type==row.dst_type];p=d.pivot_table(index='src_type',columns='dst',values='weight',aggfunc='sum',fill_value=0)
  assert np.allclose([row[k] for k in ['A','C','R']],list(matrix_metrics(p).values()))
def test_random_sparse_independent():
 rng=np.random.default_rng(87)
 for _ in range(20):
  m=rng.poisson(.7,size=(9,13));_,_,g=decompose(edges(m))
  assert np.allclose([g[k] for k in ['A','C','R']],list(matrix_metrics(m).values()),atol=1e-12)
def test_input_selective_fragmentation():
 intact=matrix_metrics([[2],[2]]);split=matrix_metrics([[2,0],[0,2]])
 assert intact['A']==split['A']==1 and intact['C']==1 and split['R']==1
 # Aggregation cannot establish whether these two identifiers are two neurons or fragments.
def test_coarsening_and_invalid():
 m=np.array([[2,0,1],[3,1,0],[0,2,4]])
 a=matrix_metrics(m)['A'];assert matrix_metrics(np.vstack([m[:2].sum(0),m[2]]))['A']<=a+1e-12
 e=edges(m,['a','b','b']);_,_,g=decompose(e);e['dst_type']='d';_,_,h=decompose(e);assert h['A']>=g['A']-1e-12
 try: matrix_metrics([[-1]])
 except ValueError: pass
 else: raise AssertionError()
def test_task_counterexample():
 # Perfectly segregated sources s1 and s2: A=R=1, C=0. A downstream
 # equal-sum readout loses the anticorrelated label (+1,-1) versus (-1,+1).
 # Mixed anatomical inputs can preserve a common-mode label; R alone does
 # not specify source responses, readout, or a universal functional ordering.
 labels=np.array([[-1,1],[1,-1]])
 assert np.allclose(labels@np.ones((2,1)),0)
 assert matrix_metrics(np.eye(2))['R']==1

def test_float32_weights_promoted():
 rng=np.random.default_rng(5);m=rng.random((32,47)).astype(np.float32);_,_,g=decompose(edges(m));assert np.allclose([g[k] for k in ["A","C","R"]],list(matrix_metrics(m).values()),atol=1e-12)
