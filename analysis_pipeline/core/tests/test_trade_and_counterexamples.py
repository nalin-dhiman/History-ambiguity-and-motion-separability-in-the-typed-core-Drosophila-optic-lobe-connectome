import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from decomposition import decompose,matrix_metrics

def test_integer_trade_preserves_margins_and_channel():
 a=np.array([[8,2],[2,8]]);b=a+np.array([[-3,3],[3,-3]])
 assert np.array_equal(a.sum(0),b.sum(0)) and np.array_equal(a.sum(1),b.sum(1))
 assert (b>0).all() and matrix_metrics(a)['A']==matrix_metrics(b)['A']
 assert matrix_metrics(a)['R']>matrix_metrics(b)['R']
 assert not np.array_equal(np.sort(a.ravel()),np.sort(b.ravel()))
def test_joint_coarsening_has_no_universal_order():
 # Original target types each have two equally weighted source types: A=1.
 # Merge targets and either pair sources 1/2 or pool sources 1/2/3.
 up=matrix_metrics([[2],[1],[1]])['A'];down=matrix_metrics([[3],[1]])['A']
 assert up>1>down

def test_functional_preference_changes_with_task_and_lesion():
 high=np.eye(2);low=np.full((2,2),.5);contrast=np.array([[1,0],[0,1]])
 assert not np.allclose((contrast@high)[0],(contrast@high)[1])
 assert np.allclose((contrast@low)[0],(contrast@low)[1])
 localized=np.array([[1,0],[0,0]])
 assert np.allclose((localized@high[:,1:])[0],(localized@high[:,1:])[1])
 assert not np.allclose((localized@low[:,1:])[0],(localized@low[:,1:])[1])
