"""Exact source-type entropy decomposition for directed weighted input tables.
All positive rows are aggregated before entropy. A,C,R are in bits.
The statistical measure is a uniformly selected retained synaptic contact.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import xlogy
LOG2=np.log(2.0)
def xlog2x(v): return xlogy(v,v)/LOG2

def decompose(edges):
    required=['src_type','dst','dst_type','weight']
    e=edges[required].copy()
    if e[required].isna().any().any(): raise ValueError('Missing endpoint/type/weight')
    if (e.weight<0).any() or not np.isfinite(e.weight).all(): raise ValueError('Invalid weights')
    e=e[e.weight>0]
    e['weight']=e.weight.astype(np.float64)
    if e.empty: return pd.DataFrame(),pd.DataFrame(),dict(A=np.nan,C=np.nan,R=np.nan,weight=0)
    if (e.groupby('dst',observed=True).dst_type.nunique()>1).any(): raise ValueError('Body has multiple target types')
    b=e.groupby(['dst_type','dst','src_type'],observed=True,sort=True).weight.sum().rename('b').reset_index()
    b['b_log_b']=xlog2x(b.b.to_numpy())
    bodies=b.groupby(['dst_type','dst'],observed=True).agg(weight=('b','sum'),sum_b_log_b=('b_log_b','sum'),source_types=('src_type','size')).reset_index()
    bodies['H']=np.log2(bodies.weight)-bodies.sum_b_log_b/bodies.weight
    bodies['weighted_H']=bodies.weight*bodies.H
    types=bodies.groupby('dst_type',observed=True).agg(weight=('weight','sum'),bodies=('dst','size'),weighted_H=('weighted_H','sum')).reset_index()
    bs=b.groupby(['dst_type','src_type'],observed=True).b.sum().rename('B').reset_index()
    bs['B_log_B']=xlog2x(bs.B.to_numpy())
    bytype=bs.groupby('dst_type',observed=True).agg(sum_B_log_B=('B_log_B','sum'),source_types=('src_type','size'))
    types=types.merge(bytype,on='dst_type',validate='one_to_one')
    types['A']=np.log2(types.weight)-types.sum_B_log_B/types.weight
    types['C']=types.weighted_H/types.weight
    # Independently evaluate weighted KL; do not define R only by subtraction.
    v=b.merge(bodies[['dst','weight']],on='dst',validate='many_to_one').merge(bs[['dst_type','src_type','B']],on=['dst_type','src_type'],validate='many_to_one').merge(types[['dst_type','weight']].rename(columns={'weight':'W'}),on='dst_type',validate='many_to_one')
    v['kl_term']=v.b/v.W*np.log2((v.b/v.weight)/(v.B/v.W))
    rr=v.groupby('dst_type',observed=True).kl_term.sum()
    types=types.merge(rr.rename('R'),on='dst_type')
    types['identity_error']=types.A-types.C-types.R
    assert np.max(np.abs(types.identity_error))<1e-10
    assert min(types.A.min(),types.C.min(),types.R.min())>-1e-10
    types['R_fraction']=np.divide(types.R,types.A,out=np.full(len(types),np.nan),where=types.A.to_numpy()>1e-12)
    W=types.weight.sum();result={k:float(np.dot(types.weight,types[k])/W) for k in ['A','C','R']}
    result.update(weight=float(W),target_bodies=len(bodies),target_types=len(types),source_types=e.src_type.nunique(),identity_error_max=float(types.identity_error.abs().max()))
    return bodies.drop(columns=['sum_b_log_b','weighted_H']),types.drop(columns=['weighted_H','sum_B_log_B']),result

def matrix_metrics(matrix):
    """Rows = source types, columns = target bodies of one type."""
    m=sparse.csr_matrix(matrix,dtype=float)
    if np.any(m.data<0): raise ValueError('negative')
    m.eliminate_zeros();W=m.sum()
    if W==0: return dict(A=np.nan,C=np.nan,R=np.nan)
    row=np.asarray(m.sum(1)).ravel();col=np.asarray(m.sum(0)).ravel()
    A=np.log2(W)-xlog2x(row).sum()/W
    C=(xlog2x(col).sum()-xlog2x(m.data).sum())/W
    coo=m.tocoo();R=np.sum(coo.data/W*np.log2(coo.data*W/(row[coo.row]*col[coo.col])))
    return dict(A=float(A),C=float(C),R=float(R))
