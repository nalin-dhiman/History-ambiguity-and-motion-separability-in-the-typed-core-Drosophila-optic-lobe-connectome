"""Graph-level and node-level irreversibility proxies."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .entropy import channel_metrics, shannon_bits
from .landauer import landauer_joules


def transition_metrics_from_adjacency(
    A: sparse.spmatrix,
    p_source: Optional[np.ndarray] = None,
    temperature_kelvin: float = 293.15,
) -> dict:
    """Compute channel metrics from row-normalized adjacency P(post|pre)."""
    A = A.tocsr().astype(float)
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    keep = row_sum > 0
    if keep.sum() == 0:
        raise ValueError("Adjacency has no positive outgoing rows")
    P = A[keep][:, :].toarray()
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    if p_source is not None:
        p = np.asarray(p_source)[keep]
        p = p / np.maximum(p.sum(), 1e-12)
    else:
        # Default source distribution weights active source neurons uniformly.
        p = np.ones(P.shape[0]) / P.shape[0]
    metrics = channel_metrics(P, p).as_dict()
    metrics["landauer_joules"] = landauer_joules(metrics["landauer_relevant_bits"], temperature_kelvin)
    metrics["n_source_states"] = int(P.shape[0])
    metrics["n_target_states"] = int(P.shape[1])
    return metrics


def local_postsynaptic_history_loss(
    edges: pd.DataFrame,
    pre_col: str = "pre_id",
    post_col: str = "post_id",
    weight_col: str = "weight",
    temperature_kelvin: float = 293.15,
) -> pd.DataFrame:
    """Compute local incoming-history ambiguity for each postsynaptic neuron.

    For each postsynaptic neuron j, normalize incoming synaptic weights to obtain
    P(pre | post=j). The entropy H(pre|j) is a local many-to-one ambiguity index.
    We weight it by q(j), the fraction of total incoming weight that lands on j,
    to obtain a graph-level contribution.

    This is a structural proxy. It is not a literal heat measurement.
    """
    required = {pre_col, post_col, weight_col}
    if not required.issubset(edges.columns):
        missing = required - set(edges.columns)
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = edges[[pre_col, post_col, weight_col]].copy()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    df = df[df[weight_col] > 0]
    total_weight = df[weight_col].sum()
    rows = []
    for post, g in df.groupby(post_col):
        w = g[weight_col].to_numpy(dtype=float)
        p = w / np.maximum(w.sum(), 1e-12)
        h = float(shannon_bits(p))
        q = float(w.sum() / max(total_weight, 1e-12))
        rows.append(
            {
                post_col: post,
                "fan_in": int(g[pre_col].nunique()),
                "weighted_in": float(w.sum()),
                "q_post": q,
                "history_loss_bits_local": h,
                "history_loss_bits_weighted": q * h,
                "landauer_joules_weighted": landauer_joules(q * h, temperature_kelvin),
            }
        )
    return pd.DataFrame(rows).sort_values("history_loss_bits_weighted", ascending=False).reset_index(drop=True)


def block_channel_from_edges(
    block_edges: pd.DataFrame,
    pre_group_col: str = "pre_group",
    post_group_col: str = "post_group",
    weight_col: str = "weight",
) -> tuple[np.ndarray, list, list]:
    """Create a dense P(post_group | pre_group) channel from block edges."""
    pre_groups = sorted(block_edges[pre_group_col].dropna().unique().tolist())
    post_groups = sorted(block_edges[post_group_col].dropna().unique().tolist())
    pre_index = {g: i for i, g in enumerate(pre_groups)}
    post_index = {g: i for i, g in enumerate(post_groups)}
    P = np.zeros((len(pre_groups), len(post_groups)), dtype=float)
    for row in block_edges.itertuples(index=False):
        i = pre_index[getattr(row, pre_group_col)]
        j = post_index[getattr(row, post_group_col)]
        P[i, j] += float(getattr(row, weight_col))
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P, pre_groups, post_groups
