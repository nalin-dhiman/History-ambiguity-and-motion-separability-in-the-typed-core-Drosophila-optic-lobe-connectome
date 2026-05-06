"""Null-model generators for connectome edge lists."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def directed_degree_preserving_rewire(
    edges: pd.DataFrame,
    pre_col: str = "pre_id",
    post_col: str = "post_id",
    weight_col: str = "weight",
    n_swaps: Optional[int] = None,
    seed: int = 0,
    max_tries_multiplier: int = 20,
    keep_weights: bool = True,
) -> pd.DataFrame:
    """Degree-preserving directed edge swap null.

    This preserves unweighted in-degree and out-degree exactly for the edge list.
    It is intended for aggregated edges. For massive FlyWire graphs, run this on
    a sampled subgraph or cell-type/neuropil-level graph, or use a compiled graph
    library.
    """
    rng = np.random.default_rng(seed)
    df = edges[[pre_col, post_col, weight_col]].copy().reset_index(drop=True)
    pairs = list(zip(df[pre_col].tolist(), df[post_col].tolist()))
    weights = df[weight_col].to_numpy(copy=True)
    edge_set = set(pairs)
    m = len(pairs)
    if n_swaps is None:
        n_swaps = 10 * m
    tries = 0
    swaps = 0
    max_tries = max_tries_multiplier * max(n_swaps, 1)
    while swaps < n_swaps and tries < max_tries:
        tries += 1
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b = pairs[i]
        c, d = pairs[j]
        if a == c or b == d:
            continue
        new1 = (a, d)
        new2 = (c, b)
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.remove(pairs[i])
        edge_set.remove(pairs[j])
        pairs[i] = new1
        pairs[j] = new2
        edge_set.add(new1)
        edge_set.add(new2)
        swaps += 1
    out = pd.DataFrame(pairs, columns=[pre_col, post_col])
    if keep_weights:
        out[weight_col] = weights
    else:
        out[weight_col] = rng.permutation(weights)
    out.attrs["n_swaps_completed"] = swaps
    out.attrs["n_swap_tries"] = tries
    return out


def strength_preserving_weight_shuffle(
    edges: pd.DataFrame,
    weight_col: str = "weight",
    group_col: Optional[str] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Shuffle edge weights, optionally within a group such as neuropil."""
    rng = np.random.default_rng(seed)
    out = edges.copy()
    if group_col is None:
        out[weight_col] = rng.permutation(out[weight_col].to_numpy())
    else:
        for _, idx in out.groupby(group_col).groups.items():
            vals = out.loc[idx, weight_col].to_numpy()
            out.loc[idx, weight_col] = rng.permutation(vals)
    return out
