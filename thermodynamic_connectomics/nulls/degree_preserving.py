"""Matched null models for directed weighted edge tables."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def target_preserving_source_shuffle(edges: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Keep target and weight fixed while shuffling source labels globally."""

    rng = np.random.default_rng(seed)
    out = edges.copy()
    out["src"] = rng.permutation(out["src"].to_numpy())
    if "src_type" in out.columns:
        out["src_type"] = rng.permutation(out["src_type"].to_numpy())
    return out


def directed_degree_preserving_rewire(
    edges: pd.DataFrame,
    n_swaps: int | None = None,
    seed: int = 0,
    keep_weights: bool = True,
    max_tries_multiplier: int = 20,
) -> pd.DataFrame:
    """Directed double-edge swap preserving unweighted in/out degree.

    Weights stay attached to edge slots by default; with `keep_weights=False`
    they are permuted after rewiring. This is intended for smoke-scale or
    sampled graphs when the raw graph has tens of millions of edges.
    """

    rng = np.random.default_rng(seed)
    df = edges[["src", "dst", "weight"]].drop_duplicates(["src", "dst"]).reset_index(drop=True)
    pairs = list(zip(df["src"].tolist(), df["dst"].tolist()))
    weights = df["weight"].to_numpy(copy=True)
    m = len(pairs)
    if m < 2:
        return df
    if n_swaps is None:
        n_swaps = min(10 * m, 1_000_000)
    max_tries = max_tries_multiplier * max(1, n_swaps)
    edge_set = set(pairs)
    swaps = 0
    tries = 0
    while swaps < n_swaps and tries < max_tries:
        tries += 1
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a, b = pairs[i]
        c, d = pairs[j]
        if a == c or b == d:
            continue
        n1 = (a, d)
        n2 = (c, b)
        if n1[0] == n1[1] or n2[0] == n2[1] or n1 in edge_set or n2 in edge_set:
            continue
        edge_set.remove(pairs[i])
        edge_set.remove(pairs[j])
        pairs[i] = n1
        pairs[j] = n2
        edge_set.add(n1)
        edge_set.add(n2)
        swaps += 1
    out = pd.DataFrame(pairs, columns=["src", "dst"])
    out["weight"] = weights if keep_weights else rng.permutation(weights)
    out.attrs["n_swaps_completed"] = swaps
    out.attrs["n_swap_tries"] = tries
    return out


def type_label_permutation(edges: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Keep graph fixed while permuting source and target cell-type labels."""

    rng = np.random.default_rng(seed)
    out = edges.copy()
    for id_col, type_col in [("src", "src_type"), ("dst", "dst_type")]:
        if type_col not in out.columns:
            continue
        labels = out[[id_col, type_col]].drop_duplicates(id_col).dropna()
        if labels.empty:
            continue
        shuffled = labels.copy()
        shuffled[type_col] = rng.permutation(shuffled[type_col].to_numpy())
        mapper = shuffled.set_index(id_col)[type_col]
        out[type_col] = out[id_col].map(mapper)
    return out


def weight_preserving_type_block_permutation(
    edges: pd.DataFrame,
    seed: int = 0,
    n_weight_bins: int = 5,
    n_degree_bins: int = 4,
) -> pd.DataFrame:
    """Shuffle endpoint type labels within weight/degree bins.

    This keeps the weighted graph fixed and permutes `src_type` and `dst_type`
    independently inside bins defined by edge weight and endpoint degree ranks.
    Therefore source-type and target-type marginal counts are preserved exactly
    within each non-empty bin, when the relevant type columns are present.
    """

    required = {"src", "dst", "weight", "src_type", "dst_type"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Cannot run type-block permutation; missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    out = edges.copy()
    work = out[["src", "dst", "weight", "src_type", "dst_type"]].copy()
    work["src_degree"] = work["src"].map(work.groupby("src")["dst"].size())
    work["dst_degree"] = work["dst"].map(work.groupby("dst")["src"].size())

    def _qbin(values: pd.Series, bins: int) -> pd.Series:
        values = pd.to_numeric(values, errors="coerce").fillna(0.0)
        if values.nunique(dropna=True) <= 1:
            return pd.Series(0, index=values.index)
        try:
            return pd.qcut(values.rank(method="first"), q=min(bins, values.nunique()), labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(0, index=values.index)

    weight_bin = _qbin(np.log1p(work["weight"]), n_weight_bins)
    src_degree_bin = _qbin(work["src_degree"], n_degree_bins)
    dst_degree_bin = _qbin(work["dst_degree"], n_degree_bins)
    bin_key = (
        weight_bin.astype("string").fillna("0")
        + "_"
        + src_degree_bin.astype("string").fillna("0")
        + "_"
        + dst_degree_bin.astype("string").fillna("0")
    )
    out["_perm_bin"] = bin_key.to_numpy()
    for _, idx in out.groupby("_perm_bin", sort=False).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        out.loc[idx, "src_type"] = rng.permutation(out.loc[idx, "src_type"].to_numpy())
        out.loc[idx, "dst_type"] = rng.permutation(out.loc[idx, "dst_type"].to_numpy())
    return out.drop(columns=["_perm_bin"])


def empirical_z_p(observed: float, null_values: Iterable[float]) -> dict[str, float]:
    """Return z-score plus empirical one/two-sided p-values."""

    vals = np.asarray(list(null_values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "observed": float(observed),
            "null_mean": np.nan,
            "null_std": np.nan,
            "z": np.nan,
            "empirical_p_high": np.nan,
            "empirical_p_low": np.nan,
            "empirical_p_two_sided": np.nan,
        }
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    z = float((observed - mean) / std) if std > 0 else np.nan
    p_high = float((1 + np.sum(vals >= observed)) / (1 + vals.size))
    p_low = float((1 + np.sum(vals <= observed)) / (1 + vals.size))
    return {
        "observed": float(observed),
        "null_mean": mean,
        "null_std": std,
        "z": z,
        "empirical_p_high": p_high,
        "empirical_p_low": p_low,
        "empirical_p_two_sided": float(min(1.0, 2.0 * min(p_high, p_low))),
    }
