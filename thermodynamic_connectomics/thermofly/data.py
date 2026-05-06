"""Data loading and aggregation helpers for FlyWire-style connectome tables."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


COLUMN_ALIASES = {
    "pre": "pre_id",
    "post": "post_id",
    "pre_pt_root_id": "pre_id",
    "post_pt_root_id": "post_id",
    "pre_root_id": "pre_id",
    "post_root_id": "post_id",
    "count": "weight",
    "syn_count": "weight",
    "n_synapses": "weight",
    "Weight": "weight",
    "Count": "weight",
}


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV/TSV/Parquet/Feather edge or metadata table."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table format: {path}")


def standardize_edge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common FlyWire/Codex columns to pre_id, post_id, weight."""
    out = df.copy()
    rename = {c: COLUMN_ALIASES[c] for c in out.columns if c in COLUMN_ALIASES}
    out = out.rename(columns=rename)
    missing = {"pre_id", "post_id"} - set(out.columns)
    if missing:
        raise ValueError(f"Edge table is missing required columns: {sorted(missing)}")
    if "weight" not in out.columns:
        out["weight"] = 1.0
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out


def aggregate_edges(
    edges: pd.DataFrame,
    pre_col: str = "pre_id",
    post_col: str = "post_id",
    weight_col: str = "weight",
    extra_group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate parallel synapses into weighted directed edges."""
    df = standardize_edge_columns(edges) if {"pre_id", "post_id"}.issubset(edges.columns) is False else edges.copy()
    group_cols = [pre_col, post_col]
    if extra_group_cols:
        group_cols.extend(list(extra_group_cols))
    return (
        df.groupby(group_cols, dropna=False, as_index=False)[weight_col]
        .sum()
        .query(f"{weight_col} > 0")
        .reset_index(drop=True)
    )


def adjacency_from_edges(
    edges: pd.DataFrame,
    min_weight: float = 0.0,
    nodes: Optional[Sequence] = None,
    pre_col: str = "pre_id",
    post_col: str = "post_id",
    weight_col: str = "weight",
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Build a sparse adjacency matrix A[pre, post] from a weighted edge table."""
    df = standardize_edge_columns(edges) if not {pre_col, post_col, weight_col}.issubset(edges.columns) else edges.copy()
    df = df[df[weight_col] > min_weight].copy()
    if nodes is None:
        node_ids = pd.Index(pd.unique(pd.concat([df[pre_col], df[post_col]], ignore_index=True)))
    else:
        node_ids = pd.Index(nodes)
    node_to_idx = pd.Series(np.arange(len(node_ids)), index=node_ids)
    pre = node_to_idx.reindex(df[pre_col]).to_numpy()
    post = node_to_idx.reindex(df[post_col]).to_numpy()
    mask = ~pd.isna(pre) & ~pd.isna(post)
    row = pre[mask].astype(int)
    col = post[mask].astype(int)
    data = df.loc[mask, weight_col].astype(float).to_numpy()
    A = sparse.coo_matrix((data, (row, col)), shape=(len(node_ids), len(node_ids))).tocsr()
    return A, node_ids.to_numpy()


def row_stochastic(A: sparse.spmatrix, eps: float = 1e-12) -> sparse.csr_matrix:
    """Normalize rows of a sparse matrix to sum to one."""
    A = A.tocsr().astype(float)
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum, dtype=float)
    nz = row_sum > eps
    inv[nz] = 1.0 / row_sum[nz]
    return sparse.diags(inv) @ A


def coarse_grain_by_metadata(
    edges: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    pre_col: str = "pre_id",
    post_col: str = "post_id",
    node_col: str = "root_id",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Aggregate neuron-level edges into group-to-group edges.

    metadata must contain node_col and group_col. This is useful for grouping by
    cell type, neuropil, super-class, hemisphere, or user-defined modules.
    """
    e = standardize_edge_columns(edges)
    meta = metadata[[node_col, group_col]].drop_duplicates().copy()
    pre_meta = meta.rename(columns={node_col: pre_col, group_col: "pre_group"})
    post_meta = meta.rename(columns={node_col: post_col, group_col: "post_group"})
    merged = e.merge(pre_meta, on=pre_col, how="left").merge(post_meta, on=post_col, how="left")
    merged = merged.dropna(subset=["pre_group", "post_group"])
    return (
        merged.groupby(["pre_group", "post_group"], as_index=False)[weight_col]
        .sum()
        .sort_values(weight_col, ascending=False)
        .reset_index(drop=True)
    )
