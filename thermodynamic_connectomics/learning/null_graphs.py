"""Null graph controls for learning-efficiency experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from thermodynamic_connectomics.learning.subgraphs import LearningSubgraph, learning_subgraph_from_edges


def degree_preserving_edge_null(edges: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Approximate degree-preserving null by permuting targets and target labels.

    This preserves source out-edge counts, target in-edge counts, and edge
    weights exactly as multisets. It may create duplicate source-target pairs,
    so it is documented as approximate rather than an exact double-edge swap.
    """

    rng = np.random.default_rng(seed)
    out = edges.copy()
    dst_cols = [c for c in out.columns if c == "dst" or c.startswith("dst_")]
    perm = rng.permutation(len(out))
    for col in dst_cols:
        out[col] = out[col].to_numpy()[perm]
    return out


def type_label_null_nodes(nodes: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute type labels while preserving body IDs and graph topology."""

    rng = np.random.default_rng(seed)
    out = nodes.copy()
    known = out["type"].notna() & out["type"].astype(str).str.len().gt(0)
    out.loc[known, "type"] = rng.permutation(out.loc[known, "type"].to_numpy())
    return out


def weight_shuffled_real_topology(edges: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Keep directed topology fixed and shuffle synapse weights across edges."""

    rng = np.random.default_rng(seed)
    out = edges.copy()
    out["weight"] = rng.permutation(out["weight"].to_numpy(dtype=float))
    return out


def weight_uniform_real_topology(edges: pd.DataFrame) -> pd.DataFrame:
    """Keep directed topology fixed and set all pre-normalization weights to one."""

    out = edges.copy()
    out["weight"] = 1.0
    return out


def apply_condition(subgraph: LearningSubgraph, condition: str, seed: int = 0) -> LearningSubgraph:
    """Return graph condition: real, approximate degree null, or type-label null."""

    if condition == "real":
        return subgraph
    if condition == "degree_preserving_null":
        edges = degree_preserving_edge_null(subgraph.edges, seed=seed)
        return learning_subgraph_from_edges(subgraph.name, edges, subgraph.node_table, max_nodes=subgraph.n_nodes, max_edges=subgraph.n_edges)
    if condition == "weight_shuffled_real_topology":
        edges = weight_shuffled_real_topology(subgraph.edges, seed=seed)
        return learning_subgraph_from_edges(subgraph.name, edges, subgraph.node_table, max_nodes=subgraph.n_nodes, max_edges=subgraph.n_edges)
    if condition in {"weight_uniform_real_topology", "uniform_weight_real_topology"}:
        edges = weight_uniform_real_topology(subgraph.edges)
        return learning_subgraph_from_edges(subgraph.name, edges, subgraph.node_table, max_nodes=subgraph.n_nodes, max_edges=subgraph.n_edges)
    if condition == "type_label_null":
        nodes = type_label_null_nodes(subgraph.node_table, seed=seed)
        edges = subgraph.edges.copy()
        mapper = nodes.drop_duplicates("bodyId").set_index("bodyId")["type"]
        edges["src_type"] = edges["src"].map(mapper)
        edges["dst_type"] = edges["dst"].map(mapper)
        return learning_subgraph_from_edges(subgraph.name, edges, nodes, max_nodes=subgraph.n_nodes, max_edges=subgraph.n_edges)
    raise ValueError(f"Unknown condition: {condition}")
