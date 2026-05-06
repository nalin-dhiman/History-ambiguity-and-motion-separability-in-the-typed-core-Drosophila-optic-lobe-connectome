"""History-ambiguity gates for local plasticity rule search."""
from __future__ import annotations

import numpy as np
import pandas as pd

from thermodynamic_connectomics.metrics.entropy import shannon_entropy

EPS = 1e-12


def compute_target_history_ambiguity(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    target_col: str = "dst",
    src_type_col: str = "src_type",
) -> pd.DataFrame:
    """Compute target-local source-type ambiguity and specificity gates.

    Each target receives a local ambiguity score:
    H_j = H(source type | target j), weighted by pre-normalization synapse
    weight. The inverse and normalized gates are finite, bounded proxies that
    can scale local update magnitude without using nonlocal labels.
    """

    if edges.empty:
        return pd.DataFrame(
            columns=[
                "node_id",
                "type",
                "H_in_src_type_bits",
                "S_inverse_ambiguity",
                "S_normalized",
                "weighted_in_degree",
                "weighted_out_degree",
                "local_type_MI_proxy",
                "target_update_budget",
            ]
        )
    work = edges.copy()
    work["weight"] = pd.to_numeric(work.get("weight", 1.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    work[src_type_col] = work[src_type_col].fillna("unknown").astype(str)
    type_map = nodes.drop_duplicates("bodyId").set_index("bodyId")["type"] if "bodyId" in nodes.columns and "type" in nodes.columns else pd.Series(dtype=object)
    out_degree = work.groupby("src")["weight"].sum()
    rows = []
    for target, group in work.groupby(target_col, sort=False):
        by_type = group.groupby(src_type_col)["weight"].sum()
        probs = by_type.to_numpy(dtype=float)
        total = float(probs.sum())
        if total > 0:
            probs = probs / total
        H = shannon_entropy(probs, base=2)
        n_types = int((by_type > 0).sum())
        h_norm = H / np.log2(n_types + EPS) if n_types > 1 else 0.0
        h_norm = float(np.clip(h_norm, 0.0, 1.0))
        weighted_in = float(group["weight"].sum())
        rows.append(
            {
                "node_id": int(target),
                "type": str(type_map.get(target, group["dst_type"].iloc[0] if "dst_type" in group.columns else "")),
                "H_in_src_type_bits": float(H),
                "S_inverse_ambiguity": float(1.0 / (1.0 + H)),
                "S_normalized": float(1.0 - h_norm),
                "weighted_in_degree": weighted_in,
                "weighted_out_degree": float(out_degree.get(target, 0.0)),
                "local_type_MI_proxy": float(max(np.log2(n_types + EPS) - H, 0.0)) if n_types > 1 else 0.0,
                "target_update_budget": float(max(weighted_in, EPS)),
            }
        )
    out = pd.DataFrame(rows)
    numeric = [
        "H_in_src_type_bits",
        "S_inverse_ambiguity",
        "S_normalized",
        "weighted_in_degree",
        "weighted_out_degree",
        "local_type_MI_proxy",
        "target_update_budget",
    ]
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["S_inverse_ambiguity"] = out["S_inverse_ambiguity"].clip(0.0, 1.0)
    out["S_normalized"] = out["S_normalized"].clip(0.0, 1.0)
    out["target_update_budget"] = out["target_update_budget"].clip(lower=EPS)
    return out.sort_values("node_id").reset_index(drop=True)


def gates_for_graph(graph, nodes: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Return target-ordered gate arrays for a LearningSubgraph."""

    table = compute_target_history_ambiguity(graph.edges, graph.node_table if nodes is None else nodes)
    indexed = table.set_index("node_id")
    target_ids = pd.Index(graph.target_nodes)
    def _col(name: str, default: float) -> np.ndarray:
        return indexed.reindex(target_ids)[name].fillna(default).to_numpy(dtype=np.float32)

    return {
        "target_ids": np.asarray(graph.target_nodes, dtype=np.int64),
        "S_inverse_ambiguity": _col("S_inverse_ambiguity", 1.0),
        "S_normalized": _col("S_normalized", 1.0),
        "local_type_MI_proxy": _col("local_type_MI_proxy", 0.0),
        "target_update_budget": np.maximum(_col("target_update_budget", 1.0), EPS),
        "H_in_src_type_bits": _col("H_in_src_type_bits", 0.0),
    }
