"""Pathway subgraph construction for learning-efficiency experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from thermodynamic_connectomics.metrics.irreversibility import compute_block_irreversibility

LAMINA_RE = r"^L[1-5](?:$|[^0-9])"
MEDULLA_RE = r"^(?:Mi|Tm|TmY)"
MOTION_RE = r"^(?:T4|T5)"
OUTPUT_RE = r"^(?:LC|LPLC|LT|DN)"


@dataclass
class LearningSubgraph:
    """A directed weighted pathway graph represented as source-target edges."""

    name: str
    edges: pd.DataFrame
    node_table: pd.DataFrame
    source_nodes: list[int]
    target_nodes: list[int]
    source_types: list[str]
    target_types: list[str]
    W: sparse.csr_matrix
    history_ambiguity_bits: float
    mutual_information_bits: float
    effective_preimage_size: float

    @property
    def n_nodes(self) -> int:
        return int(len(set(self.source_nodes) | set(self.target_nodes)))

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))


def _contains(series: pd.Series, pattern: str) -> pd.Series:
    return series.astype(str).str.contains(pattern, regex=True, na=False)


def filter_pathway_edges(edges: pd.DataFrame, pathway: str) -> pd.DataFrame:
    """Return pathway-specific edge subset using typed-core edge annotations."""

    src = edges["src_type"]
    dst = edges["dst_type"]
    roi = edges["edge_major_roi"].astype(str) if "edge_major_roi" in edges.columns else pd.Series("", index=edges.index)
    lamina_src = _contains(src, LAMINA_RE)
    medulla_dst = _contains(dst, MEDULLA_RE) | _contains(dst, MOTION_RE)
    medulla_src = _contains(src, MEDULLA_RE)
    motion_src = _contains(src, MOTION_RE)
    output_dst = _contains(dst, OUTPUT_RE)

    if pathway == "lamina_to_medulla":
        mask = lamina_src & medulla_dst & roi.eq("ME(R)")
    elif pathway == "medulla_to_lobula_plate":
        mask = (medulla_src | motion_src) & roi.eq("LOP(R)")
    elif pathway == "lobula_lobula_plate_to_projection_output":
        mask = output_dst & roi.isin(["LO(R)", "LOP(R)"])
    else:
        raise ValueError(f"Unknown pathway: {pathway}")
    return edges[mask].copy()


def _largest_weak_component(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges
    graph = nx.from_pandas_edgelist(edges, source="src", target="dst", create_using=nx.DiGraph)
    components = list(nx.weakly_connected_components(graph))
    if not components:
        return edges
    keep = max(components, key=len)
    return edges[edges["src"].isin(keep) & edges["dst"].isin(keep)].copy()


def _cap_edges_nodes(edges: pd.DataFrame, max_nodes: int, max_edges: int) -> pd.DataFrame:
    sorted_edges = edges.sort_values("weight", ascending=False).head(max_edges * 5 if max_edges > 0 else len(edges)).copy()
    if max_nodes <= 0:
        return sorted_edges.head(max_edges).reset_index(drop=True)
    kept_rows = []
    kept_nodes: set[int] = set()
    for row in sorted_edges.itertuples(index=True):
        src = int(row.src)
        dst = int(row.dst)
        new_nodes = ({src, dst} - kept_nodes)
        if len(kept_nodes) + len(new_nodes) > max_nodes:
            continue
        kept_rows.append(row.Index)
        kept_nodes.update([src, dst])
        if len(kept_rows) >= max_edges:
            break
    return sorted_edges.loc[kept_rows].reset_index(drop=True)


def _sample_matched_edges(edges: pd.DataFrame, matched_nodes: int, matched_edges: int, seed: int) -> pd.DataFrame:
    """Sample a pathway subgraph under a shared node/edge budget.

    The sampler attempts to return exactly ``matched_edges`` among at most
    ``matched_nodes`` edge-active nodes. Exact matching is only possible when
    the pathway has enough typed-core support; downstream metadata records when
    the requested budget could not be reached.
    """

    if edges.empty:
        return edges.copy()
    if matched_nodes <= 0:
        return edges.sort_values("weight", ascending=False).head(matched_edges).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    candidate = edges.copy().reset_index(drop=True)
    pool_size = min(len(candidate), max(matched_edges * 30, matched_nodes * 80, 2000))
    if len(candidate) > pool_size:
        top = candidate.sort_values("weight", ascending=False).head(pool_size * 2)
        weights = top["weight"].to_numpy(dtype=float)
        prob = weights / weights.sum() if weights.sum() > 0 else None
        candidate = top.loc[rng.choice(top.index.to_numpy(), size=pool_size, replace=False, p=prob)].reset_index(drop=True)
    all_nodes = pd.Index(pd.unique(pd.concat([candidate["src"], candidate["dst"]], ignore_index=True))).astype("int64")
    if len(all_nodes) <= matched_nodes or len(candidate) <= matched_edges:
        return _cap_edges_nodes(candidate, max_nodes=matched_nodes, max_edges=matched_edges)

    best = pd.DataFrame()
    best_score = (-1, -1)
    for _ in range(12):
        target_stats = candidate.groupby("dst").agg(degree=("src", "size"), strength=("weight", "sum")).reset_index()
        target_stats["score"] = target_stats["degree"] + 0.01 * target_stats["strength"] + rng.random(len(target_stats))
        min_targets = max(1, matched_nodes // 8)
        max_targets = max(min_targets + 1, matched_nodes // 2)
        target_budget = int(rng.integers(min_targets, max_targets + 1))
        chosen_targets = set(target_stats.sort_values("score", ascending=False)["dst"].head(target_budget).astype("int64"))
        active: set[int] = set(int(x) for x in chosen_targets)
        incoming = candidate[candidate["dst"].isin(active)]
        source_stats = incoming.groupby("src").agg(degree=("dst", "size"), strength=("weight", "sum")).reset_index()
        source_stats["score"] = source_stats["degree"] + 0.01 * source_stats["strength"] + rng.random(len(source_stats))
        for src in source_stats.sort_values("score", ascending=False)["src"].astype("int64"):
            if int(src) not in active and len(active) >= matched_nodes:
                break
            active.add(int(src))
        induced = candidate[candidate["src"].isin(active) & candidate["dst"].isin(active)].copy()
        if len(induced) > matched_edges:
            weights = induced["weight"].to_numpy(dtype=float)
            prob = weights / weights.sum() if weights.sum() > 0 else None
            sampled = induced.loc[rng.choice(induced.index.to_numpy(), size=matched_edges, replace=False, p=prob)].copy()
        else:
            sampled = induced
        active_nodes = len(set(sampled["src"]) | set(sampled["dst"]))
        score = (len(sampled), active_nodes)
        if score > best_score:
            best = sampled
            best_score = score
        if len(sampled) >= matched_edges and active_nodes >= matched_nodes:
            break

    if best.empty:
        best = _cap_edges_nodes(candidate, max_nodes=matched_nodes, max_edges=matched_edges)
    return best.sort_values("weight", ascending=False).head(matched_edges).reset_index(drop=True)


def _normalize_per_target(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    denom = out.groupby("dst")["weight"].transform("sum").replace(0, np.nan)
    out["norm_weight"] = (out["weight"] / denom).fillna(0.0)
    return out


def _matrix_from_edges(edges: pd.DataFrame, source_nodes: list[int], target_nodes: list[int]) -> sparse.csr_matrix:
    src_idx = pd.Series(np.arange(len(source_nodes)), index=source_nodes)
    dst_idx = pd.Series(np.arange(len(target_nodes)), index=target_nodes)
    row = edges["src"].map(src_idx).to_numpy()
    col = edges["dst"].map(dst_idx).to_numpy()
    data = edges["norm_weight"].to_numpy(dtype=np.float32)
    return sparse.coo_matrix((data, (row, col)), shape=(len(source_nodes), len(target_nodes))).tocsr()


def learning_subgraph_from_edges(
    pathway: str,
    sub_edges: pd.DataFrame,
    nodes: pd.DataFrame,
    max_nodes: int = 2000,
    max_edges: int = 50000,
    keep_largest_component: bool = True,
) -> LearningSubgraph:
    """Build a capped, target-normalized learning subgraph from selected edges."""

    sub = sub_edges.copy()
    sub = _cap_edges_nodes(sub, max_nodes=max_nodes, max_edges=max_edges)
    if keep_largest_component:
        sub = _largest_weak_component(sub)
    sub = _normalize_per_target(sub)
    source_nodes = pd.Index(pd.unique(sub["src"])).astype("int64").tolist()
    target_nodes = pd.Index(pd.unique(sub["dst"])).astype("int64").tolist()
    node_meta = nodes.drop_duplicates("bodyId").set_index("bodyId")
    source_types = [str(node_meta["type"].get(n, "")) for n in source_nodes]
    target_types = [str(node_meta["type"].get(n, "")) for n in target_nodes]
    W = _matrix_from_edges(sub, source_nodes, target_nodes)
    _, summary = compute_block_irreversibility(sub, group_col="type")
    return LearningSubgraph(
        name=pathway,
        edges=sub.reset_index(drop=True),
        node_table=nodes,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_types=source_types,
        target_types=target_types,
        W=W,
        history_ambiguity_bits=float(summary.get("history_ambiguity_bits", np.nan)),
        mutual_information_bits=float(summary.get("mutual_information_bits", np.nan)),
        effective_preimage_size=float(summary.get("effective_preimage_size", np.nan)),
    )


def build_learning_subgraph(pathway: str, edges: pd.DataFrame, nodes: pd.DataFrame, max_nodes: int = 2000, max_edges: int = 50000) -> LearningSubgraph:
    """Build a capped, target-normalized learning subgraph for a named pathway."""

    return learning_subgraph_from_edges(
        pathway,
        filter_pathway_edges(edges, pathway),
        nodes,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


def build_matched_learning_subgraph(
    pathway: str,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    matched_nodes: int = 500,
    matched_edges: int = 3000,
    sample_id: int = 0,
    seed: int = 42,
) -> tuple[LearningSubgraph, dict]:
    """Build one matched-budget pathway sample for controlled comparisons."""

    pathway_edges = filter_pathway_edges(edges, pathway)
    pathway_edges = _largest_weak_component(pathway_edges)
    candidate_nodes = len(set(pathway_edges["src"]) | set(pathway_edges["dst"])) if not pathway_edges.empty else 0
    candidate_edges = len(pathway_edges)
    sampled = _sample_matched_edges(pathway_edges, matched_nodes=matched_nodes, matched_edges=matched_edges, seed=seed + sample_id)
    graph = learning_subgraph_from_edges(
        pathway,
        sampled,
        nodes,
        max_nodes=matched_nodes,
        max_edges=matched_edges,
        keep_largest_component=False,
    )
    metadata = {
        "pathway": pathway,
        "sample_id": sample_id,
        "requested_matched_nodes": matched_nodes,
        "requested_matched_edges": matched_edges,
        "candidate_nodes": candidate_nodes,
        "candidate_edges": candidate_edges,
        "matched_nodes_reached": graph.n_nodes >= min(matched_nodes, candidate_nodes),
        "matched_edges_reached": graph.n_edges >= min(matched_edges, candidate_edges),
        "exploratory": bool(
            candidate_nodes < matched_nodes
            or candidate_edges < matched_edges
            or graph.n_nodes < matched_nodes
            or graph.n_edges < matched_edges
        ),
    }
    return graph, metadata


def load_typed_core(edges_path: str | Path, nodes_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = pd.read_parquet(edges_path)
    nodes = pd.read_parquet(nodes_path)
    return edges, nodes
