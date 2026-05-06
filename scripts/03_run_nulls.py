#!/usr/bin/env python
"""Run matched null models for entropy-contraction metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.metrics.irreversibility import (
    compute_block_irreversibility,
    compute_node_irreversibility,
)
from thermodynamic_connectomics.nulls.degree_preserving import (
    directed_degree_preserving_rewire,
    empirical_z_p,
    target_preserving_source_shuffle,
    type_label_permutation,
    weight_preserving_type_block_permutation,
)


def _sample_edges(edges: pd.DataFrame, max_edges: int, seed: int) -> tuple[pd.DataFrame, bool]:
    if max_edges and max_edges > 0 and len(edges) > max_edges:
        return edges.sample(n=max_edges, random_state=seed).reset_index(drop=True), True
    return edges.reset_index(drop=True), False


def _global_metrics(edges: pd.DataFrame, group_col: str) -> dict[str, float]:
    node = compute_node_irreversibility(edges, min_degree=1)
    _, summary = compute_block_irreversibility(edges, group_col=group_col)
    out = {
        "global_node_delta_sum": float(node["delta_H_node_bits"].sum()) if len(node) else 0.0,
        "global_node_delta_mean": float(node["delta_H_node_bits"].mean()) if len(node) else 0.0,
        "block_history_ambiguity": float(summary.get("history_ambiguity_bits", np.nan)),
        "block_mutual_information": float(summary.get("mutual_information_bits", np.nan)),
    }
    for key in ["block_history_ambiguity", "block_mutual_information"]:
        if not np.isfinite(out[key]):
            raise ValueError(f"{key} is not finite; block summary status={summary.get('status')!r}")
    return out


def _restore_endpoint_metadata(null_edges: pd.DataFrame, reference_edges: pd.DataFrame) -> pd.DataFrame:
    """Attach endpoint metadata after structural rewiring using reference body IDs."""

    out = null_edges.copy()
    endpoint_cols = ["type", "instance", "roi", "nt"]
    for side in ["src", "dst"]:
        id_col = side
        ref_cols = [id_col] + [f"{side}_{col}" for col in endpoint_cols if f"{side}_{col}" in reference_edges.columns]
        if len(ref_cols) == 1:
            continue
        mapper = reference_edges[ref_cols].drop_duplicates(id_col).set_index(id_col)
        for col in mapper.columns:
            out[col] = out[id_col].map(mapper[col])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", required=True)
    parser.add_argument("--node-metrics", required=True)
    parser.add_argument("--out-dir", default="outputs/nulls")
    parser.add_argument("--n-nulls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-col", default="type")
    parser.add_argument(
        "--max-edges",
        type=int,
        default=500_000,
        help="Cap expensive nulls to this many edges; use 0 for the full graph.",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges_full = pd.read_parquet(args.edges)
    edges, sampled = _sample_edges(edges_full, args.max_edges, args.seed)
    if sampled:
        print(
            f"Null models are running on a reproducible sample of {len(edges):,} "
            f"of {len(edges_full):,} edges. Set --max-edges 0 for full graph nulls."
        )

    observed = _global_metrics(edges, args.group_col)
    observed_node = compute_node_irreversibility(edges, min_degree=1).set_index("bodyId")
    observed_nodes = observed_node.index.to_numpy()
    obs_entropy = observed_node["delta_H_node_bits"].to_numpy()
    null_sum = np.zeros_like(obs_entropy, dtype=float)
    null_sumsq = np.zeros_like(obs_entropy, dtype=float)
    null_high = np.zeros_like(obs_entropy, dtype=int)
    null_low = np.zeros_like(obs_entropy, dtype=int)
    node_index = pd.Index(observed_nodes)

    summary_rows = []
    failure_rows = []
    failure_counts: dict[str, int] = {}
    block_nulls: dict[tuple[str, str], list[float]] = {}
    target_success_count = 0
    for i in trange(args.n_nulls, desc="nulls"):
        seed = args.seed + i
        null_specs = [
            ("target_preserving_source_shuffle", lambda: target_preserving_source_shuffle(edges, seed=seed)),
            (
                "degree_preserving_directed_rewire",
                lambda: _restore_endpoint_metadata(
                    directed_degree_preserving_rewire(edges, seed=seed, n_swaps=min(20_000, max(1, len(edges) // 10))),
                    edges,
                ),
            ),
            ("type_label_permutation", lambda: type_label_permutation(edges, seed=seed)),
            ("weight_preserving_type_block_permutation", lambda: weight_preserving_type_block_permutation(edges, seed=seed)),
        ]
        for null_name, make_null in null_specs:
            try:
                null_edges = make_null()
                gm = _global_metrics(null_edges, args.group_col)
            except Exception as exc:  # noqa: BLE001 - logged and excluded from null distribution
                failure_counts[null_name] = failure_counts.get(null_name, 0) + 1
                failure_rows.append({"null_model": null_name, "iteration": i, "reason": repr(exc)})
                continue
            for metric, value in gm.items():
                summary_rows.append({"null_model": null_name, "iteration": i, "metric": metric, "value": value})
            if null_name == "target_preserving_source_shuffle":
                target_success_count += 1
                nm = compute_node_irreversibility(null_edges, min_degree=1).set_index("bodyId")
                vals = nm["delta_H_node_bits"].reindex(node_index).fillna(0.0).to_numpy()
                null_sum += vals
                null_sumsq += vals * vals
                null_high += vals >= obs_entropy
                null_low += vals <= obs_entropy
            bm, _ = compute_block_irreversibility(null_edges, group_col=args.group_col)
            if "dst_type" in bm.columns:
                for row in bm.itertuples(index=False):
                    value = float(row.history_ambiguity_bits)
                    if np.isfinite(value):
                        block_nulls.setdefault((null_name, str(row.dst_type)), []).append(value)

    null_long = pd.DataFrame(summary_rows)
    rows = []
    if not null_long.empty:
        for (null_model, metric), sub in null_long.groupby(["null_model", "metric"]):
            stats = empirical_z_p(observed.get(metric, np.nan), sub["value"])
            rows.append(
                {
                    "null_model": null_model,
                    "metric": metric,
                    **stats,
                    "sampled_edges": sampled,
                    "edge_count_used": len(edges),
                    "n_success": int(sub["iteration"].nunique()),
                    "failure_count": int(failure_counts.get(null_model, 0)),
                }
            )
    null_summary = pd.DataFrame(rows)
    null_summary.to_csv(out / "null_summary.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(out / "null_failures.csv", index=False)

    n = max(target_success_count, 1)
    mean = null_sum / n
    var = np.maximum(null_sumsq / n - mean * mean, 0.0)
    std = np.sqrt(var)
    z = np.full_like(obs_entropy, np.nan, dtype=float)
    np.divide(obs_entropy - mean, std, out=z, where=std > 0)
    node_z = pd.DataFrame(
        {
            "bodyId": observed_nodes,
            "observed_delta_H_node_bits": obs_entropy,
            "null_mean": mean,
            "null_std": std,
            "z": z,
            "empirical_p_high": (1 + null_high) / (1 + n) if target_success_count else np.nan,
            "empirical_p_low": (1 + null_low) / (1 + n) if target_success_count else np.nan,
        }
    )
    node_z["empirical_p_two_sided"] = np.minimum(1.0, 2.0 * np.minimum(node_z["empirical_p_high"], node_z["empirical_p_low"]))
    node_z.to_csv(out / "node_null_zscores.csv", index=False)

    observed_block, _ = compute_block_irreversibility(edges, group_col=args.group_col)
    block_rows = []
    if "dst_type" in observed_block.columns:
        for null_name in sorted({key[0] for key in block_nulls}):
            for row in observed_block.itertuples(index=False):
                stats = empirical_z_p(float(row.history_ambiguity_bits), block_nulls.get((null_name, str(row.dst_type)), []))
                block_rows.append({"null_model": null_name, "dst_type": row.dst_type, "metric": "history_ambiguity_bits", **stats})
    pd.DataFrame(block_rows).to_csv(out / "block_null_zscores.csv", index=False)
    print(f"Wrote null outputs to {out}")


if __name__ == "__main__":
    main()
