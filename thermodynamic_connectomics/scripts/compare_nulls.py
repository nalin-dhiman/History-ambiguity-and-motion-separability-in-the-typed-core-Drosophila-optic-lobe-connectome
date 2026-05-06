#!/usr/bin/env python
"""Compare real connectome metrics with directed-degree-preserving nulls."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thermofly.data import load_table, standardize_edge_columns, aggregate_edges, adjacency_from_edges
from thermofly.graph_metrics import local_postsynaptic_history_loss, transition_metrics_from_adjacency
from thermofly.nulls import directed_degree_preserving_rewire


def summarize(edges, nodes=None):
    local = local_postsynaptic_history_loss(edges)
    A, node_ids = adjacency_from_edges(edges, nodes=nodes)
    m = transition_metrics_from_adjacency(A)
    return {
        "mean_local_history_loss_bits": local["history_loss_bits_local"].mean(),
        "weighted_history_loss_bits": local["history_loss_bits_weighted"].sum(),
        "channel_landauer_bits": m["landauer_relevant_bits"],
        "channel_history_loss_bits": m["history_loss_bits"],
    }, node_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edges", required=True)
    p.add_argument("--out", type=Path, default=Path("null_outputs"))
    p.add_argument("--n-nulls", type=int, default=25)
    p.add_argument("--swaps-per-edge", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    edges = aggregate_edges(standardize_edge_columns(load_table(args.edges)))
    real, node_ids = summarize(edges)
    rows = [{"condition": "real", "sample": 0, **real}]
    for i in range(args.n_nulls):
        rw = directed_degree_preserving_rewire(
            edges,
            n_swaps=args.swaps_per_edge * len(edges),
            seed=args.seed + i,
        )
        s, _ = summarize(rw, nodes=node_ids)
        rows.append({"condition": "degree_preserving_null", "sample": i, **s})
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "null_comparison.csv", index=False)
    print(out)
    print("Saved", args.out / "null_comparison.csv")


if __name__ == "__main__":
    main()
