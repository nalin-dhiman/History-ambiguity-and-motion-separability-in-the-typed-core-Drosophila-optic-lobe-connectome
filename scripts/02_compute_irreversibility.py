#!/usr/bin/env python
"""Compute node and block entropy-contraction metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.metrics.irreversibility import (
    compute_block_irreversibility,
    compute_node_irreversibility,
    write_metric_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--out-dir", default="outputs/metrics")
    parser.add_argument("--group-col", default="type")
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--min-degree", type=int, default=3)
    args = parser.parse_args()

    edges = pd.read_parquet(args.edges)
    edges = edges[pd.to_numeric(edges["weight"], errors="coerce").fillna(0) >= args.min_weight].copy()
    node_metrics = compute_node_irreversibility(edges, min_degree=args.min_degree)
    block_metrics, block_summary = compute_block_irreversibility(edges, group_col=args.group_col)
    nodes = pd.read_parquet(args.nodes)
    typed = nodes["type"].notna() & (nodes["type"].astype(str).str.len() > 0)
    global_summary = {
        "n_nodes_processed": int(len(nodes)),
        "n_edges_processed": int(len(edges)),
        "n_typed_segments": int(typed.sum()),
        "total_synapse_weight": float(edges["weight"].sum()),
        "node_metric_rows": int(len(node_metrics)),
        "sum_delta_H_node_bits": float(node_metrics["delta_H_node_bits"].sum()) if len(node_metrics) else 0.0,
        "mean_delta_H_node_bits": float(node_metrics["delta_H_node_bits"].mean()) if len(node_metrics) else 0.0,
        "median_delta_H_node_bits": float(node_metrics["delta_H_node_bits"].median()) if len(node_metrics) else 0.0,
        "block_channel": block_summary,
        "scientific_guardrail": (
            "All Landauer quantities are lower-bound proxies under explicit anatomical "
            "coarse-graining assumptions, not measured biological heat."
        ),
    }
    write_metric_outputs(node_metrics, block_metrics, global_summary, args.out_dir)
    print(f"Wrote node metrics: {len(node_metrics):,} rows")
    print(f"Wrote block metrics: {len(block_metrics):,} rows")


if __name__ == "__main__":
    main()
