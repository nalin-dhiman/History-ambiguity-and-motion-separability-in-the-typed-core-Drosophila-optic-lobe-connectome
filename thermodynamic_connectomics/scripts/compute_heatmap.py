#!/usr/bin/env python
"""Compute a structural Landauer-style heatmap from a FlyWire-style edge table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thermofly.data import load_table, standardize_edge_columns, aggregate_edges
from thermofly.graph_metrics import local_postsynaptic_history_loss
from thermofly.visualize import scatter_heatmap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edges", required=True, help="CSV/TSV/Parquet/Feather with pre/post/weight columns")
    p.add_argument("--metadata", help="Optional node metadata with root_id,x,y columns")
    p.add_argument("--out", type=Path, default=Path("heatmap_outputs"))
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=293.15)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    edges = standardize_edge_columns(load_table(args.edges))
    edges = aggregate_edges(edges)
    edges = edges[edges["weight"] > args.min_weight]
    metrics = local_postsynaptic_history_loss(edges, temperature_kelvin=args.temperature)
    metrics.to_csv(args.out / "local_history_loss.csv", index=False)
    if args.metadata:
        meta = load_table(args.metadata)
        scatter_heatmap(metrics, meta, args.out / "local_history_heatmap.png")
    print(metrics.head(20).to_string(index=False))
    print("Saved", args.out / "local_history_loss.csv")


if __name__ == "__main__":
    main()
