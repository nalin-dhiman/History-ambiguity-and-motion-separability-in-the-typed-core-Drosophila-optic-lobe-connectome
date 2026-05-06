#!/usr/bin/env python
"""Compare full graph, typed core, and typed-core min-weight variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.metrics.irreversibility import (
    compute_block_irreversibility,
    compute_node_irreversibility,
    write_metric_outputs,
)


def _load_summary(metrics_dir: Path) -> dict:
    path = metrics_dir / "global_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _summarize_existing(label: str, processed_dir: Path, metrics_dir: Path, min_weight: float) -> dict:
    summary = _load_summary(metrics_dir)
    block = summary.get("block_channel", {})
    return {
        "analysis": label,
        "processed_dir": str(processed_dir),
        "metrics_dir": str(metrics_dir),
        "min_weight": min_weight,
        "n_body_id_rows": summary.get("n_nodes_processed"),
        "n_typed_body_id_rows": summary.get("n_typed_segments", summary.get("n_typed_neurons")),
        "n_edges": summary.get("n_edges_processed"),
        "total_synapse_weight": summary.get("total_synapse_weight"),
        "node_metric_rows": summary.get("node_metric_rows"),
        "sum_delta_H_node_bits_proxy": summary.get("sum_delta_H_node_bits"),
        "mean_delta_H_node_bits_proxy": summary.get("mean_delta_H_node_bits"),
        "n_src_types": block.get("n_src_types"),
        "n_dst_types": block.get("n_dst_types"),
        "H_src_type_bits": block.get("H_src_type_bits"),
        "H_dst_type_bits": block.get("H_dst_type_bits"),
        "cell_type_history_ambiguity_bits": block.get("history_ambiguity_bits"),
        "cell_type_mutual_information_bits": block.get("mutual_information_bits"),
        "cell_type_effective_preimage_size": block.get("effective_preimage_size"),
        "status": block.get("status"),
        "note": "existing metrics",
    }


def _compute_variant(
    label: str,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    metrics_dir: Path,
    min_weight: float,
    min_degree: int,
    group_col: str,
) -> dict:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    use_edges = edges[pd.to_numeric(edges["weight"], errors="coerce").fillna(0) >= min_weight].copy()
    node_metrics = compute_node_irreversibility(use_edges, min_degree=min_degree)
    block_metrics, block_summary = compute_block_irreversibility(use_edges, group_col=group_col)
    typed = nodes["type"].notna() & (nodes["type"].astype(str).str.len() > 0)
    global_summary = {
        "n_nodes_processed": int(len(nodes)),
        "n_edges_processed": int(len(use_edges)),
        "n_typed_segments": int(typed.sum()),
        "total_synapse_weight": float(use_edges["weight"].sum()),
        "node_metric_rows": int(len(node_metrics)),
        "sum_delta_H_node_bits": float(node_metrics["delta_H_node_bits"].sum()) if len(node_metrics) else 0.0,
        "mean_delta_H_node_bits": float(node_metrics["delta_H_node_bits"].mean()) if len(node_metrics) else 0.0,
        "median_delta_H_node_bits": float(node_metrics["delta_H_node_bits"].median()) if len(node_metrics) else 0.0,
        "block_channel": block_summary,
        "scientific_guardrail": (
            "Main reported quantity is cell-type history ambiguity. Node-level "
            "Landauer quantities remain lower-bound proxy scales only."
        ),
    }
    write_metric_outputs(node_metrics, block_metrics, global_summary, metrics_dir)
    return _summarize_existing(label, Path("outputs/processed_typed_core"), metrics_dir, min_weight) | {
        "note": "computed from typed core"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-processed-dir", default="outputs/processed")
    parser.add_argument("--full-metrics-dir", default="outputs/metrics")
    parser.add_argument("--typed-processed-dir", default="outputs/processed_typed_core")
    parser.add_argument("--typed-metrics-dir", default="outputs/metrics_typed_core")
    parser.add_argument("--out", default="outputs/tables/full_vs_typed_core_comparison.csv")
    parser.add_argument("--group-col", default="type")
    parser.add_argument("--min-degree", type=int, default=3)
    parser.add_argument("--min-weights", type=float, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()

    typed_processed = Path(args.typed_processed_dir)
    typed_metrics = Path(args.typed_metrics_dir)
    edges = pd.read_parquet(typed_processed / "edges.parquet")
    nodes = pd.read_parquet(typed_processed / "nodes.parquet")

    rows = [
        _summarize_existing(
            "full_graph_min_weight_1",
            Path(args.full_processed_dir),
            Path(args.full_metrics_dir),
            1,
        )
    ]
    for min_weight in args.min_weights:
        metrics_dir = typed_metrics if min_weight == 1 else Path(f"{args.typed_metrics_dir}_minw_{int(min_weight)}")
        rows.append(
            _compute_variant(
                f"typed_core_min_weight_{int(min_weight)}",
                edges,
                nodes,
                metrics_dir,
                min_weight,
                args.min_degree,
                args.group_col,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote comparison table to {out}")


if __name__ == "__main__":
    main()
