#!/usr/bin/env python
"""Write a short markdown report for the optic-lobe pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _md_table(df: pd.DataFrame, n: int = 10) -> str:
    if df.empty:
        return "_No rows available._"
    return df.head(n).to_markdown(index=False)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="outputs/processed")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--nulls-dir", default="outputs/nulls")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--comparison-table", default="outputs/tables/full_vs_typed_core_comparison.csv")
    parser.add_argument("--roi-table", default="outputs/tables/roi_comparison_table.csv")
    parser.add_argument("--pathway-table", default="outputs/tables/pathway_comparison_table.csv")
    parser.add_argument("--roi-nulls", default="outputs/roi_pathway/roi_pathway_null_summary.csv")
    parser.add_argument(
        "--known-visual-validation",
        default="outputs/known_visual_pathway_validation/known_visual_pathway_validation.csv",
    )
    parser.add_argument(
        "--known-visual-interpretation",
        default="outputs/tables/known_visual_pathway_interpretation.csv",
    )
    parser.add_argument("--out", default="outputs/report.md")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    metrics = Path(args.metrics_dir)
    nulls = Path(args.nulls_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(processed / "nodes.parquet")
    edges = pd.read_parquet(processed / "edges.parquet", columns=["src", "dst", "weight", "src_type", "dst_type"])
    type_summary = pd.read_csv(processed / "type_summary.csv")
    node = pd.read_csv(metrics / "node_irreversibility.csv", low_memory=False)
    block = pd.read_csv(metrics / "block_irreversibility.csv")
    global_summary = json.loads((metrics / "global_summary.json").read_text(encoding="utf-8"))
    null_summary_path = nulls / "null_summary.csv"
    null_summary = _read_csv_or_empty(null_summary_path)
    null_failures_path = nulls / "null_failures.csv"
    null_failures = _read_csv_or_empty(null_failures_path)
    comparison_path = Path(args.comparison_table)
    comparison = _read_csv_or_empty(comparison_path)
    roi_table = _read_csv_or_empty(Path(args.roi_table))
    pathway_table = _read_csv_or_empty(Path(args.pathway_table))
    roi_nulls = _read_csv_or_empty(Path(args.roi_nulls))
    known_validation = _read_csv_or_empty(Path(args.known_visual_validation))
    known_interpretation = _read_csv_or_empty(Path(args.known_visual_interpretation))
    rescue_path = out.parent / "reversibility_rescue" / "rescue_summary.csv"
    rescue = _read_csv_or_empty(rescue_path)

    typed = nodes["type"].notna() & (nodes["type"].astype(str).str.len() > 0)
    top_nodes = node.sort_values("delta_H_node_bits", ascending=False)[
        ["bodyId", "type", "weighted_in_degree", "incoming_entropy_bits", "effective_preimage_size"]
    ]
    top_blocks = block.sort_values("history_ambiguity_bits", ascending=False)[
        ["dst_type", "total_weight", "n_src_types", "history_ambiguity_bits", "effective_preimage_size"]
    ]

    block_summary = global_summary.get("block_channel", {})
    typed_edges = edges.dropna(subset=["src_type", "dst_type"])
    typed_edges = typed_edges[
        (typed_edges["src_type"].astype(str).str.len() > 0)
        & (typed_edges["dst_type"].astype(str).str.len() > 0)
    ]

    roi_focus = roi_table[roi_table.get("group", pd.Series(dtype=str)).isin(["full_typed_core", "LO(R)", "ME(R)", "LOP(R)"])] if not roi_table.empty else roi_table
    pathway_focus = pathway_table if not pathway_table.empty else pathway_table
    roi_null_focus = roi_nulls[
        roi_nulls.get("metric", pd.Series(dtype=str)).isin(["history_ambiguity_bits", "mutual_information_bits"])
        & roi_nulls.get("group", pd.Series(dtype=str)).isin(["full_typed_core", "LO(R)", "ME(R)", "LOP(R)", "T4_T5_related", "lamina_medulla_intrinsic_L1_L2_L3_Mi_Tm_TmY", "projection_output_LC_LPLC_DN_LT"])
    ] if not roi_nulls.empty else roi_nulls

    report = f"""# Optic-Lobe Typed-Core Cell-Type Channels by ROI and Pathway

## 1. Data loaded
- Body-ID rows in processed graph: {len(nodes):,}
- Typed body-ID rows in processed graph: {int(typed.sum()):,}
- Edges in processed graph: {len(edges):,}
- Edges with source and target cell-type labels: {len(typed_edges):,}
- Total synapse weight: {float(edges['weight'].sum()):,.0f}
- Node metric rows after filtering: {len(node):,}

Top cell types by incoming weight:

{_md_table(type_summary[['type', 'n_nodes', 'incoming_weight', 'outgoing_weight']], 10)}

## 2. What was measured
- The main analysis is the typed-core cell-type channel, using only body IDs with non-null cell-type labels and only edges whose source and target are both in that typed core.
- The main claim is that cell-type channels differ in history ambiguity across optic-lobe compartments and pathway-focused subsets.
- The main quantity is `H(src_type | dst_type)`: how ambiguous the upstream cell-type history remains after observing the downstream cell type within a selected ROI or pathway subset.
- `I(src_type; dst_type)` measures retained cell-type information in the weighted type-to-type channel.
- Edge-level `roiInfo` is parsed into major compartments: `LO(R)`, `ME(R)`, `LOP(R)`, `OL(R)`, and `other`. Node `major_roi` is an incident-edge summary, not an independent anatomical proof.
- Node-level incoming entropy and Landauer-inspired columns are retained as secondary proxy scales only.
- No result here is a direct heat measurement or proof of thermodynamic optimization.

## 3. Main results
Typed-core cell-type history ambiguity: {block_summary.get('history_ambiguity_bits', float('nan')):.3f} bits.
Typed-core cell-type mutual information: {block_summary.get('mutual_information_bits', float('nan')):.3f} bits.
Typed-core effective preimage size: {block_summary.get('effective_preimage_size', float('nan')):.3f} source cell types.

ROI comparison:

{_md_table(roi_focus, 10)}

Pathway comparison:

{_md_table(pathway_focus, 10)}

Known visual pathway validation:

{_md_table(known_validation, 12)}

Biological interpretation table:

{_md_table(known_interpretation, 12)}

Comparison across analysis sets and min-weight thresholds:

{_md_table(comparison, 20)}

Top target cell-type blocks by history ambiguity:

{_md_table(top_blocks, 10)}

Secondary node-level proxy rows:

{_md_table(top_nodes, 10)}

These node-level rows are anatomical aggregation proxies. They should not be read as measured erased bits.

## 4. Null models and robustness
Global typed-core null-model summary:

{_md_table(null_summary, 30)}

ROI/pathway null-model summary:

{_md_table(roi_null_focus, 40)}

Null failures excluded from null distributions:

{_md_table(null_failures, 20)}

If `sampled_edges` is true in the null table, the null result is a reproducible preliminary estimate rather than a full-graph null.

Auxiliary task-relevance smoke test:

{_md_table(rescue, 10)}

This classifier result is not a proof of biological reversibility. It only asks whether a source cell-type label can be partly recovered from downstream neighborhood features better than a baseline on a sampled typed subgraph.

## 5. What is not defensible yet
- This analysis does not measure biological heat dissipation.
- It does not include membrane dynamics, active transport, synaptic kinetics, or metabolic accounting.
- It does not estimate the real activity distribution unless activity data are supplied separately.
- Cell-type history ambiguity is a coarse-grained anatomical/statistical proxy, not a literal record of logical state trajectories.
- The export rows are treated as body IDs/segments unless independent curation verifies neuronal identity.
- The Landauer-inspired quantities are lower-bound proxy scales under an explicit coarse-graining model, not fly thermodynamic measurements.

## 6. Strongest next experiment
The strongest next experiment is to combine typed-core anatomy with measured or carefully simulated activity distributions, then ask whether high-history-ambiguity cell-type channels selectively preserve task-relevant visual variables better than matched null networks. That would turn the current anatomical proxy into a test of selective coarse-grained computation.
"""
    out.write_text(report, encoding="utf-8")
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
