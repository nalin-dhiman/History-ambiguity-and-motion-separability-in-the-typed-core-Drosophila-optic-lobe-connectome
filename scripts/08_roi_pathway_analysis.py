#!/usr/bin/env python
"""ROI- and pathway-specific typed-core cell-type channel analyses."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.metrics.irreversibility import compute_block_irreversibility
from thermodynamic_connectomics.nulls.degree_preserving import (
    directed_degree_preserving_rewire,
    empirical_z_p,
    target_preserving_source_shuffle,
    type_label_permutation,
)


ROI_ORDER = ["full_typed_core", "LO(R)", "ME(R)", "LOP(R)", "OL(R)", "other"]
PATHWAY_PATTERNS = {
    "T4_T5_related": r"^(?:T4|T5)",
    "lamina_medulla_intrinsic_L1_L2_L3_Mi_Tm_TmY": r"^(?:L1$|L2$|L3$|Mi|Tm|TmY)",
    "projection_output_LC_LPLC_DN_LT": r"^(?:LC|LPLC|DN|LT)",
}
NULL_MODELS = {
    "target_preserving_source_shuffle": target_preserving_source_shuffle,
    "degree_preserving_directed_rewire": None,
    "type_label_permutation": type_label_permutation,
}


def _clean_roi_label(value: object) -> str:
    if pd.isna(value):
        return "other"
    text = str(value)
    return text if text in {"LO(R)", "ME(R)", "LOP(R)", "OL(R)"} else "other"


def _restore_endpoint_metadata(null_edges: pd.DataFrame, reference_edges: pd.DataFrame) -> pd.DataFrame:
    out = null_edges.copy()
    endpoint_cols = ["type", "instance", "roi", "major_roi", "nt"]
    for side in ["src", "dst"]:
        ref_cols = [side] + [f"{side}_{col}" for col in endpoint_cols if f"{side}_{col}" in reference_edges.columns]
        if len(ref_cols) == 1:
            continue
        mapper = reference_edges[ref_cols].drop_duplicates(side).set_index(side)
        for col in mapper.columns:
            out[col] = out[side].map(mapper[col])
    if "edge_major_roi" in reference_edges.columns and "edge_major_roi" not in out.columns:
        out["edge_major_roi"] = "rewired"
    return out


def _channel_summary(name: str, kind: str, edges: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    block, summary = compute_block_irreversibility(edges, group_col="type")
    row = {
        "group": name,
        "group_kind": kind,
        "n_edges": int(len(edges)),
        "total_weight": float(edges["weight"].sum()) if len(edges) else 0.0,
        "n_src_types": summary.get("n_src_types", 0),
        "n_dst_types": summary.get("n_dst_types", 0),
        "history_ambiguity_bits": summary.get("history_ambiguity_bits", np.nan),
        "mutual_information_bits": summary.get("mutual_information_bits", np.nan),
        "effective_preimage_size": summary.get("effective_preimage_size", np.nan),
        "status": summary.get("status", "unknown"),
    }
    if not block.empty:
        block = block.copy()
        block["group"] = name
        block["group_kind"] = kind
    return row, block


def _pathway_subset(edges: pd.DataFrame, pattern: str) -> pd.DataFrame:
    src = edges["src_type"].astype(str).str.contains(pattern, regex=True, na=False)
    dst = edges["dst_type"].astype(str).str.contains(pattern, regex=True, na=False)
    return edges[src | dst].copy()


def _sample_for_null(edges: pd.DataFrame, sample_edges: int, seed: int) -> tuple[pd.DataFrame, bool]:
    if sample_edges > 0 and len(edges) > sample_edges:
        return edges.sample(sample_edges, random_state=seed).reset_index(drop=True), True
    return edges.reset_index(drop=True), False


def _run_nulls_for_group(
    name: str,
    kind: str,
    edges: pd.DataFrame,
    observed: dict,
    n_nulls: int,
    sample_edges: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if len(edges) == 0 or not np.isfinite(float(observed.get("history_ambiguity_bits", np.nan))):
        return [], [{"group": name, "group_kind": kind, "null_model": "all", "iteration": -1, "reason": "empty_or_invalid_observed"}]
    base, sampled = _sample_for_null(edges, sample_edges, seed)
    obs_row, _ = _channel_summary(name, kind, base)
    null_values: dict[tuple[str, str], list[float]] = {}
    failures = []
    for i in trange(n_nulls, desc=f"nulls {name}", leave=False):
        null_seed = seed + i
        null_specs = [
            ("target_preserving_source_shuffle", lambda: target_preserving_source_shuffle(base, seed=null_seed)),
            (
                "degree_preserving_directed_rewire",
                lambda: _restore_endpoint_metadata(
                    directed_degree_preserving_rewire(
                        base,
                        seed=null_seed,
                        n_swaps=min(40_000, max(1, len(base) // 10)),
                    ),
                    base,
                ),
            ),
            ("type_label_permutation", lambda: type_label_permutation(base, seed=null_seed)),
        ]
        for null_name, make_null in null_specs:
            try:
                null_edges = make_null()
                row, _ = _channel_summary(name, kind, null_edges)
                for metric in ["history_ambiguity_bits", "mutual_information_bits", "effective_preimage_size"]:
                    value = float(row.get(metric, np.nan))
                    if not np.isfinite(value):
                        raise ValueError(f"{metric} is not finite")
                    null_values.setdefault((null_name, metric), []).append(value)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "group": name,
                        "group_kind": kind,
                        "null_model": null_name,
                        "iteration": i,
                        "reason": repr(exc),
                    }
                )

    rows = []
    for null_name in ["target_preserving_source_shuffle", "degree_preserving_directed_rewire", "type_label_permutation"]:
        for metric in ["history_ambiguity_bits", "mutual_information_bits", "effective_preimage_size"]:
            vals = null_values.get((null_name, metric), [])
            stats = empirical_z_p(float(obs_row.get(metric, np.nan)), vals)
            rows.append(
                {
                    "group": name,
                    "group_kind": kind,
                    "null_model": null_name,
                    "metric": metric,
                    **stats,
                    "n_success": len(vals),
                    "failure_count": sum(1 for f in failures if f["null_model"] == null_name),
                    "sampled_edges": sampled,
                    "edge_count_used": int(len(base)),
                    "edge_count_available": int(len(edges)),
                }
            )
    return rows, failures


def _plot_outputs(summary: pd.DataFrame, target: pd.DataFrame, nulls: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    roi = summary[summary["group_kind"].eq("roi") & summary["group"].isin(["LO(R)", "ME(R)", "LOP(R)", "full_typed_core"])].copy()
    roi["group"] = pd.Categorical(roi["group"], ["full_typed_core", "LO(R)", "ME(R)", "LOP(R)"], ordered=True)
    roi = roi.sort_values("group")

    plt.figure(figsize=(6, 4))
    sns.barplot(data=roi, x="group", y="history_ambiguity_bits", color="#4C78A8")
    plt.xlabel("")
    plt.ylabel("H(src type | dst type), bits")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_roi_ambiguity.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=roi, x="group", y="mutual_information_bits", color="#59A14F")
    plt.xlabel("")
    plt.ylabel("I(src type; dst type), bits")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_roi_mutual_information.png", dpi=220)
    plt.close()

    ranked = target[target["group"].eq("full_typed_core")].sort_values("history_ambiguity_bits")
    ranked = ranked.assign(rank=np.arange(1, len(ranked) + 1))
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=ranked, x="rank", y="history_ambiguity_bits", color="#4C78A8")
    plt.xlabel("Target type rank")
    plt.ylabel("Target-type history ambiguity, bits")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_target_type_ambiguity_ranked.png", dpi=220)
    plt.close()

    obs_null = nulls[nulls["metric"].isin(["history_ambiguity_bits", "mutual_information_bits"])].copy()
    obs_null = obs_null[obs_null["group"].isin(["full_typed_core", "LO(R)", "ME(R)", "LOP(R)", *PATHWAY_PATTERNS.keys()])]
    if not obs_null.empty:
        obs_null["metric_label"] = obs_null["metric"].map(
            {"history_ambiguity_bits": "History ambiguity", "mutual_information_bits": "Mutual information"}
        )
        plt.figure(figsize=(10, 5))
        plot_df = obs_null.melt(
            id_vars=["group", "group_kind", "null_model", "metric", "metric_label"],
            value_vars=["observed", "null_mean"],
            var_name="series",
            value_name="bits",
        )
        sns.catplot(
            data=plot_df,
            x="group",
            y="bits",
            hue="series",
            col="metric_label",
            row="null_model",
            kind="bar",
            height=2.5,
            aspect=2.2,
            sharey=False,
        )
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_observed_vs_null_roi_pathway.png", dpi=220)
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="outputs/processed_typed_core/edges.parquet")
    parser.add_argument("--out-dir", default="outputs/roi_pathway")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--n-nulls", type=int, default=30)
    parser.add_argument("--sample-edges", type=int, default=1_000_000)
    parser.add_argument("--min-target-weight", type=float, default=100.0)
    parser.add_argument("--min-target-src-types", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_parquet(args.edges)
    if "edge_major_roi" not in edges.columns:
        raise ValueError("typed-core edges are missing edge_major_roi; rerun 01_prepare_optic_lobe.py --typed-core")
    edges = edges.dropna(subset=["src_type", "dst_type"]).copy()
    edges["edge_major_roi"] = edges["edge_major_roi"].map(_clean_roi_label)

    groups: list[tuple[str, str, pd.DataFrame]] = [("full_typed_core", "roi", edges)]
    for roi in ["LO(R)", "ME(R)", "LOP(R)", "OL(R)", "other"]:
        groups.append((roi, "roi", edges[edges["edge_major_roi"].eq(roi)].copy()))
    for pathway, pattern in PATHWAY_PATTERNS.items():
        groups.append((pathway, "pathway", _pathway_subset(edges, pattern)))

    summary_rows = []
    target_blocks = []
    null_rows = []
    failure_rows = []
    for index, (name, kind, subset) in enumerate(groups):
        row, block = _channel_summary(name, kind, subset)
        summary_rows.append(row)
        if not block.empty:
            target_blocks.append(block)
        nr, fr = _run_nulls_for_group(
            name,
            kind,
            subset,
            row,
            n_nulls=args.n_nulls,
            sample_edges=args.sample_edges,
            seed=args.seed + 1000 * index,
        )
        null_rows.extend(nr)
        failure_rows.extend(fr)

    summary = pd.DataFrame(summary_rows)
    target = pd.concat(target_blocks, ignore_index=True) if target_blocks else pd.DataFrame()
    nulls = pd.DataFrame(null_rows)
    failures = pd.DataFrame(failure_rows)

    roi_comparison = summary[summary["group_kind"].eq("roi")].copy()
    pathway_comparison = summary[summary["group_kind"].eq("pathway")].copy()
    target_rank_pool = target[
        target["group"].eq("full_typed_core")
        & target["total_weight"].ge(args.min_target_weight)
        & target["n_src_types"].ge(args.min_target_src_types)
    ].copy()
    low = target_rank_pool.sort_values(["history_ambiguity_bits", "total_weight"], ascending=[True, False]).head(20)
    high = target_rank_pool.sort_values(["history_ambiguity_bits", "total_weight"], ascending=[False, False]).head(20)

    outputs = {
        out_dir / "roi_pathway_summary.csv": summary,
        out_dir / "roi_pathway_null_summary.csv": nulls,
        out_dir / "roi_pathway_null_failures.csv": failures,
        out_dir / "target_type_ambiguity_by_group.csv": target,
        tables_dir / "roi_comparison_table.csv": roi_comparison,
        tables_dir / "pathway_comparison_table.csv": pathway_comparison,
        tables_dir / "top20_lowest_ambiguity_target_types.csv": low,
        tables_dir / "top20_highest_ambiguity_target_types.csv": high,
    }
    for path, df in outputs.items():
        df.to_csv(path, index=False)

    _plot_outputs(summary, target, nulls, figures_dir)
    print(f"Wrote ROI/pathway analysis to {out_dir}, {tables_dir}, and {figures_dir}")


if __name__ == "__main__":
    main()
