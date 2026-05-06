#!/usr/bin/env python
"""Create preliminary publication-style figures and summary tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _top_or_other(series: pd.Series, n: int = 12) -> pd.Series:
    top = set(series.dropna().astype(str).value_counts().head(n).index)
    return series.astype(str).where(series.astype(str).isin(top), "Other")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="outputs/processed")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--nulls-dir", default="outputs/nulls")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    out = Path(args.out_dir)
    tables = out.parent / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    node = pd.read_csv(Path(args.metrics_dir) / "node_irreversibility.csv")
    block = pd.read_csv(Path(args.metrics_dir) / "block_irreversibility.csv")
    edges = pd.read_parquet(Path(args.processed_dir) / "edges.parquet", columns=["src_type", "dst_type", "weight"])
    null_summary_path = Path(args.nulls_dir) / "null_summary.csv"
    null_summary = pd.read_csv(null_summary_path) if null_summary_path.exists() else pd.DataFrame()

    node.sort_values("delta_H_node_bits", ascending=False).head(100).to_csv(
        tables / "top_nodes_by_irreversibility.csv", index=False
    )
    block.sort_values("history_ambiguity_bits", ascending=False).head(100).to_csv(
        tables / "top_blocks_by_irreversibility.csv", index=False
    )
    null_summary.to_csv(tables / "null_significance_summary.csv", index=False)

    plot_node = node.sample(n=min(100_000, len(node)), random_state=0) if len(node) > 100_000 else node.copy()
    plot_node["plot_type"] = _top_or_other(plot_node.get("type", pd.Series(["unknown"] * len(plot_node))))
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=plot_node,
        x="weighted_in_degree",
        y="incoming_entropy_bits",
        hue="plot_type",
        s=8,
        linewidth=0,
        alpha=0.45,
        legend=False,
    )
    plt.xscale("log")
    plt.xlabel("Weighted in-degree")
    plt.ylabel("Incoming entropy (bits)")
    plt.tight_layout()
    plt.savefig(out / "fig_degree_entropy_scatter.png", dpi=220)
    plt.close()

    top = node.sort_values("delta_H_node_bits", ascending=False).head(25).copy()
    top["label"] = top["type"].fillna(top["bodyId"]).astype(str)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top, y="label", x="delta_H_node_bits", color="#4C78A8")
    plt.xlabel("Entropy contraction proxy (bits)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out / "fig_top_landauer_nodes.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    metric_rows = null_summary[
        (null_summary.get("metric", pd.Series(dtype=str)) == "global_node_delta_sum")
        & (null_summary.get("null_model", pd.Series(dtype=str)) == "target_preserving_source_shuffle")
    ]
    if not metric_rows.empty and {"observed", "null_mean", "null_std"}.issubset(metric_rows.columns):
        row = metric_rows.iloc[0]
        xs = np.linspace(row["null_mean"] - 4 * row["null_std"], row["null_mean"] + 4 * row["null_std"], 200)
        if np.isfinite(row["null_std"]) and row["null_std"] > 0:
            ys = np.exp(-0.5 * ((xs - row["null_mean"]) / row["null_std"]) ** 2)
            plt.plot(xs, ys, label="Null normal approximation")
        plt.axvline(row["observed"], color="#D62728", linewidth=2, label="Observed")
        plt.axvline(row["null_mean"], color="#4C78A8", linewidth=2, linestyle="--", label="Null mean")
        plt.xlabel("Global node entropy contraction proxy (bits)")
        plt.yticks([])
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Null summary unavailable", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "fig_observed_vs_null.png", dpi=220)
    plt.close()

    typed = edges.dropna(subset=["src_type", "dst_type"]).copy()
    typed["src_type"] = _top_or_other(typed["src_type"], 24)
    typed["dst_type"] = _top_or_other(typed["dst_type"], 24)
    heat = typed.groupby(["dst_type", "src_type"])["weight"].sum().reset_index()
    pivot = heat.pivot_table(index="dst_type", columns="src_type", values="weight", fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": "P(src type | dst type)"})
    plt.xlabel("Source type")
    plt.ylabel("Target type")
    plt.tight_layout()
    plt.savefig(out / "fig_block_heatmap.png", dpi=220)
    plt.close()

    by_type = node.dropna(subset=["type"]).groupby("type", as_index=False)["effective_preimage_size"].median()
    by_type = by_type.sort_values("effective_preimage_size", ascending=False).head(30)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=by_type, y="type", x="effective_preimage_size", color="#59A14F")
    plt.xscale("log")
    plt.xlabel("Median effective preimage size")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out / "fig_preimage_size_by_type.png", dpi=220)
    plt.close()
    print(f"Wrote figures to {out} and tables to {tables}")


if __name__ == "__main__":
    main()
