#!/usr/bin/env python
"""Plot physical learning-efficiency proxy outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/learning_efficiency/final_summary.csv")
    parser.add_argument("--curves", default="outputs/learning_efficiency/learning_curves.csv")
    parser.add_argument("--out-dir", default="outputs/learning_efficiency/figures")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    summary = pd.read_csv(args.summary)
    curves = pd.read_csv(args.curves)
    hebb = summary[summary["mode"].eq("local_hebbian")].copy()

    plt.figure(figsize=(10, 4.5))
    sns.catplot(data=hebb, x="pathway", y="eta_phys_L1", hue="task", col="condition", kind="bar", height=4, aspect=1.1, sharey=False)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_eta_phys_by_pathway_task.png", dpi=220)
    plt.close("all")

    plot_curves = curves.groupby(["pathway", "task", "condition", "mode", "epoch"], as_index=False)["accuracy"].mean()
    g = sns.relplot(
        data=plot_curves,
        x="epoch",
        y="accuracy",
        hue="condition",
        style="mode",
        col="task",
        row="pathway",
        kind="line",
        height=2.4,
        aspect=1.5,
    )
    g.figure.tight_layout()
    g.figure.savefig(out / "fig_accuracy_curves.png", dpi=220)
    plt.close("all")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=hebb,
        x="final_accuracy",
        y="total_synaptic_update_L1",
        hue="task",
        style="pathway",
        size="eta_phys_L1",
        sizes=(30, 180),
        alpha=0.8,
    )
    plt.tight_layout()
    plt.savefig(out / "fig_learning_efficiency_tradeoff.png", dpi=220)
    plt.close()

    real = hebb[hebb["condition"].eq("real")]
    null = hebb[~hebb["condition"].eq("real")]
    null_mean = null.groupby(["pathway", "task"], as_index=False)["eta_phys_L1"].mean().rename(columns={"eta_phys_L1": "null_eta"})
    diff = real.merge(null_mean, on=["pathway", "task"], how="left")
    diff["real_minus_null_eta"] = diff["eta_phys_L1"] - diff["null_eta"]
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=diff, x="pathway", y="real_minus_null_eta", hue="task")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_real_vs_null_eta.png", dpi=220)
    plt.close()

    plot = real.groupby(["pathway", "task"], as_index=False).agg(
        eta_phys_L1=("eta_phys_L1", "mean"),
        history_ambiguity_bits=("history_ambiguity_bits", "mean"),
    )
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot, x="history_ambiguity_bits", y="eta_phys_L1", hue="task", style="pathway", s=120)
    for row in plot.itertuples(index=False):
        plt.text(row.history_ambiguity_bits, row.eta_phys_L1, f" {row.pathway}:{row.task}", fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_history_ambiguity_vs_eta.png", dpi=220)
    plt.close()
    print(f"Wrote learning figures to {out}")


if __name__ == "__main__":
    main()
