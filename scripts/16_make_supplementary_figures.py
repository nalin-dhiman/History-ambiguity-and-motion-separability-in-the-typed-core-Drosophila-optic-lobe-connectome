#!/usr/bin/env python
"""Create supplementary figures from completed analysis outputs."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "supplement_figures"


def _save(path: Path) -> None:
    plt.gcf().subplots_adjust(bottom=max(plt.gcf().subplotpars.bottom, 0.18))
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _angle_x(ax, rotation: float = 35) -> None:
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")


def _wrap(values, width: int = 24):
    return [textwrap.fill(str(v).replace("_", " "), width=width) for v in values]


def _outside_legend(ax, ncol: int = 1, y: float = -0.25) -> None:
    leg = ax.get_legend()
    if leg is not None:
        handles, labels = ax.get_legend_handles_labels()
        leg.remove()
        ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


def sfig_min_weight() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/full_vs_typed_core_comparison.csv")
    df = df[df["analysis"].str.contains("typed_core")].copy()
    plt.figure(figsize=(7.8, 4.8))
    ax = sns.lineplot(data=df, x="min_weight", y="cell_type_history_ambiguity_bits", marker="o")
    ax.set_xlabel("Minimum edge weight")
    ax.set_ylabel("History ambiguity (bits)")
    ax.set_title("Typed-core min-weight robustness")
    _save(OUT / "sfig1_min_weight_robustness.png")


def sfig_pathway_bars() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/pathway_comparison_table.csv")
    plt.figure(figsize=(8.8, 5.2))
    ax = sns.barplot(data=df, y="group", x="history_ambiguity_bits", color="#4C78A8")
    ax.set_yticklabels(_wrap(df["group"], 28))
    ax.set_xlabel("History ambiguity (bits)")
    ax.set_ylabel("Pathway subset")
    ax.set_title("Pathway-focused history ambiguity")
    _save(OUT / "sfig2_pathway_ambiguity.png")


def sfig_known_null_deltas() -> None:
    df = pd.read_csv(ROOT / "outputs/known_visual_pathway_validation/known_visual_pathway_validation.csv")
    df = df.sort_values(["group_kind", "stage_order"]).copy()
    plt.figure(figsize=(10.2, 6.6))
    ax = sns.barplot(data=df, y="group", x="observed_minus_degree_null_bits", hue="group_kind", dodge=False)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Observed minus degree-null ambiguity (bits)")
    ax.set_ylabel("Visual pathway group")
    ax.set_title("Known-pathway null comparison")
    ax.set_yticklabels(_wrap(df["group"], 30))
    _outside_legend(ax, ncol=2, y=-0.18)
    _save(OUT / "sfig3_known_pathway_null_deltas.png")


def sfig_rescue_budget() -> None:
    df = pd.read_csv(ROOT / "outputs/learning_signal_rescue/rescue_budget_summary.csv")
    df = df[df["mode"].isin(["readout_only", "local_hebbian"])].copy()
    plt.figure(figsize=(8.4, 5.0))
    ax = sns.lineplot(data=df, x="matched_edges", y="mean_real_accuracy", hue="mode", marker="o")
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Matched edge budget")
    ax.set_ylabel("Mean real accuracy")
    ax.set_title("Motion rescue budget sweep")
    _outside_legend(ax, ncol=2)
    _save(OUT / "sfig4_motion_budget_sweep.png")


def sfig_final_control_deltas() -> None:
    df = pd.read_csv(ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/final_control_decisive_table.csv")
    view = df[df["condition_comparison"].str.startswith("real_vs")].copy()
    plt.figure(figsize=(9.8, 5.2))
    ax = sns.barplot(data=view, y="condition_comparison", x="delta_MI", color="#54A24B")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Real minus control ΔMI (bits)")
    ax.set_ylabel("Control")
    ax.set_yticklabels(_wrap(view["condition_comparison"], 32))
    ax.set_title("Final motion-control ΔMI")
    _save(OUT / "sfig5_final_control_delta_mi.png")


def sfig_learning_curves() -> None:
    df = pd.read_csv(ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/learning_curves.csv")
    plot = df.groupby(["condition", "mode", "epoch"], as_index=False)["accuracy"].mean()
    plt.figure(figsize=(10.4, 5.4))
    ax = sns.lineplot(data=plot, x="epoch", y="accuracy", hue="condition", style="mode")
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1)
    ax.set_title("Final-control learning curves")
    ax.set_ylabel("Accuracy")
    _outside_legend(ax, ncol=2, y=-0.23)
    _save(OUT / "sfig6_learning_curves_final_control.png")


def sfig_rule_scores() -> None:
    df = pd.read_csv(ROOT / "outputs/rule_discovery/discovery_summary.csv").head(20)
    plt.figure(figsize=(10.2, 7.8))
    ax = sns.barplot(data=df, y="rule_name", x="score", hue="protocol", dodge=False)
    ax.set_xlabel("Discovery score")
    ax.set_ylabel("Rule")
    ax.set_title("Rule-search scores")
    ax.set_yticklabels(_wrap(df["rule_name"], 28))
    _outside_legend(ax, ncol=2, y=-0.14)
    _save(OUT / "sfig7_rule_search_scores.png")


def sfig_rule_cost_gain() -> None:
    df = pd.read_csv(ROOT / "outputs/rule_discovery/rule_search_results.csv")
    df = df[(df["condition"].eq("real")) & (df["protocol"].eq("representation_probe"))].copy()
    plot = df.groupby("rule_name", as_index=False).agg(
        plasticity_delta_MI=("plasticity_delta_MI", "mean"),
        total_update_L1=("total_update_L1", "mean"),
    )
    plt.figure(figsize=(8.2, 5.2))
    ax = sns.scatterplot(data=plot, x="total_update_L1", y="plasticity_delta_MI", s=80)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean update cost L1")
    ax.set_ylabel("Representation-probe plasticity ΔMI")
    ax.set_title("Rule-search cost vs gain")
    _save(OUT / "sfig8_rule_cost_vs_gain.png")


def sfig_subgraph_realization() -> None:
    df = pd.read_csv(ROOT / "outputs/learning_signal_rescue/rescue_subgraph_summary.csv")
    plt.figure(figsize=(8.8, 5.2))
    ax = sns.scatterplot(data=df, x="subgraph_n_nodes", y="subgraph_n_edges", hue="budget", s=90)
    ax.set_xlabel("Realized nodes")
    ax.set_ylabel("Realized edges")
    ax.set_title("Matched subgraph realization")
    _outside_legend(ax, ncol=3)
    _save(OUT / "sfig9_subgraph_realization.png")


def sfig_roi_mi_scatter() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/roi_comparison_table.csv")
    df = df[df["status"].eq("ok")].copy()
    plt.figure(figsize=(7.8, 5.0))
    ax = sns.scatterplot(data=df, x="history_ambiguity_bits", y="mutual_information_bits", hue="group", s=120)
    ax.set_xlabel("History ambiguity (bits)")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title("ROI ambiguity and retained information")
    _outside_legend(ax, ncol=2)
    _save(OUT / "sfig10_roi_mi_scatter.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    sfig_min_weight()
    sfig_pathway_bars()
    sfig_known_null_deltas()
    sfig_rescue_budget()
    sfig_final_control_deltas()
    sfig_learning_curves()
    sfig_rule_scores()
    sfig_rule_cost_gain()
    sfig_subgraph_realization()
    sfig_roi_mi_scatter()
    print(f"Wrote supplementary figures to {OUT}")


if __name__ == "__main__":
    main()
