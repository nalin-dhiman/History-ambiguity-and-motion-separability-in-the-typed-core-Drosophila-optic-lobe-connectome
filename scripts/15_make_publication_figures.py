#!/usr/bin/env python
"""Create main-text publication figures from completed analysis outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def fig1_roi_ambiguity() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/roi_comparison_table.csv")
    df = df[df["group"].isin(["ME(R)", "LO(R)", "LOP(R)", "OL(R)"])].copy()
    order = ["ME(R)", "LO(R)", "LOP(R)", "OL(R)"]
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    df = df.sort_values("group")
    plt.figure(figsize=(6.2, 4.0))
    ax = sns.barplot(data=df, x="group", y="history_ambiguity_bits", color="#4C78A8")
    ax.set_xlabel("Optic-lobe compartment")
    ax.set_ylabel(r"History ambiguity $H(\mathrm{src\ type}\mid\mathrm{dst\ type})$ (bits)")
    ax.set_title("Typed-core ROI history ambiguity")
    for row in df.itertuples(index=False):
        ax.text(order.index(row.group), row.history_ambiguity_bits + 0.06, f"{row.history_ambiguity_bits:.2f}", ha="center", fontsize=9)
    _save(OUT / "fig1_roi_history_ambiguity.png")


def fig2_visual_hierarchy() -> None:
    df = pd.read_csv(ROOT / "outputs/known_visual_pathway_validation/known_visual_pathway_validation.csv")
    df = df[df["group_kind"].eq("directional_stage")].sort_values("stage_order").copy()
    labels = ["Lamina->ME", "ME->LO", "ME->LOP", "LO/LOP->Out"]
    plt.figure(figsize=(7.0, 4.1))
    ax = plt.gca()
    ax.plot(labels, df["history_ambiguity_bits"], marker="o", linewidth=2.4, label="Observed", color="#2F4B7C")
    ax.plot(labels, df["degree_null_mean_history_ambiguity_bits"], marker="s", linewidth=1.8, label="Degree null", color="#A05195")
    ax.fill_between(
        labels,
        df["degree_null_mean_history_ambiguity_bits"] - df["degree_null_std_history_ambiguity_bits"],
        df["degree_null_mean_history_ambiguity_bits"] + df["degree_null_std_history_ambiguity_bits"],
        color="#A05195",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_ylabel(r"$H(\mathrm{src\ type}\mid\mathrm{dst\ type})$ (bits)")
    ax.set_xlabel("Directional visual stage")
    ax.set_title("Visual pathway history-ambiguity hierarchy")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    _save(OUT / "fig2_visual_pathway_hierarchy.png")


def fig3_motion_separability() -> None:
    df = pd.read_csv(ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/final_control_mode_summary.csv")
    order = ["real", "degree_preserving_null", "weight_shuffled_real_topology", "weight_uniform_real_topology"]
    label_map = {
        "real": "Real",
        "degree_preserving_null": "Degree null",
        "weight_shuffled_real_topology": "Weight shuffled",
        "weight_uniform_real_topology": "Uniform weight",
    }
    df = df[df["condition"].isin(order)].copy()
    df["condition_label"] = df["condition"].map(label_map)
    df["condition_label"] = pd.Categorical(df["condition_label"], [label_map[x] for x in order], ordered=True)
    plt.figure(figsize=(7.2, 4.2))
    ax = sns.barplot(data=df, x="condition_label", y="accuracy_mean", hue="mode", palette=["#7A5195", "#EF5675"])
    ax.axhline(0.25, color="black", linewidth=1.0, linestyle="--", label="Chance")
    ax.set_xlabel("Graph condition")
    ax.set_ylabel("Motion-direction accuracy")
    ax.set_title("Medulla-to-lobula-plate motion separability")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title="")
    _save(OUT / "fig3_motion_separability_controls.png")


def fig4_plasticity_failure() -> None:
    df = pd.read_csv(ROOT / "outputs/rule_discovery/rule_search_results.csv")
    df = df[(df["condition"].eq("real")) & (df["protocol"].eq("representation_probe"))].copy()
    rules = [
        "no_plasticity",
        "hebbian",
        "oja",
        "bcm",
        "ambiguity_gated_hebbian",
        "ambiguity_gated_oja",
        "energy_normalized_ambiguity_oja",
    ]
    df = df[df["rule_name"].isin(rules)]
    means = df.groupby("rule_name", as_index=False).agg(
        plasticity_delta_MI=("plasticity_delta_MI", "mean"),
        total_update_L1=("total_update_L1", "mean"),
    )
    means["rule_name"] = pd.Categorical(means["rule_name"], categories=rules, ordered=True)
    means = means.sort_values("rule_name")
    plt.figure(figsize=(7.4, 4.0))
    ax = sns.barplot(data=means, x="rule_name", y="plasticity_delta_MI", color="#F58518")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Local rule")
    ax.set_ylabel("Representation-probe plasticity ΔMI (bits)")
    ax.set_title("No tested plasticity rule improves beyond readout-only")
    ax.tick_params(axis="x", rotation=35)
    _save(OUT / "fig4_plasticity_rule_search_failure.png")


def fig5_concept() -> None:
    plt.figure(figsize=(7.2, 3.2))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "Typed-core\ncell-type graph"),
        (0.35, 0.55, "History ambiguity\nH(src type | dst type)"),
        (0.65, 0.55, "Visual hierarchy\nvalidation"),
        (0.35, 0.10, "Motion separability\npositive"),
        (0.65, 0.10, "Plasticity search\nnegative control"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.24, 0.26, facecolor="#F2F2F2", edgecolor="#333333", linewidth=1.2))
        ax.text(x + 0.12, y + 0.13, text, ha="center", va="center", fontsize=10)
    arrows = [((0.29, 0.68), (0.35, 0.68)), ((0.59, 0.68), (0.65, 0.68)), ((0.47, 0.55), (0.47, 0.36)), ((0.59, 0.23), (0.65, 0.23))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))
    _save(OUT / "fig5_concept_summary.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    fig1_roi_ambiguity()
    fig2_visual_hierarchy()
    fig3_motion_separability()
    fig4_plasticity_failure()
    fig5_concept()
    print(f"Wrote main publication figures to {OUT}")


if __name__ == "__main__":
    main()
