#!/usr/bin/env python
"""Regenerate main figures in a Network Neuroscience friendly style.

The emphasis here is clean vector PDF output, readable labels, and legends
outside plotting panels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "network_neuroscience_submission_package" / "manuscript" / "figs"

PALETTE = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#777777",
    "black": "#222222",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def sem(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1) / np.sqrt(len(series)))


def fig1_roi() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/authoritative_channel_metrics_with_ci.csv")
    order = ["ME(R)", "LO(R)", "LOP(R)", "OL(R)"]
    df = df[df["group"].isin(order)].copy()
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    df = df.sort_values("group")

    y = np.arange(len(df))
    vals = df["H_source_given_target_bits"].to_numpy()
    lo = vals - df["H_source_given_target_ci_low"].to_numpy()
    hi = df["H_source_given_target_ci_high"].to_numpy() - vals

    fig, ax = plt.subplots(figsize=(5.7, 3.1))
    ax.barh(y, vals, color=[PALETTE["cyan"], PALETTE["blue"], PALETTE["green"], PALETTE["grey"]], height=0.62)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=PALETTE["black"], elinewidth=1.0, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["group"])
    ax.invert_yaxis()
    ax.set_xlabel(r"History ambiguity $H(S\mid D)$ (bits)")
    ax.set_title("Typed-core compartment-level history ambiguity")
    for yi, value, pre in zip(y, vals, df["effective_preimage_size"]):
        ax.text(value + 0.04, yi, f"{value:.2f} bits; $2^H$={pre:.1f}", va="center", fontsize=7)
    ax.set_xlim(0, max(vals) + 1.4)
    fig.subplots_adjust(right=0.88)
    save(fig, "fig1_roi_history_ambiguity")


def fig2_hierarchy() -> None:
    df = pd.read_csv(ROOT / "outputs/known_visual_pathway_validation/known_visual_pathway_validation.csv")
    df = df[df["group_kind"].eq("directional_stage")].sort_values("stage_order").copy()
    labels = ["Lamina→ME", "ME→LO", "ME→LOP", "LO/LOP→Output"]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    obs = df["history_ambiguity_bits"].to_numpy()
    null = df["degree_null_mean_history_ambiguity_bits"].to_numpy()
    null_sd = df["degree_null_std_history_ambiguity_bits"].to_numpy()

    ax.plot(x, obs, marker="o", color=PALETTE["blue"], lw=2.2, label="Observed")
    ax.plot(x, null, marker="s", color=PALETTE["purple"], lw=1.8, label="Degree-preserving null mean")
    ax.fill_between(x, null - null_sd, null + null_sd, color=PALETTE["purple"], alpha=0.18, label="Null ±1 SD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(r"$H(S\mid D)$ (bits)")
    ax.set_title("Directional visual hierarchy")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.subplots_adjust(right=0.68, bottom=0.24)
    save(fig, "fig2_visual_pathway_hierarchy")


def fig3_motion() -> None:
    summary = pd.read_csv(ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/final_control_mode_summary.csv")
    keep = ["real", "degree_preserving_null", "weight_shuffled_real_topology", "weight_uniform_real_topology"]
    labels = {
        "real": "Real",
        "degree_preserving_null": "Degree null",
        "weight_shuffled_real_topology": "Weight shuffled",
        "weight_uniform_real_topology": "Uniform weight",
    }
    modes = ["readout_only", "local_hebbian"]
    mode_labels = {"readout_only": "Readout only", "local_hebbian": "Local Hebbian"}
    summary = summary[summary["condition"].isin(keep)].copy()

    pairs = pd.read_csv(ROOT / "outputs/tables/motion_probe_reviewer_control_pair_summary.csv")
    pairs = pairs[
        (pairs["comparison"].eq("real_vs_degree_preserving_null"))
        & (pairs["metric"].eq("delta_task_MI_bits"))
    ].copy()
    pairs["label"] = pairs["template_mode"].str.replace("_", " ") + "\n" + pairs["feature_mode"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45), gridspec_kw={"width_ratios": [1.12, 1.0]})

    ax = axes[0]
    x = np.arange(len(keep))
    width = 0.34
    colors = [PALETTE["blue"], PALETTE["red"]]
    for i, mode in enumerate(modes):
        sub = summary[summary["mode"].eq(mode)].set_index("condition").loc[keep]
        xpos = x + (i - 0.5) * width
        y = sub["accuracy_mean"].to_numpy()
        err = sub["accuracy_sd"].to_numpy() / np.sqrt(sub["n"].to_numpy())
        ax.bar(xpos, y, width, color=colors[i], label=mode_labels[mode])
        ax.errorbar(xpos, y, yerr=err, fmt="none", ecolor=PALETTE["black"], elinewidth=0.9, capsize=2)
    ax.axhline(0.25, color=PALETTE["black"], lw=1, ls="--", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keep], rotation=22, ha="right")
    ax.set_ylabel("Motion-direction accuracy")
    ax.set_title("A. Matched 1500-node/12000-edge probe")
    ax.set_ylim(0, 0.82)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    ax = axes[1]
    pairs = pairs.sort_values(["template_mode", "feature_mode"])
    x2 = np.arange(len(pairs))
    vals = pairs["mean"].to_numpy()
    lo = vals - pairs["ci_low"].to_numpy()
    hi = pairs["ci_high"].to_numpy() - vals
    colors2 = np.where(vals >= 0, PALETTE["green"], PALETTE["red"])
    ax.bar(x2, vals, color=colors2, width=0.68)
    ax.errorbar(x2, vals, yerr=[lo, hi], fmt="none", ecolor=PALETTE["black"], elinewidth=0.8, capsize=2)
    ax.axhline(0, color=PALETTE["black"], lw=1)
    ax.set_xticks(x2)
    ax.set_xticklabels(pairs["label"], rotation=55, ha="right")
    ax.set_ylabel(r"Real − degree null $\Delta I$ (bits)")
    ax.set_title("B. Reviewer template/feature controls")

    fig.subplots_adjust(wspace=0.72, right=0.84, bottom=0.34)
    save(fig, "fig3_motion_separability_controls")


def fig4_plasticity() -> None:
    df = pd.read_csv(ROOT / "outputs/rule_discovery/rule_search_results.csv")
    df = df[(df["condition"].eq("real")) & (df["protocol"].eq("representation_probe"))].copy()
    rules = [
        "no_plasticity",
        "hebbian",
        "oja",
        "bcm",
        "reward_hebbian",
        "ambiguity_gated_hebbian",
        "ambiguity_gated_oja",
        "energy_normalized_ambiguity_oja",
    ]
    df = df[df["rule_name"].isin(rules)].copy()
    agg = df.groupby("rule_name", as_index=False).agg(
        plasticity_delta_MI=("plasticity_delta_MI", "mean"),
        se=("plasticity_delta_MI", sem),
        update_cost=("total_update_L1", "mean"),
    )
    agg["rule_name"] = pd.Categorical(agg["rule_name"], categories=rules, ordered=True)
    agg = agg.sort_values("rule_name")
    labels = [r.replace("_", "\n") for r in agg["rule_name"]]

    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    x = np.arange(len(agg))
    vals = agg["plasticity_delta_MI"].to_numpy()
    ax.bar(x, vals, color=PALETTE["grey"], width=0.68)
    ax.errorbar(x, vals, yerr=agg["se"].to_numpy(), fmt="none", ecolor=PALETTE["black"], elinewidth=0.8, capsize=2)
    ax.axhline(0, color=PALETTE["black"], lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Representation-probe plasticity ΔMI (bits)")
    ax.set_title("Local plasticity rules do not beat the fixed representation")
    fig.subplots_adjust(bottom=0.35)
    save(fig, "fig4_plasticity_rule_search_failure")


def fig5_concept() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_axis_off()
    boxes = [
        (0.03, 0.58, 0.22, 0.25, "Typed-core\ncell-type graph", PALETTE["cyan"]),
        (0.31, 0.58, 0.24, 0.25, r"Channel metric\n$H(S\mid D)$", PALETTE["blue"]),
        (0.63, 0.58, 0.25, 0.25, "Visual hierarchy\nvalidation", PALETTE["green"]),
        (0.31, 0.12, 0.24, 0.25, "Motion-template\nseparability", PALETTE["yellow"]),
        (0.63, 0.12, 0.25, 0.25, "Plasticity rules\nnegative control", PALETTE["red"]),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.22, edgecolor=PALETTE["black"], linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    arrows = [((0.25, 0.705), (0.31, 0.705)), ((0.55, 0.705), (0.63, 0.705)), ((0.43, 0.58), (0.43, 0.37)), ((0.55, 0.245), (0.63, 0.245))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2, color=PALETTE["black"]))
    ax.text(0.02, 0.01, "Interpretive rule: structure ≠ representation ≠ learning.", fontsize=8, color=PALETTE["black"])
    save(fig, "fig5_concept_summary")


def main() -> None:
    setup_style()
    fig1_roi()
    fig2_hierarchy()
    fig3_motion()
    fig4_plasticity()
    fig5_concept()
    print(f"Wrote Network Neuroscience figures to {OUT}")


if __name__ == "__main__":
    main()
