#!/usr/bin/env python
"""Create Nature/npj-style figures without changing analysis results.

This script only reads completed analysis outputs and rewrites visualization
assets as roomy vector PDFs.  Layout choices are deliberately conservative:
legends and explanatory labels sit outside data axes wherever possible.
"""
from __future__ import annotations

import math
import shutil
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIG = ROOT / "paper" / "figures"
PAPER_SUPP = ROOT / "paper" / "supplement_figures"
NETN_FIG = ROOT / "network_neuroscience_submission_package" / "manuscript" / "figs"
NETN_SUPP = ROOT / "network_neuroscience_submission_package" / "manuscript" / "supplement_figures"

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
PURPLE = "#9467bd"
GREY = "#7f7f7f"
DARK = "#222222"
LIGHT = "#e9edf2"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, GREY]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "figure.constrained_layout.use": False,
        }
    )


def ensure_dirs() -> None:
    for d in [PAPER_FIG, PAPER_SUPP, NETN_FIG, NETN_SUPP]:
        d.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")


def save_all_main(fig: plt.Figure, stem: str, aliases: list[str] | None = None) -> None:
    aliases = aliases or []
    for outdir in [PAPER_FIG, NETN_FIG]:
        save(fig, outdir, stem)
        for alias in aliases:
            shutil.copy2(outdir / f"{stem}.pdf", outdir / f"{alias}.pdf")
    plt.close(fig)


def save_all_supp(fig: plt.Figure, stem: str, aliases: list[str] | None = None) -> None:
    aliases = aliases or []
    for outdir in [PAPER_SUPP, NETN_SUPP]:
        save(fig, outdir, stem)
        for alias in aliases:
            shutil.copy2(outdir / f"{stem}.pdf", outdir / f"{alias}.pdf")
    plt.close(fig)


def angle_xlabels(ax: plt.Axes, rotation: float = 35) -> None:
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")


def wrapped(values: pd.Series | list[str], width: int = 22) -> list[str]:
    return [textwrap.fill(str(v).replace("_", " "), width=width) for v in values]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.13,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=color,
        edgecolor=DARK,
        linewidth=1.2,
        alpha=0.14,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=DARK)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = DARK, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_observed_type_subgraph(ax: plt.Axes) -> None:
    """Draw a data-derived type-level network from the typed-core edge table."""
    edges = pd.read_parquet(
        ROOT / "outputs/processed_typed_core/edges.parquet",
        columns=["src_type", "dst_type", "weight"],
    )
    agg = (
        edges.groupby(["src_type", "dst_type"], as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )
    degree = pd.concat(
        [
            agg[["src_type", "weight"]].rename(columns={"src_type": "type"}),
            agg[["dst_type", "weight"]].rename(columns={"dst_type": "type"}),
        ],
        ignore_index=True,
    ).groupby("type")["weight"].sum().sort_values(ascending=False)
    required = {"L1", "L2", "L3", "Mi1", "Mi4", "Tm3", "T4a", "T5a", "LC10a", "LPLC4"}
    top_nodes = set(degree.head(36).index).union(required)
    plot_edges = (
        agg[agg["src_type"].isin(top_nodes) & agg["dst_type"].isin(top_nodes) & (agg["src_type"] != agg["dst_type"])]
        .head(95)
        .copy()
    )

    graph = nx.DiGraph()
    for node in sorted(set(plot_edges["src_type"]).union(plot_edges["dst_type"])):
        graph.add_node(node)
    for row in plot_edges.itertuples(index=False):
        graph.add_edge(row.src_type, row.dst_type, weight=float(np.log1p(row.weight)))

    pos_raw = nx.spring_layout(graph, seed=17, weight="weight", k=0.72, iterations=250)
    xs = np.array([p[0] for p in pos_raw.values()])
    ys = np.array([p[1] for p in pos_raw.values()])
    xspan = max(float(xs.max() - xs.min()), 1e-9)
    yspan = max(float(ys.max() - ys.min()), 1e-9)
    pos = {
        n: (0.08 + 0.84 * (p[0] - xs.min()) / xspan, 0.16 + 0.72 * (p[1] - ys.min()) / yspan)
        for n, p in pos_raw.items()
    }

    max_w = max(float(plot_edges["weight"].max()), 1.0)
    for row in plot_edges.sort_values("weight").itertuples(index=False):
        x0, y0 = pos[row.src_type]
        x1, y1 = pos[row.dst_type]
        strength = math.sqrt(float(row.weight) / max_w)
        ax.plot([x0, x1], [y0, y1], color="#8f8f8f", lw=0.25 + 2.4 * strength, alpha=0.10 + 0.38 * strength, zorder=1)

    node_df = pd.DataFrame({"type": list(graph.nodes)})
    node_df["degree"] = node_df["type"].map(degree).fillna(0.0)
    node_df = node_df.sort_values("degree", ascending=False)

    def type_color(name: str) -> str:
        if name.startswith("L") and not name.startswith(("LC", "LPLC", "LT")):
            return "#4a90c2"
        if name.startswith(("Mi", "Tm", "Dm")):
            return "#d9903d"
        if name.startswith(("T4", "T5")):
            return "#4f9a55"
        if name.startswith(("LC", "LPLC", "LT", "DN")):
            return "#8d6ab8"
        return "#9a9a9a"

    sizes = 16 + 92 * np.sqrt(node_df["degree"].to_numpy() / max(float(node_df["degree"].max()), 1.0))
    ax.scatter(
        [pos[n][0] for n in node_df["type"]],
        [pos[n][1] for n in node_df["type"]],
        s=sizes,
        c=[type_color(n) for n in node_df["type"]],
        edgecolors="white",
        linewidths=0.45,
        alpha=0.95,
        zorder=3,
    )
    label_nodes = set(node_df.head(13)["type"]).union({"L1", "L2", "Mi1", "T4a", "T5a", "LC10a", "LPLC4"})
    for name in sorted(label_nodes.intersection(pos)):
        x, y = pos[name]
        ax.text(
            x + 0.012,
            y + 0.010,
            name,
            fontsize=6.6,
            color=DARK,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=0.5),
            zorder=4,
        )
    ax.text(0.50, 0.07, "Observed type-type graph; node size and edge width scale with synapse weight", ha="center", fontsize=8.3, color=GREY)


def fig1_conceptual_schematic() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.2))
    for ax in axs.ravel():
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax = axs[0, 0]
    panel_label(ax, "A")
    ax.set_title("Observed typed-core subgraph", loc="left", pad=8)
    draw_observed_type_subgraph(ax)

    ax = axs[0, 1]
    panel_label(ax, "B")
    ax.set_title("Cell-type channel abstraction", loc="left", pad=8)
    sources = ["Lamina", "Medulla", "Tm/Mi"]
    targets = ["ME", "LOP", "Output"]
    for i, s in enumerate(sources):
        y = 0.76 - i * 0.23
        draw_box(ax, (0.06, y - 0.07), (0.24, 0.12), s, BLUE)
    for i, t in enumerate(targets):
        y = 0.76 - i * 0.23
        draw_box(ax, (0.70, y - 0.07), (0.24, 0.12), t, PURPLE)
    weights = np.array([[0.9, 0.1, 0.05], [0.35, 0.55, 0.18], [0.1, 0.5, 0.7]])
    for i in range(3):
        for j in range(3):
            lw = 0.8 + 3.0 * weights[i, j]
            alpha = 0.16 + 0.55 * weights[i, j]
            arrow(ax, (0.31, 0.76 - i * 0.23), (0.69, 0.76 - j * 0.23), DARK, rad=0.05 * (j - i))
            ax.patches[-1].set_linewidth(lw)
            ax.patches[-1].set_alpha(alpha)
    ax.text(0.50, 0.10, r"$P(S,D)\propto$ synapse weight", ha="center", fontsize=12)

    ax = axs[1, 0]
    panel_label(ax, "C")
    ax.set_title("History ambiguity", loc="left", pad=8)
    labels = ["S1", "S2", "S3", "S4"]
    low = np.array([0.78, 0.10, 0.07, 0.05])
    high = np.array([0.28, 0.25, 0.24, 0.23])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, low, 0.34, color=BLUE, label="Specific target")
    ax.bar(x + 0.18, high, 0.34, color=ORANGE, label="Integrative target")
    ax.set_axis_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$P(S\mid D)$")
    ax.set_ylim(0, 0.88)
    ax.grid(axis="y", color="#dddddd")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)

    ax = axs[1, 1]
    panel_label(ax, "D")
    ax.set_title("Visual hierarchy mapping", loc="left", pad=8)
    stages = [("Lamina", "low H", BLUE), ("Medulla", "intermediate", ORANGE), ("LOP", "motion", GREEN), ("Output", "high H", PURPLE)]
    xs = [0.12, 0.38, 0.64, 0.88]
    for i, ((name, desc, color), x0) in enumerate(zip(stages, xs)):
        ax.add_patch(Circle((x0, 0.58), 0.075, facecolor=color, edgecolor="white", linewidth=2.0))
        ax.text(x0, 0.58, name, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(x0, 0.36, desc, ha="center", va="center", fontsize=10, color=DARK)
        if i < len(stages) - 1:
            arrow(ax, (x0 + 0.08, 0.58), (xs[i + 1] - 0.08, 0.58), GREY)
    ax.text(0.12, 0.23, "lower ambiguity", ha="center", fontsize=10, color=BLUE)
    ax.text(0.88, 0.23, "higher ambiguity", ha="center", fontsize=10, color=PURPLE)
    arrow(ax, (0.25, 0.23), (0.75, 0.23), GREY)

    fig.suptitle("Typed connectomes as weighted cell-type channels", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(wspace=0.42, hspace=0.48, right=0.96, bottom=0.08, top=0.91)
    save_all_main(fig, "fig1_conceptual_schematic", aliases=["fig5_concept_summary"])


def fig2_roi_ambiguity() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/authoritative_channel_metrics_with_ci.csv")
    df = df[df["group"].isin(["ME(R)", "LO(R)", "LOP(R)", "OL(R)"])].copy()
    df = df.sort_values("H_source_given_target_bits", ascending=True)
    vals = df["H_source_given_target_bits"].to_numpy()
    lo = vals - df["H_source_given_target_ci_low"].to_numpy()
    hi = df["H_source_given_target_ci_high"].to_numpy() - vals
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [BLUE, GREEN, ORANGE, PURPLE][: len(df)]
    ax.barh(y, vals, color=colors, height=0.62, alpha=0.93)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=DARK, elinewidth=1.1, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped(df["group"], width=14))
    ax.set_xlabel(r"Cell-type history ambiguity, $H(S\mid D)$ (bits)")
    ax.set_title("Cell-type history ambiguity across optic-lobe compartments", loc="left", fontweight="bold")
    for yi, v, pre in zip(y, vals, df["effective_preimage_size"]):
        ax.text(v + 0.05, yi, f"{v:.2f} bits  |  $2^H$={pre:.1f}", va="center", fontsize=10)
    ax.set_xlim(0, max(vals) + 1.70)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.24, top=0.84)
    save_all_main(fig, "fig2_roi_ambiguity", aliases=["fig1_roi_history_ambiguity"])


def fig3_pathway_hierarchy() -> None:
    df = pd.read_csv(ROOT / "outputs/known_visual_pathway_validation/known_visual_pathway_validation.csv")
    curated = df[df["group_kind"].eq("curated_pathway")].sort_values("stage_order").copy()
    directional = df[df["group_kind"].eq("directional_stage")].sort_values("stage_order").copy()
    curated_labels = ["Lamina\ninput", "Medulla\nintrinsic", "T4/T5 +\npartners", "Output /\nprojection"]
    dir_labels = ["Lamina→ME", "ME→LO", "ME→LOP", "LO/LOP→Out"]

    fig, axs = plt.subplots(1, 3, figsize=(16.4, 5.3), gridspec_kw={"width_ratios": [1.08, 1.16, 1.04]})

    ax = axs[0]
    panel_label(ax, "A")
    x = np.arange(len(curated))
    y = curated["history_ambiguity_bits"].to_numpy()
    ax.plot(x, y, color=BLUE, marker="o", lw=2.4, ms=7)
    ax.fill_between(x, 0, y, color=BLUE, alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(curated_labels)
    angle_xlabels(ax, 18)
    ax.set_ylabel(r"$H(S\mid D)$ (bits)")
    ax.set_title("Ordered pathway groups", loc="left", fontweight="bold")
    ax.set_ylim(0, max(y) + 0.9)

    ax = axs[1]
    panel_label(ax, "B")
    obs = directional["history_ambiguity_bits"].to_numpy()
    null = directional["degree_null_mean_history_ambiguity_bits"].to_numpy()
    null_sd = directional["degree_null_std_history_ambiguity_bits"].to_numpy()
    x = np.arange(len(directional))
    ax.errorbar(x - 0.08, null, yerr=null_sd, fmt="s", color=GREY, ecolor=GREY, capsize=3, label="Degree null")
    ax.plot(x + 0.08, obs, "o", color=ORANGE, label="Observed")
    for xi, o, n in zip(x, obs, null):
        ax.plot([xi - 0.08, xi + 0.08], [n, o], color="#bbbbbb", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(dir_labels, rotation=25, ha="right")
    ax.set_ylabel(r"$H(S\mid D)$ (bits)")
    ax.set_title("Observed vs matched null", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    ax = axs[2]
    panel_label(ax, "C")
    y = directional["history_ambiguity_bits"].to_numpy()
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    ax.bar(np.arange(len(y)), y, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(y)))
    ax.set_xticklabels(dir_labels)
    angle_xlabels(ax, 35)
    ax.set_ylabel(r"$H(S\mid D)$ (bits)")
    ax.set_title("Directional hierarchy", loc="left", fontweight="bold")
    ax.set_ylim(0, max(y) + 0.75)

    fig.suptitle("History ambiguity follows known visual pathway organization", fontsize=16, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.30, top=0.82, wspace=0.54)
    save_all_main(fig, "fig3_pathway_hierarchy", aliases=["fig2_visual_pathway_hierarchy"])


def fig4_motion_separability() -> None:
    summary = pd.read_csv(ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/final_control_mode_summary.csv")
    decisive = pd.read_csv(ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/final_control_decisive_table.csv")
    curves = pd.read_csv(ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/learning_curves.csv")
    keep = ["real", "degree_preserving_null", "weight_shuffled_real_topology", "weight_uniform_real_topology"]
    labels = {
        "real": "Real",
        "degree_preserving_null": "Degree\nnull",
        "weight_shuffled_real_topology": "Weight\nshuffled",
        "weight_uniform_real_topology": "Uniform\nweight",
    }
    modes = ["readout_only", "local_hebbian"]
    mode_labels = {"readout_only": "Readout only", "local_hebbian": "Local Hebbian"}
    mode_colors = {"readout_only": BLUE, "local_hebbian": ORANGE}

    fig, axs = plt.subplots(1, 3, figsize=(16.8, 5.4), gridspec_kw={"width_ratios": [1.18, 1.0, 1.22]})

    ax = axs[0]
    panel_label(ax, "A")
    x = np.arange(len(keep))
    width = 0.34
    for i, mode in enumerate(modes):
        sub = summary[summary["mode"].eq(mode)].set_index("condition").loc[keep]
        xpos = x + (i - 0.5) * width
        vals = sub["accuracy_mean"].to_numpy()
        se = sub["accuracy_sd"].to_numpy() / np.sqrt(sub["n"].to_numpy())
        ax.bar(xpos, vals, width, color=mode_colors[mode], label=mode_labels[mode], alpha=0.92)
        ax.errorbar(xpos, vals, yerr=se, fmt="none", ecolor=DARK, elinewidth=1.0, capsize=2.5)
    ax.axhline(0.25, color=DARK, lw=1.1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keep])
    angle_xlabels(ax, 25)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.84)
    ax.set_title("Accuracy by graph condition", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    ax = axs[1]
    panel_label(ax, "B")
    view = decisive[decisive["condition_comparison"].str.startswith("real_vs")].copy()
    view = view[~view["condition_comparison"].str.contains("readout")]
    comp_order = [
        "real_vs_degree_preserving_null",
        "real_vs_weight_shuffled_real_topology",
        "real_vs_weight_uniform_real_topology",
    ]
    comp_labels = ["Degree\nnull", "Weight\nshuffled", "Uniform\nweight"]
    view = view.set_index("condition_comparison").loc[comp_order].reset_index()
    y = np.arange(len(view))
    vals = view["delta_MI"].to_numpy()
    lo = vals - view["delta_MI_ci_low"].to_numpy()
    hi = view["delta_MI_ci_high"].to_numpy() - vals
    ax.barh(y, vals, color=[GREEN if v > 0 else GREY for v in vals], alpha=0.9)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=DARK, elinewidth=1.0, capsize=3)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped(comp_labels, width=14))
    ax.set_xlabel(r"Real − control $\Delta I$ (bits)")
    ax.set_title("Task-information contrast", loc="left", fontweight="bold")

    ax = axs[2]
    panel_label(ax, "C")
    line_specs = [
        ("real", "readout_only", BLUE, "Real, readout"),
        ("degree_preserving_null", "readout_only", GREY, "Degree null, readout"),
        ("real", "local_hebbian", ORANGE, "Real, Hebbian"),
    ]
    for cond, mode, color, label in line_specs:
        sub = curves[(curves["condition"].eq(cond)) & (curves["mode"].eq(mode))]
        agg = sub.groupby("epoch", as_index=False).agg(mean=("accuracy", "mean"), se=("accuracy", lambda s: s.std(ddof=1) / math.sqrt(len(s))))
        ls = ":" if mode == "local_hebbian" else "-"
        ax.plot(agg["epoch"], agg["mean"], color=color, lw=2.4, ls=ls, label=label)
        ax.fill_between(agg["epoch"], agg["mean"] - agg["se"], agg["mean"] + agg["se"], color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.25, color=DARK, lw=1.0, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning curves", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    ax.set_ylim(0.18, 0.78)

    fig.suptitle("Medulla-to-lobula-plate supports synthetic motion separability, not plasticity-driven learning", fontsize=16, fontweight="bold", y=1.03)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.34, top=0.80, wspace=0.52)
    save_all_main(fig, "fig4_motion_separability", aliases=["fig3_motion_separability_controls"])


def fig5_plasticity_failure() -> None:
    rs = pd.read_csv(ROOT / "outputs/rule_discovery/rule_search_results.csv")
    disc = pd.read_csv(ROOT / "outputs/rule_discovery/discovery_summary.csv")
    rep = rs[(rs["condition"].eq("real")) & (rs["protocol"].eq("representation_probe"))].copy()

    fig, axs = plt.subplots(1, 3, figsize=(17.0, 5.5), gridspec_kw={"width_ratios": [0.95, 1.35, 1.55]})

    standard = ["no_plasticity", "hebbian", "oja", "bcm"]
    labels = ["Readout\nonly", "Hebbian", "Oja", "BCM"]
    agg = rep[rep["rule_name"].isin(standard)].groupby("rule_name", as_index=False).agg(
        acc=("final_accuracy", "mean"),
        se=("final_accuracy", lambda s: s.std(ddof=1) / math.sqrt(len(s))),
    )
    agg["rule_name"] = pd.Categorical(agg["rule_name"], categories=standard, ordered=True)
    agg = agg.sort_values("rule_name")
    ax = axs[0]
    panel_label(ax, "A")
    x = np.arange(len(agg))
    ax.bar(x, agg["acc"], color=[BLUE, ORANGE, GREEN, PURPLE], alpha=0.9)
    ax.errorbar(x, agg["acc"], yerr=agg["se"], fmt="none", ecolor=DARK, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    angle_xlabels(ax, 20)
    ax.set_ylabel("Probe accuracy")
    ax.set_ylim(0, 0.9)
    ax.set_title("Standard local rules", loc="left", fontweight="bold")

    selected = ["no_plasticity", "hebbian", "oja", "bcm", "reward_hebbian", "ambiguity_gated_hebbian", "ambiguity_gated_oja", "energy_normalized_ambiguity_oja"]
    names = {
        "no_plasticity": "Readout",
        "hebbian": "Hebbian",
        "oja": "Oja",
        "bcm": "BCM",
        "reward_hebbian": "Reward Hebb.",
        "ambiguity_gated_hebbian": "Ambig.-gated Hebb.",
        "ambiguity_gated_oja": "Ambig.-gated Oja",
        "energy_normalized_ambiguity_oja": "Energy-norm. Oja",
    }
    agg = rep[rep["rule_name"].isin(selected)].groupby("rule_name", as_index=False).agg(
        mi=("plasticity_delta_MI", "mean"),
        se=("plasticity_delta_MI", lambda s: s.std(ddof=1) / math.sqrt(len(s))),
    )
    agg["rule_name"] = pd.Categorical(agg["rule_name"], categories=selected, ordered=True)
    agg = agg.sort_values("rule_name")
    ax = axs[1]
    panel_label(ax, "B")
    y = np.arange(len(agg))
    ax.scatter(agg["mi"], y, color=PURPLE, s=46, zorder=3)
    ax.errorbar(agg["mi"], y, xerr=0.001, fmt="none", ecolor=PURPLE, alpha=0.45, capsize=2, zorder=2)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped([names[r] for r in agg["rule_name"]], width=18), fontsize=8.5)
    ax.set_xlabel(r"Plasticity $\Delta I$ (bits)")
    ax.set_title("Representation-probe gain", loc="left", fontweight="bold")
    ax.set_xlim(-0.004, 0.004)

    rank = disc[disc["protocol"].eq("representation_probe")].head(10).copy()
    rank["plot_name"] = rank["rule_name"].str.replace("_", " ", regex=False)
    ax = axs[2]
    panel_label(ax, "C")
    y = np.arange(len(rank))[::-1]
    vals = rank["score"].to_numpy()[::-1]
    ax.barh(y, vals, color=[BLUE if "no plasticity" in n else GREY for n in rank["plot_name"].to_numpy()[::-1]], alpha=0.9)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped(rank["plot_name"].to_numpy()[::-1], width=24), fontsize=8.3)
    ax.set_xlabel("Rule-search score")
    ax.set_title("Rule-search ranking", loc="left", fontweight="bold")

    fig.suptitle("No tested rule improves the fixed representation", fontsize=16, fontweight="bold", y=1.03)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.80, wspace=0.58)
    save_all_main(fig, "fig5_plasticity_failure", aliases=["fig4_plasticity_rule_search_failure"])


def fig4_target_type_integrative_blocks() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/target_type_distribution_summary.csv")
    top = df.sort_values("history_ambiguity_bits", ascending=False).head(14).iloc[::-1].copy()
    y = np.arange(len(top))
    vals = top["history_ambiguity_bits"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    ax.barh(y, vals, color=PURPLE, alpha=0.88, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped(top["dst_type"], width=18), fontsize=9)
    ax.set_xlabel(r"Target-specific history ambiguity, $H(S\mid D=d)$ (bits)")
    ax.set_title("High-ambiguity target types", loc="left", fontweight="bold")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for yi, v, pre in zip(y, vals, top["effective_preimage_size"].to_numpy()):
        ax.text(v + 0.07, yi, rf"$2^H$={pre:.0f}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) + 1.20)
    fig.subplots_adjust(left=0.22, right=0.94, bottom=0.14, top=0.90)
    save_all_main(fig, "fig4_target_type_integrative_blocks")


def sfig1_null_model_diagram() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/null_ensemble_and_multiplicity.csv")
    h = df[
        df["metric"].eq("block_history_ambiguity")
        & df["source_table"].eq("typed_core_global")
        & df["null_model"].isin(["degree_preserving_directed_rewire", "target_preserving_source_shuffle", "type_label_permutation"])
    ].copy()
    labels = {
        "degree_preserving_directed_rewire": "Degree-preserving",
        "target_preserving_source_shuffle": "Target-preserving\nsource shuffle",
        "type_label_permutation": "Type-label\npermutation",
    }
    h["label"] = h["null_model"].map(labels)

    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.8), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})
    ax = axs[0]
    panel_label(ax, "A")
    x = np.arange(len(h))
    ax.bar(x - 0.17, h["observed"], 0.34, color=BLUE, label="Observed")
    ax.bar(x + 0.17, h["null_mean"], 0.34, color=GREY, label="Null mean")
    ax.errorbar(x + 0.17, h["null_mean"], yerr=h["null_std"], fmt="none", ecolor=DARK, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(h["label"])
    angle_xlabels(ax, 25)
    ax.set_ylabel(r"$H(S\mid D)$ (bits)")
    ax.set_title("Global null comparison", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2)

    ax = axs[1]
    panel_label(ax, "B")
    colors = [BLUE if v < 0 else ORANGE for v in h["observed_minus_null_mean"]]
    ax.barh(np.arange(len(h)), h["observed_minus_null_mean"], color=colors, alpha=0.9)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_yticks(np.arange(len(h)))
    ax.set_yticklabels(h["label"], fontsize=9)
    ax.set_xlabel("Observed − null (bits)")
    ax.set_title("Effect size", loc="left", fontweight="bold")

    ax = axs[2]
    panel_label(ax, "C")
    roi = pd.read_csv(ROOT / "outputs/roi_pathway/roi_pathway_null_summary.csv")
    plot = roi[
        roi["metric"].eq("history_ambiguity_bits")
        & roi["null_model"].eq("degree_preserving_directed_rewire")
        & roi["group"].isin(["ME(R)", "LO(R)", "LOP(R)"])
    ].sort_values("group")
    y = np.arange(len(plot))
    ax.errorbar(plot["null_mean"], y, xerr=plot["null_std"], fmt="s", color=GREY, capsize=3, label="Degree null")
    ax.scatter(plot["observed"], y, color=ORANGE, s=42, label="Observed", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["group"])
    ax.set_xlabel(r"$H(S\mid D)$ (bits)")
    ax.set_title("ROI null checks", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2)
    fig.suptitle("Null comparisons are data-derived, not schematic", fontsize=15, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.36, top=0.78, wspace=0.48)
    save_all_supp(fig, "sfig1_null_model_diagram")


def sfig2_budget_sweep() -> None:
    df = pd.read_csv(ROOT / "outputs/learning_signal_rescue/rescue_budget_summary.csv")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for mode, color, label in [("readout_only", BLUE, "Readout only"), ("local_hebbian", ORANGE, "Local Hebbian")]:
        sub = df[df["mode"].eq(mode)].sort_values("matched_edges")
        x = sub["matched_edges"].to_numpy()
        y = sub["mean_real_accuracy"].to_numpy()
        lo = sub["real_accuracy_ci_low"].to_numpy()
        hi = sub["real_accuracy_ci_high"].to_numpy()
        ls = ":" if mode == "local_hebbian" else "-"
        ax.plot(x, y, marker="o", lw=2.4, ms=7, color=color, ls=ls, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0.25, color=DARK, lw=1.1, ls="--")
    ax.set_xlabel("Matched edge budget")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Budget sweep for motion separability", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_ylim(0.20, 0.75)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.27, top=0.88)
    save_all_supp(fig, "sfig2_budget_sweep", aliases=["sfig4_motion_budget_sweep"])


def sfig3_distribution() -> None:
    df = pd.read_csv(ROOT / "outputs/tables/target_type_distribution_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.3), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axs[0]
    panel_label(ax, "A")
    vals = df["history_ambiguity_bits"].dropna()
    ax.hist(vals, bins=32, density=True, color=BLUE, alpha=0.72, edgecolor="white")
    vals.plot(kind="kde", ax=ax, color=DARK, lw=2.0)
    ax.set_xlabel(r"Target-type $H(S\mid D)$ (bits)")
    ax.set_ylabel("Density")
    ax.set_title("Target-type ambiguity distribution", loc="left", fontweight="bold")
    ax = axs[1]
    panel_label(ax, "B")
    top = df.sort_values("history_ambiguity_bits", ascending=False).head(15).iloc[::-1]
    ax.barh(np.arange(len(top)), top["history_ambiguity_bits"], color=PURPLE, alpha=0.88)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(wrapped(top["dst_type"], width=22), fontsize=8.4)
    ax.set_xlabel(r"$H(S\mid D)$ (bits)")
    ax.set_title("Highest-ambiguity target types", loc="left", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.48)
    save_all_supp(fig, "sfig3_target_type_distribution")


def sfig4_scatter() -> None:
    roi = pd.read_csv(ROOT / "outputs/tables/roi_handling_sensitivity.csv")
    target = pd.read_csv(ROOT / "outputs/tables/target_type_distribution_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axs[0]
    panel_label(ax, "A")
    sub = roi[roi["roi"].isin(["LO(R)", "ME(R)", "LOP(R)"])].copy()
    for handling, group in sub.groupby("roi_handling"):
        ax.scatter(group["H_source_given_target_bits"], group["mutual_information_bits"], s=55, alpha=0.72, label=handling.replace("_", " "))
    ax.set_xlabel(r"$H(S\mid D)$ (bits)")
    ax.set_ylabel(r"$I(S;D)$ (bits)")
    ax.set_title("ROI handling sensitivity", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), fontsize=8, ncol=2)

    ax = axs[1]
    panel_label(ax, "B")
    x = target["log10_total_weight"].to_numpy()
    y = target["history_ambiguity_bits"].to_numpy()
    ax.scatter(x, y, s=18, alpha=0.28, color=GREY, edgecolor="none")
    m, b = np.polyfit(x, y, deg=1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xx, m * xx + b, color=ORANGE, lw=2.2, label="linear trend")
    ax.set_xlabel(r"log$_{10}$(target input weight)")
    ax.set_ylabel(r"Target-type $H(S\mid D)$ (bits)")
    ax.set_title("Weight vs ambiguity", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=1)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.88, wspace=0.42)
    save_all_supp(fig, "sfig4_scatter_diagnostics")


def main() -> None:
    setup_style()
    ensure_dirs()
    fig1_conceptual_schematic()
    fig2_roi_ambiguity()
    fig3_pathway_hierarchy()
    fig4_motion_separability()
    fig5_plasticity_failure()
    fig4_target_type_integrative_blocks()
    sfig1_null_model_diagram()
    sfig2_budget_sweep()
    sfig3_distribution()
    sfig4_scatter()
    print("Wrote Nature/npj-style main and supplementary figures.")


if __name__ == "__main__":
    main()
