#!/usr/bin/env python
"""Make a real neuron-level optic-lobe backbone figure.

This figure is intentionally different from the conceptual type-channel
schematic: nodes are typed-core body IDs, edges are real synapse-weighted
connections, and the layout is computed from a weighted graph backbone.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

BLUE = "#1f77b4"
GREEN = "#228B22"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREY = "#b8b8b8"
DARK = "#222222"

ROI_COLORS = {"ME(R)": BLUE, "LO(R)": GREEN, "LOP(R)": ORANGE, "DN": RED}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_backbone() -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    nodes = pd.read_parquet(ROOT / "outputs/processed_typed_core/nodes.parquet")
    edges = pd.read_parquet(
        ROOT / "outputs/processed_typed_core/edges.parquet",
        columns=["src", "dst", "weight", "src_major_roi", "dst_major_roi"],
    )
    rois = ["ME(R)", "LO(R)", "LOP(R)"]
    roi_nodes = nodes[nodes["major_roi"].isin(rois)].copy()

    degree = pd.concat(
        [
            edges[["src", "weight"]].rename(columns={"src": "bodyId"}),
            edges[["dst", "weight"]].rename(columns={"dst": "bodyId"}),
        ],
        ignore_index=True,
    ).groupby("bodyId")["weight"].sum()

    caps = {"ME(R)": 1500, "LO(R)": 750, "LOP(R)": 650}
    selected: set[int] = set()
    for roi, cap in caps.items():
        roi_ids = roi_nodes.loc[roi_nodes["major_roi"].eq(roi), "bodyId"]
        ranked = degree.reindex(roi_ids).fillna(0).sort_values(ascending=False).head(cap)
        selected.update(int(x) for x in ranked.index)

    dn_ids = set(nodes.loc[nodes["type"].fillna("").str.startswith("DN"), "bodyId"].astype(int))
    dn_in = edges[edges["dst"].isin(dn_ids) & edges["src_major_roi"].isin(rois)]
    target_ids = set(dn_in.groupby("dst")["weight"].sum().sort_values(ascending=False).head(18).index.astype(int))
    selected.update(target_ids)

    node_table = nodes[nodes["bodyId"].isin(selected)].copy()
    node_table["plot_group"] = node_table["major_roi"]
    node_table.loc[node_table["bodyId"].isin(target_ids), "plot_group"] = "DN"
    node_table["degree_weight"] = degree.reindex(node_table["bodyId"]).fillna(0).to_numpy()

    body_edges = edges[edges["src"].isin(selected) & edges["dst"].isin(selected)].copy()
    target_edges = body_edges[body_edges["dst"].isin(target_ids)].sort_values("weight", ascending=False).head(700)
    non_target = body_edges[~body_edges["dst"].isin(target_ids)].sort_values("weight", ascending=False).head(10000)
    plot_edges = pd.concat([non_target, target_edges], ignore_index=True).drop_duplicates(["src", "dst"])

    # Keep the visible figure focused on the connected backbone rather than on
    # high-degree neurons that have no retained edge after backbone filtering.
    graph = nx.Graph()
    graph.add_edges_from((int(r.src), int(r.dst)) for r in plot_edges.itertuples(index=False))
    if graph.number_of_nodes():
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        connected_ids = set(components[0])
        target_ids = target_ids & connected_ids
        plot_edges = plot_edges[plot_edges["src"].isin(connected_ids) & plot_edges["dst"].isin(connected_ids)].copy()
        node_table = node_table[node_table["bodyId"].isin(connected_ids)].copy()
    return node_table, plot_edges, target_ids


def compute_layout(nodes: pd.DataFrame, edges: pd.DataFrame, target_ids: set[int]) -> dict[int, tuple[float, float]]:
    graph = nx.Graph()
    for body in nodes["bodyId"].astype(int):
        graph.add_node(int(body))
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.src), int(row.dst), weight=float(math.log1p(row.weight)))

    rng = np.random.default_rng(7)
    centers = {
        "ME(R)": np.array([-0.38, 0.08]),
        "LO(R)": np.array([0.12, 0.02]),
        "LOP(R)": np.array([0.30, -0.18]),
        "DN": np.array([0.92, 0.04]),
    }
    init = {}
    for row in nodes.itertuples(index=False):
        center = centers.get(row.plot_group, np.array([0.0, 0.0]))
        spread = 0.22 if row.plot_group != "DN" else 0.10
        init[int(row.bodyId)] = center + rng.normal(0, spread, size=2)

    pos = nx.spring_layout(graph, pos=init, seed=9, weight="weight", k=0.040, iterations=120)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    xspan = max(float(xs.max() - xs.min()), 1e-9)
    yspan = max(float(ys.max() - ys.min()), 1e-9)
    return {n: ((p[0] - xs.min()) / xspan, (p[1] - ys.min()) / yspan) for n, p in pos.items()}


def draw_panel(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    pos: dict[int, tuple[float, float]],
    target_ids: set[int],
    keep_group: str | None,
    label: str,
) -> None:
    ax.set_axis_off()
    if keep_group is None:
        keep = set(nodes["bodyId"].astype(int))
    else:
        keep = set(nodes.loc[nodes["plot_group"].eq(keep_group), "bodyId"].astype(int)) | target_ids

    sub_edges = edges[edges["src"].isin(keep) & edges["dst"].isin(keep)].copy()
    target_mask = sub_edges["dst"].isin(target_ids)
    connected_targets = set(sub_edges.loc[target_mask, "dst"].astype(int))

    def line_segments(frame: pd.DataFrame) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return [(pos[int(r.src)], pos[int(r.dst)]) for r in frame.itertuples(index=False) if int(r.src) in pos and int(r.dst) in pos]

    grey_edges = sub_edges[~target_mask]
    red_edges = sub_edges[target_mask]
    if len(grey_edges):
        widths = 0.08 + 0.75 * np.sqrt(grey_edges["weight"].to_numpy() / max(float(grey_edges["weight"].max()), 1.0))
        lc = LineCollection(line_segments(grey_edges), colors=GREY, linewidths=widths, alpha=0.30, zorder=1)
        ax.add_collection(lc)
    if len(red_edges):
        widths = 0.22 + 1.25 * np.sqrt(red_edges["weight"].to_numpy() / max(float(red_edges["weight"].max()), 1.0))
        lc = LineCollection(line_segments(red_edges), colors=RED, linewidths=widths, alpha=0.72, zorder=2)
        ax.add_collection(lc)

    plot_nodes = nodes[nodes["bodyId"].isin(keep)].copy()
    if keep_group is not None:
        plot_nodes = plot_nodes[
            plot_nodes["plot_group"].eq(keep_group) | plot_nodes["bodyId"].isin(connected_targets)
        ].copy()
    for group, color in ROI_COLORS.items():
        sub = plot_nodes[plot_nodes["plot_group"].eq(group)]
        if sub.empty:
            continue
        size_base = 2.4 if group != "DN" else 20.0
        sizes = size_base + 18.0 * np.sqrt(sub["degree_weight"].to_numpy() / max(float(nodes["degree_weight"].max()), 1.0))
        ax.scatter(
            [pos[int(x)][0] for x in sub["bodyId"]],
            [pos[int(x)][1] for x in sub["bodyId"]],
            s=sizes,
            c=color,
            edgecolors="black" if group == "DN" else "none",
            linewidths=0.35,
            alpha=0.92 if group == "DN" else 0.86,
            zorder=3 if group != "DN" else 4,
        )

    ax.text(0.01, 0.99, f"({label})", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold", color=DARK)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    nodes, edges, target_ids = load_backbone()
    pos = compute_layout(nodes, edges, target_ids)

    fig = plt.figure(figsize=(12.4, 7.1))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.88],
        height_ratios=[1.05, 0.80],
        hspace=0.20,
        wspace=0.10,
    )
    ax_a = fig.add_subplot(gs[0, :3])
    ax_leg = fig.add_subplot(gs[0, 3])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_leg.set_axis_off()

    draw_panel(ax_a, nodes, edges, pos, target_ids, None, "a")
    draw_panel(ax_b, nodes, edges, pos, target_ids, "ME(R)", "b")
    draw_panel(ax_c, nodes, edges, pos, target_ids, "LO(R)", "c")
    draw_panel(ax_d, nodes, edges, pos, target_ids, "LOP(R)", "d")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=6, label="Medulla neurons"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=6, label="Lobula neurons"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=6, label="Lobula plate neurons"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor=RED, markeredgewidth=0.4, markersize=8, label=r"Descending neurons (target set $\mathcal{T}$)"),
        Line2D([0], [0], color=GREY, lw=1.5, label="Backbone edges"),
        Line2D([0], [0], color=RED, lw=2.2, label="Backbone edges terminating in targets"),
    ]
    ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=10, handlelength=1.4)
    fig.suptitle("Neuron-level optic-lobe backbone to descending targets", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(OUT / "fig6_connectome_backbone_network.pdf", bbox_inches="tight")
    print(f"Wrote {OUT / 'fig6_connectome_backbone_network.pdf'}")


if __name__ == "__main__":
    main()
