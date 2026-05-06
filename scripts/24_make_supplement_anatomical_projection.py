#!/usr/bin/env python
"""Supplementary anatomical-style projection of optic-lobe target fan-in.

The local processed tables do not contain skeleton/soma coordinates. This
figure therefore uses real neurons and real synapse-weighted edges, but places
neurons in an ROI-constrained 2D projection rather than in physical brain
coordinates. The goal is visual auditability: readers can see which ROI groups
feed the descending target set and which source types dominate that fan-in.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "supplement_figures"

BLUE = "#1f77b4"
GREEN = "#228B22"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREY = "#9a9a9a"
DARK = "#222222"

ROIS = ["ME(R)", "LO(R)", "LOP(R)"]
ROI_COLORS = {"ME(R)": BLUE, "LO(R)": GREEN, "LOP(R)": ORANGE}
ROI_NAMES = {"ME(R)": "Medulla", "LO(R)": "Lobula", "LOP(R)": "Lobula plate"}
ROI_GEOMETRY = {
    "ME(R)": {"center": (0.28, 0.56), "width": 0.43, "height": 0.62, "angle": -13},
    "LO(R)": {"center": (0.56, 0.58), "width": 0.34, "height": 0.47, "angle": 7},
    "LOP(R)": {"center": (0.69, 0.35), "width": 0.38, "height": 0.27, "angle": -18},
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_projection_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[int]]:
    nodes = pd.read_parquet(ROOT / "outputs/processed_typed_core/nodes.parquet")
    edges = pd.read_parquet(
        ROOT / "outputs/processed_typed_core/edges.parquet",
        columns=["src", "dst", "weight", "src_major_roi", "dst_major_roi"],
    )
    node_lookup = nodes.set_index("bodyId")

    dn_ids = set(nodes.loc[nodes["type"].fillna("").str.startswith("DN"), "bodyId"].astype(int))
    target_edge_pool = edges[edges["dst"].isin(dn_ids) & edges["src_major_roi"].isin(ROIS)].copy()
    target_ids = set(
        target_edge_pool.groupby("dst")["weight"].sum().sort_values(ascending=False).head(18).index.astype(int)
    )

    target_edges = target_edge_pool[target_edge_pool["dst"].isin(target_ids)].copy()
    target_source_score = target_edges.groupby("src")["weight"].sum()
    degree = pd.concat(
        [
            edges[["src", "weight"]].rename(columns={"src": "bodyId"}),
            edges[["dst", "weight"]].rename(columns={"dst": "bodyId"}),
        ],
        ignore_index=True,
    ).groupby("bodyId")["weight"].sum()

    selected: set[int] = set(target_ids)
    caps = {"ME(R)": 1000, "LO(R)": 720, "LOP(R)": 520}
    for roi, cap in caps.items():
        roi_ids = nodes.loc[nodes["major_roi"].eq(roi), "bodyId"].astype(int)
        score = 7.0 * target_source_score.reindex(roi_ids).fillna(0) + 0.025 * degree.reindex(roi_ids).fillna(0)
        selected.update(int(x) for x in score.sort_values(ascending=False).head(cap).index)

    plot_nodes = nodes[nodes["bodyId"].isin(selected)].copy()
    plot_nodes["degree_weight"] = degree.reindex(plot_nodes["bodyId"]).fillna(0).to_numpy()
    plot_nodes["target_input_weight"] = target_source_score.reindex(plot_nodes["bodyId"]).fillna(0).to_numpy()
    plot_nodes["plot_group"] = plot_nodes["major_roi"]
    plot_nodes.loc[plot_nodes["bodyId"].isin(target_ids), "plot_group"] = "DN"

    body_edges = edges[edges["src"].isin(selected) & edges["dst"].isin(selected)].copy()
    context_edges = body_edges[~body_edges["dst"].isin(target_ids)].sort_values("weight", ascending=False).head(5200)
    red_edges = body_edges[body_edges["dst"].isin(target_ids)].sort_values("weight", ascending=False).head(1300)
    plot_edges = pd.concat([context_edges, red_edges], ignore_index=True).drop_duplicates(["src", "dst"])

    target_edges = red_edges.copy()
    target_edges["src_type"] = target_edges["src"].map(node_lookup["type"]).fillna("unknown")
    target_edges["src_roi"] = target_edges["src"].map(node_lookup["major_roi"]).fillna("unknown")
    target_edges["target_type"] = target_edges["dst"].map(node_lookup["type"]).fillna("DN")
    target_edges["target_instance"] = target_edges["dst"].map(node_lookup["instance"]).fillna(target_edges["target_type"])
    return plot_nodes, plot_edges, target_edges, target_ids


def ellipse_points(
    n: int,
    center: tuple[float, float],
    width: float,
    height: float,
    angle_deg: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    rank = np.arange(n)
    theta = rank * golden + rng.normal(0, 0.055, size=n)
    radius = np.sqrt((rank + 0.5) / max(n, 1))
    radius = np.clip(radius + rng.normal(0, 0.018, size=n), 0.02, 0.985)
    x = radius * np.cos(theta) * width / 2.0
    y = radius * np.sin(theta) * height / 2.0
    angle = math.radians(angle_deg)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    xy = np.vstack([x, y]).T @ rot.T
    xy[:, 0] += center[0]
    xy[:, 1] += center[1]
    return xy


def compute_positions(nodes: pd.DataFrame, target_ids: set[int], target_edges: pd.DataFrame) -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    for seed, roi in enumerate(ROIS, start=11):
        sub = nodes[nodes["plot_group"].eq(roi)].sort_values(
            ["target_input_weight", "degree_weight"], ascending=False
        )
        geom = ROI_GEOMETRY[roi]
        xy = ellipse_points(
            len(sub),
            center=geom["center"],
            width=geom["width"],
            height=geom["height"],
            angle_deg=geom["angle"],
            seed=seed,
        )
        for body, point in zip(sub["bodyId"].astype(int), xy):
            pos[int(body)] = (float(point[0]), float(point[1]))

    target_order = (
        target_edges.groupby("dst")["weight"].sum().reindex(list(target_ids)).fillna(0).sort_values(ascending=False)
    )
    yvals = np.linspace(0.80, 0.20, max(len(target_order), 1))
    for body, y in zip(target_order.index.astype(int), yvals):
        pos[int(body)] = (0.92, float(y))
    return pos


def curved_segments(
    edges: pd.DataFrame,
    pos: dict[int, tuple[float, float]],
    curvature: float,
    samples: int = 18,
) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for row in edges.itertuples(index=False):
        src = int(row.src)
        dst = int(row.dst)
        if src not in pos or dst not in pos:
            continue
        p0 = np.array(pos[src])
        p1 = np.array(pos[dst])
        delta = p1 - p0
        dist = max(float(np.linalg.norm(delta)), 1e-6)
        normal = np.array([-delta[1], delta[0]]) / dist
        sign = -1.0 if p0[1] > p1[1] else 1.0
        ctrl = (p0 + p1) / 2.0 + sign * curvature * dist * normal
        t = np.linspace(0, 1, samples)[:, None]
        curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1
        segments.append(curve)
    return segments


def draw_projection(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    target_edges: pd.DataFrame,
    target_ids: set[int],
    pos: dict[int, tuple[float, float]],
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0.03, 1.01)
    ax.set_ylim(0.05, 0.95)

    ax.add_patch(Ellipse((0.49, 0.53), 0.88, 0.78, angle=-7, facecolor="#f7f7f7", edgecolor="#d0d0d0", lw=1.0, zorder=0))
    for roi in ROIS:
        geom = ROI_GEOMETRY[roi]
        ax.add_patch(
            Ellipse(
                geom["center"],
                geom["width"],
                geom["height"],
                angle=geom["angle"],
                facecolor=ROI_COLORS[roi],
                edgecolor=ROI_COLORS[roi],
                alpha=0.075,
                lw=1.3,
                zorder=0.5,
            )
        )
        cx, cy = geom["center"]
        ax.text(cx, cy + geom["height"] / 2.0 + 0.035, ROI_NAMES[roi], color=ROI_COLORS[roi], fontsize=9, fontweight="bold", ha="center")

    red_pairs = set(zip(target_edges["src"].astype(int), target_edges["dst"].astype(int)))
    edge_pairs = list(zip(edges["src"].astype(int), edges["dst"].astype(int)))
    red_mask = np.array([pair in red_pairs for pair in edge_pairs])
    context = edges.loc[~red_mask].copy()
    targets = edges.loc[red_mask].copy()

    if len(context):
        widths = 0.06 + 0.40 * np.sqrt(context["weight"].to_numpy() / max(float(context["weight"].max()), 1.0))
        ax.add_collection(
            LineCollection(curved_segments(context, pos, curvature=0.05), colors=GREY, linewidths=widths, alpha=0.12, zorder=1)
        )
    if len(targets):
        widths = 0.16 + 1.10 * np.sqrt(targets["weight"].to_numpy() / max(float(targets["weight"].max()), 1.0))
        ax.add_collection(
            LineCollection(curved_segments(targets, pos, curvature=0.12), colors=RED, linewidths=widths, alpha=0.38, zorder=2)
        )

    max_degree = max(float(nodes["degree_weight"].max()), 1.0)
    for roi in ROIS:
        sub = nodes[nodes["plot_group"].eq(roi)]
        sizes = 2.0 + 14.0 * np.sqrt(sub["degree_weight"].to_numpy() / max_degree)
        ax.scatter(
            [pos[int(x)][0] for x in sub["bodyId"]],
            [pos[int(x)][1] for x in sub["bodyId"]],
            s=sizes,
            c=ROI_COLORS[roi],
            edgecolors="none",
            alpha=0.78,
            zorder=3,
        )

    targets = nodes[nodes["bodyId"].isin(target_ids)]
    ax.scatter(
        [pos[int(x)][0] for x in targets["bodyId"]],
        [pos[int(x)][1] for x in targets["bodyId"]],
        s=36,
        c=RED,
        edgecolors="black",
        linewidths=0.45,
        alpha=0.98,
        zorder=4,
    )
    ax.text(0.925, 0.855, "descending\ntargets", ha="center", va="bottom", fontsize=8, color=RED, fontweight="bold")
    ax.text(0.01, 0.99, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=13, fontweight="bold", color=DARK)


def draw_heatmap(ax: plt.Axes, target_edges: pd.DataFrame) -> None:
    source_rank = target_edges.groupby("src_type")["weight"].sum().sort_values(ascending=False).head(14)
    target_rank = target_edges.groupby("target_type")["weight"].sum().sort_values(ascending=False).head(10)
    matrix = (
        target_edges[target_edges["src_type"].isin(source_rank.index) & target_edges["target_type"].isin(target_rank.index)]
        .pivot_table(index="src_type", columns="target_type", values="weight", aggfunc="sum", fill_value=0)
        .reindex(index=source_rank.index, columns=target_rank.index, fill_value=0)
    )
    image = ax.imshow(np.log10(matrix.to_numpy() + 1.0), aspect="auto", cmap="magma")
    ax.set_title("Source-type fan-in to descending targets", fontsize=10, pad=10)
    ax.set_xlabel("Target type")
    ax.set_ylabel("Source type")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=38, ha="right", fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.tick_params(length=0)
    ax.text(-0.10, 1.05, "(b)", transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold", color=DARK)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(r"$\log_{10}(1+w)$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def draw_roi_composition(ax: plt.Axes, target_edges: pd.DataFrame) -> None:
    target_weight = target_edges.groupby("target_type")["weight"].sum().sort_values(ascending=True).tail(12)
    comp = (
        target_edges[target_edges["target_type"].isin(target_weight.index)]
        .pivot_table(index="target_type", columns="src_roi", values="weight", aggfunc="sum", fill_value=0)
        .reindex(index=target_weight.index, columns=ROIS, fill_value=0)
    )
    frac = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    left = np.zeros(len(frac))
    y = np.arange(len(frac))
    for roi in ROIS:
        ax.barh(y, frac[roi], left=left, color=ROI_COLORS[roi], edgecolor="white", linewidth=0.35, label=ROI_NAMES[roi])
        left += frac[roi].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels(frac.index, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of target input weight")
    ax.set_title("ROI composition of target fan-in", fontsize=10, pad=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="x", color="#e1e1e1", lw=0.6)
    ax.text(-0.13, 1.05, "(c)", transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold", color=DARK)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    nodes, edges, target_edges, target_ids = load_projection_data()
    pos = compute_positions(nodes, target_ids, target_edges)

    fig = plt.figure(figsize=(12.6, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.45, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_projection(ax_a, nodes, edges, target_edges, target_ids, pos)
    draw_heatmap(ax_b, target_edges)
    draw_roi_composition(ax_c, target_edges)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=6, label="Medulla neurons"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=6, label="Lobula neurons"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=6, label="Lobula plate neurons"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor=RED, markeredgewidth=0.4, markersize=7, label="Descending targets"),
        Line2D([0], [0], color=GREY, lw=1.4, alpha=0.55, label="Retained backbone edges"),
        Line2D([0], [0], color=RED, lw=2.0, alpha=0.80, label="Edges terminating in targets"),
    ]
    ax_a.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=True, framealpha=0.93, fontsize=8, ncol=2)
    fig.suptitle("Anatomical-style projection of optic-lobe fan-in to descending targets", fontsize=14, fontweight="bold")
    out = OUT / "sfig19_anatomical_projection_target_fanin.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
