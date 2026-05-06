#!/usr/bin/env python
"""Reviewer-revision analyses for channel uncertainty and motion controls."""
from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.learning.null_graphs import apply_condition
from thermodynamic_connectomics.learning.plasticity_metrics import split_train_test, train_linear_probe
from thermodynamic_connectomics.learning.rule_search import representation
from thermodynamic_connectomics.learning.subgraphs import build_matched_learning_subgraph, load_typed_core
from thermodynamic_connectomics.learning.synthetic_tasks import generate_task_data
from thermodynamic_connectomics.metrics.entropy import shannon_entropy


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
SUPP_FIGS = ROOT / "paper" / "supplement_figures"

LAMINA_RE = r"^L[1-5](?:$|[^0-9])"
MEDULLA_RE = r"^(?:Mi|Tm|TmY)"
MOTION_RE = r"^(?:T4|T5)"
OUTPUT_RE = r"^(?:LC|LPLC|LT|DN)"


def _contains(series: pd.Series, pattern: str) -> pd.Series:
    return series.astype(str).str.contains(pattern, regex=True, na=False)


def _channel_from_block(block: pd.DataFrame, src_col: str = "src_type", dst_col: str = "dst_type") -> dict[str, float]:
    if block.empty or block["weight"].sum() <= 0:
        return {
            "n_src_types": 0,
            "n_dst_types": 0,
            "H_source_bits": np.nan,
            "H_target_bits": np.nan,
            "H_source_given_target_bits": np.nan,
            "mutual_information_bits": np.nan,
            "effective_preimage_size": np.nan,
        }
    total = float(block["weight"].sum())
    pjoint = block["weight"].to_numpy(dtype=float) / total
    hs = shannon_entropy(block.groupby(src_col)["weight"].sum().to_numpy(dtype=float), base=2)
    hd = shannon_entropy(block.groupby(dst_col)["weight"].sum().to_numpy(dtype=float), base=2)
    hsd = shannon_entropy(pjoint, base=2)
    mi = max(0.0, hs + hd - hsd)
    h_s_given_d = max(0.0, hsd - hd)
    return {
        "n_src_types": int(block[src_col].nunique()),
        "n_dst_types": int(block[dst_col].nunique()),
        "H_source_bits": float(hs),
        "H_target_bits": float(hd),
        "H_source_given_target_bits": float(h_s_given_d),
        "mutual_information_bits": float(mi),
        "effective_preimage_size": float(2.0**h_s_given_d),
    }


def _block_table(edges: pd.DataFrame) -> pd.DataFrame:
    work = edges[["src_type", "dst_type", "weight"]].dropna().copy()
    work["src_type"] = work["src_type"].astype(str)
    work["dst_type"] = work["dst_type"].astype(str)
    work = work[work["src_type"].str.len().gt(0) & work["dst_type"].str.len().gt(0)]
    return work.groupby(["src_type", "dst_type"], as_index=False)["weight"].sum()


def _multinomial_ci(block: pd.DataFrame, seed: int, n_boot: int = 300) -> dict[str, float]:
    """Conditional multinomial uncertainty over observed type-pair proportions.

    This is not a biological sampling model. It captures finite-table
    estimation uncertainty conditional on the observed weighted type-pair
    distribution and is reported as a lower-bound uncertainty proxy.
    """

    if block.empty or block["weight"].sum() <= 0:
        return {}
    rng = np.random.default_rng(seed)
    weights = block["weight"].to_numpy(dtype=float)
    p = weights / weights.sum()
    n = int(max(1000, round(weights.sum())))
    h_vals = np.empty(n_boot, dtype=float)
    mi_vals = np.empty(n_boot, dtype=float)
    eff_vals = np.empty(n_boot, dtype=float)
    sim = block[["src_type", "dst_type"]].copy()
    for i in range(n_boot):
        counts = rng.multinomial(n, p)
        keep = counts > 0
        sim_i = sim.loc[keep].copy()
        sim_i["weight"] = counts[keep].astype(float)
        m = _channel_from_block(sim_i)
        h_vals[i] = m["H_source_given_target_bits"]
        mi_vals[i] = m["mutual_information_bits"]
        eff_vals[i] = m["effective_preimage_size"]
    observed = _channel_from_block(block)
    h_std = float(np.std(h_vals, ddof=1))
    mi_std = float(np.std(mi_vals, ddof=1))
    eff_std = float(np.std(eff_vals, ddof=1))
    return {
        "H_source_given_target_ci_low": float(observed["H_source_given_target_bits"] - 1.96 * h_std),
        "H_source_given_target_ci_high": float(observed["H_source_given_target_bits"] + 1.96 * h_std),
        "mutual_information_ci_low": float(observed["mutual_information_bits"] - 1.96 * mi_std),
        "mutual_information_ci_high": float(observed["mutual_information_bits"] + 1.96 * mi_std),
        "effective_preimage_ci_low": float(observed["effective_preimage_size"] - 1.96 * eff_std),
        "effective_preimage_ci_high": float(observed["effective_preimage_size"] + 1.96 * eff_std),
        "ci_method": f"centered conditional multinomial SE over type-pair weights, n_boot={n_boot}, n_draws={n}",
    }


def _pathway_groups(edges: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    src = edges["src_type"]
    dst = edges["dst_type"]
    roi = edges["edge_major_roi"].astype(str)
    lamina_src = _contains(src, LAMINA_RE)
    lamina_dst = _contains(dst, LAMINA_RE)
    medulla_src = _contains(src, MEDULLA_RE)
    medulla_dst = _contains(dst, MEDULLA_RE)
    motion_src = _contains(src, MOTION_RE)
    motion_dst = _contains(dst, MOTION_RE)
    output_src = _contains(src, OUTPUT_RE)
    output_dst = _contains(dst, OUTPUT_RE)
    motion_types = set(edges.loc[motion_src, "src_type"].astype(str)) | set(edges.loc[motion_dst, "dst_type"].astype(str))
    partner_edges = edges[motion_src | motion_dst]
    partner_types = set(partner_edges["src_type"].astype(str)) | set(partner_edges["dst_type"].astype(str))
    motion_partner_mask = src.astype(str).isin(motion_types | partner_types) | dst.astype(str).isin(motion_types | partner_types)
    return [
        ("full_typed_core", "global", edges),
        ("LO(R)", "roi", edges[roi.eq("LO(R)")]),
        ("ME(R)", "roi", edges[roi.eq("ME(R)")]),
        ("LOP(R)", "roi", edges[roi.eq("LOP(R)")]),
        ("lamina_input_L1_L2_L3_L4_L5", "curated_pathway", edges[lamina_src | lamina_dst]),
        ("medulla_intrinsic_Mi_Tm_TmY", "curated_pathway", edges[medulla_src | medulla_dst]),
        ("motion_T4_T5_plus_direct_partners", "curated_pathway", edges[motion_partner_mask]),
        ("output_projection_LC_LPLC_LT_DN", "curated_pathway", edges[output_src | output_dst]),
        ("lamina_to_medulla", "directional_stage", edges[lamina_src & (medulla_dst | motion_dst) & roi.eq("ME(R)")]),
        ("medulla_to_lobula", "directional_stage", edges[medulla_src & roi.eq("LO(R)")]),
        ("medulla_to_lobula_plate", "directional_stage", edges[(medulla_src | motion_src) & roi.eq("LOP(R)")]),
        ("lobula_lobula_plate_to_projection_output", "directional_stage", edges[output_dst & roi.isin(["LO(R)", "LOP(R)"])]),
    ]


def authoritative_channel_metrics(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (name, kind, subset) in enumerate(_pathway_groups(edges)):
        block = _block_table(subset)
        row = {
            "group": name,
            "group_kind": kind,
            "n_edges": int(len(subset)),
            "total_weight": float(subset["weight"].sum()) if len(subset) else 0.0,
            **_channel_from_block(block),
            **_multinomial_ci(block, seed=1000 + i),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "authoritative_channel_metrics_with_ci.csv", index=False)
    return out


def target_distribution_outputs() -> pd.DataFrame:
    target_path = ROOT / "outputs/roi_pathway/target_type_ambiguity_by_group.csv"
    target = pd.read_csv(target_path)
    full = target[target["group"].eq("full_typed_core")].copy()
    full = full[full["total_weight"].ge(100) & full["n_src_types"].ge(2)].copy()
    full["log10_total_weight"] = np.log10(full["total_weight"].clip(lower=1))
    full.to_csv(TABLES / "target_type_distribution_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(9.4, 5.2))
    sns.histplot(full["history_ambiguity_bits"], bins=40, color="#4C78A8")
    plt.xlabel("Target-type history ambiguity H(S | D=d), bits")
    plt.ylabel("Target type count")
    plt.subplots_adjust(left=0.12, right=0.96, bottom=0.18, top=0.92)
    plt.savefig(SUPP_FIGS / "sfig15_target_type_ambiguity_distribution.pdf", bbox_inches="tight")
    plt.close()

    top = full.sort_values("history_ambiguity_bits", ascending=False).head(30)
    plt.figure(figsize=(11.8, 7.6))
    ax = sns.scatterplot(
        data=full,
        x="log10_total_weight",
        y="history_ambiguity_bits",
        size="n_src_types",
        sizes=(10, 90),
        alpha=0.35,
        color="#4C78A8",
        legend=False,
    )
    offsets = [(0.035, 0.08), (0.04, -0.10), (-0.34, 0.08), (-0.36, -0.10)]
    for i, row in enumerate(top.head(12).itertuples(index=False)):
        dx, dy = offsets[i % len(offsets)]
        ax.text(
            row.log10_total_weight + dx,
            row.history_ambiguity_bits + dy,
            str(row.dst_type),
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=1.0),
        )
    plt.xlabel("log10 total incoming weight")
    plt.ylabel("Target-type H(S | D=d), bits")
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.92)
    plt.savefig(SUPP_FIGS / "sfig16_target_type_weight_vs_ambiguity.pdf", bbox_inches="tight")
    plt.close()
    return full


def _random_template_data(graph, n_samples: int, noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_source = len(graph.source_nodes)
    n_classes = 4
    y = rng.integers(0, n_classes, size=n_samples)
    X = rng.normal(0.0, noise, size=(n_samples, n_source)).astype(np.float32)
    frac = min(0.35, max(0.05, 50 / max(n_source, 1)))
    templates = []
    for _ in range(n_classes):
        templates.append(rng.random(n_source) < frac)
    for cls, active in enumerate(templates):
        if not active.any():
            active[rng.integers(0, n_source)] = True
        X[np.ix_(y == cls, active)] += 1.0
    X = 1.0 / (1.0 + np.exp(-X))
    return {
        "X_source": X.astype(np.float32),
        "y": y.astype(np.int64),
        "source_nodes": graph.source_nodes,
        "target_nodes": graph.target_nodes,
        "task_metadata": {"task_name": "motion_direction_random_source_groups", "n_classes": n_classes},
    }


def _task_for_mode(graph, mode: str, n_samples: int, seed: int) -> dict:
    if mode == "type_structured":
        return generate_task_data(graph, graph.node_table, "motion_direction", n_samples=n_samples, noise=0.25, seed=seed)
    if mode == "permuted_source_types":
        rng = np.random.default_rng(seed + 999)
        shuffled = list(np.asarray(graph.source_types, dtype=object)[rng.permutation(len(graph.source_types))])
        graph_perm = replace(graph, source_types=shuffled)
        return generate_task_data(graph_perm, graph_perm.node_table, "motion_direction", n_samples=n_samples, noise=0.25, seed=seed)
    if mode == "random_source_groups":
        return _random_template_data(graph, n_samples=n_samples, noise=0.25, seed=seed)
    raise ValueError(mode)


def _features(X: np.ndarray, W, feature_mode: str) -> np.ndarray:
    H = representation(X, W)
    if feature_mode == "linear":
        return H
    if feature_mode == "tanh":
        return np.tanh(H).astype(np.float32)
    if feature_mode == "relu":
        return np.maximum(H, 0).astype(np.float32)
    raise ValueError(feature_mode)


def motion_reviewer_controls(edges: pd.DataFrame, nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    conditions = ["real", "degree_preserving_null", "weight_shuffled_real_topology", "uniform_weight_real_topology"]
    template_modes = ["type_structured", "permuted_source_types", "random_source_groups"]
    feature_modes = ["linear", "tanh", "relu"]
    for sample_id in range(5):
        base_graph, meta = build_matched_learning_subgraph(
            "medulla_to_lobula_plate",
            edges,
            nodes,
            matched_nodes=1500,
            matched_edges=12000,
            sample_id=sample_id,
            seed=4242,
        )
        for seed in range(5):
            task_seed = 10_000 + seed + 1000 * sample_id
            for condition in conditions:
                graph = apply_condition(base_graph, condition, seed=20_000 + seed + 1000 * sample_id)
                for template_mode in template_modes:
                    task = _task_for_mode(graph, template_mode, n_samples=2000, seed=task_seed)
                    X_train, X_test, y_train, y_test = split_train_test(task["X_source"], task["y"], seed=task_seed)
                    for feature_mode in feature_modes:
                        H_train = _features(X_train, graph.W, feature_mode)
                        H_test = _features(X_test, graph.W, feature_mode)
                        metrics = train_linear_probe(
                            H_train,
                            y_train,
                            H_test,
                            y_test,
                            seed=task_seed,
                            epochs=80,
                            lr=0.03,
                        )
                        rows.append(
                            {
                                "pathway": "medulla_to_lobula_plate",
                                "task": "motion_direction",
                                "condition": condition,
                                "template_mode": template_mode,
                                "feature_mode": feature_mode,
                                "seed": seed,
                                "sample_id": sample_id,
                                "n_samples": 2000,
                                "train_fraction": 0.65,
                                "test_fraction": 0.35,
                                "exploratory_subgraph": meta["exploratory"],
                                **metrics,
                            }
                        )
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "motion_probe_reviewer_controls.csv", index=False)
    pairs = []
    key = ["template_mode", "feature_mode", "seed", "sample_id"]
    real = results[results["condition"].eq("real")].set_index(key)
    for control in ["degree_preserving_null", "weight_shuffled_real_topology", "uniform_weight_real_topology"]:
        ctrl = results[results["condition"].eq(control)].set_index(key)
        for idx in real.index.intersection(ctrl.index):
            r = real.loc[idx]
            c = ctrl.loc[idx]
            row = dict(zip(key, idx))
            row.update(
                {
                    "comparison": f"real_vs_{control}",
                    "delta_accuracy": float(r["accuracy"] - c["accuracy"]),
                    "delta_task_MI_bits": float(r["task_MI_bits"] - c["task_MI_bits"]),
                    "real_accuracy": float(r["accuracy"]),
                    "control_accuracy": float(c["accuracy"]),
                    "real_task_MI_bits": float(r["task_MI_bits"]),
                    "control_task_MI_bits": float(c["task_MI_bits"]),
                }
            )
            pairs.append(row)
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(TABLES / "motion_probe_reviewer_control_pairs.csv", index=False)

    summary_rows = []
    for (template_mode, feature_mode, condition), g in results.groupby(["template_mode", "feature_mode", "condition"]):
        summary_rows.append(
            {
                "template_mode": template_mode,
                "feature_mode": feature_mode,
                "condition": condition,
                "n": int(len(g)),
                "accuracy_mean": float(g["accuracy"].mean()),
                "accuracy_std": float(g["accuracy"].std(ddof=1)),
                "task_MI_bits_mean": float(g["task_MI_bits"].mean()),
                "task_MI_bits_std": float(g["task_MI_bits"].std(ddof=1)),
            }
        )
    pair_summary = []
    for (template_mode, feature_mode, comparison), g in pair_df.groupby(["template_mode", "feature_mode", "comparison"]):
        for metric in ["delta_accuracy", "delta_task_MI_bits"]:
            vals = g[metric].to_numpy(float)
            rng = np.random.default_rng(abs(hash((template_mode, feature_mode, comparison, metric))) % (2**32))
            boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
            pair_summary.append(
                {
                    "template_mode": template_mode,
                    "feature_mode": feature_mode,
                    "comparison": comparison,
                    "metric": metric,
                    "n_pairs": int(len(vals)),
                    "mean": float(vals.mean()),
                    "ci_low": float(np.percentile(boots, 2.5)),
                    "ci_high": float(np.percentile(boots, 97.5)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    pair_summary = pd.DataFrame(pair_summary)
    summary.to_csv(TABLES / "motion_probe_reviewer_control_summary.csv", index=False)
    pair_summary.to_csv(TABLES / "motion_probe_reviewer_control_pair_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    plot = summary[summary["feature_mode"].isin(["linear", "tanh"])].copy()
    condition_labels = {
        "real": "Real",
        "degree_preserving_null": "Degree null",
        "weight_shuffled_real_topology": "Shuffled weights",
        "weight_uniform_real_topology": "Uniform weights",
    }
    template_labels = {
        "permuted_source_types": "Source types\npermuted",
        "random_source_groups": "Random source\ngroups",
        "type_structured": "Type\nstructured",
    }
    plot["condition_label"] = plot["condition"].map(condition_labels).fillna(plot["condition"])
    plot["template_label"] = plot["template_mode"].map(template_labels).fillna(plot["template_mode"])
    g = sns.catplot(
        data=plot,
        x="condition_label",
        y="accuracy_mean",
        hue="feature_mode",
        col="template_label",
        kind="bar",
        height=4.7,
        aspect=1.24,
        sharey=True,
        legend=False,
    )
    g.set_axis_labels("Graph condition", "Accuracy")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        for tick in ax.get_xticklabels():
            tick.set_rotation(24)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    g.fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    plt.ylim(0.2, max(0.75, float(plot["accuracy_mean"].max()) + 0.05))
    g.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.30, top=0.82, wspace=0.22)
    g.fig.savefig(SUPP_FIGS / "sfig17_motion_template_and_feature_controls.pdf", bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(10.6, 5.4))
    plot = pair_summary[
        pair_summary["comparison"].eq("real_vs_degree_preserving_null")
        & pair_summary["metric"].eq("delta_task_MI_bits")
    ].copy()
    plot["template_label"] = plot["template_mode"].map(
        {
            "permuted_source_types": "Source types\npermuted",
            "random_source_groups": "Random source\ngroups",
            "type_structured": "Type\nstructured",
        }
    ).fillna(plot["template_mode"])
    ax = sns.pointplot(data=plot, x="template_label", y="mean", hue="feature_mode", dodge=0.35)
    for i, row in enumerate(plot.itertuples(index=False)):
        pass
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Real - degree null task MI, bits")
    plt.xlabel("Template control")
    for tick in ax.get_xticklabels():
        tick.set_rotation(18)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, title="")
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.36, top=0.92)
    plt.savefig(SUPP_FIGS / "sfig18_motion_real_minus_degree_controls.pdf", bbox_inches="tight")
    plt.close()
    return results, pair_summary


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    SUPP_FIGS.mkdir(parents=True, exist_ok=True)
    edges, nodes = load_typed_core(ROOT / "outputs/processed_typed_core/edges.parquet", ROOT / "outputs/processed_typed_core/nodes.parquet")
    metrics = authoritative_channel_metrics(edges)
    target_distribution_outputs()
    motion_reviewer_controls(edges, nodes)
    print(f"Wrote reviewer revision controls for {len(metrics)} channel groups")


if __name__ == "__main__":
    main()
