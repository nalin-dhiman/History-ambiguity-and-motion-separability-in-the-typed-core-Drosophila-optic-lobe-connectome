#!/usr/bin/env python
"""Reviewer-facing sensitivity analyses and statistical summaries.

These analyses do not introduce new biological claims. They quantify how the
reported history-ambiguity hierarchy changes under alternative type taxonomies,
ROI assignment choices, complementary information metrics, and existing null
ensemble uncertainty.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.metrics.entropy import shannon_entropy
from thermodynamic_connectomics.metrics.irreversibility import compute_block_irreversibility


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
SUPP_FIGS = ROOT / "paper" / "supplement_figures"


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values preserving original order."""

    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return q
    n = len(valid)
    ranked = valid.to_frame("p")
    ranked["rank"] = np.arange(1, n + 1)
    ranked["q_raw"] = ranked["p"] * n / ranked["rank"]
    ranked["q"] = ranked["q_raw"][::-1].cummin()[::-1].clip(upper=1.0)
    q.loc[ranked.index] = ranked["q"]
    return q


def _split_family(label: object) -> str:
    text = str(label)
    if re.match(r"^L[1-5](?:$|[^0-9])", text):
        return re.match(r"^L[1-5]", text).group(0)
    for pattern, family in [
        (r"^TmY\d*", "TmY"),
        (r"^Tm\d*", "Tm"),
        (r"^Mi\d*", "Mi"),
        (r"^T4[a-zA-Z0-9-]*", "T4"),
        (r"^T5[a-zA-Z0-9-]*", "T5"),
        (r"^LC\d*", "LC"),
        (r"^LPLC\d*", "LPLC"),
        (r"^LT\d*", "LT"),
        (r"^DN[a-zA-Z0-9-]*", "DN"),
        (r"^VS[a-zA-Z0-9-]*", "VS"),
        (r"^HS[a-zA-Z0-9-]*", "HS"),
    ]:
        if re.match(pattern, text):
            return family
    m = re.match(r"^[A-Za-z]+", text)
    return m.group(0) if m else text


def _coarse_visual_class(label: object) -> str:
    text = str(label)
    if re.match(r"^L[1-5](?:$|[^0-9])", text):
        return "lamina_L1_L5"
    if re.match(r"^(?:Mi|Tm|TmY)", text):
        return "medulla_intrinsic"
    if re.match(r"^(?:T4|T5)", text):
        return "motion_T4_T5"
    if re.match(r"^(?:LC|LPLC|LT|DN)", text):
        return "projection_output"
    if re.match(r"^(?:VS|HS)", text):
        return "lobula_plate_tangential"
    if re.match(r"^(?:CT|C|Me|Lo|Lop)", text):
        return "optic_lobe_other_named"
    return "other_or_unclassified"


def _alpha_prefix(label: object) -> str:
    text = str(label)
    m = re.match(r"^[A-Za-z]+", text)
    return m.group(0) if m else text


def _community_taxonomy(edges: pd.DataFrame) -> dict[str, str]:
    block = edges.groupby(["src_type", "dst_type"], as_index=False)["weight"].sum()
    graph = nx.Graph()
    for row in block.itertuples(index=False):
        s = str(row.src_type)
        d = str(row.dst_type)
        w = float(row.weight)
        if graph.has_edge(s, d):
            graph[s][d]["weight"] += w
        else:
            graph.add_edge(s, d, weight=w)
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    mapping: dict[str, str] = {}
    for idx, community in enumerate(communities):
        label = f"comm_{idx:03d}"
        for node in community:
            mapping[str(node)] = label
    return mapping


def _channel_metrics(edges: pd.DataFrame, src_col: str, dst_col: str) -> dict[str, float]:
    block = edges[[src_col, dst_col, "weight"]].dropna().copy()
    block[src_col] = block[src_col].astype(str)
    block[dst_col] = block[dst_col].astype(str)
    block = block[(block[src_col].str.len() > 0) & (block[dst_col].str.len() > 0)]
    if block.empty:
        return {
            "n_src_classes": 0,
            "n_dst_classes": 0,
            "H_source_bits": np.nan,
            "H_target_bits": np.nan,
            "H_source_given_target_bits": np.nan,
            "mutual_information_bits": np.nan,
            "effective_preimage_size": np.nan,
            "normalized_MI_sqrt": np.nan,
            "uncertainty_coeff_source": np.nan,
            "uncertainty_coeff_target": np.nan,
            "mean_KL_Ps_given_d_vs_Ps_bits": np.nan,
        }
    grouped = block.groupby([src_col, dst_col], as_index=False)["weight"].sum()
    total = float(grouped["weight"].sum())
    ps = grouped.groupby(src_col)["weight"].sum().to_numpy(dtype=float)
    pdst = grouped.groupby(dst_col)["weight"].sum().to_numpy(dtype=float)
    pjoint = grouped["weight"].to_numpy(dtype=float) / total
    hs = shannon_entropy(ps, base=2)
    hd = shannon_entropy(pdst, base=2)
    hsd = shannon_entropy(pjoint, base=2)
    mi = max(0.0, hs + hd - hsd)
    h_s_given_d = max(0.0, hsd - hd)
    return {
        "n_src_classes": int(grouped[src_col].nunique()),
        "n_dst_classes": int(grouped[dst_col].nunique()),
        "H_source_bits": float(hs),
        "H_target_bits": float(hd),
        "H_source_given_target_bits": float(h_s_given_d),
        "mutual_information_bits": float(mi),
        "effective_preimage_size": float(2.0**h_s_given_d),
        "normalized_MI_sqrt": float(mi / math.sqrt(max(hs * hd, 1e-12))),
        "uncertainty_coeff_source": float(mi / max(hs, 1e-12)),
        "uncertainty_coeff_target": float(mi / max(hd, 1e-12)),
        "mean_KL_Ps_given_d_vs_Ps_bits": float(mi),
    }


def _taxonomy_sensitivity(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    community_map = _community_taxonomy(edges)
    taxonomies = {
        "original_cell_type": lambda x: str(x),
        "family_merge_regex": _split_family,
        "coarse_visual_class": _coarse_visual_class,
        "alpha_prefix": _alpha_prefix,
        "weighted_type_community": lambda x: community_map.get(str(x), "community_unassigned"),
    }
    summary_rows = []
    roi_rows = []
    comp_rows = []
    for name, mapper in taxonomies.items():
        work = edges[["src_type", "dst_type", "weight", "edge_major_roi"]].copy()
        work["src_tax"] = work["src_type"].map(mapper)
        work["dst_tax"] = work["dst_type"].map(mapper)
        metrics = _channel_metrics(work, "src_tax", "dst_tax")
        metrics["taxonomy"] = name
        metrics["n_edges"] = int(len(work))
        metrics["total_weight"] = float(work["weight"].sum())
        summary_rows.append(metrics)
        comp_rows.append({"group": "full_typed_core", "taxonomy": name, **metrics})
        for roi in ["LO(R)", "ME(R)", "LOP(R)"]:
            subset = work[work["edge_major_roi"].eq(roi)].copy()
            roi_metrics = _channel_metrics(subset, "src_tax", "dst_tax")
            roi_rows.append({"taxonomy": name, "roi": roi, **roi_metrics})
            comp_rows.append({"group": roi, "taxonomy": name, **roi_metrics})
    return pd.DataFrame(summary_rows), pd.DataFrame(roi_rows), pd.DataFrame(comp_rows)


def _roi_handling_sensitivity(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    schemes = {
        "edge_dominant_roi": ("edge_major_roi", edges),
        "source_major_roi": ("src_major_roi", edges),
        "target_major_roi": ("dst_major_roi", edges),
        "edge_specific_only": ("edge_major_roi", edges[edges["edge_major_roi"].isin(["LO(R)", "ME(R)", "LOP(R)"])]),
        "strict_edge_endpoint_consensus": (
            "edge_major_roi",
            edges[
                edges["edge_major_roi"].isin(["LO(R)", "ME(R)", "LOP(R)"])
                & edges["edge_major_roi"].eq(edges["src_major_roi"])
                & edges["edge_major_roi"].eq(edges["dst_major_roi"])
            ],
        ),
    }
    for scheme, (roi_col, subset) in schemes.items():
        for roi in ["LO(R)", "ME(R)", "LOP(R)"]:
            roi_edges = subset[subset[roi_col].astype(str).eq(roi)].copy()
            metrics = _channel_metrics(roi_edges, "src_type", "dst_type")
            rows.append(
                {
                    "roi_handling": scheme,
                    "roi": roi,
                    "n_edges": int(len(roi_edges)),
                    "total_weight": float(roi_edges["weight"].sum()) if len(roi_edges) else 0.0,
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    out["available_multi_roi_membership"] = False
    out["multi_roi_note"] = (
        "Processed typed-core edges contain a single parsed major ROI only; original multi-ROI membership weights were not retained."
    )
    return out


def _null_multiplicity_summary() -> pd.DataFrame:
    rows = []
    sources = [
        ("typed_core_global", ROOT / "outputs/nulls_typed_core/null_summary.csv"),
        ("roi_pathway", ROOT / "outputs/roi_pathway/roi_pathway_null_summary.csv"),
    ]
    for source, path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "empirical_p_two_sided" in df.columns:
            df["p_uncorrected"] = pd.to_numeric(df["empirical_p_two_sided"], errors="coerce")
        else:
            df["p_uncorrected"] = np.nan
        df["source_table"] = source
        if "group" not in df.columns:
            df["group"] = "full_typed_core"
        rows.append(df)
    nulls = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if nulls.empty:
        return nulls
    nulls["q_bh_all_tests"] = _bh_fdr(nulls["p_uncorrected"])
    nulls["observed_minus_null_mean"] = pd.to_numeric(nulls["observed"], errors="coerce") - pd.to_numeric(
        nulls["null_mean"], errors="coerce"
    )
    n = pd.to_numeric(nulls.get("n_success", np.nan), errors="coerce")
    std = pd.to_numeric(nulls.get("null_std", np.nan), errors="coerce")
    se = std / np.sqrt(n.replace(0, np.nan))
    nulls["observed_minus_null_ci_low"] = nulls["observed_minus_null_mean"] - 1.96 * se
    nulls["observed_minus_null_ci_high"] = nulls["observed_minus_null_mean"] + 1.96 * se
    cols = [
        "source_table",
        "group",
        "group_kind",
        "null_model",
        "metric",
        "observed",
        "null_mean",
        "null_std",
        "n_success",
        "failure_count",
        "sampled_edges",
        "edge_count_used",
        "edge_count_available",
        "observed_minus_null_mean",
        "observed_minus_null_ci_low",
        "observed_minus_null_ci_high",
        "p_uncorrected",
        "q_bh_all_tests",
    ]
    for col in cols:
        if col not in nulls.columns:
            nulls[col] = np.nan
    return nulls[cols]


def _known_pathway_ci_summary() -> pd.DataFrame:
    path = ROOT / "outputs/known_visual_pathway_validation/known_visual_pathway_validation.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for null_prefix in ["degree_null", "type_label_null"]:
        n = pd.to_numeric(df[f"{null_prefix}_n_success"], errors="coerce")
        mean = pd.to_numeric(df[f"{null_prefix}_mean_history_ambiguity_bits"], errors="coerce")
        std = pd.to_numeric(df[f"{null_prefix}_std_history_ambiguity_bits"], errors="coerce")
        obs = pd.to_numeric(df["history_ambiguity_bits"], errors="coerce")
        diff = obs - mean
        se = std / np.sqrt(n.replace(0, np.nan))
        tmp = pd.DataFrame(
            {
                "group": df["group"],
                "group_kind": df["group_kind"],
                "null_model": null_prefix,
                "metric": "history_ambiguity_bits",
                "observed": obs,
                "null_mean": mean,
                "null_std": std,
                "n_success": n,
                "observed_minus_null_mean": diff,
                "observed_minus_null_ci_low": diff - 1.96 * se,
                "observed_minus_null_ci_high": diff + 1.96 * se,
            }
        )
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def _motion_variance_summary() -> pd.DataFrame:
    paths = [
        ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/final_summary.csv",
        ROOT / "outputs/learning_signal_rescue/final_control_1500n_12000e/paired_comparisons.csv",
    ]
    if not all(path.exists() for path in paths):
        return pd.DataFrame()
    final = pd.read_csv(paths[0])
    paired = pd.read_csv(paths[1])
    rows = []
    for cols, df, metric_cols in [
        (
            ["condition", "mode"],
            final,
            ["final_accuracy", "final_task_MI_bits", "delta_task_MI_bits", "total_synaptic_update_L1"],
        ),
        (
            ["null_condition", "mode"],
            paired,
            ["real_minus_null_accuracy", "real_minus_null_delta_MI", "real_minus_null_cost", "real_minus_null_eta"],
        ),
    ]:
        for key, group in df.groupby(cols):
            if not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(cols, key))
            base["n_seed_sample_pairs"] = int(group[["seed", "sample_id"]].drop_duplicates().shape[0])
            for metric in metric_cols:
                vals = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                base[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
                base[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            rows.append(base)
    return pd.DataFrame(rows)


def _motion_estimator_details() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "task_MI_bits in physical-learning script",
                "definition": "plug-in mutual information I(y_true; y_pred) from sklearn.metrics.mutual_info_score divided by ln(2)",
                "estimator": "empirical discrete contingency table of true class and argmax predicted class",
                "finite_sample_correction": "none",
                "folds": "none in scripts/09 physical-learning runs; same generated samples are used for SGD and reporting",
            },
            {
                "item": "delta_task_MI_bits",
                "definition": "final task_MI_bits minus initial task_MI_bits from the randomly initialized readout",
                "estimator": "difference of two plug-in predicted-label MI estimates",
                "finite_sample_correction": "none; can be biased upward for small samples, so interpreted as a functional score",
                "folds": "no cross-validation in the original rescue/final-control run",
            },
            {
                "item": "representation-probe plasticity_delta_MI",
                "definition": "held-out probe MI after plasticity minus held-out probe MI before plasticity",
                "estimator": "plug-in predicted-label MI on the 35% test split",
                "finite_sample_correction": "none",
                "folds": "single deterministic 65/35 train/test split per seed/subgraph; no early stopping",
            },
            {
                "item": "message passing architecture",
                "definition": "one linear source-to-target propagation H = X_source W, target-normalized incoming weights, softmax linear readout",
                "estimator": "readout trained by full-batch SGD cross-entropy",
                "finite_sample_correction": "not applicable",
                "folds": "synthetic inputs have sigmoid-transformed Gaussian noise plus class templates; no temporal delay lines",
            },
        ]
    )


def _plot_taxonomy(summary: pd.DataFrame, roi: pd.DataFrame) -> None:
    SUPP_FIGS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10.2, 5.0))
    plot = summary.sort_values("H_source_given_target_bits")
    sns.barplot(data=plot, y="taxonomy", x="H_source_given_target_bits", color="#4C78A8")
    plt.xlabel("H(S | D), bits")
    plt.ylabel("Taxonomy")
    plt.subplots_adjust(left=0.26, right=0.96, bottom=0.16, top=0.92)
    plt.savefig(SUPP_FIGS / "sfig11_taxonomy_sensitivity.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10.8, 5.6))
    plot = roi[roi["taxonomy"].isin(["original_cell_type", "family_merge_regex", "coarse_visual_class"])].copy()
    ax = sns.barplot(data=plot, x="roi", y="H_source_given_target_bits", hue="taxonomy")
    plt.xlabel("ROI")
    plt.ylabel("H(S | D), bits")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, title="")
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.34, top=0.92)
    plt.savefig(SUPP_FIGS / "sfig12_taxonomy_by_roi.pdf", bbox_inches="tight")
    plt.close()


def _plot_roi_handling(roi: pd.DataFrame, comp: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(11.8, 5.8))
    plot = roi[roi["roi_handling"].isin(["edge_dominant_roi", "source_major_roi", "target_major_roi", "strict_edge_endpoint_consensus"])]
    ax = sns.barplot(data=plot, x="roi", y="H_source_given_target_bits", hue="roi_handling")
    plt.xlabel("ROI")
    plt.ylabel("H(S | D), bits")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, title="")
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.92)
    plt.savefig(SUPP_FIGS / "sfig13_roi_handling_sensitivity.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9.2, 5.6))
    plot = comp[comp["taxonomy"].eq("original_cell_type")]
    ax = sns.scatterplot(
        data=plot,
        x="mean_KL_Ps_given_d_vs_Ps_bits",
        y="H_source_given_target_bits",
        hue="group",
        s=80,
    )
    plt.xlabel("E_D KL[P(S|D) || P(S)] = I(S;D), bits")
    plt.ylabel("H(S | D), bits")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, title="")
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.32, top=0.92)
    plt.savefig(SUPP_FIGS / "sfig14_kl_vs_ambiguity.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    SUPP_FIGS.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(ROOT / "outputs/processed_typed_core/edges.parquet")
    edges = edges.dropna(subset=["src_type", "dst_type"]).copy()

    taxonomy, taxonomy_roi, complementary = _taxonomy_sensitivity(edges)
    roi_handling = _roi_handling_sensitivity(edges)
    nulls = _null_multiplicity_summary()
    known_ci = _known_pathway_ci_summary()
    motion_var = _motion_variance_summary()
    motion_method = _motion_estimator_details()

    taxonomy.to_csv(TABLES / "taxonomy_sensitivity_summary.csv", index=False)
    taxonomy_roi.to_csv(TABLES / "taxonomy_sensitivity_by_roi.csv", index=False)
    complementary.to_csv(TABLES / "complementary_information_metrics.csv", index=False)
    roi_handling.to_csv(TABLES / "roi_handling_sensitivity.csv", index=False)
    nulls.to_csv(TABLES / "null_ensemble_and_multiplicity.csv", index=False)
    known_ci.to_csv(TABLES / "known_pathway_null_ci_summary.csv", index=False)
    motion_var.to_csv(TABLES / "motion_probe_variance_summary.csv", index=False)
    motion_method.to_csv(TABLES / "motion_probe_estimator_details.csv", index=False)

    _plot_taxonomy(taxonomy, taxonomy_roi)
    _plot_roi_handling(roi_handling, complementary)
    print("Wrote reviewer sensitivity controls to outputs/tables and paper/supplement_figures")


if __name__ == "__main__":
    main()
