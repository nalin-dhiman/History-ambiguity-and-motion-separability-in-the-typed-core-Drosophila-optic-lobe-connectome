#!/usr/bin/env python
"""Known visual pathway validation for typed-core optic-lobe channels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.metrics.irreversibility import compute_block_irreversibility
from thermodynamic_connectomics.nulls.degree_preserving import (
    directed_degree_preserving_rewire,
    type_label_permutation,
)


LAMINA_RE = r"^L[1-5](?:$|[^0-9])"
MEDULLA_RE = r"^(?:Mi|Tm|TmY)"
MOTION_RE = r"^(?:T4|T5)"
OUTPUT_RE = r"^(?:LC|LPLC|LT|DN)"


def _contains(series: pd.Series, pattern: str) -> pd.Series:
    return series.astype(str).str.contains(pattern, regex=True, na=False)


def _channel(name: str, kind: str, stage: int, edges: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    block, summary = compute_block_irreversibility(edges, group_col="type")
    row = {
        "group": name,
        "group_kind": kind,
        "stage_order": stage,
        "n_edges": int(len(edges)),
        "total_weight": float(edges["weight"].sum()) if len(edges) else 0.0,
        "n_src_types": summary.get("n_src_types", 0),
        "n_dst_types": summary.get("n_dst_types", 0),
        "history_ambiguity_bits": summary.get("history_ambiguity_bits", np.nan),
        "mutual_information_bits": summary.get("mutual_information_bits", np.nan),
        "effective_preimage_size": summary.get("effective_preimage_size", np.nan),
        "status": summary.get("status", "unknown"),
    }
    return row, block


def _restore_endpoint_metadata(null_edges: pd.DataFrame, reference_edges: pd.DataFrame) -> pd.DataFrame:
    out = null_edges.copy()
    endpoint_cols = ["type", "instance", "roi", "major_roi", "nt"]
    for side in ["src", "dst"]:
        ref_cols = [side] + [f"{side}_{col}" for col in endpoint_cols if f"{side}_{col}" in reference_edges.columns]
        if len(ref_cols) == 1:
            continue
        mapper = reference_edges[ref_cols].drop_duplicates(side).set_index(side)
        for col in mapper.columns:
            out[col] = out[side].map(mapper[col])
    if "edge_major_roi" in reference_edges.columns and "edge_major_roi" not in out.columns:
        out["edge_major_roi"] = "rewired"
    return out


def _sample(edges: pd.DataFrame, max_edges: int, seed: int) -> tuple[pd.DataFrame, bool]:
    if max_edges > 0 and len(edges) > max_edges:
        return edges.sample(max_edges, random_state=seed).reset_index(drop=True), True
    return edges.reset_index(drop=True), False


def _null_stats(name: str, kind: str, stage: int, edges: pd.DataFrame, n_nulls: int, sample_edges: int, seed: int) -> tuple[dict, list[dict]]:
    if len(edges) == 0:
        return {}, [{"group": name, "null_model": "all", "reason": "empty edge subset"}]
    base, sampled = _sample(edges, sample_edges, seed)
    observed, _ = _channel(name, kind, stage, base)
    values = {"degree_preserving_directed_rewire": [], "type_label_permutation": []}
    failures = []
    for i in trange(n_nulls, desc=f"validation nulls {name}", leave=False):
        null_seed = seed + i
        specs = [
            (
                "degree_preserving_directed_rewire",
                lambda: _restore_endpoint_metadata(
                    directed_degree_preserving_rewire(
                        base,
                        seed=null_seed,
                        n_swaps=min(40_000, max(1, len(base) // 10)),
                    ),
                    base,
                ),
            ),
            ("type_label_permutation", lambda: type_label_permutation(base, seed=null_seed)),
        ]
        for null_name, make_null in specs:
            try:
                row, _ = _channel(name, kind, stage, make_null())
                value = float(row["history_ambiguity_bits"])
                if not np.isfinite(value):
                    raise ValueError("history_ambiguity_bits is not finite")
                values[null_name].append(value)
            except Exception as exc:  # noqa: BLE001
                failures.append({"group": name, "null_model": null_name, "iteration": i, "reason": repr(exc)})
    row = {
        **observed,
        "sampled_edges": sampled,
        "edge_count_used": int(len(base)),
        "edge_count_available": int(len(edges)),
    }
    for null_name, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        prefix = "degree_null" if null_name.startswith("degree") else "type_label_null"
        row[f"{prefix}_mean_history_ambiguity_bits"] = float(arr.mean()) if arr.size else np.nan
        row[f"{prefix}_std_history_ambiguity_bits"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        row[f"observed_minus_{prefix}_bits"] = (
            float(row["history_ambiguity_bits"] - arr.mean()) if arr.size else np.nan
        )
        row[f"{prefix}_n_success"] = int(arr.size)
    return row, failures


def _build_groups(edges: pd.DataFrame) -> list[tuple[str, str, int, str, pd.DataFrame]]:
    src = edges["src_type"]
    dst = edges["dst_type"]
    lamina_src = _contains(src, LAMINA_RE)
    lamina_dst = _contains(dst, LAMINA_RE)
    medulla_src = _contains(src, MEDULLA_RE)
    medulla_dst = _contains(dst, MEDULLA_RE)
    motion_src = _contains(src, MOTION_RE)
    motion_dst = _contains(dst, MOTION_RE)
    output_src = _contains(src, OUTPUT_RE)
    output_dst = _contains(dst, OUTPUT_RE)
    roi = edges["edge_major_roi"].astype(str)

    motion_types = set(edges.loc[motion_src, "src_type"].astype(str)) | set(edges.loc[motion_dst, "dst_type"].astype(str))
    partner_edges = edges[motion_src | motion_dst]
    partner_types = set(partner_edges["src_type"].astype(str)) | set(partner_edges["dst_type"].astype(str))
    motion_partner_mask = src.astype(str).isin(motion_types | partner_types) | dst.astype(str).isin(motion_types | partner_types)

    groups = [
        (
            "lamina_input_L1_L2_L3_L4_L5",
            "curated_pathway",
            1,
            "Lamina input channels expected to be relatively specific entry points into medulla circuits.",
            edges[lamina_src | lamina_dst].copy(),
        ),
        (
            "medulla_intrinsic_Mi_Tm_TmY",
            "curated_pathway",
            2,
            "Medulla intrinsic channels are expected to integrate lamina and local visual features.",
            edges[medulla_src | medulla_dst].copy(),
        ),
        (
            "motion_T4_T5_plus_direct_partners",
            "curated_pathway",
            3,
            "T4/T5 motion pathways should show selective convergence rather than arbitrary type mixing.",
            edges[motion_partner_mask].copy(),
        ),
        (
            "output_projection_LC_LPLC_LT_DN",
            "curated_pathway",
            4,
            "Projection/output channels are expected to pool broader upstream histories for downstream readout.",
            edges[output_src | output_dst].copy(),
        ),
        (
            "lamina_to_medulla",
            "directional_stage",
            1,
            "Feedforward lamina-to-medulla edges should preserve structured lamina input identity.",
            edges[lamina_src & (medulla_dst | motion_dst) & roi.eq("ME(R)")].copy(),
        ),
        (
            "medulla_to_lobula",
            "directional_stage",
            2,
            "Medulla-to-lobula edges should increase pooling into object/projection-related channels.",
            edges[medulla_src & roi.eq("LO(R)")].copy(),
        ),
        (
            "medulla_to_lobula_plate",
            "directional_stage",
            3,
            "Medulla-to-lobula-plate edges include motion-associated transformations.",
            edges[(medulla_src | motion_src) & roi.eq("LOP(R)")].copy(),
        ),
        (
            "lobula_lobula_plate_to_projection_output",
            "directional_stage",
            4,
            "Lobula/lobula-plate output edges should pool information into projection/output types.",
            edges[output_dst & roi.isin(["LO(R)", "LOP(R)"])].copy(),
        ),
    ]
    return groups


def _interpretation(row: pd.Series) -> str:
    obs = row["history_ambiguity_bits"]
    degree_diff = row.get("observed_minus_degree_null_bits", np.nan)
    if not np.isfinite(obs):
        return "Insufficient typed edges for interpretation."
    if np.isfinite(degree_diff) and degree_diff < -0.25:
        return "Observed channel is more type-specific than degree-preserving rewiring, consistent with organized visual pathway structure."
    if np.isfinite(degree_diff) and abs(degree_diff) <= 0.25:
        return "Observed ambiguity is close to the degree-preserving null; anatomical degree/weight structure may explain much of the pattern."
    return "Observed channel is more ambiguous than the degree-preserving null; interpret as broad pooling or heterogeneous pathway composition."


def _make_figure(df: pd.DataFrame, out_path: Path) -> None:
    stages = df[df["group_kind"].eq("directional_stage")].sort_values("stage_order").copy()
    if stages.empty:
        return
    sns.set_theme(style="whitegrid", context="paper")
    x = np.arange(len(stages))
    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        x,
        stages["degree_null_mean_history_ambiguity_bits"],
        yerr=stages["degree_null_std_history_ambiguity_bits"].fillna(0),
        fmt="o-",
        color="#9E9E9E",
        label="Degree-preserving null",
        capsize=4,
    )
    plt.scatter(x, stages["history_ambiguity_bits"], color="#D62728", s=55, label="Observed", zorder=3)
    plt.plot(x, stages["history_ambiguity_bits"], color="#D62728", linewidth=1.5)
    plt.xticks(x, stages["group"].str.replace("_", " "), rotation=20, ha="right")
    plt.ylabel("H(src type | dst type), bits")
    plt.xlabel("Visual hierarchy stage")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="outputs/processed_typed_core/edges.parquet")
    parser.add_argument("--out-dir", default="outputs/known_visual_pathway_validation")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--n-nulls", type=int, default=20)
    parser.add_argument("--sample-edges", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_parquet(args.edges)
    groups = _build_groups(edges)
    rows = []
    failures = []
    expectations = {}
    for i, (name, kind, stage, expected, subset) in enumerate(groups):
        row, fail = _null_stats(
            name,
            kind,
            stage,
            subset,
            n_nulls=args.n_nulls,
            sample_edges=args.sample_edges,
            seed=args.seed + i * 1000,
        )
        if row:
            rows.append(row)
            expectations[name] = expected
        failures.extend(fail)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "known_visual_pathway_validation.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "known_visual_pathway_null_failures.csv", index=False)

    interp = results.copy()
    interp["expected_computation"] = interp["group"].map(expectations)
    interp["observed_ambiguity"] = interp["history_ambiguity_bits"]
    interp["interpretation"] = interp.apply(_interpretation, axis=1)
    interp[
        ["group", "expected_computation", "observed_ambiguity", "interpretation"]
    ].to_csv(tables_dir / "known_visual_pathway_interpretation.csv", index=False)

    _make_figure(results, figures_dir / "fig_known_visual_hierarchy_validation.png")
    print(f"Wrote known visual pathway validation to {out_dir}, {tables_dir}, and {figures_dir}")


if __name__ == "__main__":
    main()
