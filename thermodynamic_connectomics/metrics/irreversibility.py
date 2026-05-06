"""Landauer-inspired entropy contraction metrics for directed connectomes."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from thermodynamic_connectomics.metrics.entropy import shannon_entropy

K_B = 1.380649e-23
T_ROOM = 300.0
LN2 = math.log(2.0)


def _entropy_from_group_sums(df: pd.DataFrame, group_col: str, weight_col: str = "weight") -> pd.DataFrame:
    work = df[[group_col, weight_col]].copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    work = work[work[weight_col] > 0]
    work["w_log2_w"] = work[weight_col] * np.log2(work[weight_col])
    work["w_squared"] = work[weight_col] * work[weight_col]
    grouped = work.groupby(group_col, sort=False)
    out = grouped.agg(
        weighted_in_degree=(weight_col, "sum"),
        in_degree=(weight_col, "size"),
        sum_w_log2_w=("w_log2_w", "sum"),
        sum_w_squared=("w_squared", "sum"),
    ).reset_index()
    total = out["weighted_in_degree"].to_numpy(dtype=float)
    out["incoming_entropy_bits"] = np.log2(total) - out["sum_w_log2_w"].to_numpy(dtype=float) / total
    out["effective_preimage_size"] = np.power(2.0, out["incoming_entropy_bits"])
    out["history_ambiguity_bits"] = out["incoming_entropy_bits"]
    out["normalized_compression_index"] = np.where(
        out["in_degree"] > 1,
        out["incoming_entropy_bits"] / np.log2(out["in_degree"]),
        0.0,
    )
    out["hhi"] = out["sum_w_squared"].to_numpy(dtype=float) / np.square(total)
    return out.drop(columns=["sum_w_log2_w", "sum_w_squared"])


def compute_node_irreversibility(
    edges: pd.DataFrame,
    min_degree: int = 3,
    temperature_kelvin: float = T_ROOM,
) -> pd.DataFrame:
    """Compute node-level candidate coarse-graining proxies.

    `delta_H_node_bits` is the entropy of weighted incoming source identities.
    It is a statistical lower-bound proxy under the explicit model that the
    node aggregates distinguishable upstream histories into one downstream
    anatomical state; it is not a measured biological heat.
    """

    needed = {"src", "dst", "weight"}
    missing = needed - set(edges.columns)
    if missing:
        raise ValueError(f"edges missing required columns: {sorted(missing)}")
    base = _entropy_from_group_sums(edges[["dst", "weight"]].rename(columns={"dst": "node"}), "node")
    base = base.rename(columns={"node": "bodyId"})
    base = base[base["in_degree"] >= min_degree].copy()

    if "dst_type" in edges.columns:
        meta = edges[["dst", "dst_type", "dst_instance", "dst_roi", "dst_nt"]].drop_duplicates("dst")
        meta = meta.rename(
            columns={
                "dst": "bodyId",
                "dst_type": "type",
                "dst_instance": "instance",
                "dst_roi": "roi",
                "dst_nt": "nt",
            }
        )
        base = base.merge(meta, on="bodyId", how="left")
    for col in ["type", "instance", "roi", "nt"]:
        if col not in base.columns:
            base[col] = pd.NA

    if "src_type" in edges.columns:
        typed = edges.dropna(subset=["src_type"]).copy()
        typed["src_type"] = typed["src_type"].astype(str)
        typed = typed[typed["src_type"].str.len() > 0]
        if not typed.empty:
            by_type = typed.groupby(["dst", "src_type"], sort=False, as_index=False)["weight"].sum()
            type_h = _entropy_from_group_sums(by_type.rename(columns={"dst": "bodyId"}), "bodyId")
            type_h = type_h.rename(
                columns={
                    "incoming_entropy_bits": "src_type_entropy_before_bits",
                    "effective_preimage_size": "src_type_effective_preimage_size",
                }
            )[["bodyId", "src_type_entropy_before_bits", "src_type_effective_preimage_size"]]
            base = base.merge(type_h, on="bodyId", how="left")
    for col in ["src_type_entropy_before_bits", "src_type_effective_preimage_size"]:
        if col not in base.columns:
            base[col] = np.nan

    base["delta_H_node_bits"] = base["incoming_entropy_bits"].clip(lower=0)
    base["landauer_floor_kT"] = base["delta_H_node_bits"] * LN2
    base["landauer_floor_J_room_temp"] = base["delta_H_node_bits"] * K_B * temperature_kelvin * LN2
    base = base.sort_values("delta_H_node_bits", ascending=False).reset_index(drop=True)
    return base


def _weighted_entropy_by_category(df: pd.DataFrame, category_col: str, weight_col: str = "weight") -> float:
    weights = df.groupby(category_col, dropna=False)[weight_col].sum()
    return shannon_entropy(weights.to_numpy(), base=2)


def compute_block_irreversibility(edges: pd.DataFrame, group_col: str = "type") -> tuple[pd.DataFrame, dict]:
    """Compute cell-type/block history ambiguity metrics.

    The global channel is `P(dst_type | src_type)` weighted by synapse count.
    The principal compression metric reported here is `H(src_type | dst_type)`,
    the remaining ambiguity about upstream class after observing the downstream
    class.
    """

    src_col = f"src_{group_col}" if f"src_{group_col}" in edges.columns else "src_type"
    dst_col = f"dst_{group_col}" if f"dst_{group_col}" in edges.columns else "dst_type"
    if src_col not in edges.columns or dst_col not in edges.columns:
        empty = pd.DataFrame(
            columns=[
                "dst_type",
                "total_weight",
                "n_src_types",
                "history_ambiguity_bits",
                "compression_bits",
                "effective_preimage_size",
                "landauer_floor_kT",
                "landauer_floor_J_room_temp",
            ]
        )
        return empty, {"status": "missing_group_columns", "group_col": group_col}

    typed = edges[[src_col, dst_col, "weight"]].dropna().copy()
    typed[src_col] = typed[src_col].astype(str)
    typed[dst_col] = typed[dst_col].astype(str)
    typed = typed[(typed[src_col].str.len() > 0) & (typed[dst_col].str.len() > 0)]
    if typed.empty:
        return pd.DataFrame(), {"status": "no_typed_edges", "group_col": group_col}

    block = typed.groupby([src_col, dst_col], as_index=False, sort=False)["weight"].sum()
    total = float(block["weight"].sum())
    h_src = _weighted_entropy_by_category(block, src_col)
    h_dst = _weighted_entropy_by_category(block, dst_col)
    joint_p = block["weight"].to_numpy(dtype=float) / total
    h_joint = shannon_entropy(joint_p, base=2)
    mi = max(0.0, h_src + h_dst - h_joint)
    h_src_given_dst = max(0.0, h_joint - h_dst)
    h_dst_given_src = max(0.0, h_joint - h_src)

    rows = []
    for dst_value, sub in block.groupby(dst_col, sort=False):
        h = shannon_entropy(sub["weight"].to_numpy(dtype=float), base=2)
        rows.append(
            {
                "dst_type": dst_value,
                "total_weight": float(sub["weight"].sum()),
                "n_src_types": int(sub[src_col].nunique()),
                "history_ambiguity_bits": h,
                "compression_bits": h,
                "effective_preimage_size": float(2.0**h),
                "landauer_floor_kT": h * LN2,
                "landauer_floor_J_room_temp": h * K_B * T_ROOM * LN2,
            }
        )
    per_dst = pd.DataFrame(rows).sort_values(["history_ambiguity_bits", "total_weight"], ascending=False)
    summary = {
        "status": "ok",
        "group_col": group_col,
        "total_typed_weight": total,
        "n_src_types": int(block[src_col].nunique()),
        "n_dst_types": int(block[dst_col].nunique()),
        "H_src_type_bits": float(h_src),
        "H_dst_type_bits": float(h_dst),
        "H_src_type_given_dst_type_bits": float(h_src_given_dst),
        "H_dst_type_given_src_type_bits": float(h_dst_given_src),
        "mutual_information_bits": float(mi),
        "compression_bits": float(h_src - mi),
        "history_ambiguity_bits": float(h_src_given_dst),
        "effective_preimage_size": float(2.0**h_src_given_dst),
        "landauer_floor_kT": float(max(0.0, h_src - mi) * LN2),
        "landauer_floor_J_room_temp": float(max(0.0, h_src - mi) * K_B * T_ROOM * LN2),
    }
    return per_dst.reset_index(drop=True), summary


def write_metric_outputs(
    node_metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    global_summary: dict,
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    node_metrics.to_csv(out / "node_irreversibility.csv", index=False)
    block_metrics.to_csv(out / "block_irreversibility.csv", index=False)
    (out / "global_summary.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")
