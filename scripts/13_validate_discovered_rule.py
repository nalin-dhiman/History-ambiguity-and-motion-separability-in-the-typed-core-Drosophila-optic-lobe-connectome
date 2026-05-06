#!/usr/bin/env python
"""Validate a discovered local plasticity rule on held-out seeds/samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.learning.null_graphs import apply_condition
from thermodynamic_connectomics.learning.rule_search import evaluate_rule
from thermodynamic_connectomics.learning.rules import make_rule, rule_formula, serialize_rule
from thermodynamic_connectomics.learning.subgraphs import build_matched_learning_subgraph, load_typed_core
from thermodynamic_connectomics.learning.synthetic_tasks import generate_task_data


def _ci(values: pd.Series, seed: int = 0, n_boot: int = 3000) -> tuple[float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(clean, size=len(clean), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _paired(results: pd.DataFrame, rule: str, control_rule: str, protocol: str, condition: str = "real") -> pd.DataFrame:
    key = ["seed", "sample_id"]
    a = results[(results["rule_name"].eq(rule)) & (results["protocol"].eq(protocol)) & (results["condition"].eq(condition))].set_index(key)
    b = results[(results["rule_name"].eq(control_rule)) & (results["protocol"].eq(protocol)) & (results["condition"].eq(condition))].set_index(key)
    common = a.index.intersection(b.index)
    rows = []
    for k in common:
        ar = a.loc[k]
        br = b.loc[k]
        rows.append(
            {
                "seed": k[0],
                "sample_id": k[1],
                "accuracy_gain": float(ar["final_accuracy"] - br["final_accuracy"]),
                "MI_gain": float(ar["final_task_MI_bits"] - br["final_task_MI_bits"]),
                "cost": float(ar["total_update_L1"]),
            }
        )
    return pd.DataFrame(rows)


def _beats(results: pd.DataFrame, rule: str, controls: list[str], protocol: str, seed: int) -> bool:
    for control in controls:
        if control not in set(results["rule_name"]):
            continue
        paired = _paired(results, rule, control, protocol)
        if paired.empty:
            return False
        acc_low, _ = _ci(paired["accuracy_gain"], seed=seed)
        mi_low, _ = _ci(paired["MI_gain"], seed=seed + 17)
        if not (acc_low >= 0 and mi_low > 0):
            return False
    return True


def _real_vs_null(results: pd.DataFrame, rule: str, protocol: str, null_condition: str, seed: int) -> bool:
    key = ["seed", "sample_id"]
    real = results[(results["rule_name"].eq(rule)) & (results["protocol"].eq(protocol)) & (results["condition"].eq("real"))].set_index(key)
    null = results[(results["rule_name"].eq(rule)) & (results["protocol"].eq(protocol)) & (results["condition"].eq(null_condition))].set_index(key)
    common = real.index.intersection(null.index)
    if len(common) == 0:
        return False
    rows = []
    for k in common:
        rows.append(
            {
                "accuracy_gain": float(real.loc[k]["final_accuracy"] - null.loc[k]["final_accuracy"]),
                "MI_gain": float(real.loc[k]["final_task_MI_bits"] - null.loc[k]["final_task_MI_bits"]),
            }
        )
    paired = pd.DataFrame(rows)
    acc_low, _ = _ci(paired["accuracy_gain"], seed=seed)
    mi_low, _ = _ci(paired["MI_gain"], seed=seed + 31)
    return bool(acc_low >= 0 and mi_low > 0)


def decisive_table(results: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    baseline_rules = set(results["rule_name"])
    for (rule, protocol), _group in results.groupby(["rule_name", "protocol"]):
        if rule == "no_plasticity":
            continue
        paired = _paired(results, rule, "no_plasticity", protocol)
        if paired.empty:
            continue
        acc_low, acc_high = _ci(paired["accuracy_gain"], seed=seed)
        mi_low, mi_high = _ci(paired["MI_gain"], seed=seed + 1)
        eta = paired["MI_gain"].clip(lower=0) / (paired["cost"] + 1e-12)
        eta_low, eta_high = _ci(eta, seed=seed + 2)
        beats_readout = bool(acc_low >= 0 and mi_low > 0)
        beats_hebb = _beats(results, rule, ["hebbian", "reward_hebbian"], protocol, seed + 3)
        beats_oja = _beats(results, rule, ["oja", "reward_oja"], protocol, seed + 5)
        beats_bcm = _beats(results, rule, ["bcm", "reward_bcm"], protocol, seed + 7)
        beats_degree = _real_vs_null(results, rule, protocol, "degree_preserving_null", seed + 11)
        beats_weight = _real_vs_null(results, rule, protocol, "weight_shuffled_real_topology", seed + 13)
        beats_uniform = _real_vs_null(results, rule, protocol, "uniform_weight_real_topology", seed + 17)
        supported = bool(beats_readout and beats_hebb and beats_oja and beats_bcm and beats_degree and mi_low > 0 and acc_low >= 0)
        if not beats_readout:
            interpretation = "connectome motion separability, not learned plasticity"
        elif not beats_degree:
            interpretation = "plasticity helps, but not specific to real topology"
        elif not beats_weight:
            interpretation = "topology matters more than synaptic weights"
        elif beats_uniform:
            interpretation = "binary topology may be sufficient"
        else:
            interpretation = "candidate history-ambiguity-gated learning algorithm" if supported else "partial proxy improvement"
        rows.append(
            {
                "rule_name": rule,
                "protocol": protocol,
                "condition_comparison": "real_rule_vs_readout_baselines_nulls",
                "mean_accuracy_gain": float(paired["accuracy_gain"].mean()),
                "accuracy_gain_ci_low": acc_low,
                "accuracy_gain_ci_high": acc_high,
                "mean_MI_gain": float(paired["MI_gain"].mean()),
                "MI_gain_ci_low": mi_low,
                "MI_gain_ci_high": mi_high,
                "mean_update_cost": float(paired["cost"].mean()),
                "mean_eta_plastic": float(eta.mean()),
                "eta_ci_low": eta_low,
                "eta_ci_high": eta_high,
                "beats_readout_only": beats_readout,
                "beats_hebbian": beats_hebb,
                "beats_oja": beats_oja,
                "beats_bcm": beats_bcm,
                "beats_degree_null": beats_degree,
                "beats_weight_shuffled": beats_weight,
                "beats_uniform_weight": beats_uniform,
                "supported": supported,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _write_figures(out: Path, results: pd.DataFrame, decisive: pd.DataFrame) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    if decisive.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=decisive, x="rule_name", y="mean_MI_gain", hue="protocol")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_validated_rule_vs_readout.png", dpi=220)
    plt.close()
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results, x="condition", y="final_task_MI_bits", hue="rule_name")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_validated_rule_vs_nulls.png", dpi=220)
    plt.close()
    plt.figure(figsize=(7, 4))
    sns.barplot(data=decisive, x="rule_name", y="MI_gain_ci_low", hue="protocol")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_plasticity_delta_MI_CI.png", dpi=220)
    plt.close()
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=decisive, x="mean_update_cost", y="mean_MI_gain", hue="rule_name", style="protocol", s=100)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_accuracy_MI_cost_tradeoff.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="outputs/processed_typed_core/edges.parquet")
    parser.add_argument("--nodes", default="outputs/processed_typed_core/nodes.parquet")
    parser.add_argument("--candidate-rule", required=True)
    parser.add_argument("--out-dir", default="outputs/rule_discovery_validation")
    parser.add_argument("--pathway", default="medulla_to_lobula_plate")
    parser.add_argument("--task", default="motion_direction")
    parser.add_argument("--matched-nodes", type=int, default=1500)
    parser.add_argument("--matched-edges", type=int, default=12000)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[3, 4, 5, 6, 7])
    parser.add_argument("--validation-subgraph-samples", nargs="+", type=int, default=[3, 4, 5, 6, 7])
    parser.add_argument("--conditions", nargs="+", default=["real", "degree_preserving_null", "weight_shuffled_real_topology", "uniform_weight_real_topology"])
    parser.add_argument("--baseline-rules", nargs="+", default=["no_plasticity", "hebbian", "reward_hebbian", "oja", "reward_oja", "bcm", "reward_bcm", "ambiguity_gated_hebbian", "ambiguity_gated_oja", "energy_normalized_ambiguity_oja"])
    parser.add_argument("--protocols", nargs="+", default=["online_readout", "representation_probe"])
    parser.add_argument("--readout-lr", type=float, default=0.03)
    parser.add_argument("--plasticity-lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.0001)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    candidate = json.loads(Path(args.candidate_rule).read_text(encoding="utf-8"))
    rules = [make_rule(name) for name in args.baseline_rules]
    candidate_rule = make_rule(candidate.get("name", "candidate"), active_basis=candidate.get("active_basis") or None)
    if candidate_rule.name not in {r.name for r in rules}:
        rules.append(candidate_rule)
    edges, nodes = load_typed_core(args.edges, args.nodes)
    subgraphs = {}
    for sample_id in args.validation_subgraph_samples:
        graph, _ = build_matched_learning_subgraph(
            args.pathway,
            edges,
            nodes,
            matched_nodes=args.matched_nodes,
            matched_edges=args.matched_edges,
            sample_id=sample_id,
            seed=args.seed,
        )
        subgraphs[sample_id] = graph
    rows = []
    iterator = tqdm(
        [
            (sample_id, seed, condition, protocol, rule)
            for sample_id in args.validation_subgraph_samples
            for seed in args.validation_seeds
            for condition in args.conditions
            for protocol in args.protocols
            for rule in rules
        ],
        desc="validation",
    )
    for sample_id, seed, condition, protocol, rule in iterator:
        graph = apply_condition(subgraphs[sample_id], condition, seed=args.seed + seed + 10_000 * sample_id)
        task_data = generate_task_data(
            graph,
            graph.node_table,
            task_name=args.task,
            n_samples=args.n_samples,
            noise=args.noise,
            seed=args.seed + seed + 100_000 * sample_id,
        )
        metrics = evaluate_rule(
            graph,
            task_data,
            rule=rule,
            protocol=protocol,
            seed=args.seed + seed,
            epochs=args.epochs,
            readout_lr=args.readout_lr,
            plasticity_lr=args.plasticity_lr,
            decay=args.decay,
        )
        rows.append(
            {
                "pathway": args.pathway,
                "task": args.task,
                "sample_id": sample_id,
                "seed": seed,
                "condition": condition,
                "protocol": protocol,
                "rule_name": rule.name,
                "rule_formula": rule_formula(serialize_rule(rule)),
                **metrics,
            }
        )
    results = pd.DataFrame(rows)
    decisive = decisive_table(results, seed=args.seed)
    results.to_csv(out / "validation_results.csv", index=False)
    decisive.to_csv(out / "validation_decisive_table.csv", index=False)
    # Long-form paired table for external checks.
    paired_rows = []
    for row in decisive.itertuples(index=False):
        paired_rows.append(row._asdict())
    pd.DataFrame(paired_rows).to_csv(out / "validation_paired_comparisons.csv", index=False)
    _write_figures(out, results, decisive)
    report = [
        "# Rule Discovery Validation",
        "",
        "Held-out validation of the candidate local rule against readout-only, standard local rules, and matched nulls.",
        "",
        decisive.to_markdown(index=False) if not decisive.empty else "_No decisive rows._",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote validation outputs to {out}")


if __name__ == "__main__":
    main()
