#!/usr/bin/env python
"""Discover candidate local plasticity rules in the rescued motion setting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.learning.history_gates import compute_target_history_ambiguity
from thermodynamic_connectomics.learning.null_graphs import apply_condition
from thermodynamic_connectomics.learning.rule_search import evaluate_rule, make_rules, score_rule_results, symbolic_rule_specs
from thermodynamic_connectomics.learning.rules import rule_formula, serialize_rule
from thermodynamic_connectomics.learning.subgraphs import build_matched_learning_subgraph, load_typed_core
from thermodynamic_connectomics.learning.synthetic_tasks import generate_task_data


def _write_figures(out: Path, results: pd.DataFrame, scores: pd.DataFrame) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    if not scores.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=scores.head(25), x="score", y="rule_name", hue="protocol", dodge=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_rule_search_score.png", dpi=220)
        plt.close()
    if not results.empty:
        real = results[results["condition"].eq("real")]
        plt.figure(figsize=(9, 4))
        sns.barplot(data=real, x="rule_name", y="plasticity_delta_MI", hue="protocol")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_rule_plasticity_gain.png", dpi=220)
        plt.close()
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=real, x="total_update_L1", y="eta_plastic", hue="rule_name", style="protocol")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_rule_efficiency_vs_cost.png", dpi=220)
        plt.close()
        top = scores.head(8)["rule_name"].tolist() if not scores.empty else real["rule_name"].unique()[:8].tolist()
        plt.figure(figsize=(9, 4))
        sns.barplot(data=real[real["rule_name"].isin(top)], x="rule_name", y="final_accuracy", hue="protocol")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_best_rule_vs_baselines.png", dpi=220)
        plt.close()
        gate_cols = [c for c in ["mean_gate_S", "plasticity_delta_MI"] if c in real.columns]
        if len(gate_cols) == 2:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=real, x="mean_gate_S", y="plasticity_delta_MI", hue="rule_name")
            plt.tight_layout()
            plt.savefig(fig_dir / "fig_history_gate_effect.png", dpi=220)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="outputs/processed_typed_core/edges.parquet")
    parser.add_argument("--nodes", default="outputs/processed_typed_core/nodes.parquet")
    parser.add_argument("--out-dir", default="outputs/rule_discovery")
    parser.add_argument("--pathway", default="medulla_to_lobula_plate")
    parser.add_argument("--task", default="motion_direction")
    parser.add_argument("--matched-nodes", type=int, default=1500)
    parser.add_argument("--matched-edges", type=int, default=12000)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--discovery-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--discovery-subgraph-samples", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", default=["real", "degree_preserving_null", "weight_shuffled_real_topology", "uniform_weight_real_topology"])
    parser.add_argument("--rules", nargs="+", default=["no_plasticity", "hebbian", "reward_hebbian", "oja", "reward_oja", "bcm", "reward_bcm", "ambiguity_gated_hebbian", "ambiguity_gated_oja", "energy_normalized_ambiguity_oja"])
    parser.add_argument("--protocols", nargs="+", default=["online_readout", "representation_probe"])
    parser.add_argument("--run-symbolic-search", action="store_true")
    parser.add_argument("--max-symbolic-rules", type=int, default=200)
    parser.add_argument("--readout-lr", type=float, default=0.03)
    parser.add_argument("--plasticity-lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.0001)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    edges, nodes = load_typed_core(args.edges, args.nodes)
    symbolic_specs = symbolic_rule_specs(args.max_symbolic_rules) if args.run_symbolic_search else []
    rules = make_rules(args.rules, symbolic_specs)
    subgraphs = {}
    gate_rows = []
    for sample_id in args.discovery_subgraph_samples:
        graph, meta = build_matched_learning_subgraph(
            args.pathway,
            edges,
            nodes,
            matched_nodes=args.matched_nodes,
            matched_edges=args.matched_edges,
            sample_id=sample_id,
            seed=args.seed,
        )
        subgraphs[sample_id] = graph
        gates = compute_target_history_ambiguity(graph.edges, graph.node_table)
        gates["sample_id"] = sample_id
        gates["pathway"] = args.pathway
        gates["exploratory"] = meta["exploratory"]
        gate_rows.append(gates)
    gate_table = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    gate_table.to_csv(out / "history_gates.csv", index=False)

    rows = []
    iterator = tqdm(
        [
            (sample_id, seed, condition, protocol, rule)
            for sample_id in args.discovery_subgraph_samples
            for seed in args.discovery_seeds
            for condition in args.conditions
            for protocol in args.protocols
            for rule in rules
        ],
        desc="rule discovery",
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
                "mean_gate_S": float(gate_table[gate_table["sample_id"].eq(sample_id)]["S_inverse_ambiguity"].mean()) if not gate_table.empty else 1.0,
                **metrics,
            }
        )
    results = pd.DataFrame(rows)
    baseline = results[~results["rule_name"].str.startswith("symbolic:")].copy()
    symbolic = results[results["rule_name"].str.startswith("symbolic:")].copy()
    scores = score_rule_results(results)
    results.to_csv(out / "rule_search_results.csv", index=False)
    baseline.to_csv(out / "baseline_rule_results.csv", index=False)
    symbolic.to_csv(out / "symbolic_rule_results.csv", index=False)
    scores.to_csv(out / "discovery_summary.csv", index=False)
    best_name = scores.iloc[0]["rule_name"] if not scores.empty else (args.rules[0] if args.rules else "no_plasticity")
    best_rule = next((r for r in rules if r.name == best_name), rules[0])
    best_spec = serialize_rule(best_rule)
    best_spec["score"] = float(scores.iloc[0]["score"]) if not scores.empty else 0.0
    best_spec["formula"] = rule_formula(best_spec)
    (out / "best_candidate_rule.json").write_text(json.dumps(best_spec, indent=2), encoding="utf-8")
    _write_figures(out, results, scores)
    report = [
        "# Rule Discovery: History-Ambiguity-Gated Plasticity",
        "",
        "Discovery split only. This search is a synthetic-task, connectome-constrained proxy.",
        "",
        "## Best Candidates",
        scores.head(20).to_markdown(index=False) if not scores.empty else "_No scores available._",
        "",
        "## Guardrail",
        "A high discovery score is not validation. Validate only if the candidate improves representation beyond no_plasticity/readout-only.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote rule discovery outputs to {out}")


if __name__ == "__main__":
    main()
