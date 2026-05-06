#!/usr/bin/env python
"""Run connectome-constrained physical learning-efficiency proxy experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.learning.local_rules import train_message_passing_model
from thermodynamic_connectomics.learning.null_graphs import apply_condition
from thermodynamic_connectomics.learning.subgraphs import build_matched_learning_subgraph, load_typed_core
from thermodynamic_connectomics.learning.synthetic_tasks import generate_task_data


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_ci(values: pd.Series, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = rng.choice(clean, size=len(clean), replace=True).mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _paired_outputs(final: pd.DataFrame, seed: int, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired real-minus-null tables over seed x subgraph_sample."""

    rows = []
    key_cols = ["pathway", "task", "mode", "seed", "sample_id"]
    real = final[final["condition"].eq("real")].set_index(key_cols)
    for null_condition in ["degree_preserving_null", "type_label_null"]:
        null = final[final["condition"].eq(null_condition)].set_index(key_cols)
        common = real.index.intersection(null.index)
        for key in common:
            r = real.loc[key]
            n = null.loc[key]
            cost_real = float(r["total_synaptic_update_L1"])
            cost_null = float(n["total_synaptic_update_L1"])
            delta_cost = cost_real - cost_null
            delta_mi = float(r["delta_task_MI_bits"]) - float(n["delta_task_MI_bits"])
            row = dict(zip(key_cols, key))
            row.update(
                {
                    "null_condition": null_condition,
                    "real_minus_null_accuracy": float(r["final_accuracy"]) - float(n["final_accuracy"]),
                    "real_minus_null_delta_MI": delta_mi,
                    "real_minus_null_cost": delta_cost,
                    "real_minus_null_eta": float(r["eta_phys_L1"]) - float(n["eta_phys_L1"]),
                    "conditional_eta_gain": delta_mi / (delta_cost + 1e-12),
                    "real_accuracy": float(r["final_accuracy"]),
                    "null_accuracy": float(n["final_accuracy"]),
                    "real_delta_MI": float(r["delta_task_MI_bits"]),
                    "null_delta_MI": float(n["delta_task_MI_bits"]),
                    "real_cost": cost_real,
                    "null_cost": cost_null,
                    "exploratory": bool(r.get("exploratory", False)) or bool(n.get("exploratory", False)),
                }
            )
            rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "paired_comparisons.csv", index=False)

    decisive_rows = []
    if not paired.empty:
        degree = paired[(paired["mode"].eq("local_hebbian")) & (paired["null_condition"].eq("degree_preserving_null"))]
        metrics = [
            "real_minus_null_accuracy",
            "real_minus_null_delta_MI",
            "real_minus_null_cost",
            "real_minus_null_eta",
            "conditional_eta_gain",
        ]
        for (pathway, task), group in degree.groupby(["pathway", "task"]):
            out_row = {"pathway": pathway, "task": task}
            for metric in metrics:
                out_row[metric] = float(group[metric].replace([np.inf, -np.inf], np.nan).mean())
                digest = hashlib.sha1(f"{pathway}|{task}|{metric}".encode("utf-8")).hexdigest()
                low, high = _bootstrap_ci(group[metric], seed=seed + int(digest[:8], 16) % 100_000)
                out_row[f"{metric}_ci_low"] = low
                out_row[f"{metric}_ci_high"] = high
            out_row["CI"] = (
                f"conditional_eta_gain "
                f"[{out_row['conditional_eta_gain_ci_low']:.4g}, {out_row['conditional_eta_gain_ci_high']:.4g}]"
            )
            if group["exploratory"].any():
                support = "exploratory"
            elif (
                out_row["real_minus_null_accuracy_ci_low"] > 0
                and out_row["real_minus_null_delta_MI_ci_low"] > 0
                and out_row["real_minus_null_eta_ci_low"] > 0
            ):
                support = "yes"
            else:
                support = "no"
            out_row["supported?"] = support
            decisive_rows.append(out_row)
    decisive = pd.DataFrame(decisive_rows)
    decisive.to_csv(out / "matched_decisive_table.csv", index=False)
    return paired, decisive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="outputs/processed_typed_core/edges.parquet")
    parser.add_argument("--nodes", default="outputs/processed_typed_core/nodes.parquet")
    parser.add_argument("--out-dir", default="outputs/learning_efficiency")
    parser.add_argument("--pathways", nargs="+", default=["lamina_to_medulla", "medulla_to_lobula_plate", "lobula_lobula_plate_to_projection_output"])
    parser.add_argument("--tasks", nargs="+", default=["polarity_contrast", "motion_direction", "global_pattern"])
    parser.add_argument("--conditions", nargs="+", default=["real", "degree_preserving_null", "type_label_null"])
    parser.add_argument("--modes", nargs="+", default=["readout_only", "local_hebbian"])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--max-edges", type=int, default=50000)
    parser.add_argument("--matched-nodes", type=int, default=500)
    parser.add_argument("--matched-edges", type=int, default=3000)
    parser.add_argument("--n-subgraph-samples", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hebbian-lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.0001)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    config = vars(args).copy()
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    edges, nodes = load_typed_core(args.edges, args.nodes)
    logs = [
        "Physical learning efficiency proxy experiment.",
        "No biological heat or real fly learning is claimed.",
    ]
    subgraph_rows = []
    subgraphs = {}
    for pathway in args.pathways:
        for sample_id in range(args.n_subgraph_samples):
            graph, meta = build_matched_learning_subgraph(
                pathway,
                edges,
                nodes,
                matched_nodes=args.matched_nodes,
                matched_edges=args.matched_edges,
                sample_id=sample_id,
                seed=args.seed + 100_000 * args.pathways.index(pathway),
            )
            subgraphs[(pathway, sample_id)] = graph
            subgraph_rows.append(
                {
                    **meta,
                    "subgraph_n_nodes": graph.n_nodes,
                    "subgraph_n_edges": graph.n_edges,
                    "n_source_nodes": len(graph.source_nodes),
                    "n_target_nodes": len(graph.target_nodes),
                    "history_ambiguity_bits": graph.history_ambiguity_bits,
                    "mutual_information_bits": graph.mutual_information_bits,
                    "effective_preimage_size": graph.effective_preimage_size,
                }
            )
            if meta["exploratory"]:
                logs.append(
                    f"{pathway} sample {sample_id}: exploratory, only {meta['candidate_nodes']} candidate nodes "
                    f"and {meta['candidate_edges']} candidate edges for requested "
                    f"{args.matched_nodes} nodes/{args.matched_edges} edges"
                )
            else:
                logs.append(f"{pathway} sample {sample_id}: {graph.n_nodes} nodes, {graph.n_edges} edges")

    curves_rows = []
    final_rows = []
    iterator = tqdm(
        [
            (pathway, sample_id, task, condition, seed_i, mode)
            for pathway in args.pathways
            for sample_id in range(args.n_subgraph_samples)
            for task in args.tasks
            for condition in args.conditions
            for seed_i in range(args.n_seeds)
            for mode in args.modes
        ],
        desc="learning runs",
    )
    for pathway, sample_id, task, condition, seed_i, mode in iterator:
        base = subgraphs[(pathway, sample_id)]
        run_seed = args.seed + seed_i + 1000 * sample_id
        graph = apply_condition(base, condition, seed=run_seed + 10_000)
        # Same task-data seed across real/null conditions for a pathway/task/seed/sample.
        task_data = generate_task_data(
            graph,
            graph.node_table,
            task_name=task,
            n_samples=args.n_samples,
            noise=args.noise,
            seed=args.seed + seed_i + 1000 * args.tasks.index(task) + 100_000 * sample_id,
        )
        curves, summary = train_message_passing_model(
            graph,
            task_data,
            mode=mode,
            epochs=args.epochs,
            lr=args.lr,
            hebbian_lr=args.hebbian_lr,
            decay=args.decay,
            seed=run_seed,
        )
        common = {
            "pathway": pathway,
            "sample_id": sample_id,
            "task": task,
            "condition": condition,
            "seed": seed_i,
            "mode": mode,
        }
        for row in curves:
            curves_rows.append({**common, **row})
        final_rows.append(
            {
                **common,
                **summary,
                "history_ambiguity_bits": graph.history_ambiguity_bits,
                "effective_preimage_size": graph.effective_preimage_size,
                "mutual_information_bits": graph.mutual_information_bits,
                "subgraph_n_nodes": graph.n_nodes,
                "subgraph_n_edges": graph.n_edges,
                "exploratory": bool(
                    next(
                        row["exploratory"]
                        for row in subgraph_rows
                        if row["pathway"] == pathway and row["sample_id"] == sample_id
                    )
                ),
            }
        )

    final = pd.DataFrame(final_rows)
    pd.DataFrame(curves_rows).to_csv(out / "learning_curves.csv", index=False)
    final.to_csv(out / "final_summary.csv", index=False)
    pd.DataFrame(subgraph_rows).to_csv(out / "subgraph_summary.csv", index=False)
    paired, decisive = _paired_outputs(final, args.seed, out)
    logs.append(f"paired comparison rows: {len(paired)}")
    logs.append(f"decisive table rows: {len(decisive)}")
    _write_log(out / "logs.txt", logs)
    print(f"Wrote learning-efficiency outputs to {out}")


if __name__ == "__main__":
    main()
