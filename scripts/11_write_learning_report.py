#!/usr/bin/env python
"""Write report for physical learning-efficiency proxy experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _md(df: pd.DataFrame, n: int = 12) -> str:
    if df.empty:
        return "_No rows available._"
    return df.head(n).to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/learning_efficiency/final_summary.csv")
    parser.add_argument("--curves", default="outputs/learning_efficiency/learning_curves.csv")
    parser.add_argument("--subgraphs", default="outputs/learning_efficiency/subgraph_summary.csv")
    parser.add_argument("--out", default="outputs/learning_efficiency/report.md")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    subgraphs = pd.read_csv(args.subgraphs)
    decisive_path = out.parent / "matched_decisive_table.csv"
    paired_path = out.parent / "paired_comparisons.csv"
    decisive = pd.read_csv(decisive_path) if decisive_path.exists() else pd.DataFrame()
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    hebb = summary[summary["mode"].eq("local_hebbian")].copy()
    real = hebb[hebb["condition"].eq("real")]
    best = real.sort_values(["eta_phys_L1", "final_balanced_accuracy"], ascending=False)[
        ["pathway", "task", "final_accuracy", "final_balanced_accuracy", "delta_task_MI_bits", "eta_phys_L1", "history_ambiguity_bits"]
    ]
    if not paired.empty:
        comparison = paired[(paired["mode"].eq("local_hebbian")) & (paired["null_condition"].eq("degree_preserving_null"))]
        comparison = comparison.groupby(["pathway", "task"], as_index=False).agg(
            real_minus_null_accuracy=("real_minus_null_accuracy", "mean"),
            real_minus_null_delta_MI=("real_minus_null_delta_MI", "mean"),
            real_minus_null_cost=("real_minus_null_cost", "mean"),
            real_minus_null_eta=("real_minus_null_eta", "mean"),
            conditional_eta_gain=("conditional_eta_gain", "mean"),
        )
    else:
        comparison = pd.DataFrame()
    report = f"""# Physical Learning Efficiency Proxy in Optic-Lobe Pathway Subgraphs

## 1. Motivation
Static anatomical entropy showed a hierarchy of history ambiguity. Here we test whether that hierarchy predicts learning efficiency under controlled local updates.

## 2. Methods
- Subgraphs are typed-core pathway subsets sampled under a matched node/edge budget and target-normalized weights.
- Tasks are synthetic: polarity/contrast, motion direction, and global pattern classification.
- Conditions are real, approximate degree-preserving null, and type-label null.
- Modes are readout-only and reward-modulated local Hebbian learning.
- The main metric is task information gained per synaptic-update cost: a physical learning efficiency proxy, not real biological learning.
- Paired real-minus-null comparisons are computed over matched seed x subgraph-sample rows, with bootstrapped 95% confidence intervals.

Subgraph summary:

{_md(subgraphs, 10)}

## 3. Main Results
Decisive matched-size table, real vs degree-preserving null:

{_md(decisive[["pathway", "task", "real_minus_null_accuracy", "real_minus_null_delta_MI", "real_minus_null_cost", "conditional_eta_gain", "CI", "supported?"]] if not decisive.empty else decisive, 20)}

Best real local-Hebbian pathway-task rows:

{_md(best, 12)}

Mean paired real vs degree-null comparison:

{_md(comparison.sort_values("real_minus_null_eta", ascending=False) if not comparison.empty else comparison, 12)}

## 4. Interpretation
These results estimate pathway-specific learning profiles under the same simplified local rule and matched-size sampling. Positive paired real-minus-degree-null efficiency with a confidence interval above zero is the conservative support criterion. Exploratory rows, including pathways that cannot reach the matched budget, should not be treated as confirmatory.

## 5. Failure Modes and Limits
- Synthetic tasks may bias the result toward the hand-coded pathway groups.
- The local rule is simplified and not a model of fly synaptic biophysics.
- No measured activity, stimuli, or behavior are used.
- Degree-preserving nulls are approximate when exact rewiring would be too slow.
- Synaptic-update cost is a mathematical proxy, not thermodynamic heat.
- Readout-only performance can dominate if tasks are linearly separable without graph updates.

## 6. Next Experiment
Use real visual stimulus/activity datasets, then revisit this proxy with a more biophysical SNN only after the controlled local-learning result remains pathway-specific under stronger nulls.
"""
    out.write_text(report, encoding="utf-8")
    print(f"Wrote learning report to {out}")


if __name__ == "__main__":
    main()
