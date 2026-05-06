#!/usr/bin/env python
"""Write a conservative report for rule discovery/validation outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.learning.rules import rule_formula


def _md(df: pd.DataFrame, n: int = 20) -> str:
    if df.empty:
        return "_No rows available._"
    return df.head(n).to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-dir", default="outputs/rule_discovery")
    parser.add_argument("--validation-dir", default="outputs/rule_discovery_validation")
    parser.add_argument("--out", default="outputs/rule_discovery_validation/report.md")
    args = parser.parse_args()
    discovery = Path(args.discovery_dir)
    validation = Path(args.validation_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(discovery / "discovery_summary.csv") if (discovery / "discovery_summary.csv").exists() else pd.DataFrame()
    decisive = pd.read_csv(validation / "validation_decisive_table.csv") if (validation / "validation_decisive_table.csv").exists() else pd.DataFrame()
    candidate_path = discovery / "best_candidate_rule.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8")) if candidate_path.exists() else {"name": "none"}
    supported = bool(decisive["supported"].any()) if not decisive.empty and "supported" in decisive.columns else False
    skipped = (validation / "validation_skipped.json").exists()
    if supported:
        answer = "A local history-ambiguity-gated rule is a candidate algorithm under this synthetic proxy."
    elif not decisive.empty and decisive["beats_readout_only"].any():
        answer = "A candidate rule partially improves the proxy but does not clear all null/baseline controls."
    elif skipped:
        answer = "No new learning algorithm is supported; validation was skipped because discovery did not beat readout-only in the representation-probe protocol."
    else:
        answer = "No new learning algorithm is supported; the result is representational separability."
    report = f"""# Rule Discovery: History-Ambiguity-Gated Plasticity

## 1. Motivation
The anatomical hierarchy showed structured history ambiguity. The rescue run showed a medulla-to-lobula-plate motion signal. Here we ask whether a local rule can exploit this better than standard local baselines and readout-only.

## 2. Candidate Rules
The search includes no-plasticity/readout-only, Hebbian, reward-Hebbian, Oja, reward-Oja, BCM, reward-BCM, ambiguity-gated Hebbian, ambiguity-gated Oja, and energy-normalized ambiguity-gated Oja. Symbolic rules are sparse combinations of local basis terms when requested.

## 3. Search Method
Discovery scores reward positive plasticity delta MI, real-minus-degree-null gains, and accuracy gains, while penalizing update cost, instability, and rule complexity. Eta is not rewarded when delta MI is non-positive.

Best discovery candidate:

```json
{json.dumps(candidate, indent=2)}
```

Formula:

`{rule_formula(candidate)}`

Top discovery rows:

{_md(summary, 15)}

## 4. Validation Design
Validation uses held-out seeds and subgraph samples, the real graph, degree-preserving nulls, weight-shuffled real topology, uniform-weight real topology, and the representation probe as the stricter test.

## 5. Main Validation Result
Validation was skipped when the discovery screen failed the readout-only/plasticity-gain gate.

{_md(decisive, 30)}

## 6. Did We Discover a New Algorithm?
{answer}

## 7. Limits
- Synthetic motion task.
- Simplified one-step message passing.
- No measured activity, stimulus movie, behavior, or synaptic biophysics.
- Approximate degree-preserving nulls.
- The result is a plasticity-efficiency proxy, not biological learning or thermodynamic heat.

## 8. Next Experiment
Use real visual stimulus/activity datasets or a biophysical SNN only after a rule clears readout-only and null controls in the representation-probe protocol.
"""
    out.write_text(report, encoding="utf-8")
    print(f"Wrote rule discovery report to {out}")


if __name__ == "__main__":
    main()
