#!/usr/bin/env python
"""Run a synthetic ThermoFly demo.

Outputs:
- synthetic_edges.csv and synthetic_metadata.csv
- local_history_loss.csv
- transition_metrics.csv
- null_comparison.csv
- synthetic_heatmap.png

Optional: --train-snn trains a tiny connectome-masked LIF model for a smoke test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thermofly.data import adjacency_from_edges
from thermofly.graph_metrics import local_postsynaptic_history_loss, transition_metrics_from_adjacency
from thermofly.nulls import directed_degree_preserving_rewire
from thermofly.synthetic import make_layered_connectome
from thermofly.visualize import scatter_heatmap


def run_core(out_dir: Path, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    edges, meta = make_layered_connectome(seed=seed)
    edges.to_csv(out_dir / "synthetic_edges.csv", index=False)
    meta.to_csv(out_dir / "synthetic_metadata.csv", index=False)

    local = local_postsynaptic_history_loss(edges)
    local.to_csv(out_dir / "local_history_loss.csv", index=False)

    A, node_ids = adjacency_from_edges(edges)
    metrics = transition_metrics_from_adjacency(A)
    pd.DataFrame([metrics]).to_csv(out_dir / "transition_metrics.csv", index=False)

    null_rows = []
    for i in range(5):
        rewired = directed_degree_preserving_rewire(edges, n_swaps=10 * len(edges), seed=seed + i)
        local_null = local_postsynaptic_history_loss(rewired)
        A_null, _ = adjacency_from_edges(rewired, nodes=node_ids)
        m_null = transition_metrics_from_adjacency(A_null)
        null_rows.append(
            {
                "null_id": i,
                "mean_local_history_loss_bits": local_null["history_loss_bits_local"].mean(),
                "weighted_history_loss_bits": local_null["history_loss_bits_weighted"].sum(),
                "channel_landauer_bits": m_null["landauer_relevant_bits"],
                "channel_history_loss_bits": m_null["history_loss_bits"],
            }
        )
    real_row = {
        "null_id": "real",
        "mean_local_history_loss_bits": local["history_loss_bits_local"].mean(),
        "weighted_history_loss_bits": local["history_loss_bits_weighted"].sum(),
        "channel_landauer_bits": metrics["landauer_relevant_bits"],
        "channel_history_loss_bits": metrics["history_loss_bits"],
    }
    comp = pd.concat([pd.DataFrame([real_row]), pd.DataFrame(null_rows)], ignore_index=True)
    comp.to_csv(out_dir / "null_comparison.csv", index=False)
    scatter_heatmap(local, meta, out_dir / "synthetic_heatmap.png")

    print("Wrote demo outputs to", out_dir)
    print(pd.DataFrame([metrics]).T.rename(columns={0: "value"}).head(12))
    print("\nNull comparison:")
    print(comp)


def run_snn(out_dir: Path, seed: int = 0) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from thermofly.snn import ConnectomeMaskedLIF, ThermoLossWeights, thermodynamic_loss
    from thermofly.synthetic import make_toy_event_dataset

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = make_toy_event_dataset(seed=seed)
    X = torch.tensor(x)
    Y = torch.tensor(y)
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    n_inputs = X.shape[-1]
    n_hidden = 32
    # Sparse recurrent mask: ring plus random edges.
    mask = torch.zeros(n_hidden, n_hidden)
    for i in range(n_hidden):
        mask[(i + 1) % n_hidden, i] = 1
        mask[(i - 1) % n_hidden, i] = 1
    random_edges = rng.integers(0, n_hidden, size=(80, 2))
    for a, b in random_edges:
        if a != b:
            mask[b, a] = 1
    model = ConnectomeMaskedLIF(n_inputs, n_hidden, 2, mask)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_weights = ThermoLossWeights(spike=1e-4, landauer_proxy=1e-3)
    logs = []
    for epoch in range(3):
        total = 0.0
        correct = 0
        n = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits, stats = model(xb)
            input_rates = xb.mean(dim=1)
            hidden_rates = stats["rates"]
            loss, parts = thermodynamic_loss(logits, yb, input_rates, hidden_rates, stats, loss_weights)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum())
            n += xb.size(0)
        row = {"epoch": epoch, "loss": total / n, "accuracy": correct / n}
        row.update({k: float(v) for k, v in parts.items()})
        logs.append(row)
        print(row)
    pd.DataFrame(logs).to_csv(out_dir / "snn_smoke_test_log.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("demo_outputs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-snn", action="store_true")
    args = parser.parse_args()
    run_core(args.out, seed=args.seed)
    if args.train_snn:
        run_snn(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
