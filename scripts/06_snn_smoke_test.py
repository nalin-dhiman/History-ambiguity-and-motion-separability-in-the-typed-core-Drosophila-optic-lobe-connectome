#!/usr/bin/env python
"""Lightweight connectome-masked recurrent smoke test.

This is a graph-dynamics sanity check, not a biologically faithful SNN model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermodynamic_connectomics.metrics.entropy import shannon_entropy


def _simulate(mask: sparse.csr_matrix, seed: int = 0, steps: int = 80) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = mask.shape[0]
    weights = mask.astype(float).copy()
    scale = 1.0 / max(1.0, float(np.sqrt(mask.getnnz(axis=1).mean() + 1)))
    weights.data = rng.normal(0.0, scale, size=weights.nnz)
    state = rng.random(n) < 0.05
    spikes = []
    for _ in range(steps):
        drive = weights.T @ state.astype(float) + rng.normal(0, 0.5, n)
        state = drive > 1.0
        spikes.append(state.astype(float))
    seq = np.vstack(spikes)
    rates = seq.mean(axis=0)
    return {
        "spike_count": float(seq.sum()),
        "activity_sparsity": float(1.0 - rates.mean()),
        "state_entropy": shannon_entropy(np.bincount(seq.sum(axis=1).astype(int)), base=2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="outputs/processed/edges.parquet")
    parser.add_argument("--out-dir", default="outputs/snn_smoke")
    parser.add_argument("--n-nodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(args.edges, columns=["src", "dst", "weight"])
    top = pd.concat(
        [
            edges.groupby("src")["weight"].sum(),
            edges.groupby("dst")["weight"].sum(),
        ],
        axis=1,
    ).fillna(0).sum(axis=1).sort_values(ascending=False).head(args.n_nodes).index
    sub = edges[edges["src"].isin(top) & edges["dst"].isin(top)].copy()
    ids = pd.Index(top)
    idx = pd.Series(np.arange(len(ids)), index=ids)
    row = sub["src"].map(idx).to_numpy()
    col = sub["dst"].map(idx).to_numpy()
    mask = sparse.coo_matrix((np.ones(len(sub)), (row, col)), shape=(len(ids), len(ids))).tocsr()
    rng = np.random.default_rng(args.seed)
    random_col = rng.permutation(col)
    random_mask = sparse.coo_matrix((np.ones(len(sub)), (row, random_col)), shape=(len(ids), len(ids))).tocsr()
    summary = {
        "connectome_masked_random_recurrent": _simulate(mask, seed=args.seed),
        "degree_matched_random_masked": _simulate(random_mask, seed=args.seed + 1),
        "interpretation": "Smoke-test dynamics only; no biological accuracy or heat claim is made.",
    }
    (out / "snn_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote SNN smoke summary to {out}")


if __name__ == "__main__":
    main()
