"""Synthetic connectome and data generators for smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_layered_connectome(
    n_in: int = 80,
    n_mid: int = 40,
    n_out: int = 16,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a toy retina-medulla-lobula graph with convergent structure."""
    rng = np.random.default_rng(seed)
    rows = []
    meta = []
    for i in range(n_in):
        meta.append({"root_id": f"R{i}", "group": "retina", "x": rng.normal(0, 1), "y": rng.normal(0, 1), "z": 0})
    for i in range(n_mid):
        meta.append({"root_id": f"M{i}", "group": "medulla", "x": rng.normal(0, 0.7), "y": rng.normal(0, 0.7), "z": 1})
    for i in range(n_out):
        meta.append({"root_id": f"L{i}", "group": "lobula", "x": rng.normal(0, 0.5), "y": rng.normal(0, 0.5), "z": 2})
    # Structured convergence retina -> medulla.
    for m in range(n_mid):
        center = int((m / n_mid) * n_in)
        pres = (center + rng.integers(-6, 7, size=10)) % n_in
        for p in np.unique(pres):
            rows.append({"pre_id": f"R{p}", "post_id": f"M{m}", "weight": int(rng.integers(1, 8)), "neuropil": "ME"})
    # Stronger convergence medulla -> lobula.
    for l in range(n_out):
        center = int((l / n_out) * n_mid)
        pres = (center + rng.integers(-8, 9, size=12)) % n_mid
        for p in np.unique(pres):
            rows.append({"pre_id": f"M{p}", "post_id": f"L{l}", "weight": int(rng.integers(1, 10)), "neuropil": "LO"})
    # Mild recurrence in medulla.
    for _ in range(n_mid * 2):
        a, b = rng.choice(n_mid, size=2, replace=False)
        rows.append({"pre_id": f"M{a}", "post_id": f"M{b}", "weight": int(rng.integers(1, 4)), "neuropil": "ME"})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def make_toy_event_dataset(
    n_samples: int = 256,
    time: int = 20,
    n_inputs: int = 20,
    seed: int = 0,
):
    """Toy event sequences: label is which half has more events."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_samples)
    x = rng.random((n_samples, time, n_inputs)) < 0.05
    for i, y in enumerate(labels):
        if y == 0:
            x[i, :, : n_inputs // 2] |= rng.random((time, n_inputs // 2)) < 0.10
        else:
            x[i, :, n_inputs // 2 :] |= rng.random((time, n_inputs - n_inputs // 2)) < 0.10
    return x.astype("float32"), labels.astype("int64")
