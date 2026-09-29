"""Evaluation helpers for plasticity rule discovery."""
from __future__ import annotations

import numpy as np

from thermodynamic_connectomics.learning.metrics import EPS, classification_metrics, efficiency_metrics, softmax


def split_train_test(X: np.ndarray, y: np.ndarray, seed: int, test_fraction: float = 0.35) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(round(len(y) * test_fraction)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if len(train_idx) == 0:
        train_idx, test_idx = test_idx, idx[:0]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_linear_probe(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int = 80,
    lr: float = 0.05,
) -> dict[str, float]:
    """Train a deterministic softmax probe and evaluate on held-out data."""

    rng = np.random.default_rng(seed)
    n_classes = int(max(y_train.max(), y_test.max() if len(y_test) else y_train.max())) + 1
    readout = rng.normal(0, 0.01, size=(H_train.shape[1], n_classes)).astype(np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    onehot = np.eye(n_classes, dtype=np.float32)[y_train]
    for _ in range(max(1, epochs)):
        logits = H_train @ readout + bias
        prob = softmax(logits)
        grad = (prob - onehot) / max(len(y_train), 1)
        readout -= lr * (H_train.T @ grad)
        bias -= lr * grad.sum(axis=0)
    eval_H = H_test if len(y_test) else H_train
    eval_y = y_test if len(y_test) else y_train
    return classification_metrics(eval_y, eval_H @ readout + bias)


def plasticity_efficiency(delta_mi: float, update_cost_l1: float) -> float:
    """Plasticity-only MI gain per update-cost proxy.

    Returns zero for non-positive gains so eta cannot look good when plasticity
    failed to add task information.
    """

    if delta_mi <= 0:
        return 0.0
    return float(delta_mi / (update_cost_l1 + EPS))


def paired_delta(real: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    dmi = float(real["plasticity_delta_MI"] - control["plasticity_delta_MI"])
    dcost = float(real["total_update_L1"] - control["total_update_L1"])
    return {
        "delta_accuracy": float(real["plasticity_delta_accuracy"] - control["plasticity_delta_accuracy"]),
        "delta_MI": dmi,
        "delta_cost": dcost,
        "delta_eta": float(real["eta_plastic"] - control["eta_plastic"]),
        "conditional_eta_gain": float(dmi / (dcost + EPS)),
    }


def finite_summary(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            out[key] = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        else:
            out[key] = value
    return out
