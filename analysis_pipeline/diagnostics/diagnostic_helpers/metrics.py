"""Metrics for physical learning-efficiency proxy experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, mutual_info_score

EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def task_mutual_information_bits(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mutual_info_score(y_true, y_pred) / np.log(2.0))


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prob = softmax(logits)
    pred = prob.argmax(axis=1)
    labels = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(y_true, prob, labels=labels)),
        "task_MI_bits": task_mutual_information_bits(y_true, pred),
    }


def efficiency_metrics(delta_mi: float, update_l1: float, update_l2: float, activity_energy: float) -> dict[str, float]:
    return {
        "eta_phys_L1": float(delta_mi / (update_l1 + EPS)),
        "eta_phys_L2": float(delta_mi / (update_l2 + EPS)),
        "eta_activity": float(delta_mi / (activity_energy + EPS)),
    }


def activity_energy_proxy(X_source: np.ndarray, W) -> float:
    propagated = X_source @ W
    return float(np.abs(propagated).sum())
