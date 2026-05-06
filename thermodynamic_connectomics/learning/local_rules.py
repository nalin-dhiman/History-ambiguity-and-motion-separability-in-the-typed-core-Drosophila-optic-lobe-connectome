"""Local learning rules and compact message-passing readout model."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from thermodynamic_connectomics.learning.metrics import (
    activity_energy_proxy,
    classification_metrics,
    efficiency_metrics,
    softmax,
)


def normalize_per_target(W: sparse.csr_matrix) -> sparse.csr_matrix:
    W = W.tocsc(copy=True)
    col_sums = np.asarray(W.sum(axis=0)).ravel()
    for j in range(W.shape[1]):
        start, end = W.indptr[j], W.indptr[j + 1]
        if end > start and col_sums[j] > 0:
            W.data[start:end] /= col_sums[j]
    return W.tocsr()


def hebbian_update(
    W: sparse.csr_matrix,
    pre_mean: np.ndarray,
    post_mean: np.ndarray,
    reward: float,
    lr: float,
    decay: float,
    reward_modulated: bool = True,
) -> tuple[sparse.csr_matrix, float, float]:
    """Apply sparse Hebbian update on existing graph edges only."""

    coo = W.tocoo(copy=True)
    mod = reward if reward_modulated else 1.0
    delta = lr * mod * pre_mean[coo.row] * post_mean[coo.col] - decay * coo.data
    new_data = np.clip(coo.data + delta, 0.0, None)
    update = new_data - coo.data
    out = sparse.coo_matrix((new_data, (coo.row, coo.col)), shape=W.shape).tocsr()
    out = normalize_per_target(out)
    return out, float(np.abs(update).sum()), float(np.sqrt(np.square(update).sum()))


def _representation(X: np.ndarray, W: sparse.csr_matrix, steps: int = 1) -> np.ndarray:
    H = X @ W
    if steps > 1:
        H = np.tanh(H)
    return np.asarray(H, dtype=np.float32)


def train_message_passing_model(
    graph,
    task_data: dict,
    mode: str,
    epochs: int,
    lr: float,
    hebbian_lr: float,
    decay: float,
    seed: int,
) -> tuple[list[dict], dict]:
    """Train readout-only or reward-modulated local-Hebbian model."""

    rng = np.random.default_rng(seed)
    X = task_data["X_source"]
    y = task_data["y"]
    n_classes = int(y.max()) + 1
    W_graph = normalize_per_target(graph.W.copy())
    H0 = _representation(X, W_graph)
    readout = rng.normal(0, 0.01, size=(H0.shape[1], n_classes)).astype(np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    init_logits = H0 @ readout + bias
    initial_mi = classification_metrics(y, init_logits)["task_MI_bits"]

    curves = []
    total_l1 = 0.0
    total_l2_sq = 0.0
    total_energy = 0.0
    for epoch in range(epochs):
        H = _representation(X, W_graph)
        logits = H @ readout + bias
        prob = softmax(logits)
        onehot = np.eye(n_classes, dtype=np.float32)[y]
        grad_logits = (prob - onehot) / len(y)
        readout -= lr * (H.T @ grad_logits)
        bias -= lr * grad_logits.sum(axis=0)

        logits = H @ readout + bias
        metrics = classification_metrics(y, logits)
        pred = logits.argmax(axis=1)
        reward = 1.0 if metrics["accuracy"] >= (1.0 / n_classes) else -1.0
        l1_epoch = 0.0
        l2_epoch = 0.0
        if mode in {"local_hebbian", "unsupervised_hebbian"}:
            W_graph, l1_epoch, l2_epoch = hebbian_update(
                W_graph,
                pre_mean=X.mean(axis=0),
                post_mean=np.asarray(H).mean(axis=0),
                reward=reward,
                lr=hebbian_lr,
                decay=decay,
                reward_modulated=mode == "local_hebbian",
            )
        elif mode != "readout_only":
            raise ValueError(f"Unknown mode: {mode}")
        energy_epoch = activity_energy_proxy(X, W_graph)
        total_l1 += l1_epoch
        total_l2_sq += l2_epoch * l2_epoch
        total_energy += energy_epoch
        delta_mi = metrics["task_MI_bits"] - initial_mi
        eta = efficiency_metrics(delta_mi, l1_epoch, l2_epoch, energy_epoch)
        curves.append(
            {
                "epoch": epoch,
                **metrics,
                "delta_task_MI_bits": float(delta_mi),
                "synaptic_update_L1_epoch": l1_epoch,
                "synaptic_update_L2_epoch": l2_epoch,
                "activity_energy_proxy_epoch": energy_epoch,
                "eta_phys_L1_epoch": eta["eta_phys_L1"],
                "eta_activity_epoch": eta["eta_activity"],
            }
        )

    final = curves[-1].copy()
    final_delta = final["task_MI_bits"] - initial_mi
    total_l2 = float(np.sqrt(total_l2_sq))
    eta_total = efficiency_metrics(final_delta, total_l1, total_l2, total_energy)
    summary = {
        "final_accuracy": final["accuracy"],
        "final_balanced_accuracy": final["balanced_accuracy"],
        "final_macro_f1": final["macro_f1"],
        "final_cross_entropy": final["cross_entropy"],
        "initial_task_MI_bits": float(initial_mi),
        "final_task_MI_bits": final["task_MI_bits"],
        "delta_task_MI_bits": float(final_delta),
        "total_synaptic_update_L1": total_l1,
        "total_synaptic_update_L2": total_l2,
        "total_activity_energy_proxy": total_energy,
        **eta_total,
    }
    return curves, summary
