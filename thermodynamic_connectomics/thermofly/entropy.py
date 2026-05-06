"""Information-theoretic metrics for connectome channels and activity states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-12


def _as_prob(p: np.ndarray, axis: Optional[int] = None, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.maximum(arr, 0.0)
    denom = arr.sum(axis=axis, keepdims=True)
    return arr / np.maximum(denom, eps)


def shannon_bits(p: np.ndarray, axis: Optional[int] = None, eps: float = EPS) -> np.ndarray:
    """Shannon entropy in bits for a probability vector or array.

    Zero probabilities are ignored numerically by clipping inside the log.
    """
    prob = _as_prob(p, axis=axis, eps=eps)
    return -np.sum(prob * np.log2(np.maximum(prob, eps)), axis=axis)


def bernoulli_entropy_bits(prob: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Entropy of independent Bernoulli variables with probabilities prob."""
    p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


@dataclass
class ChannelMetrics:
    """Information metrics for a channel P(Y|X) under source distribution p(X)."""

    h_x_bits: float
    h_y_bits: float
    h_y_given_x_bits: float
    h_x_given_y_bits: float
    mutual_information_bits: float
    entropy_contraction_bits: float
    landauer_relevant_bits: float
    history_loss_bits: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def joint_from_channel(p_x: np.ndarray, p_y_given_x: np.ndarray) -> np.ndarray:
    """Return joint P(X,Y) from p(X) and row-stochastic P(Y|X)."""
    px = _as_prob(np.asarray(p_x, dtype=float))
    P = _as_prob(np.asarray(p_y_given_x, dtype=float), axis=1)
    if P.shape[0] != px.shape[0]:
        raise ValueError(f"P has {P.shape[0]} rows but p_x has length {px.shape[0]}")
    return px[:, None] * P


def channel_metrics(p_y_given_x: np.ndarray, p_x: Optional[np.ndarray] = None) -> ChannelMetrics:
    """Compute entropy-contraction and history-loss metrics for P(Y|X).

    Important interpretation:
    - entropy_contraction_bits = H(X) - H(Y). This is the Landauer-style
      entropy decrease only when the effective operation is entropy reducing.
    - landauer_relevant_bits = max(0, H(X) - H(Y)). Use this for lower-bound
      heat conversion.
    - history_loss_bits = H(X|Y), the amount of upstream state not recoverable
      from downstream state. For deterministic many-to-one maps, this equals
      the entropy contraction. For noisy channels, it is a non-invertibility
      index, not automatically a heat bound.
    """
    P = _as_prob(np.asarray(p_y_given_x, dtype=float), axis=1)
    if p_x is None:
        px = np.ones(P.shape[0], dtype=float) / P.shape[0]
    else:
        px = _as_prob(np.asarray(p_x, dtype=float))
    joint = joint_from_channel(px, P)
    py = joint.sum(axis=0)

    h_x = float(shannon_bits(px))
    h_y = float(shannon_bits(py))
    h_xy = float(shannon_bits(joint.ravel()))
    h_y_given_x = h_xy - h_x
    h_x_given_y = h_xy - h_y
    mi = h_x + h_y - h_xy
    contraction = h_x - h_y
    return ChannelMetrics(
        h_x_bits=h_x,
        h_y_bits=h_y,
        h_y_given_x_bits=float(max(0.0, h_y_given_x)),
        h_x_given_y_bits=float(max(0.0, h_x_given_y)),
        mutual_information_bits=float(max(0.0, mi)),
        entropy_contraction_bits=float(contraction),
        landauer_relevant_bits=float(max(0.0, contraction)),
        history_loss_bits=float(max(0.0, h_x_given_y)),
    )


def discretize_quantiles(x: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Discretize a continuous vector/array into quantile bins per column.

    For multivariate activity matrices with shape (samples, features), each
    feature is discretized independently. Ties are handled by np.digitize.
    """
    arr = np.asarray(x)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if arr.ndim == 1:
        qs = np.quantile(arr, np.linspace(0, 1, n_bins + 1)[1:-1])
        return np.digitize(arr, qs)
    out = np.zeros_like(arr, dtype=int)
    for j in range(arr.shape[1]):
        qs = np.quantile(arr[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[:, j] = np.digitize(arr[:, j], qs)
    return out


def hash_rows(x: np.ndarray) -> np.ndarray:
    """Map rows of a discrete matrix to compact integer state IDs."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    _, inv = np.unique(arr, axis=0, return_inverse=True)
    return inv


def mutual_information_from_states(x_state: np.ndarray, y_state: np.ndarray) -> float:
    """Mutual information I(X;Y) in bits for two discrete state vectors."""
    x = np.asarray(x_state).ravel()
    y = np.asarray(y_state).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x_state and y_state must have the same number of samples")
    x_ids, x_inv = np.unique(x, return_inverse=True)
    y_ids, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_ids), len(y_ids)), dtype=float)
    np.add.at(joint, (x_inv, y_inv), 1.0)
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    h_x = shannon_bits(px)
    h_y = shannon_bits(py)
    h_xy = shannon_bits(joint.ravel())
    return float(max(0.0, h_x + h_y - h_xy))
