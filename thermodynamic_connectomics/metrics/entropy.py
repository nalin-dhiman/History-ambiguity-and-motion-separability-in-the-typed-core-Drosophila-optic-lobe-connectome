"""Entropy/diversity helpers used by the optic-lobe pipeline."""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def _probabilities(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr / arr.sum()


def shannon_entropy(p, base: float = 2) -> float:
    """Shannon entropy of a probability vector."""

    prob = _probabilities(np.asarray(p, dtype=float))
    if prob.size == 0:
        return 0.0
    logs = np.log(prob) / math.log(base)
    return float(-(prob * logs).sum())


def weighted_entropy(weights, base: float = 2) -> float:
    """Entropy of non-negative weights after normalizing to probabilities."""

    return shannon_entropy(_probabilities(np.asarray(weights, dtype=float)), base=base)


def effective_number(p) -> float:
    """Hill number equivalent, `2 ** H_bits`."""

    return float(2.0 ** shannon_entropy(p, base=2))


def normalized_entropy(p) -> float:
    """Entropy divided by the maximum possible entropy for nonzero bins."""

    prob = _probabilities(np.asarray(p, dtype=float))
    n = prob.size
    if n <= 1:
        return 0.0
    return float(shannon_entropy(prob, base=2) / math.log2(n))


def gini_or_hhi(weights) -> dict[str, float]:
    """Return concentration metrics for a weighted distribution."""

    prob = _probabilities(np.asarray(weights, dtype=float))
    if prob.size == 0:
        return {"hhi": 0.0, "gini": 0.0}
    hhi = float(np.square(prob).sum())
    sorted_prob = np.sort(prob)
    n = sorted_prob.size
    gini = float((2 * np.arange(1, n + 1) @ sorted_prob) / (n * sorted_prob.sum()) - (n + 1) / n)
    return {"hhi": hhi, "gini": max(0.0, gini)}
