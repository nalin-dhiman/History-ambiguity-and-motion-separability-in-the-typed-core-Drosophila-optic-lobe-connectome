"""Activity-conditioned irreversibility metrics."""
from __future__ import annotations

import numpy as np

from .entropy import discretize_quantiles, hash_rows, mutual_information_from_states, shannon_bits


def activity_irreversibility(
    upstream_activity: np.ndarray,
    downstream_activity: np.ndarray,
    labels: np.ndarray | None = None,
    n_bins: int = 4,
) -> dict:
    """Estimate empirical information loss from paired activity samples.

    Parameters
    ----------
    upstream_activity, downstream_activity:
        Arrays with shape (samples, features). Samples must be temporally paired
        or trial-paired.
    labels:
        Optional task or stimulus labels used to estimate task leakage.
    n_bins:
        Number of quantile bins per feature before hashing multivariate states.
    """
    Xd = discretize_quantiles(upstream_activity, n_bins=n_bins)
    Yd = discretize_quantiles(downstream_activity, n_bins=n_bins)
    xs = hash_rows(Xd)
    ys = hash_rows(Yd)
    # Entropies of empirical state IDs.
    _, x_counts = np.unique(xs, return_counts=True)
    _, y_counts = np.unique(ys, return_counts=True)
    h_x = float(shannon_bits(x_counts))
    h_y = float(shannon_bits(y_counts))
    i_xy = mutual_information_from_states(xs, ys)
    out = {
        "h_upstream_bits": h_x,
        "h_downstream_bits": h_y,
        "mutual_information_bits": i_xy,
        "history_loss_bits": max(0.0, h_x - i_xy),
        "entropy_contraction_bits": h_x - h_y,
        "landauer_relevant_bits": max(0.0, h_x - h_y),
    }
    if labels is not None:
        lab = np.asarray(labels).ravel()
        i_x_task = mutual_information_from_states(xs, lab)
        i_y_task = mutual_information_from_states(ys, lab)
        out.update(
            {
                "i_upstream_task_bits": i_x_task,
                "i_downstream_task_bits": i_y_task,
                "task_relevant_leakage_bits": max(0.0, i_x_task - i_y_task),
                "compression_usefulness": i_y_task / max(out["landauer_relevant_bits"], 1e-9),
            }
        )
    return out
