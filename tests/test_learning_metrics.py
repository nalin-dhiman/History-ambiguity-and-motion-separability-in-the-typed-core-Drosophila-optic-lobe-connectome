import numpy as np

from thermodynamic_connectomics.learning.metrics import efficiency_metrics, task_mutual_information_bits


def test_efficiency_metrics_finite_when_denominator_nonzero():
    vals = efficiency_metrics(0.5, 2.0, 1.0, 4.0)
    assert all(np.isfinite(v) for v in vals.values())


def test_task_mutual_information_positive_for_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    assert task_mutual_information_bits(y, y) > 0
