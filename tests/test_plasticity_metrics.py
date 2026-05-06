import numpy as np

from thermodynamic_connectomics.learning.plasticity_metrics import plasticity_efficiency, split_train_test, train_linear_probe


def test_plasticity_efficiency_no_blowup_for_nonpositive_gain():
    assert plasticity_efficiency(-1.0, 0.0) == 0.0
    assert plasticity_efficiency(0.0, 0.0) == 0.0
    val = plasticity_efficiency(1.0, 2.0)
    assert np.isfinite(val)
    assert val > 0


def test_linear_probe_returns_finite_metrics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    Xtr, Xte, ytr, yte = split_train_test(X, y, seed=2)
    metrics = train_linear_probe(Xtr, ytr, Xte, yte, seed=3, epochs=5)
    assert np.isfinite(list(metrics.values())).all()
    assert 0 <= metrics["accuracy"] <= 1
