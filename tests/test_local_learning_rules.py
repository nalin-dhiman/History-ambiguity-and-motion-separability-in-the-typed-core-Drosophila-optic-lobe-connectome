import numpy as np
from scipy import sparse

from thermodynamic_connectomics.learning.local_rules import hebbian_update, normalize_per_target


def test_hebbian_update_non_negative_and_finite():
    W = sparse.csr_matrix([[0.5, 0.0], [0.5, 1.0]], dtype=float)
    W2, l1, l2 = hebbian_update(W, np.array([0.2, 0.8]), np.array([0.7, 0.4]), reward=1.0, lr=0.01, decay=0.001)
    assert np.isfinite(W2.data).all()
    assert (W2.data >= 0).all()
    assert l1 >= 0
    assert l2 >= 0


def test_normalize_per_target_column_sums():
    W = sparse.csr_matrix([[2.0, 0.0], [2.0, 3.0]], dtype=float)
    Wn = normalize_per_target(W)
    sums = np.asarray(Wn.sum(axis=0)).ravel()
    assert np.allclose(sums, [1.0, 1.0])
