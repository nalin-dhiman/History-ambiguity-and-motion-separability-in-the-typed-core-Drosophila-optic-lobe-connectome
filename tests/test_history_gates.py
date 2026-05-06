import numpy as np
import pandas as pd

from thermodynamic_connectomics.learning.history_gates import compute_target_history_ambiguity


def test_history_gates_are_finite_and_bounded():
    edges = pd.DataFrame(
        {
            "src": [1, 2, 3, 4],
            "dst": [10, 10, 11, 11],
            "weight": [1.0, 3.0, 2.0, 2.0],
            "src_type": ["Mi1", "Tm1", "Mi1", "Mi1"],
            "dst_type": ["T4", "T4", "T5", "T5"],
        }
    )
    nodes = pd.DataFrame({"bodyId": [10, 11], "type": ["T4", "T5"]})
    gates = compute_target_history_ambiguity(edges, nodes)
    assert {"node_id", "H_in_src_type_bits", "S_inverse_ambiguity", "S_normalized"}.issubset(gates.columns)
    assert np.isfinite(gates["H_in_src_type_bits"]).all()
    assert gates["S_inverse_ambiguity"].between(0, 1).all()
    assert gates["S_normalized"].between(0, 1).all()
