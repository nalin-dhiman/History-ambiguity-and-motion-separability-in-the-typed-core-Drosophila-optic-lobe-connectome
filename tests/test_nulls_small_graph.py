import pandas as pd

from thermodynamic_connectomics.nulls.degree_preserving import (
    directed_degree_preserving_rewire,
    target_preserving_source_shuffle,
    weight_preserving_type_block_permutation,
)


def test_target_preserving_source_shuffle_keeps_targets_and_weights():
    edges = pd.DataFrame({"src": [1, 2, 3, 4], "dst": [9, 9, 8, 8], "weight": [1, 2, 3, 4]})
    shuffled = target_preserving_source_shuffle(edges, seed=1)
    assert shuffled["dst"].tolist() == edges["dst"].tolist()
    assert shuffled["weight"].tolist() == edges["weight"].tolist()
    assert sorted(shuffled["src"]) == sorted(edges["src"])


def test_degree_preserving_rewire_preserves_degree_counts():
    edges = pd.DataFrame(
        {
            "src": [1, 1, 2, 2, 3, 3],
            "dst": [4, 5, 5, 6, 4, 6],
            "weight": [1, 1, 1, 1, 1, 1],
        }
    )
    rewired = directed_degree_preserving_rewire(edges, seed=2, n_swaps=5)
    assert rewired.groupby("src").size().to_dict() == edges.groupby("src").size().to_dict()
    assert rewired.groupby("dst").size().to_dict() == edges.groupby("dst").size().to_dict()


def test_weight_preserving_type_block_permutation_keeps_weights_and_type_marginals():
    edges = pd.DataFrame(
        {
            "src": [1, 2, 3, 4, 5, 6],
            "dst": [7, 8, 9, 10, 11, 12],
            "weight": [1, 1, 2, 2, 3, 3],
            "src_type": ["A", "A", "B", "B", "C", "C"],
            "dst_type": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )
    permuted = weight_preserving_type_block_permutation(edges, seed=1, n_weight_bins=1, n_degree_bins=1)
    assert permuted["weight"].tolist() == edges["weight"].tolist()
    assert permuted["src_type"].value_counts().to_dict() == edges["src_type"].value_counts().to_dict()
    assert permuted["dst_type"].value_counts().to_dict() == edges["dst_type"].value_counts().to_dict()
