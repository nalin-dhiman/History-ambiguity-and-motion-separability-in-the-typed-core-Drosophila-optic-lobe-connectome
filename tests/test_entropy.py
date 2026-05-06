import math

from thermodynamic_connectomics.metrics.entropy import effective_number, normalized_entropy, weighted_entropy


def test_weighted_entropy_fair_binary():
    assert math.isclose(weighted_entropy([1, 1]), 1.0)


def test_effective_number_four_equal_bins():
    assert math.isclose(effective_number([0.25, 0.25, 0.25, 0.25]), 4.0)


def test_normalized_entropy_single_bin_zero():
    assert normalized_entropy([10]) == 0.0
