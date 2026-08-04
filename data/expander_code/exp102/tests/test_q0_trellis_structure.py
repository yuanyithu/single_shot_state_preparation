"""Exact small-code checks for linear-code trellis state dimensions."""

import itertools

import numpy as np

from data.expander_code.exp102.exp102_pipeline.q0_trellis_structure import (
    gf2_column_rank,
    tanner_min_degree_order,
    trellis_state_profile,
)


def _kernel_vectors(matrix):
    matrix = np.asarray(matrix, dtype=np.uint8)
    result = []
    for bits in itertools.product((0, 1), repeat=matrix.shape[1]):
        vector = np.asarray(bits, dtype=np.uint8)
        if not np.any(matrix.astype(np.int64) @ vector.astype(np.int64) % 2):
            result.append(vector)
    return result


def _subcode_dimension(vectors, positions, length):
    selected = [vector for vector in vectors if all(
        int(vector[index]) == 0 for index in range(length) if index not in positions
    )]
    if not selected:
        return 0
    matrix = np.asarray(selected, dtype=np.uint8)
    return gf2_column_rank(matrix.T)


def test_trellis_state_formula_matches_bruteforce_subcode_dimensions():
    matrix = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    order = (2, 0, 1)
    profile = trellis_state_profile(matrix, order)
    vectors = _kernel_vectors(matrix)
    dimension = int(round(np.log2(len(vectors))))
    for cut, observed in enumerate(profile["state_exponents"]):
        past = set(order[:cut])
        future = set(order[cut:])
        past_dimension = _subcode_dimension(vectors, past, matrix.shape[1])
        future_dimension = _subcode_dimension(vectors, future, matrix.shape[1])
        assert int(observed) == dimension - past_dimension - future_dimension


def test_profiles_have_trivial_endpoints_and_matrix_rank():
    matrix = np.asarray([[1, 1, 1], [0, 1, 1]], dtype=np.uint8)
    profile = trellis_state_profile(matrix, (0, 1, 2))
    assert profile["rank"] == 2
    assert tuple(profile["state_exponents"][[0, -1]]) == (0, 0)
    assert np.all(profile["state_exponents"] >= 0)


def test_tanner_min_degree_is_a_deterministic_permutation():
    matrix = np.asarray([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    assert tanner_min_degree_order(matrix) == (0, 1, 2, 3)
