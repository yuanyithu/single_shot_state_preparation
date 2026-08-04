"""Exact small-code tests for collapsed-B generalized Houdayer moves."""

from __future__ import annotations

import itertools
import math

import numpy as np

from data.expander_code.exp102.exp102_pipeline.q0_collapsed_houdayer import (
    b_bits_to_masks,
    b_masks_to_bits,
    build_collapsed_b_houdayer_kernel,
    collapsed_b_component_swap,
    collapsed_b_factor_masks,
    collapsed_b_factor_scopes,
    collapsed_b_houdayer_components,
    collapsed_b_houdayer_transition_distribution,
    collapsed_b_pair_key,
    hgp_syndrome_to_columns,
    initialize_collapsed_b_houdayer_pair,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _ctt_y_columns,
    _initial_collapsed_masks,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


H_SMALL = np.asarray(((1, 1, 0), (0, 1, 1)), dtype=np.uint8)
Y_SMALL = np.asarray((1, 2, 3), dtype=np.uint32)
H_DISCONNECTED = np.asarray(((1, 0), (0, 1)), dtype=np.uint8)
Y_DISCONNECTED = np.asarray((1, 2), dtype=np.uint32)


def _all_b_states(rows):
    result = []
    for integer in range(1 << (rows * rows)):
        bits = np.asarray(
            [(integer >> position) & 1 for position in range(rows * rows)],
            dtype=np.uint8,
        )
        result.append(b_bits_to_masks(bits, rows))
    return tuple(result)


def _collapsed_posterior(H, y_columns, p):
    rows = H.shape[0]
    states = _all_b_states(rows)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_odds = math.log(p / (1.0 - p))
    log_weights = []
    for state in states:
        factors = collapsed_b_factor_masks(H, state, y_columns)
        log_weights.append(
            sum(int(value).bit_count() for value in state) * log_odds
            + sum(math.log(float(mass[int(value)])) for value in factors),
        )
    maximum = max(log_weights)
    weights = np.exp(np.asarray(log_weights) - maximum)
    return states, weights / weights.sum()


def _component_index(components, expected):
    expected = np.asarray(expected, dtype=np.int32)
    for index, component in enumerate(components):
        if np.array_equal(component, expected):
            return index
    raise AssertionError("component partition changed under its own swap")


def test_b_mask_roundtrip_and_factor_scopes_are_canonical():
    rows = H_SMALL.shape[0]
    for state in _all_b_states(rows):
        assert np.array_equal(b_bits_to_masks(b_masks_to_bits(state, rows), rows), state)
    scopes = collapsed_b_factor_scopes(H_SMALL)
    assert tuple(tuple(scope.tolist()) for scope in scopes) == ((0, 2), (0, 1, 2, 3), (1, 3))


def test_collapsed_factor_masks_match_the_existing_hgp_factorization():
    model, _frame = build_model(H_SMALL)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    state = model.logical_sector_section.apply(syndrome, strict=True).astype(np.uint8)
    state ^= model.stabilizer_rows[0]
    b_columns, a_syndromes, _unused = _initial_collapsed_masks(state, syndrome, H_SMALL)
    assert np.array_equal(
        collapsed_b_factor_masks(H_SMALL, b_columns, _ctt_y_columns(syndrome, H_SMALL)),
        a_syndromes,
    )
    assert np.array_equal(hgp_syndrome_to_columns(syndrome, H_SMALL),
                          _ctt_y_columns(syndrome, H_SMALL))


def test_every_complete_component_preserves_collapsed_factor_pairs_and_is_an_involution():
    # This deliberately factorized code supplies a non-whole-swap component.
    kernel = build_collapsed_b_houdayer_kernel(H_DISCONNECTED, Y_DISCONNECTED)
    states = _all_b_states(H_DISCONNECTED.shape[0])
    saw_nontrivial = False
    for left, right in itertools.product(states, repeat=2):
        original = initialize_collapsed_b_houdayer_pair(kernel, left, right)
        components = collapsed_b_houdayer_components(original, kernel)
        for component_index, component in enumerate(components):
            moved = original.copy()
            result = collapsed_b_component_swap(moved, kernel, component_index)
            saw_nontrivial |= bool(result["new_unordered_b_pair"])
            reverse_index = _component_index(
                collapsed_b_houdayer_components(moved, kernel), component,
            )
            collapsed_b_component_swap(moved, kernel, reverse_index)
            assert np.array_equal(moved.left, original.left)
            assert np.array_equal(moved.right, original.right)
    assert saw_nontrivial


def test_complete_collapsed_b_houdayer_transition_has_small_code_stationarity_and_balance():
    p = 0.17
    kernel = build_collapsed_b_houdayer_kernel(H_SMALL, Y_SMALL)
    states, posterior = _collapsed_posterior(H_SMALL, Y_SMALL, p)
    state_index = {state.tobytes(): index for index, state in enumerate(states)}
    pair_count = len(states) ** 2
    transition = np.zeros((pair_count, pair_count), dtype=np.float64)
    for left_index, left in enumerate(states):
        for right_index, right in enumerate(states):
            source = left_index * len(states) + right_index
            pair = initialize_collapsed_b_houdayer_pair(kernel, left, right)
            for key, probability in collapsed_b_houdayer_transition_distribution(pair, kernel).items():
                width = left.nbytes
                target_left = state_index[key[:width]]
                target_right = state_index[key[width:]]
                target = target_left * len(states) + target_right
                transition[source, target] += probability
    pair_posterior = np.outer(posterior, posterior).reshape(-1)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-13
    assert np.max(np.abs(pair_posterior @ transition - pair_posterior)) <= 1e-13
    flow = pair_posterior[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 1e-13


def test_no_difference_is_an_exact_self_loop():
    kernel = build_collapsed_b_houdayer_kernel(H_SMALL, Y_SMALL)
    state = _all_b_states(H_SMALL.shape[0])[7]
    pair = initialize_collapsed_b_houdayer_pair(kernel, state, state)
    distribution = collapsed_b_houdayer_transition_distribution(pair, kernel)
    assert distribution == {collapsed_b_pair_key(pair): 1.0}
