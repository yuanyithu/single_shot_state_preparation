"""Exact finite-state tests for the reduced-coordinate Houdayer pair kernel."""

import itertools

import numpy as np

from data.expander_code.exp102.exp102_pipeline.q0_houdayer import (
    coordinates_to_state,
    houdayer_components,
)
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    build_reduced_houdayer_pair_kernel,
    coordinate_flip_probability,
    deterministic_low_energy_logical_starts,
    houdayer_component_swap,
    initialize_houdayer_pair,
    pair_clock_transition_distribution,
    pair_coordinate_key,
    run_houdayer_pair_trajectory,
    validate_houdayer_pair_state,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _context(H, p=0.25, nonzero_syndrome=False):
    model, frame = build_model(H)
    if nonzero_syndrome:
        planted = np.zeros(model.num_qubits, dtype=np.uint8)
        planted[0] = 1
        syndrome = (
            model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2
        ).astype(np.uint8)
    else:
        syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    kernel = build_reduced_houdayer_pair_kernel(H, model, frame, syndrome, p)
    return model, frame, kernel


def _coordinates(dimension):
    return tuple(
        np.asarray([(value >> bit) & 1 for bit in range(dimension)], dtype=np.uint8)
        for value in range(1 << dimension)
    )


def test_coordinate_heatbath_satisfies_exact_binary_detailed_balance():
    H = np.asarray([[1, 1]], dtype=np.uint8)
    _model, _frame, kernel = _context(H)
    coordinates = _coordinates(kernel.coordinate_count)
    b = kernel.p / (1.0 - kernel.p)
    for values in coordinates:
        state = coordinates_to_state(kernel.base_state, kernel.generators, values)
        for coordinate, support in enumerate(kernel.supports):
            flipped_values = values.copy()
            flipped_values[coordinate] ^= np.uint8(1)
            flipped_state = coordinates_to_state(
                kernel.base_state, kernel.generators, flipped_values,
            )
            delta = int(flipped_state.sum()) - int(state.sum())
            forward = coordinate_flip_probability(delta, kernel.log_odds)
            reverse = coordinate_flip_probability(-delta, kernel.log_odds)
            assert abs(b ** int(state.sum()) * forward
                       - b ** int(flipped_state.sum()) * reverse) <= 1e-13


def test_logical_start_catalog_is_legal_distinct_and_deterministic():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame, kernel = _context(H, nonzero_syndrome=True)
    planted = kernel.base_state ^ kernel.generators[-1]
    first = deterministic_low_energy_logical_starts(model, frame, planted, count=4)
    second = deterministic_low_energy_logical_starts(model, frame, planted, count=4)
    assert [row["signature"] for row in first] == [row["signature"] for row in second]
    assert len({row["signature"] for row in first}) == 4
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left["state"], right["state"])
        assert np.array_equal(
            kernel.H_check.astype(np.int64) @ left["state"].astype(np.int64) % 2,
            kernel.syndrome,
        )


def test_complete_pair_clock_has_exact_small_code_stationarity():
    """The composed two-local-update plus HCA clock preserves pi times pi."""
    H = np.asarray([[1, 1]], dtype=np.uint8)
    _model, _frame, kernel = _context(H)
    coordinates = _coordinates(kernel.coordinate_count)
    pairs = tuple(itertools.product(coordinates, repeat=2))
    pair_states = {}
    unnormalized = {}
    b = kernel.p / (1.0 - kernel.p)
    for left_coordinates, right_coordinates in pairs:
        left = coordinates_to_state(kernel.base_state, kernel.generators, left_coordinates)
        right = coordinates_to_state(kernel.base_state, kernel.generators, right_coordinates)
        pair = initialize_houdayer_pair(kernel, left, right)
        key = pair_coordinate_key(pair)
        pair_states[key] = pair
        unnormalized[key] = b ** int(left.sum() + right.sum())
    normalizer = sum(unnormalized.values())
    inflow = {key: 0.0 for key in pair_states}
    for source_key, pair in pair_states.items():
        distribution = pair_clock_transition_distribution(pair, kernel)
        assert abs(sum(distribution.values()) - 1.0) <= 1e-13
        for target_key, probability in distribution.items():
            inflow[target_key] += unnormalized[source_key] / normalizer * probability
    for key, weight in unnormalized.items():
        assert abs(inflow[key] - weight / normalizer) <= 1e-13


def test_component_swap_creates_a_new_pair_when_small_reduced_components_split():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    _model, _frame, kernel = _context(H)
    coordinates = _coordinates(kernel.coordinate_count)
    found = False
    for left_coordinates, right_coordinates in itertools.combinations(coordinates, 2):
        left = coordinates_to_state(kernel.base_state, kernel.generators, left_coordinates)
        right = coordinates_to_state(kernel.base_state, kernel.generators, right_coordinates)
        pair = initialize_houdayer_pair(kernel, left, right)
        components = houdayer_components(
            pair.left_coordinates, pair.right_coordinates, kernel.factor_scopes,
        )
        if len(components) < 2:
            continue
        before_total = int(pair.left.sum() + pair.right.sum())
        result = houdayer_component_swap(pair, kernel, 0)
        if result["new_unordered_pair"]:
            assert int(pair.left.sum() + pair.right.sum()) == before_total
            validate_houdayer_pair_state(kernel, pair)
            found = True
            break
    assert found


def test_fixed_clock_reference_trajectory_is_replayable_and_hard_legal():
    H = np.asarray([[1, 1]], dtype=np.uint8)
    _model, _frame, kernel = _context(H, nonzero_syndrome=True)
    left = kernel.base_state.copy()
    right = kernel.base_state ^ kernel.generators[-1]
    first = run_houdayer_pair_trajectory(kernel, left, right, 12345, 4, 16, 3)
    second = run_houdayer_pair_trajectory(kernel, left, right, 12345, 4, 16, 3)
    for key in first:
        if key == "counters":
            assert first[key] == second[key]
        elif key == "kernel":
            assert first[key] == second[key]
        else:
            assert np.array_equal(first[key], second[key]), key
    assert not np.any(first["measurement_residual_weights"])
    assert first["counters"]["local_attempts"] == 2 * 3 * (4 + 16)
