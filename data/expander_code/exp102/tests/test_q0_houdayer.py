"""Exact small-HGP checks for generalized Houdayer pair components."""

import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_houdayer import (
    build_sparse_hgp_coordinate_basis,
    build_sparse_hgp_reduced_logical_coordinate_basis,
    coordinate_factor_scopes,
    coordinates_from_kernel_delta,
    coordinates_to_state,
    houdayer_components,
    houdayer_swap_one,
    prepare_coordinate_readout,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _small_context(basis_builder=build_sparse_hgp_coordinate_basis):
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = build_model(H)
    basis = basis_builder(H, model, frame)
    factors = coordinate_factor_scopes(basis["generators"])
    base = model.logical_sector_section.apply(np.zeros(model.num_checks, dtype=np.uint8), strict=True)
    return model, basis, factors, base


def test_sparse_coordinate_basis_spans_the_small_hard_kernel():
    model, basis, _factors, _base = _small_context()
    generators = basis["generators"]
    assert generators.shape[0] == model.num_qubits - model.num_checks
    assert not np.any(model.H_check.astype(np.int64) @ generators.T.astype(np.int64) % 2)
    assert basis["logical_count"] == model.k


def test_reduced_logical_coordinate_basis_spans_the_small_hard_kernel():
    model, basis, _factors, _base = _small_context(
        build_sparse_hgp_reduced_logical_coordinate_basis,
    )
    generators = basis["generators"]
    assert generators.shape[0] == model.num_qubits - model.num_checks
    assert not np.any(model.H_check.astype(np.int64) @ generators.T.astype(np.int64) % 2)
    assert basis["logical_count"] == model.k


def test_houdayer_component_swap_preserves_each_pair_energy_exactly():
    _model, basis, factors, base = _small_context()
    generators = basis["generators"]
    dimension = generators.shape[0]
    b = 1.0 / 24.0
    for left_bits in itertools.product((0, 1), repeat=dimension):
        left = np.asarray(left_bits, dtype=np.uint8)
        left_state = coordinates_to_state(base, generators, left)
        for right_bits in itertools.product((0, 1), repeat=dimension):
            right = np.asarray(right_bits, dtype=np.uint8)
            right_state = coordinates_to_state(base, generators, right)
            components = houdayer_components(left, right, factors)
            for index in range(len(components)):
                swapped_left, swapped_right = houdayer_swap_one(left, right, components, index)
                swapped_left_state = coordinates_to_state(base, generators, swapped_left)
                swapped_right_state = coordinates_to_state(base, generators, swapped_right)
                before = b ** int(left_state.sum() + right_state.sum())
                after = b ** int(swapped_left_state.sum() + swapped_right_state.sum())
                assert math.isclose(before, after, rel_tol=0.0, abs_tol=1e-15)


def test_houdayer_swap_is_an_involution_with_same_component_partition():
    _model, basis, factors, _base = _small_context()
    dimension = basis["generators"].shape[0]
    left = np.asarray([(index >> 0) & 1 for index in range(dimension)], dtype=np.uint8)
    right = np.asarray([(index >> 1) & 1 for index in range(dimension)], dtype=np.uint8)
    components = houdayer_components(left, right, factors)
    for index in range(len(components)):
        once_left, once_right = houdayer_swap_one(left, right, components, index)
        again_components = houdayer_components(once_left, once_right, factors)
        twice_left, twice_right = houdayer_swap_one(once_left, once_right, again_components, index)
        assert np.array_equal(twice_left, left)
        assert np.array_equal(twice_right, right)


def test_coordinate_roundtrip_recovers_every_small_kernel_coordinate():
    _model, basis, _factors, _base = _small_context()
    generators = basis["generators"]
    readout = prepare_coordinate_readout(generators)
    for bits in itertools.product((0, 1), repeat=generators.shape[0]):
        expected = np.asarray(bits, dtype=np.uint8)
        delta = (expected.astype(np.int64) @ generators.astype(np.int64) % 2).astype(np.uint8)
        assert np.array_equal(coordinates_from_kernel_delta(delta, generators), expected)
        assert np.array_equal(
            coordinates_from_kernel_delta(delta, generators, readout=readout), expected,
        )


@pytest.mark.parametrize("basis_builder", [
    build_sparse_hgp_coordinate_basis,
    build_sparse_hgp_reduced_logical_coordinate_basis,
])
def test_houdayer_pair_kernel_has_exact_small_code_stationarity(basis_builder):
    """Check the full sparse transition matrix without conflating swaps with mixing."""
    _model, basis, factors, base = _small_context(basis_builder)
    generators = basis["generators"]
    dimension = generators.shape[0]
    coordinates = tuple(
        np.asarray([(value >> bit) & 1 for bit in range(dimension)], dtype=np.uint8)
        for value in range(1 << dimension)
    )
    state_weights = {
        value.tobytes(): int(coordinates_to_state(base, generators, value).sum())
        for value in coordinates
    }
    pairs = tuple(itertools.product(coordinates, repeat=2))
    pair_key = lambda left, right: left.tobytes() + right.tobytes()
    unnormalized = {
        pair_key(left, right): 0.2 ** (
            state_weights[left.tobytes()] + state_weights[right.tobytes()]
        )
        for left, right in pairs
    }
    normalizer = sum(unnormalized.values())
    inflow = {key: 0.0 for key in unnormalized}

    for left, right in pairs:
        source_key = pair_key(left, right)
        components = houdayer_components(left, right, factors)
        destinations = ((left, right),) if not components else tuple(
            houdayer_swap_one(left, right, components, index)
            for index in range(len(components))
        )
        probability = 1.0 / len(destinations)
        assert math.isclose(probability * len(destinations), 1.0, abs_tol=1e-15)
        for target_left, target_right in destinations:
            target_key = pair_key(target_left, target_right)
            inflow[target_key] += unnormalized[source_key] / normalizer * probability
            reverse_components = houdayer_components(target_left, target_right, factors)
            reverse_destinations = ((target_left, target_right),) if not reverse_components else tuple(
                houdayer_swap_one(target_left, target_right, reverse_components, index)
                for index in range(len(reverse_components))
            )
            reverse_probability = 1.0 / len(reverse_destinations)
            assert any(
                pair_key(back_left, back_right) == source_key
                for back_left, back_right in reverse_destinations
            )
            assert abs(
                unnormalized[source_key] / normalizer * probability
                - unnormalized[target_key] / normalizer * reverse_probability
            ) <= 1e-13

    for key, weight in unnormalized.items():
        assert abs(inflow[key] - weight / normalizer) <= 1e-13


def test_coordinate_readout_rejects_a_different_generator_basis():
    _model, basis, _factors, _base = _small_context()
    generators = basis["generators"]
    readout = prepare_coordinate_readout(generators)
    altered = generators.copy()
    altered[0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="another generator basis"):
        coordinates_from_kernel_delta(np.zeros(generators.shape[1], dtype=np.uint8), altered,
                                      readout=readout)
