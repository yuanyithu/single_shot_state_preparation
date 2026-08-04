"""Algebra and replay checks for independent hard-coset MIS draws."""

from dataclasses import replace
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_crossfit_importance import (
    crossfit_collision_ratio,
)
from data.expander_code.exp102.exp102_pipeline.q0_iid_importance import (
    IidImportanceError,
    draw_stratified_iid_mixture,
    mixture_log_probability_state,
    validate_stratified_iid_mixture,
    weight_diagnostics,
)


class _Model:
    def __init__(self, k=2):
        self.num_qubits = 2
        self.num_checks = 1
        self.k = int(k)
        self.H_check = np.zeros((1, 2), dtype=np.uint8)


class _Frame:
    def __init__(self, k=2):
        self.k = int(k)

    def label_of(self, state):
        bits = np.zeros(self.k, dtype=np.uint8)
        bits[:2] = np.asarray(state, dtype=np.uint8)
        return bits


class _Bit63Frame(_Frame):
    def __init__(self):
        super().__init__(64)

    def label_of(self, state):
        bits = np.zeros(64, dtype=np.uint8)
        bits[63] = np.asarray(state, dtype=np.uint8)[0]
        return bits


class _Coordinates:
    def state_from_coordinates(self, coordinate):
        coordinate = np.asarray(coordinate, dtype=np.uint8)
        if coordinate.shape != (2,):
            raise ValueError("coordinate shape")
        return coordinate.copy()


class _Proposal:
    _states = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.coordinates = _Coordinates()

    @staticmethod
    def _index(state):
        state = np.asarray(state, dtype=np.uint8)
        return int(state[0]) + 2 * int(state[1])

    def log_probability_state(self, state):
        return math.log(float(self.probabilities[self._index(state)]))

    def log_probability_coordinates(self, coordinate):
        return self.log_probability_state(coordinate)

    def sample(self, rng):
        threshold = rng.random()
        cumulative = 0.0
        index = 0
        for index, probability in enumerate(self.probabilities):
            cumulative += float(probability)
            if threshold < cumulative:
                break
        state = self._states[index].copy()
        return {
            "state": state,
            "coordinate": state.copy(),
            "log_q": self.log_probability_state(state),
        }


def _draws(*, frame=None, blocks=5, draws_per_source=11):
    model = _Model(k=frame.k if frame is not None else 2)
    frame = _Frame() if frame is None else frame
    proposals = (
        _Proposal([0.70, 0.10, 0.10, 0.10]),
        _Proposal([0.10, 0.20, 0.30, 0.40]),
    )
    seeds = np.arange(blocks * len(proposals), dtype=np.uint64).reshape(blocks, len(proposals)) + 100
    draws = draw_stratified_iid_mixture(
        model, frame, np.zeros(1, dtype=np.uint8), 0.20, proposals, [0.5, 0.5],
        seeds, block_count=blocks, draws_per_proposal_per_block=draws_per_source,
    )
    return model, frame, proposals, draws


def test_mixture_log_probability_matches_the_explicit_sum():
    first = _Proposal([0.70, 0.10, 0.10, 0.10])
    second = _Proposal([0.10, 0.20, 0.30, 0.40])
    state = np.asarray([1, 0], dtype=np.uint8)
    actual = mixture_log_probability_state((first, second), [0.25, 0.75], state)
    expected = math.log(0.25 * 0.10 + 0.75 * 0.20)
    assert actual == pytest.approx(expected, abs=1e-15)


def test_stratified_iid_draws_replay_without_calling_the_rng_again():
    model, frame, proposals, draws = _draws()
    assert validate_stratified_iid_mixture(
        draws, model, frame, np.zeros(1, dtype=np.uint8), 0.20, proposals, [0.5, 0.5],
    )
    assert draws.states_packed.shape == (5 * 2 * 11, 1)
    assert np.array_equal(draws.block_indices, np.repeat(np.arange(5), 22))
    assert np.array_equal(draws.source_indices, np.tile(np.repeat([0, 1], 11), 5))

    altered = draws.mixture_log_importance.copy()
    altered[0] += 1.0
    corrupted = replace(draws, mixture_log_importance=altered)
    with pytest.raises(IidImportanceError, match="log weights"):
        validate_stratified_iid_mixture(
            corrupted, model, frame, np.zeros(1, dtype=np.uint8), 0.20,
            proposals, [0.5, 0.5],
        )


def test_equal_source_schedule_rejects_a_nonuniform_mixture_density():
    model = _Model()
    frame = _Frame()
    proposals = (_Proposal([0.70, 0.10, 0.10, 0.10]), _Proposal([0.10, 0.20, 0.30, 0.40]))
    seeds = np.arange(6, dtype=np.uint64).reshape(3, 2) + 100
    with pytest.raises(IidImportanceError, match="equal source allocation"):
        draw_stratified_iid_mixture(
            model, frame, np.zeros(1, dtype=np.uint8), 0.20, proposals, [0.25, 0.75],
            seeds, block_count=3, draws_per_proposal_per_block=2,
        )


def test_iid_mis_crossfit_converges_to_the_tiny_exact_purity():
    model, frame, proposals, draws = _draws(blocks=16, draws_per_source=512)
    assert validate_stratified_iid_mixture(
        draws, model, frame, np.zeros(1, dtype=np.uint8), 0.20, proposals, [0.5, 0.5],
    )
    result = crossfit_collision_ratio(
        draws.labels, draws.mixture_log_importance, block_count=16, logical_dimension=2,
    )
    b = 0.20 / 0.80
    exact_masses = np.asarray([1.0, b, b, b * b])
    exact_purity = float(np.dot(exact_masses, exact_masses) / exact_masses.sum() ** 2)
    assert result.purity == pytest.approx(exact_purity, abs=0.015)
    diagnostics = weight_diagnostics(draws.mixture_log_importance, block_count=16)
    assert diagnostics["minimum_block_effective_sample_size"] > 100.0
    assert diagnostics["maximum_block_normalized_weight"] < 0.02


def test_iid_mis_preserves_a_bit63_logical_label():
    frame = _Bit63Frame()
    model, _, proposals, draws = _draws(frame=frame)
    assert draws.labels.dtype == np.uint64
    assert np.any(draws.labels & (np.uint64(1) << np.uint64(63)))
    assert validate_stratified_iid_mixture(
        draws, model, frame, np.zeros(1, dtype=np.uint8), 0.20, proposals, [0.5, 0.5],
    )
