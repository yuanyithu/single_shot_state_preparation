"""Replay tests for IID MIS component/coordinate provenance records."""

from dataclasses import replace

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    build_bp_systematic_proposal,
)
from data.expander_code.exp102.exp102_pipeline.q0_iid_provenance import (
    IidProvenanceError,
    draw_provenanced_stratified_iid_mixture,
    validate_provenanced_stratified_iid_mixture,
)


class _Model:
    def __init__(self):
        self.H_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        self.num_qubits = 3
        self.num_checks = 2
        self.k = 1


class _Frame:
    k = 1

    @staticmethod
    def label_of(state):
        return np.asarray([np.asarray(state, dtype=np.uint8)[0]], dtype=np.uint8)


def _proposals(model, syndrome):
    common = {
        "bp_iterations": 8,
        "bp_damping": 0.25,
        "bp_llr_cap": 30.0,
        "min_probability": 1e-5,
        "component_weights": (0.90, 0.09, 0.01),
    }
    return (
        build_bp_systematic_proposal(
            model, syndrome, 0.20, column_order=np.asarray([0, 1, 2]), **common,
        ),
        build_bp_systematic_proposal(
            model, syndrome, 0.20, column_order=np.asarray([2, 1, 0]), **common,
        ),
    )


def test_provenanced_iid_draws_replay_coordinates_and_component_ids():
    model = _Model()
    frame = _Frame()
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    proposals = _proposals(model, syndrome)
    seeds = np.arange(8, dtype=np.uint64).reshape(4, 2) + 100
    draws = draw_provenanced_stratified_iid_mixture(
        model, frame, syndrome, 0.20, proposals, [0.5, 0.5], seeds,
        block_count=4, draws_per_proposal_per_block=7,
    )
    assert validate_provenanced_stratified_iid_mixture(
        draws, model, frame, syndrome, 0.20, proposals, [0.5, 0.5],
    )
    assert np.all(draws.anchor_indices == -1)
    assert np.all((draws.component_indices >= 0) & (draws.component_indices < 3))
    assert draws.coordinates_packed.shape == (56, 1)


def test_provenanced_iid_rejects_a_corrupted_coordinate():
    model = _Model()
    frame = _Frame()
    syndrome = np.asarray([1, 0], dtype=np.uint8)
    proposals = _proposals(model, syndrome)
    seeds = np.arange(8, dtype=np.uint64).reshape(4, 2) + 100
    draws = draw_provenanced_stratified_iid_mixture(
        model, frame, syndrome, 0.20, proposals, [0.5, 0.5], seeds,
        block_count=4, draws_per_proposal_per_block=7,
    )
    altered = draws.coordinates_packed.copy()
    altered[0, 0] ^= np.uint8(1)
    with pytest.raises(IidProvenanceError, match="coordinate"):
        validate_provenanced_stratified_iid_mixture(
            replace(draws, coordinates_packed=altered), model, frame, syndrome,
            0.20, proposals, [0.5, 0.5],
        )
