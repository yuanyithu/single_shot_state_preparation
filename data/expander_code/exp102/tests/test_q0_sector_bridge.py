"""Small-HGP exact oracles for fixed-sector free-energy bridge identities."""

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_sector_bridge import (
    bridge_step_ratio,
    exact_fixed_sector_bridge,
    logical_bridge_prefixes,
    reverse_bridge_step_ratio,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import state_label
from data.expander_code.exp102.exp102_pipeline.worker import build_model


SMALL_H = (
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
)


def _hard_coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, generator in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= generator
    packed = np.packbits(states, axis=1, bitorder="little")
    assert len({row.tobytes() for row in packed}) == states.shape[0]
    return states


def _syndrome(model, nonzero):
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    return (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)


def test_prefixes_toggle_each_support_bit_once_in_order():
    move = np.asarray([0, 1, 0, 1, 1], dtype=np.uint8)
    support, prefixes = logical_bridge_prefixes(move)
    assert np.array_equal(support, np.asarray([1, 3, 4], dtype=np.int32))
    assert np.array_equal(prefixes[0], np.zeros(5, dtype=np.uint8))
    assert np.array_equal(prefixes[-1], move)
    assert np.array_equal(prefixes[1:] ^ prefixes[:-1], np.eye(5, dtype=np.uint8)[support])


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_exact_fixed_sector_bridge_matches_partition_ratios(H, p, nonzero_syndrome):
    model, frame = build_model(H)
    syndrome = _syndrome(model, nonzero_syndrome)
    states = _hard_coset_states(model, syndrome)
    labels = np.asarray([state_label(frame, state) for state in states], dtype=np.uint64)
    base = model.logical_sector_section.apply(syndrome, strict=True)
    base_label = state_label(frame, base)
    move = np.asarray(model.logical_move_basis[0], dtype=np.uint8)
    report = exact_fixed_sector_bridge(states, labels, base_label, move, p)

    assert np.max(np.abs(
        report["expected_step_ratios"] - report["actual_step_ratios"],
    )) <= 2e-13
    assert np.max(np.abs(
        report["reverse_expected_step_ratios"] - report["actual_step_ratios"],
    )) <= 2e-13
    assert abs(report["product_ratio"] - report["endpoint_ratio"]) <= 3e-13

    endpoint_label = np.uint64(base_label) ^ np.uint64(state_label(frame, move))
    endpoint_mass = (p / (1.0 - p)) ** states[labels == endpoint_label].sum(axis=1)
    start_mass = (p / (1.0 - p)) ** states[labels == base_label].sum(axis=1)
    assert abs(
        report["endpoint_ratio"]
        - float(endpoint_mass.sum(dtype=np.float64) / start_mass.sum(dtype=np.float64))
    ) <= 3e-13


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_bridge_step_ratio_is_the_binary_likelihood_expectation(p):
    odds = p / (1.0 - p)
    for probability in (0.0, 0.125, 0.5, 0.875, 1.0):
        expected = (1.0 - probability) * odds + probability / odds
        assert bridge_step_ratio(p, probability) == pytest.approx(expected, abs=1e-15)
        assert reverse_bridge_step_ratio(p, probability) == pytest.approx(
            1.0 / expected, abs=1e-15,
        )
