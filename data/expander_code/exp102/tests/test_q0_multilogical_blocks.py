from dataclasses import replace
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_global import (
    GlobalConflictError,
    GlobalSeedIdentity,
    _signature_rank_masks,
    build_logical_proposal_catalog,
    canonical_global_trajectory_digest,
)
from data.expander_code.exp102.exp102_pipeline.q0_multilogical_blocks import (
    MultiLogicalBlockConfig,
    build_multilogical_empty_catalog,
    build_multilogical_blocks,
    run_multilogical_block_trajectory,
    validate_multilogical_empty_catalog,
    validate_multilogical_blocks,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model


REGISTRY_PATH = "data/expander_code/exp102/registry/registry.json"


def _model(classical):
    return build_model(np.asarray(classical, dtype=np.uint8))


def _coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, row in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= row
    return states


def _key(state):
    return np.packbits(state, bitorder="little").tobytes()


def _block_transition(states, generators, K):
    lookup = {_key(state): index for index, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    moves = []
    for mask in range(1 << generators.shape[0]):
        move = np.zeros(states.shape[1], dtype=np.uint8)
        for bit, generator in enumerate(generators):
            if (mask >> bit) & 1:
                move ^= generator
        moves.append(move)
    for index, state in enumerate(states):
        targets = np.asarray([lookup[_key(state ^ move)] for move in moves])
        weights = np.exp(-K * states[targets].sum(axis=1))
        weights /= weights.sum()
        for target, probability in zip(targets, weights):
            transition[index, target] += probability
    return transition


@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_multilogical_blocks_are_exact_conditional_heatbaths(classical, nonzero, p):
    model, frame = _model(classical)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    logicals_per_block = min(2, model.k)
    block_size = logicals_per_block + 1
    blocks = build_multilogical_blocks(
        model, frame, logicals_per_block=logicals_per_block, block_size=block_size,
    )
    assert validate_multilogical_blocks(
        model, frame, blocks, logicals_per_block=logicals_per_block,
        block_size=block_size,
    )
    states = _coset_states(model, syndrome)
    K = math.log((1.0 - p) / p)
    posterior = np.exp(-K * states.sum(axis=1)); posterior /= posterior.sum()
    transitions = []
    for generators in blocks.generators:
        transition = _block_transition(states, generators, K)
        flow = posterior[:, None] * transition
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-13
        assert np.max(np.abs(flow - flow.T)) <= 1e-13
        assert np.max(np.abs(posterior @ transition - posterior)) <= 1e-13
        transitions.append(transition)
    mixture = sum(transitions) / len(transitions)
    assert np.max(np.abs(posterior @ mixture - posterior)) <= 1e-13


def _seed(method):
    return GlobalSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=method,
        resource_tier="T1",
        init_family="P",
        trajectory_index=0,
        trajectory_namespace="multi_logical_block_test",
    )


def test_mlb8_j16_reference_numba_trace_is_bit_identical_on_real_m3():
    _, _, H = load_frozen_code(REGISTRY_PATH, "m03_c00")
    model, frame = build_model(H)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    config = MultiLogicalBlockConfig(
        p=0.10, burn_sweeps=1, measurement_sweeps=8,
    )
    blocks = build_multilogical_blocks(model, frame)
    catalog = build_multilogical_empty_catalog(model, frame)
    assert catalog.size == 0
    assert validate_multilogical_empty_catalog(model, frame, catalog)
    assert _signature_rank_masks(
        blocks.signatures[:, :config.logicals_per_block].reshape(-1), model.k,
    ) == model.k
    assert not blocks.signatures[:, config.logicals_per_block:].any()
    results = [
        run_multilogical_block_trajectory(
            model, frame, syndrome, config, _seed(config.method_id), initial,
            engine=engine, catalog=catalog, blocks=blocks,
        )
        for engine in ("reference", "numba")
    ]
    assert canonical_global_trajectory_digest(results[0]) == canonical_global_trajectory_digest(results[1])
    assert not results[0]["measurement_residual_weights"].any()
    assert not results[0]["measurement_counters"][2:4].any()


def test_mlb8_j16_builds_complete_uint64_blocks_on_hard_m8():
    _, _, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    blocks = build_multilogical_blocks(model, frame)
    logical_signatures = blocks.signatures[:, :8].reshape(-1)
    assert _signature_rank_masks(logical_signatures, model.k) == 64
    assert any((int(value) >> 63) & 1 for value in logical_signatures)
    assert not blocks.signatures[:, 8:].any()


def test_mlb8_j16_reference_numba_trace_matches_with_m8_bit63_initial_label():
    _, _, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    blocks = build_multilogical_blocks(model, frame)
    location = next(
        (block, row)
        for block in range(blocks.num_blocks)
        for row in range(8)
        if (int(blocks.signatures[block, row]) >> 63) & 1
    )
    initial = blocks.generators[location].copy()
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    config = MultiLogicalBlockConfig(p=0.04, burn_sweeps=1, measurement_sweeps=8)
    catalog = build_multilogical_empty_catalog(model, frame)
    results = [
        run_multilogical_block_trajectory(
            model, frame, syndrome, config, _seed(config.method_id), initial,
            engine=engine, catalog=catalog, blocks=blocks,
        )
        for engine in ("reference", "numba")
    ]
    assert (int(results[0]["initial_label"]) >> 63) & 1
    assert results[0]["measurement_labels"].dtype == np.uint64
    assert canonical_global_trajectory_digest(results[0]) == canonical_global_trajectory_digest(results[1])
    assert not results[0]["measurement_counters"][2:4].any()


def test_multilogical_catalog_replay_rejects_tampered_generator_or_signature():
    _, _, H = load_frozen_code(REGISTRY_PATH, "m03_c00")
    model, frame = build_model(H)
    blocks = build_multilogical_blocks(model, frame)

    generators = blocks.generators.copy()
    generators[0, 0, 0] ^= np.uint8(1)
    with pytest.raises(GlobalConflictError, match="catalog replay"):
        validate_multilogical_blocks(
            model, frame, replace(blocks, generators=generators),
            logicals_per_block=8, block_size=16,
        )

    signatures = blocks.signatures.copy()
    signatures[0, 0] ^= np.uint64(1)
    with pytest.raises(GlobalConflictError, match="catalog replay"):
        validate_multilogical_blocks(
            model, frame, replace(blocks, signatures=signatures),
            logicals_per_block=8, block_size=16,
        )


def test_mlb_rejects_any_nonempty_or_tampered_logical_catalog():
    _, _, H = load_frozen_code(REGISTRY_PATH, "m03_c00")
    model, frame = build_model(H)
    empty = build_multilogical_empty_catalog(model, frame)
    assert validate_multilogical_empty_catalog(model, frame, empty)

    with pytest.raises(GlobalConflictError, match="empty catalog replay"):
        validate_multilogical_empty_catalog(
            model, frame, build_logical_proposal_catalog(model, frame),
        )

    supports = empty.support_offsets.copy()
    supports[-1] = 1
    with pytest.raises(GlobalConflictError, match="empty catalog replay"):
        validate_multilogical_empty_catalog(
            model, frame, replace(empty, support_offsets=supports),
        )
