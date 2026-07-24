import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    CollapsedConflictError,
    _bits_to_mask,
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_physical_pt import (
    CollapsedPhysicalPtConfig,
    PhysicalPtMassArtifact,
    PhysicalPtSeedIdentity,
    _reference_log_target,
    build_physical_pt_mass_artifact,
    physical_pt_resource_requirements,
    run_collapsed_physical_pt_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _seed(method="CPPT32", *, family="P", trajectory=0, namespace="cppt_test"):
    return PhysicalPtSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=method,
        resource_tier="TEST",
        init_family=family,
        trajectory_index=trajectory,
        trajectory_namespace=namespace,
    )


def _hard_coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, row in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= row
    return states


def _b_matrices(rows):
    result = []
    for value in range(1 << (rows * rows)):
        bits = np.asarray(
            [(value >> bit) & 1 for bit in range(rows * rows)],
            dtype=np.uint8,
        )
        result.append(bits.reshape(rows, rows))
    return result


def _collapsed_scores(H, syndrome, p):
    rows, columns = H.shape
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(p / (1.0 - p))
    matrices = _b_matrices(rows)
    scores = []
    for B in matrices:
        a_syndromes = Y ^ (
            B.astype(np.int64) @ H.astype(np.int64) % 2
        ).astype(np.uint8)
        masks = np.asarray(
            [_bits_to_mask(a_syndromes[:, column]) for column in range(columns)],
            dtype=np.uint32,
        )
        b_columns = np.asarray(
            [_bits_to_mask(B[:, column]) for column in range(rows)],
            dtype=np.uint32,
        )
        scores.append(
            _reference_log_target(b_columns, masks, log_mass, log_odds)
        )
    return matrices, np.asarray(scores, dtype=np.float64)


def _deterministic_column_sweep(H, syndrome, p):
    matrices, scores = _collapsed_scores(H, syndrome, p)
    rows = H.shape[0]
    keys = {matrix.tobytes(): index for index, matrix in enumerate(matrices)}
    transition = np.zeros((len(matrices), len(matrices)), dtype=np.float64)
    for source_index, source in enumerate(matrices):
        partial = {source.tobytes(): 1.0}
        for column in range(rows):
            updated = {}
            for packed, prefix_probability in partial.items():
                base = np.frombuffer(packed, dtype=np.uint8).reshape(rows, rows)
                candidate_indices = []
                for value in range(1 << rows):
                    candidate = base.copy()
                    candidate[:, column] = [
                        (value >> bit) & 1 for bit in range(rows)
                    ]
                    candidate_indices.append(keys[candidate.tobytes()])
                conditional = np.exp(
                    scores[candidate_indices] - np.max(scores[candidate_indices])
                )
                conditional /= conditional.sum()
                for candidate_index, probability in zip(
                    candidate_indices, conditional,
                ):
                    key = matrices[candidate_index].tobytes()
                    updated[key] = updated.get(key, 0.0) + (
                        prefix_probability * float(probability)
                    )
            partial = updated
        for packed, probability in partial.items():
            transition[source_index, keys[packed]] = probability
    return scores, transition


def test_physical_pt_config_ladder_seed_and_resource_identity():
    config32 = CollapsedPhysicalPtConfig("CPPT32", 0.04, 8, 16)
    config64 = CollapsedPhysicalPtConfig("CPPT64", 0.04, 8, 16)
    assert config32.p_values[0] == 0.5
    assert config32.p_values[-1] == 0.04
    assert np.all(np.diff(config32.beta_values) > 0.0)
    assert np.all(np.diff(config32.p_values) < 0.0)
    assert config64.num_replicas == 64
    seed = _seed()
    assert seed.seed("burn", "replica", 0) != seed.seed(
        "measurement", "replica", 0,
    )
    assert seed.seed("burn", "replica", 0) != _seed(trajectory=1).seed(
        "burn", "replica", 0,
    )

    H = np.zeros((24, 32), dtype=np.uint8)
    requirements32 = physical_pt_resource_requirements(H, config32)
    requirements64 = physical_pt_resource_requirements(H, config64)
    assert requirements32["mass_table_bytes"] == 4 * 1024 ** 3
    assert requirements64["mass_table_bytes"] == 8 * 1024 ** 3


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25, 0.50])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_physical_p_collapsed_density_matches_full_hard_coset(p, nonzero_syndrome):
    H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    model, _ = build_model(H)
    planted = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero_syndrome:
        planted[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2
    ).astype(np.uint8)
    states = _hard_coset_states(model, syndrome)
    odds = p / (1.0 - p)
    full = odds ** states.sum(axis=1)
    full /= full.sum()
    matrices, scores = _collapsed_scores(H, syndrome, p)
    collapsed = np.exp(scores - scores.max())
    collapsed /= collapsed.sum()
    grouped = np.zeros(len(matrices), dtype=np.float64)
    keys = {matrix.tobytes(): index for index, matrix in enumerate(matrices)}
    for state, probability in zip(states, full):
        _, B = split_hgp_state(state, H)
        grouped[keys[B.tobytes()]] += probability
    assert np.max(np.abs(grouped - collapsed)) <= 2e-13


def test_physical_p_local_sweeps_and_adjacent_swap_preserve_product_target():
    H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    model, _ = build_model(H)
    planted = np.zeros(model.num_qubits, dtype=np.uint8)
    planted[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2
    ).astype(np.uint8)
    score_hot, kernel_hot = _deterministic_column_sweep(H, syndrome, 0.5)
    score_cold, kernel_cold = _deterministic_column_sweep(H, syndrome, 0.10)
    hot = np.exp(score_hot - score_hot.max())
    cold = np.exp(score_cold - score_cold.max())
    hot /= hot.sum()
    cold /= cold.sum()
    assert np.max(np.abs(kernel_hot - 1.0 / kernel_hot.shape[1])) <= 2e-15
    assert np.max(np.abs(hot @ kernel_hot - hot)) <= 2e-15
    assert np.max(np.abs(cold @ kernel_cold - cold)) <= 2e-15

    states = len(hot)
    product = np.asarray([
        hot[left] * cold[right]
        for left, right in itertools.product(range(states), repeat=2)
    ])
    swap = np.zeros((states * states, states * states), dtype=np.float64)
    for left, right in itertools.product(range(states), repeat=2):
        source = left * states + right
        target = right * states + left
        delta = (
            score_hot[right] + score_cold[left]
            - score_hot[left] - score_cold[right]
        )
        acceptance = min(1.0, math.exp(delta))
        swap[source, target] += acceptance
        swap[source, source] += 1.0 - acceptance
    flow = product[:, None] * swap
    assert np.max(np.abs(flow - flow.T)) <= 2e-15
    local = np.kron(kernel_hot, kernel_cold)
    complete = local @ swap
    assert np.max(np.abs(complete.sum(axis=1) - 1.0)) <= 3e-15
    assert np.max(np.abs(product @ complete - product)) <= 3e-15


def test_reference_numba_trajectory_identity_through_k64_bit63():
    H = np.ones((1, 9), dtype=np.uint8)
    model, frame = build_model(H)
    assert model.k == 64
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = model.logical_move_basis[63].copy()
    config = CollapsedPhysicalPtConfig("CPPT32", 0.40, 1, 8)
    seed = _seed(namespace="cppt_k64_bit63")
    artifact = build_physical_pt_mass_artifact(
        H, config.p_values, "reference",
    )
    assert artifact.log_mass_tables.flags.writeable is False
    reference = run_collapsed_physical_pt_trajectory(
        model, frame, H, syndrome, config, seed, initial,
        engine="reference", mass_artifact=artifact,
    )
    accelerated = run_collapsed_physical_pt_trajectory(
        model, frame, H, syndrome, config, seed, initial,
        engine="numba", mass_artifact=artifact,
    )
    assert int(reference["initial_label"]) == 1 << 63
    bit63 = np.uint64(1) << np.uint64(63)
    assert np.any((reference["measurement_labels"] & bit63) != 0)
    for field in reference:
        if field != "engine":
            assert np.array_equal(
                np.asarray(reference[field]), np.asarray(accelerated[field]),
            ), field
    assert not reference["measurement_residual_weights"].any()
    assert reference["swap_attempts"].sum() == 5 * 16 + 4 * 15


def test_physical_pt_rejects_seed_mass_and_support_tampering():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = build_model(H)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    config = CollapsedPhysicalPtConfig("CPPT32", 0.10, 1, 8)
    with pytest.raises(CollapsedConflictError, match="config/seed method"):
        run_collapsed_physical_pt_trajectory(
            model, frame, H, syndrome, config, _seed("CPPT64"), initial,
            engine="reference",
        )
    artifact = build_physical_pt_mass_artifact(
        H, config.p_values, "reference",
    )
    tampered = artifact.log_mass_tables.copy()
    tampered[1, 0] += 1e-4
    with pytest.raises(CollapsedConflictError, match="artifact SHA changed"):
        PhysicalPtMassArtifact(
            artifact.h_sha256, artifact.p_values_sha256, tampered,
            artifact.log_mass_tables_sha256,
        )
    foreign = CollapsedPhysicalPtConfig("CPPT32", 0.04, 1, 8)
    with pytest.raises(CollapsedConflictError, match="artifact binding"):
        run_collapsed_physical_pt_trajectory(
            model, frame, H, syndrome, foreign, _seed(), initial,
            engine="reference", mass_artifact=artifact,
        )
    outside = initial.copy()
    outside[0] = 1
    with pytest.raises(CollapsedConflictError, match="outside the hard coset"):
        run_collapsed_physical_pt_trajectory(
            model, frame, H, syndrome, config, _seed(), outside,
            engine="reference",
        )


@pytest.mark.parametrize("bad", [
    ("CPPT16", 0.10, 1, 8),
    ("CPPT32", 0.50, 1, 8),
    ("CPPT32", 0.10, 0, 8),
    ("CPPT32", 0.10, 1, 7),
])
def test_physical_pt_config_rejects_unfrozen_variants(bad):
    with pytest.raises(ValueError):
        CollapsedPhysicalPtConfig(*bad)
