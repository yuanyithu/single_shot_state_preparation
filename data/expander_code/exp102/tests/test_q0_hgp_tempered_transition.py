import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_global import (
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    CollapsedConflictError,
    CollapsedTemperedTransitionConfig,
    CollapsedTemperedTransitionSeedIdentity,
    _bits_to_mask,
    build_classical_coset_mass,
    run_collapsed_tempered_transition_trajectory,
    split_hgp_state,
    tempered_transition_log_acceptance,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


REGISTRY_PATH = "data/expander_code/exp102/registry/registry.json"


def _model(H):
    H = np.asarray(H, dtype=np.uint8)
    model, frame = build_model(H)
    return H, model, frame


def _seed(method, family="P", trajectory=0, namespace="q0_hgp_ctt_test"):
    return CollapsedTemperedTransitionSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=method,
        resource_tier="test",
        init_family=family,
        trajectory_index=trajectory,
        trajectory_namespace=namespace,
    )


def _coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, row in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= row
    return states


def _key(array):
    return np.packbits(np.asarray(array, dtype=np.uint8), bitorder="little").tobytes()


def _all_b_states(r):
    result = np.empty((1 << (r * r), r, r), dtype=np.uint8)
    for value in range(result.shape[0]):
        for bit in range(r * r):
            result[value, bit // r, bit % r] = (value >> bit) & 1
    return result


def _b_likelihoods(H, syndrome, p, b_states):
    r, n = H.shape
    y = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    log_mass = np.log(build_classical_coset_mass(H, p, engine="reference"))
    result = np.empty(b_states.shape[0], dtype=np.float64)
    for index, B in enumerate(b_states):
        a_syndromes = y ^ ((B.astype(np.int64) @ H.astype(np.int64)) % 2).astype(np.uint8)
        result[index] = sum(log_mass[int(_bits_to_mask(a_syndromes[:, column]))]
                            for column in range(n))
    return result


def _random_column_heatbath_matrix(H, p, b_states, likelihoods, power):
    """The production CTT random-scan kernel on tiny full-column blocks."""
    r, _ = H.shape
    lookup = {_key(B): index for index, B in enumerate(b_states)}
    log_odds = math.log(p / (1.0 - p))
    log_target = np.asarray([
        log_odds * int(B.sum()) + power * likelihood
        for B, likelihood in zip(b_states, likelihoods)
    ])
    transition = np.zeros((len(b_states), len(b_states)), dtype=np.float64)
    for source, B in enumerate(b_states):
        for column in range(r):
            targets = []
            for value in range(1 << r):
                candidate = B.copy()
                candidate[:, column] = [
                    (value >> row) & 1 for row in range(r)
                ]
                targets.append(lookup[_key(candidate)])
            weights = np.exp(log_target[targets] - log_target[targets].max())
            weights /= weights.sum()
            for target, probability in zip(targets, weights):
                transition[source, target] += probability / r
    return transition


def _tempered_transition_matrix(transitions, lambdas, likelihoods):
    """Enumerate the actual CTT path plus reject branch on a tiny B space."""
    levels = len(lambdas)
    states = transitions[0].shape[0]
    output = np.zeros((states, states), dtype=np.float64)

    def descend(source, forward, current, probability, level):
        if level == levels:
            reverse = np.empty(levels - 1, dtype=np.float64)
            reverse[-1] = likelihoods[current]
            ascend(source, forward, reverse, current, probability, levels - 2)
            return
        for target, transition_probability in enumerate(transitions[level][current]):
            if transition_probability == 0.0:
                continue
            updated = forward.copy()
            if level < levels - 1:
                updated[level] = likelihoods[target]
            descend(
                source, updated, target,
                probability * transition_probability, level + 1,
            )

    def ascend(source, forward, reverse, current, probability, level):
        # The palindromic proposal ends at x'_1; applying a final T_0 would
        # make the forward and reverse schedules differ.
        if level == 0:
            log_acceptance = tempered_transition_log_acceptance(
                lambdas, forward, reverse,
            )
            acceptance = min(1.0, math.exp(log_acceptance))
            output[source, current] += probability * acceptance
            output[source, source] += probability * (1.0 - acceptance)
            return
        for target, transition_probability in enumerate(transitions[level][current]):
            if transition_probability == 0.0:
                continue
            updated = reverse.copy()
            updated[level - 1] = likelihoods[target]
            ascend(
                source, forward, updated, target,
                probability * transition_probability, level - 1,
            )

    for source in range(states):
        forward = np.empty(levels - 1, dtype=np.float64)
        forward[0] = likelihoods[source]
        descend(source, forward, source, 1.0, 1)
    return output


def _full_ctt_transition(H, model, syndrome, p):
    """Lift an exact B CTT kernel through the exact A|B reconstruction."""
    states = _coset_states(model, syndrome)
    posterior = (p / (1.0 - p)) ** states.sum(axis=1)
    posterior /= posterior.sum()
    r, _ = H.shape
    b_states = _all_b_states(r)
    b_lookup = {_key(B): index for index, B in enumerate(b_states)}
    groups = [[] for _ in b_states]
    for index, state in enumerate(states):
        _, B = split_hgp_state(state, H)
        groups[b_lookup[_key(B)]].append(index)
    assert all(group for group in groups)
    b_mass = np.asarray([posterior[group].sum() for group in groups])
    likelihoods = _b_likelihoods(H, syndrome, p, b_states)
    lambdas = np.asarray([1.0, 0.25, 0.0], dtype=np.float64)
    local = [
        np.linalg.matrix_power(
            _random_column_heatbath_matrix(H, p, b_states, likelihoods, power),
            r,
        )
        for power in lambdas[:-1]
    ]
    prior = (p / (1.0 - p)) ** b_states.reshape(len(b_states), -1).sum(axis=1)
    prior /= prior.sum()
    b_transition = _tempered_transition_matrix(
        [*local, np.repeat(prior[None, :], len(b_states), axis=0)],
        lambdas, likelihoods,
    )
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    for source, state in enumerate(states):
        _, B = split_hgp_state(state, H)
        source_b = b_lookup[_key(B)]
        for target_b, group in enumerate(groups):
            conditional = posterior[group] / b_mass[target_b]
            transition[source, group] += b_transition[source_b, target_b] * conditional
    return posterior, transition


@pytest.mark.parametrize("H", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_ctt_complete_transition_preserves_exact_hgp_posterior(
        H, p, nonzero_syndrome):
    H, model, _ = _model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero_syndrome:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    assert bool(syndrome.any()) == nonzero_syndrome
    posterior, transition = _full_ctt_transition(H, model, syndrome, p)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-13
    assert np.max(np.abs(posterior @ transition - posterior)) <= 1e-13
    flow = posterior[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 1e-13


def test_ctt_log_hastings_path_formula_and_shape_rejection():
    lambdas = np.asarray([1.0, 0.25, 0.0])
    forward = np.asarray([-7.0, -3.0])
    reverse = np.asarray([-5.0, -4.0])
    expected = ((0.25 - 1.0) * (-7.0 + 5.0)
                + (0.0 - 0.25) * (-3.0 + 4.0))
    assert tempered_transition_log_acceptance(lambdas, forward, reverse) == expected
    with pytest.raises(ValueError, match="incompatible"):
        tempered_transition_log_acceptance(lambdas, forward[:1], reverse)


def test_ctt_config_seed_and_reference_numba_transcript_identity():
    H, model, frame = _model([[1, 1, 1]])
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, 4]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    config = CollapsedTemperedTransitionConfig(0.10, 2, 8, num_levels=3)
    seed = _seed(config.method_id, "U")
    initial = uniform_hard_coset_state(model, syndrome, seed.seed("initialize"))
    results = [
        run_collapsed_tempered_transition_trajectory(
            model, frame, H, syndrome, config, seed, initial, engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    for field in results[0]:
        if field != "engine":
            assert np.array_equal(
                np.asarray(results[0][field]), np.asarray(results[1][field]),
            ), field
    result = results[0]
    assert not result["measurement_residual_weights"].any()
    assert result["lambda_values"].tolist() == [1.0, 0.25, 0.0]
    assert result["burn_tt_counters"][0] == 2
    assert result["measurement_tt_counters"][0] == 8
    assert result["measurement_tt_prior_refresh_bit_changes"].shape == (8,)
    assert np.all(result["measurement_tt_prior_refresh_bit_changes"] >= 0)


def test_ctt_reference_numba_identity_handles_k64_bit63():
    _, _, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = model.logical_move_basis[63].copy()
    config = CollapsedTemperedTransitionConfig(0.40, 1, 8, num_levels=2)
    seed = _seed(config.method_id, namespace="ctt_k64_bit63")
    reference = run_collapsed_tempered_transition_trajectory(
        model, frame, H, syndrome, config, seed, initial, engine="reference",
    )
    accelerated = run_collapsed_tempered_transition_trajectory(
        model, frame, H, syndrome, config, seed, initial, engine="numba",
    )
    assert int(reference["initial_label"]) == 1 << 63
    assert reference["measurement_labels"].dtype == np.uint64
    for field in reference:
        if field != "engine":
            assert np.array_equal(
                np.asarray(reference[field]), np.asarray(accelerated[field]),
            ), field


def test_ctt_actual_m8_p004_64_level_reference_numba_identity():
    """Exercise the frozen V0 path, not only the small-code oracle path."""
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    uniform_seed = derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], 0, "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < 0.04
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    config = CollapsedTemperedTransitionConfig(
        0.04, 1, 8, num_levels=64, reversible_sweeps_per_level=1,
        block_size=8, method_id="CTT64-S1",
    )
    seed = CollapsedTemperedTransitionSeedIdentity(
        source_commit="a" * 40,
        config_sha256="b" * 64,
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint="c" * 64,
        method_id=config.method_id,
        resource_tier="actual_path_test",
        init_family="P",
        trajectory_index=0,
        trajectory_namespace="q0_hgp_ctt_actual_path_test",
    )
    reference = run_collapsed_tempered_transition_trajectory(
        model, frame, H, syndrome, config, seed, epsilon, engine="reference",
    )
    accelerated = run_collapsed_tempered_transition_trajectory(
        model, frame, H, syndrome, config, seed, epsilon, engine="numba",
    )
    for field in reference:
        if field != "engine":
            assert np.array_equal(
                np.asarray(reference[field]), np.asarray(accelerated[field]),
            ), field


@pytest.mark.parametrize("config", [
    lambda: CollapsedTemperedTransitionConfig(0.10, 1, 7),
    lambda: CollapsedTemperedTransitionConfig(0.10, 1, 8, num_levels=1),
    lambda: CollapsedTemperedTransitionConfig(0.10, 1, 8, block_size=4),
    lambda: CollapsedTemperedTransitionConfig(
        0.10, 1, 8, num_levels=3, method_id="CTT64-S1",
    ),
])
def test_ctt_config_rejects_nonfrozen_shapes(config):
    with pytest.raises(ValueError):
        config()


def test_ctt_rejects_mismatched_mass_or_seed_method():
    H, model, frame = _model([[1, 1, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    config = CollapsedTemperedTransitionConfig(0.10, 1, 8, num_levels=2)
    seed = _seed(config.method_id)
    mass = build_classical_coset_mass(H, 0.10)
    mass[0] *= 1.0001
    with pytest.raises(CollapsedConflictError, match="does not match H and p"):
        run_collapsed_tempered_transition_trajectory(
            model, frame, H, syndrome, config, seed, initial,
            engine="reference", mass=mass,
        )
    bad_seed = _seed("CTT03-S1")
    with pytest.raises(CollapsedConflictError, match="config/seed"):
        run_collapsed_tempered_transition_trajectory(
            model, frame, H, syndrome, config, bad_seed, initial,
            engine="reference",
        )
