import math
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _bits_to_mask,
    _initial_collapsed_masks,
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    FULL_ROW_METHOD_ID,
    FullRowGibbsConfig,
    FullRowGibbsSeedIdentity,
    brute_force_full_row_conditional,
    build_full_row_elimination_plan,
    full_row_conditional_probabilities,
    full_row_current_assignment_log_probability,
    run_full_row_gibbs_trajectory,
    select_low_energy_logical_start,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


SMALL_H = (
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
)


def _model(H):
    H = np.asarray(H, dtype=np.uint8)
    model, frame = build_model(H)
    return H, model, frame


def _syndrome(model, nonzero):
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    return epsilon, syndrome


def _all_b_columns(rank):
    result = np.zeros((1 << (rank * rank), rank), dtype=np.uint32)
    for integer in range(result.shape[0]):
        for column in range(rank):
            for row in range(rank):
                if (integer >> (column * rank + row)) & 1:
                    result[integer, column] |= np.uint32(1) << np.uint32(row)
    return result


def _b_index(columns, rank):
    result = 0
    for column, value in enumerate(columns):
        for row in range(rank):
            result |= ((int(value) >> row) & 1) << (column * rank + row)
    return result


def _a_syndromes(syndrome, H, b_columns):
    rows, columns = H.shape
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    result = np.asarray([_bits_to_mask(Y[:, column]) for column in range(columns)], dtype=np.uint32)
    for column in range(columns):
        for variable in np.flatnonzero(H[:, column]):
            result[column] ^= b_columns[int(variable)]
    return result


def _collapsed_b_posterior(H, syndrome, p):
    rows, _ = H.shape
    b_states = _all_b_columns(rows)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_mass = np.log(mass)
    log_odds = math.log(p / (1.0 - p))
    log_weights = np.empty(b_states.shape[0], dtype=np.float64)
    a_values = []
    for index, b_columns in enumerate(b_states):
        a_syndromes = _a_syndromes(syndrome, H, b_columns)
        a_values.append(a_syndromes)
        log_weights[index] = (
            sum(int(value).bit_count() for value in b_columns) * log_odds
            + sum(float(log_mass[int(value)]) for value in a_syndromes)
        )
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum(dtype=np.float64)
    return b_states, a_values, weights, mass, log_odds


def _replace_b_row(columns, row, assignment):
    result = np.asarray(columns, dtype=np.uint32).copy()
    mask = np.uint32(1 << row)
    clear = np.uint32(~int(mask) & 0xFFFFFFFF)
    for variable in range(result.size):
        result[variable] &= clear
        if (assignment >> variable) & 1:
            result[variable] |= mask
    return result


def _row_b_transition(H, syndrome, p, row):
    plan = build_full_row_elimination_plan(H)
    b_states, a_values, posterior, mass, log_odds = _collapsed_b_posterior(H, syndrome, p)
    transition = np.zeros((b_states.shape[0], b_states.shape[0]), dtype=np.float64)
    for source, (b_columns, a_syndromes) in enumerate(zip(b_states, a_values)):
        conditional, _ = full_row_conditional_probabilities(
            H, plan, b_columns, a_syndromes, row, np.log(mass), log_odds,
        )
        for assignment, probability in enumerate(conditional):
            target = _b_index(_replace_b_row(b_columns, row, assignment), H.shape[0])
            transition[source, target] += probability
    return b_states, a_values, posterior, mass, log_odds, transition


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


def _full_joint_row_transition(H, model, syndrome, p, row):
    states = _hard_coset_states(model, syndrome)
    posterior = (p / (1.0 - p)) ** states.sum(axis=1)
    posterior /= posterior.sum(dtype=np.float64)
    b_by_state = [split_hgp_state(state, H)[1] for state in states]
    groups = {}
    for index, B in enumerate(b_by_state):
        groups.setdefault(_b_index(np.asarray([
            _bits_to_mask(B[:, column]) for column in range(B.shape[1])
        ], dtype=np.uint32), H.shape[0]), []).append(index)
    b_mass = {key: float(posterior[indices].sum()) for key, indices in groups.items()}
    plan = build_full_row_elimination_plan(H)
    mass = build_classical_coset_mass(H, p, engine="reference")
    log_odds = math.log(p / (1.0 - p))
    transition = np.zeros((states.shape[0], states.shape[0]), dtype=np.float64)
    for source, (state, B) in enumerate(zip(states, b_by_state)):
        b_columns, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, H)
        conditional, _ = full_row_conditional_probabilities(
            H, plan, b_columns, a_syndromes, row, np.log(mass), log_odds,
        )
        for assignment, probability in enumerate(conditional):
            target_b = _replace_b_row(b_columns, row, assignment)
            target_key = _b_index(target_b, H.shape[0])
            for target in groups[target_key]:
                transition[source, target] += (
                    probability * posterior[target] / b_mass[target_key]
                )
    return posterior, transition


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_full_row_conditionals_match_brute_force_oracle(H, p, nonzero_syndrome):
    H, model, _ = _model(H)
    _, syndrome = _syndrome(model, nonzero_syndrome)
    plan = build_full_row_elimination_plan(H)
    b_states, a_values, _, mass, log_odds = _collapsed_b_posterior(H, syndrome, p)
    for b_columns, a_syndromes in zip(b_states, a_values):
        for row in range(H.shape[0]):
            exact, exact_log_normalizer = brute_force_full_row_conditional(
                H, b_columns, a_syndromes, row, np.log(mass), log_odds,
            )
            eliminated, eliminated_log_normalizer = full_row_conditional_probabilities(
                H, plan, b_columns, a_syndromes, row, np.log(mass), log_odds,
            )
            assert abs(exact_log_normalizer - eliminated_log_normalizer) <= 3e-14
            assert np.max(np.abs(exact - eliminated)) <= 3e-14
            current = sum(
                ((int(b_columns[variable]) >> row) & 1) << variable
                for variable in range(H.shape[0])
            )
            assert abs(
                full_row_current_assignment_log_probability(
                    H, plan, b_columns, a_syndromes, row, np.log(mass), log_odds,
                ) - math.log(exact[current])
            ) <= 3e-14


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_full_row_b_kernel_and_sweep_are_exact(H, p, nonzero_syndrome):
    H, model, _ = _model(H)
    _, syndrome = _syndrome(model, nonzero_syndrome)
    transitions = []
    posterior = None
    for row in range(H.shape[0]):
        _, _, current, _, _, transition = _row_b_transition(H, syndrome, p, row)
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 5e-15
        flow = current[:, None] * transition
        assert np.max(np.abs(flow - flow.T)) <= 5e-14
        transitions.append(transition)
        posterior = current
    sweep = transitions[0]
    for transition in transitions[1:]:
        sweep = sweep @ transition
    assert np.max(np.abs(posterior @ sweep - posterior)) <= 8e-14


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_full_hgp_exact_enumeration_validates_row_kernel(H, p, nonzero_syndrome):
    H, model, _ = _model(H)
    _, syndrome = _syndrome(model, nonzero_syndrome)
    transitions = []
    posterior = None
    for row in range(H.shape[0]):
        current, transition = _full_joint_row_transition(H, model, syndrome, p, row)
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 6e-15
        assert np.max(np.abs(current @ transition - current)) <= 8e-14
        flow = current[:, None] * transition
        assert np.max(np.abs(flow - flow.T)) <= 8e-14
        transitions.append(transition)
        posterior = current
    sweep = transitions[0]
    for transition in transitions[1:]:
        sweep = sweep @ transition
    assert np.max(np.abs(posterior @ sweep - posterior)) <= 1e-13


def _seed(config):
    return FullRowGibbsSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=config.method_id,
        resource_tier="test",
        init_family="P",
        trajectory_index=0,
        trajectory_namespace="q0_hgp_full_row_gibbs_test",
    )


def test_reference_trajectory_is_replayable_and_stays_in_the_hard_coset():
    H, model, frame = _model(SMALL_H[1])
    epsilon, syndrome = _syndrome(model, True)
    config = FullRowGibbsConfig(0.10, 16, 32)
    first = run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon,
    )
    second = run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon,
    )
    for field in first:
        if isinstance(first[field], np.ndarray):
            assert np.array_equal(first[field], second[field]), field
        else:
            assert first[field] == second[field], field
    assert first["method_id"] == FULL_ROW_METHOD_ID
    assert np.all(first["measurement_residual_weights"] == 0)
    assert np.all(first["measurement_b_columns"] < (1 << H.shape[0]))
    assert np.all(first["measurement_a_syndromes"] < (1 << H.shape[0]))
    assert first["burn_counters"][0] == config.burn_sweeps * H.shape[0]
    assert first["measurement_counters"][0] == config.measurement_sweeps * H.shape[0]


def test_l_start_is_legal_low_energy_and_logically_separated():
    H, model, frame = _model(SMALL_H[1])
    epsilon, syndrome = _syndrome(model, True)
    start, metadata = select_low_energy_logical_start(epsilon, model, frame)
    residual = (
        model.H_check.astype(np.int64) @ start.astype(np.int64) % 2
    ).astype(np.uint8)
    assert np.array_equal(residual, syndrome)
    assert int(start.sum()) == metadata["selected_absolute_weight"]
    assert metadata["candidate_count"] > 0
    assert not np.array_equal(frame.label_of(start), frame.label_of(epsilon))


def test_reference_numba_transcript_identity_on_nonzero_small_hgp():
    pytest.importorskip("numba")
    H, model, frame = _model(SMALL_H[1])
    epsilon, syndrome = _syndrome(model, True)
    config = FullRowGibbsConfig(0.10, 16, 32)
    reference = run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon,
        engine="reference",
    )
    accelerated = run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, config, _seed(config), epsilon,
        engine="numba",
    )
    for field in reference:
        if field != "engine":
            assert np.array_equal(
                np.asarray(reference[field]), np.asarray(accelerated[field]),
            ), field


def test_reference_numba_transcript_identity_preserves_m8_bit63_label():
    pytest.importorskip("numba")
    registry_root = Path("data/expander_code/exp102/registry")
    with np.load(registry_root / "codes" / "m08_c06.npz", allow_pickle=False) as data:
        H = data["H"].copy()
    model, frame = build_model(H)
    initial = next(
        row.copy() for row in model.logical_move_basis
        if (int(state_label(frame, row)) >> 63) & 1
    )
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    config = FullRowGibbsConfig(0.04, 1, 8)
    seed = FullRowGibbsSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=config.method_id,
        resource_tier="test",
        init_family="L",
        trajectory_index=0,
        trajectory_namespace="q0_hgp_full_row_gibbs_m8_bit63_test",
    )
    reference = run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, config, seed, initial, engine="reference",
    )
    accelerated = run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, config, seed, initial, engine="numba",
    )
    assert (int(reference["initial_label"]) >> 63) & 1
    assert reference["measurement_labels"].dtype == np.uint64
    for field in reference:
        if field != "engine":
            assert np.array_equal(
                np.asarray(reference[field]), np.asarray(accelerated[field]),
            ), field


def test_plan_is_deterministic_and_result_independent():
    H = np.asarray([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ], dtype=np.uint8)
    first = build_full_row_elimination_plan(H)
    second = build_full_row_elimination_plan(H.copy())
    assert first.as_dict() == second.as_dict()
    assert first.sha256 == second.sha256
    assert first.order == (0, 1, 2)
    assert first.max_table_entries == 8


def test_frozen_registry_min_fill_bounds_and_hard_m8_plan_are_stable():
    registry_root = Path("data/expander_code/exp102/registry")
    registry = load_registry(registry_root / "registry.json")
    maxima = {}
    hard_plan = None
    for code in registry["codes"]:
        with np.load(registry_root / "codes" / f'{code["code_id"]}.npz', allow_pickle=False) as data:
            plan = build_full_row_elimination_plan(data["H"])
        maxima[code["m"]] = max(maxima.get(code["m"], 0), plan.max_width)
        if code["code_id"] == "m08_c06":
            hard_plan = plan
    assert maxima == {3: 6, 4: 7, 5: 9, 6: 10, 7: 11, 8: 13}
    assert hard_plan is not None
    assert hard_plan.max_width == 12
    assert hard_plan.max_table_entries == 8192
    assert hard_plan.structural_table_cells == 37118
    assert hard_plan.order == (
        3, 16, 6, 23, 9, 15, 4, 13, 12, 8, 22, 0,
        1, 2, 5, 7, 10, 11, 14, 17, 18, 19, 20, 21,
    )
