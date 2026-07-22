import itertools
import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_global import (
    GlobalSeedIdentity,
    uniform_hard_coset_state,
    validate_observable_frame,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    CollapsedConflictError,
    CollapsedConfig,
    CollapsedPowerPtConfig,
    _bits_to_mask,
    _classical_row_neighbors,
    _hp_advance_transport,
    _initial_collapsed_masks,
    _qubit_signatures,
    _reference_advance_transport,
    _reference_categorical_draw,
    _reference_power_b_sweep,
    _reference_sample_full_state,
    _section_and_kernel_masks,
    build_classical_coset_mass,
    hgp_syndrome_matrix,
    join_hgp_state,
    run_collapsed_power_pt_trajectory,
    run_collapsed_trajectory,
    split_hgp_state,
    validate_hgp_wiring,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _model(H):
    H = np.asarray(H, dtype=np.uint8)
    model, frame = build_model(H)
    return H, model, frame


def _seed(method, family="P", trajectory=0, namespace="q0_hgp_collapsed_test"):
    return GlobalSeedIdentity(
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


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("H", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
def test_classical_coset_mass_matches_brute_force_and_numba_reference(H, p):
    H = np.asarray(H, dtype=np.uint8)
    r, n = H.shape
    brute = np.zeros(1 << r, dtype=np.float64)
    for integer in range(1 << n):
        state = np.asarray([(integer >> bit) & 1 for bit in range(n)], dtype=np.uint8)
        syndrome = (H.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8)
        weight = int(state.sum())
        brute[int(_bits_to_mask(syndrome))] += p ** weight * (1.0 - p) ** (n - weight)
    reference = build_classical_coset_mass(H, p, engine="reference")
    accelerated = build_classical_coset_mass(H, p, engine="numba")
    assert np.array_equal(reference, accelerated)
    assert np.max(np.abs(reference - brute)) <= 2e-16
    assert abs(float(reference.sum()) - 1.0) <= 3e-16


@pytest.mark.parametrize("H", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
def test_hgp_matrix_factorization_matches_canonical_hz(H):
    H, model, _ = _model(H)
    validate_hgp_wiring(H, model)
    r, n = H.shape
    rng = np.random.default_rng(321)
    A = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
    B = rng.integers(0, 2, size=(r, r), dtype=np.uint8)
    state = join_hgp_state(A, B)
    recovered_A, recovered_B = split_hgp_state(state, H)
    assert np.array_equal(recovered_A, A)
    assert np.array_equal(recovered_B, B)
    direct = (model.H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8)
    assert np.array_equal(direct.reshape(r, n), hgp_syndrome_matrix(A, B, H))


def _b_key(B):
    return np.packbits(B, bitorder="little").tobytes()


def _collapsed_full_transition(H, model, syndrome, p):
    r, n = H.shape
    states = _coset_states(model, syndrome)
    posterior = (p / (1.0 - p)) ** states.sum(axis=1)
    posterior /= posterior.sum()
    blocks = [split_hgp_state(state, H) for state in states]
    by_b = {}
    for index, (_, B) in enumerate(blocks):
        by_b.setdefault(_b_key(B), []).append(index)
    b_mass = {key: float(posterior[indices].sum()) for key, indices in by_b.items()}
    transition = np.zeros((states.shape[0], states.shape[0]), dtype=np.float64)
    for source, (_, B) in enumerate(blocks):
        for column in range(r):
            candidates = []
            candidate_weights = []
            for integer in range(1 << r):
                proposed = B.copy()
                proposed[:, column] = [(integer >> bit) & 1 for bit in range(r)]
                key = _b_key(proposed)
                candidates.append(key)
                candidate_weights.append(b_mass[key])
            candidate_weights = np.asarray(candidate_weights)
            candidate_weights /= candidate_weights.sum()
            for key, b_probability in zip(candidates, candidate_weights):
                indices = by_b[key]
                conditional = posterior[indices] / b_mass[key]
                transition[source, indices] += b_probability * conditional / r
    return states, posterior, transition


class _FixedPermutationRng:
    def __init__(self, order):
        self.order = np.asarray(order, dtype=np.int32)

    def permutation(self, size):
        assert int(size) == self.order.size
        return self.order.copy()

    def random(self):
        raise AssertionError("the exact transition oracle supplies categorical choices")


class _SingleColumnPermutationRng:
    """Restrict the production sweep to one column for a block-DB oracle."""

    def __init__(self, size, column):
        self.size = int(size)
        self.column = int(column)
        self.calls = 0

    def permutation(self, size):
        assert int(size) == self.size
        self.calls += 1
        if self.calls == 1:
            return np.asarray([self.column], dtype=np.int32)
        return np.arange(self.size, dtype=np.int32)

    def random(self):
        raise AssertionError("the exact transition oracle supplies categorical choices")


class _ScriptedCategorical:
    def __init__(self, choices):
        self._choices = iter(choices)
        self.probability = 1.0
        self.draws = 0

    def __call__(self, weights, _rng):
        category = next(self._choices)
        weights = np.asarray(weights, dtype=np.float64)
        self.probability *= float(weights[category] / weights.sum())
        self.draws += 1
        return category


def _production_reference_transition(H, model, frame, syndrome, p, column_order):
    """Enumerate the real reference B categorical and A reconstruction path."""
    states = _coset_states(model, syndrome)
    packed_to_index = {
        np.packbits(state, bitorder="little").tobytes(): index
        for index, state in enumerate(states)
    }
    assert len(packed_to_index) == len(states)
    r, n = H.shape
    section_masks, kernel_combinations = _section_and_kernel_masks(H)
    neighbors, neighbor_counts = _classical_row_neighbors(H)
    mass = build_classical_coset_mass(H, p, engine="reference")
    odds = p / (1.0 - p)
    odds_powers = odds ** np.arange(max(H.shape) + 1, dtype=np.float64)
    signatures = _qubit_signatures(frame)
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    rng = _FixedPermutationRng(column_order)
    choice_ranges = [range(1 << r)] * r + [range(len(kernel_combinations))] * n

    for source_index, source in enumerate(states):
        initial_b, initial_syndromes, _ = _initial_collapsed_masks(
            source, syndrome, H,
        )
        for choices in itertools.product(*choice_ranges):
            b_columns = initial_b.copy()
            a_syndromes = initial_syndromes.copy()
            categorical = _ScriptedCategorical(choices)
            _reference_power_b_sweep(
                b_columns, a_syndromes, rng, 8, 1.0, neighbors,
                neighbor_counts, np.log(mass), np.log(odds),
                categorical_draw=categorical,
            )
            target, _, _ = _reference_sample_full_state(
                source, b_columns, a_syndromes, rng, n, r, section_masks,
                kernel_combinations, odds_powers, signatures,
                categorical_draw=categorical,
            )
            assert categorical.draws == r + n
            target_index = packed_to_index[
                np.packbits(target, bitorder="little").tobytes()
            ]
            transition[source_index, target_index] += categorical.probability
    return states, transition


def _production_reference_single_column_transition(
        H, model, frame, syndrome, p, column):
    """Enumerate one real B-column heatbath followed by exact A|B sampling."""
    states = _coset_states(model, syndrome)
    packed_to_index = {
        np.packbits(state, bitorder="little").tobytes(): index
        for index, state in enumerate(states)
    }
    r, n = H.shape
    section_masks, kernel_combinations = _section_and_kernel_masks(H)
    neighbors, neighbor_counts = _classical_row_neighbors(H)
    mass = build_classical_coset_mass(H, p, engine="reference")
    odds = p / (1.0 - p)
    odds_powers = odds ** np.arange(max(H.shape) + 1, dtype=np.float64)
    signatures = _qubit_signatures(frame)
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    choice_ranges = (
        [range(1 << r)] + [range(len(kernel_combinations))] * n
    )

    for source_index, source in enumerate(states):
        initial_b, initial_syndromes, _ = _initial_collapsed_masks(
            source, syndrome, H,
        )
        for choices in itertools.product(*choice_ranges):
            b_columns = initial_b.copy()
            a_syndromes = initial_syndromes.copy()
            categorical = _ScriptedCategorical(choices)
            rng = _SingleColumnPermutationRng(r, column)
            _reference_power_b_sweep(
                b_columns, a_syndromes, rng, 8, 1.0, neighbors,
                neighbor_counts, np.log(mass), np.log(odds),
                categorical_draw=categorical,
            )
            target, _, _ = _reference_sample_full_state(
                source, b_columns, a_syndromes, rng, n, r, section_masks,
                kernel_combinations, odds_powers, signatures,
                categorical_draw=categorical,
            )
            assert categorical.draws == 1 + n
            target_index = packed_to_index[
                np.packbits(target, bitorder="little").tobytes()
            ]
            transition[source_index, target_index] += categorical.probability
    return states, transition


@pytest.mark.parametrize("H", [
    np.array([[1, 1, 1]], dtype=np.uint8),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_collapsed_random_column_heatbath_full_transition_detailed_balance(H, p):
    model, _ = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    _, posterior, transition = _collapsed_full_transition(H, model, syndrome, p)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 2e-15
    assert np.max(np.abs(posterior @ transition - posterior)) <= 2e-15
    flow = posterior[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 2e-15


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
@pytest.mark.parametrize("H", [
    np.array([[1, 1, 1]], dtype=np.uint8),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
])
def test_production_reference_b_sweep_and_a_path_match_exact_hgp_posterior(
        H, p, nonzero_syndrome):
    H, model, frame = _model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero_syndrome:
        epsilon[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    assert bool(syndrome.any()) == nonzero_syndrome
    for column_order in (np.arange(H.shape[0]), np.arange(H.shape[0] - 1, -1, -1)):
        states, transition = _production_reference_transition(
            H, model, frame, syndrome, p, column_order,
        )
        posterior = (p / (1.0 - p)) ** states.sum(axis=1)
        posterior /= posterior.sum()
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-13
        assert np.max(np.abs(posterior @ transition - posterior)) <= 1e-13
        if H.shape[0] == 1:
            assert np.max(np.abs(transition - posterior[None, :])) <= 1e-13


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero_syndrome", [False, True])
def test_production_reference_single_b_column_has_detailed_balance(
        p, nonzero_syndrome):
    H, model, frame = _model([[1, 1, 0], [0, 1, 1]])
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero_syndrome:
        epsilon[0] = 1
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    for column in range(H.shape[0]):
        states, transition = _production_reference_single_column_transition(
            H, model, frame, syndrome, p, column,
        )
        posterior = (p / (1.0 - p)) ** states.sum(axis=1)
        posterior /= posterior.sum()
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-13
        assert np.max(np.abs(posterior @ transition - posterior)) <= 1e-13
        flow = posterior[:, None] * transition
        assert np.max(np.abs(flow - flow.T)) <= 1e-13


def test_reference_categorical_draw_uses_exact_half_open_cdf_intervals():
    class FixedRng:
        def __init__(self, value):
            self.value = value

        def random(self):
            return self.value

    weights = np.asarray([1.0, 2.0, 3.0])
    assert _reference_categorical_draw(weights, FixedRng(0.0)) == 0
    assert _reference_categorical_draw(weights, FixedRng(1.0 / 6.0)) == 1
    assert _reference_categorical_draw(weights, FixedRng(0.5)) == 2
    assert _reference_categorical_draw(weights, FixedRng(np.nextafter(1.0, 0.0))) == 2


def test_likelihood_power_local_refresh_and_swap_preserve_product_target():
    H = np.array([[1, 1, 1]], dtype=np.uint8)
    p = 0.10
    mass = build_classical_coset_mass(H, p)
    # A nonuniform syndrome makes the likelihood tempering nontrivial.
    Y = np.array([[1, 0, 1]], dtype=np.uint8)
    likelihood = np.empty(2, dtype=np.float64)
    for B in (0, 1):
        syndromes = Y[0] ^ B
        likelihood[B] = float(np.prod(mass[syndromes]))
    lambdas = np.array([0.0, 1.0])
    distributions = []
    for power in lambdas:
        values = np.asarray([
            (1.0 - p) * likelihood[0] ** power,
            p * likelihood[1] ** power,
        ])
        distributions.append(values / values.sum())
    product = np.asarray([
        distributions[0][left] * distributions[1][right]
        for left, right in itertools.product((0, 1), repeat=2)
    ])
    pairs = list(itertools.product((0, 1), repeat=2))
    transition = np.zeros((4, 4), dtype=np.float64)
    for source in range(4):
        for proposed_index, (left, right) in enumerate(pairs):
            refresh = distributions[0][left] * distributions[1][right]
            delta = ((lambdas[0] - lambdas[1])
                     * (math.log(likelihood[right]) - math.log(likelihood[left])))
            acceptance = min(1.0, math.exp(delta))
            swapped_index = pairs.index((right, left))
            transition[source, swapped_index] += refresh * acceptance
            transition[source, proposed_index] += refresh * (1.0 - acceptance)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 2e-15
    assert np.max(np.abs(product @ transition - product)) <= 2e-15


def test_collapsed_configs_and_short_numba_trajectories_replay():
    H, model, frame = _model([[1, 1, 1]])
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, 4]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    for method in ("HC08", "HM08"):
        config = CollapsedConfig(method, 0.10, 8, 16)
        result = run_collapsed_trajectory(
            model, frame, H, syndrome, config, _seed(method), epsilon,
        )
        assert not result["measurement_residual_weights"].any()
        assert result["measurement_states_packed"].shape == (16, 2)
        assert result["measurement_block"].tolist() == list(itertools.chain.from_iterable([[i] * 2 for i in range(8)]))
    config = CollapsedPowerPtConfig("HP16", 0.10, 8, 16)
    seed = _seed("HP16", "U")
    uniform = uniform_hard_coset_state(model, syndrome, seed.seed("initialize"))
    reference = run_collapsed_power_pt_trajectory(
        model, frame, H, syndrome, config, seed, uniform, engine="reference",
    )
    result = run_collapsed_power_pt_trajectory(
        model, frame, H, syndrome, config, seed, uniform, engine="numba",
    )
    for field in reference:
        if field != "engine":
            assert np.array_equal(np.asarray(reference[field]), np.asarray(result[field])), field
    assert not result["measurement_residual_weights"].any()
    assert result["lambda_values"][0] == 0.0
    assert result["lambda_values"][-1] == 1.0
    assert np.all(np.diff(result["lambda_values"]) > 0.0)
    assert result["swap_attempts"].sum() == 15 * (8 + 16) // 2


def test_power_pt_reference_numba_bit_identity_through_k64_bit63():
    registry = "data/expander_code/exp102/registry/registry.json"
    _, _, H = load_frozen_code(registry, "m08_c06")
    model, frame = build_model(H)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = model.logical_move_basis[63].copy()
    seed = _seed("HP16", "P", namespace="k64_bit63")
    # A warm oracle point makes bit 63 observable in the production transcript.
    config = CollapsedPowerPtConfig("HP16", 0.40, 1, 8)
    mass = build_classical_coset_mass(H, 0.40)
    reference = run_collapsed_power_pt_trajectory(
        model, frame, H, syndrome, config, seed, initial,
        engine="reference", mass=mass,
    )
    accelerated = run_collapsed_power_pt_trajectory(
        model, frame, H, syndrome, config, seed, initial,
        engine="numba", mass=mass,
    )
    assert int(reference["initial_label"]) == 1 << 63
    bit63 = np.uint64(1) << np.uint64(63)
    assert np.any((reference["burn_labels"] & bit63) != 0)
    assert np.any((reference["measurement_labels"] & bit63) != 0)
    for field in reference:
        if field != "engine":
            assert np.array_equal(
                np.asarray(reference[field]), np.asarray(accelerated[field]),
            ), field


def test_collapsed_runner_rejects_a_foreign_same_dimension_frame():
    H, model, _ = _model([[1, 1, 0], [0, 1, 1]])
    _, _, foreign = _model([[1, 1, 1], [1, 0, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    with pytest.raises(CollapsedConflictError, match="observable frame"):
        run_collapsed_power_pt_trajectory(
            model, foreign, H, syndrome,
            CollapsedPowerPtConfig("HP16", 0.10, 1, 8),
            _seed("HP16"), initial, engine="reference",
        )


def test_observable_frame_validation_rejects_matrix_and_section_tampering():
    from exp101_certified_src.observables import ObservableFrame

    _, model, frame = _model([[1, 1, 0], [0, 1, 1]])
    assert validate_observable_frame(model, frame)
    matrix = frame.W_basis.copy()
    matrix[0, 0] ^= 1
    bad_matrix = ObservableFrame(
        matrix, frame.k, frame.num_qubits, frame.section_fingerprint,
    )
    with pytest.raises(ValueError, match="observable frame"):
        validate_observable_frame(model, bad_matrix)
    bad_section = ObservableFrame(
        frame.W_basis.copy(), frame.k, frame.num_qubits, "0" * 64,
    )
    with pytest.raises(ValueError, match="observable frame"):
        validate_observable_frame(model, bad_section)


def test_strict_round_trip_counter_requires_cold_hot_cold():
    phase_reference = np.asarray([0, 0, 1], dtype=np.uint8)
    trips_reference = np.zeros(3, dtype=np.int64)
    phase_numba = phase_reference.copy()
    trips_numba = trips_reference.copy()
    observations = [
        (0, 2),  # Origin 0 starts hot; reaching cold later is not a round trip.
        (1, 0),
        (0, 2),
        (2, 0),  # Origin 0 has now completed cold -> hot -> cold.
        (2, 1),
        (0, 2),  # Origin 2 started cold and has completed its own cycle.
    ]
    for hot, cold in observations:
        _reference_advance_transport(
            phase_reference, trips_reference, hot, cold,
        )
        _hp_advance_transport(phase_numba, trips_numba, hot, cold)
    assert np.array_equal(phase_reference, phase_numba)
    assert np.array_equal(trips_reference, trips_numba)
    assert trips_reference.tolist() == [1, 0, 1]


@pytest.mark.parametrize("method", ["HC08", "HP16"])
@pytest.mark.parametrize("tamper", ["single_entry", "wrong_p"])
def test_supplied_mass_table_is_bound_to_h_and_p(method, tamper):
    H, model, frame = _model([[1, 1, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    p = 0.10
    mass = build_classical_coset_mass(H, p)
    if tamper == "single_entry":
        mass = mass.copy()
        mass[0] *= 1.0001
    else:
        mass = build_classical_coset_mass(H, 0.04)
    if method == "HP16":
        config = CollapsedPowerPtConfig(method, p, 1, 8)
        runner = run_collapsed_power_pt_trajectory
    else:
        config = CollapsedConfig(method, p, 1, 8)
        runner = run_collapsed_trajectory
    with pytest.raises(CollapsedConflictError, match="does not match H and p"):
        runner(
            model, frame, H, syndrome, config, _seed(method), initial,
            mass=mass,
        )


@pytest.mark.parametrize("constructor,args", [
    (CollapsedConfig, ("HC08", 0.1, 8, 15)),
    (CollapsedPowerPtConfig, ("HP16", 0.1, 8, 15)),
])
def test_collapsed_configs_reject_non_eight_block_measurements(constructor, args):
    with pytest.raises(ValueError, match="eight time blocks"):
        constructor(*args)
