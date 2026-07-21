import concurrent.futures
import itertools
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    BIAS_RAW_FIELDS,
    DEFECT_RAW_FIELDS,
    EASY_PANEL_SHA256,
    GAP_PANEL_SHA256,
    HARD_RAW_FIELDS,
    SMALL_PANEL_SHA256,
    _parallel_validate_measurement_entry,
    bias_binding_from_raw,
    bias_task_identity,
    character_seed,
    global_task_identity,
    load_global_discovery_config,
    run_bias_task,
    run_defect_task,
    run_hard_task,
    validate_bias_raw,
    validate_defect_raw,
    validate_hard_raw,
)
from data.expander_code.exp102.exp102_pipeline.io import atomic_npz, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    DEFECT_METHODS,
    HARD_METHODS,
    CharacterSet,
    DefectTraceConfig,
    GlobalSeedIdentity,
    HardCosetConfig,
    _reference_cluster_move,
    build_joint_blocks,
    build_logical_proposal_catalog,
    canonical_global_trajectory_digest,
    character_d2_estimate,
    character_means,
    character_qtop_estimate,
    character_values,
    frozen_character_set,
    label_collision_diagnostic,
    qubit_signatures,
    reduce_logical_basis,
    run_defect_trace_trajectory,
    run_hardcoset_trajectory,
    tune_defect_bias,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_global.discovery.v1.json"
SOURCE_COMMIT = "1" * 40


def _model(classical):
    return build_model(np.asarray(classical, dtype=np.uint8))


def _seed(method, family="P", trajectory=0, namespace="test"):
    return GlobalSeedIdentity(
        source_commit=SOURCE_COMMIT,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        method_id=method,
        resource_tier="T1",
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


def _state_key(state):
    return np.packbits(state, bitorder="little").tobytes()


def test_frozen_global_config_panels_methods_resources_and_versions():
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    assert [value["method_id"] for value in config["hard_methods"]] == list(HARD_METHODS)
    assert [value["method_id"] for value in config["defect_methods"]] == list(DEFECT_METHODS)
    assert config["resource_tiers"] == {
        "T1": {"burn_sweeps": 2048, "measurement_sweeps": 8192},
        "T2": {"burn_sweeps": 4096, "measurement_sweeps": 16384},
        "T3": {"burn_sweeps": 8192, "measurement_sweeps": 32768},
    }
    expected = {
        "EASY3": EASY_PANEL_SHA256,
        "CONF17": "8f2c1a6d60f346ecc5bf703f7e5d0d17d068462f978c78dd937ace0fb98b41be",
        "RES6": "03f9b16dbc0cc52ee18313cdf57fd25ea4db50f44687971bedac53662b275c22",
        "GAP8": GAP_PANEL_SHA256,
        "SMALL6": SMALL_PANEL_SHA256,
    }
    for name, digest in expected.items():
        assert config["panels"][name]["ordered_panel_sha256"] == digest
    assert len(config["panels"]["CONF17"]["cells"]) == 17
    assert len(config["panels"]["RES6"]["cells"]) == 6
    assert len(config["panels"]["GAP8"]["cells"]) == 8
    assert len(config["panels"]["SMALL6"]["cells"]) == 6


@pytest.mark.parametrize(
    "classical",
    [
        np.array([[1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
)
def test_catalog_is_deterministic_kernel_signature_complete_and_unique(classical):
    model, frame = _model(classical)
    first = build_logical_proposal_catalog(model, frame)
    second = build_logical_proposal_catalog(model, frame)
    assert first.catalog_sha256 == second.catalog_sha256
    assert np.array_equal(first.moves, second.moves)
    assert np.array_equal(first.signatures, second.signatures)
    assert not ((model.H_check.astype(np.int64) @ first.moves.T.astype(np.int64)) % 2).any()
    recomputed = []
    for move in first.moves:
        bits = (frame.W_basis.astype(np.int64) @ move.astype(np.int64) % 2).astype(np.uint8)
        value = sum(int(bit) << index for index, bit in enumerate(bits))
        recomputed.append(np.uint64(value))
    assert np.array_equal(first.signatures, recomputed)
    signature_matrix = np.asarray([
        [(int(value) >> bit) & 1 for bit in range(model.k)]
        for value in first.signatures
    ], dtype=np.uint8)
    from exp101_certified_src.gf2 import gf2_rank
    assert gf2_rank(signature_matrix) == model.k
    assert np.unique(np.packbits(first.moves, axis=1), axis=0).shape[0] == first.size
    assert first.size == min(8 * model.k, 512, sum(math.comb(model.k, j) for j in (1, 2, 3)))


def test_local_basis_reduction_uses_strict_deterministic_descent():
    model, _ = _model([[1, 1, 1]])
    reduced = reduce_logical_basis(model.logical_move_basis)
    assert int(reduced.sum()) <= int(model.logical_move_basis.sum())
    for i, j in itertools.permutations(range(model.k), 2):
        assert np.count_nonzero(reduced[i] ^ reduced[j]) >= reduced[i].sum()


def test_k0_affine_initializer_coordinate_map_is_bijective_and_seed_replayable():
    model, _ = _model([[1, 1, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    states = _coset_states(model, syndrome)
    assert states.shape == (128, 10)
    assert np.unique(np.packbits(states, axis=1), axis=0).shape[0] == 128
    first = uniform_hard_coset_state(model, syndrome, 12345)
    second = uniform_hard_coset_state(model, syndrome, 12345)
    assert np.array_equal(first, second)
    assert not ((model.H_check.astype(np.int64) @ first.astype(np.int64)) % 2).any()


def _exact_cluster_transition(model, syndrome, p):
    from exp101_certified_src.gf2 import gf2_nullspace

    states = _coset_states(model, syndrome)
    lookup = {_state_key(state): index for index, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    b = p / (1.0 - p)
    for row_index, state in enumerate(states):
        zeros = np.flatnonzero(state == 0)
        ones = np.flatnonzero(state == 1)
        for unpinned_mask in range(1 << zeros.size):
            unpinned = [zeros[index] for index in range(zeros.size) if (unpinned_mask >> index) & 1]
            free = np.asarray([*ones, *unpinned], dtype=np.int32)
            probability = b ** len(unpinned) * (1.0 - b) ** (zeros.size - len(unpinned))
            restricted = model.H_check[:, free]
            kernel = gf2_nullspace(restricted)
            for coefficient in range(1 << kernel.shape[0]):
                move = np.zeros(model.num_qubits, dtype=np.uint8)
                for bit, basis in enumerate(kernel):
                    if (coefficient >> bit) & 1:
                        move[free] ^= basis
                column = lookup[_state_key(state ^ move)]
                transition[row_index, column] += probability / (1 << kernel.shape[0])
    return states, transition


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
def test_q0_cluster_complete_transition_stationarity_and_detailed_balance(
        classical, p, nonzero):
    model, _ = _model(classical)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    states, transition = _exact_cluster_transition(model, syndrome, p)
    K = math.log((1.0 - p) / p)
    posterior = np.exp(-K * states.sum(axis=1))
    posterior /= posterior.sum()
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 1e-13
    assert np.max(np.abs(posterior @ transition - posterior)) <= 1e-13
    flow = posterior[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 1e-13


def _joint_transition(states, generators, K):
    lookup = {_state_key(state): index for index, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)))
    moves = []
    for mask in range(1 << len(generators)):
        move = np.zeros(states.shape[1], dtype=np.uint8)
        for bit, row in enumerate(generators):
            if (mask >> bit) & 1:
                move ^= row
        moves.append(move)
    for index, state in enumerate(states):
        targets = np.asarray([lookup[_state_key(state ^ move)] for move in moves])
        weights = np.exp(-K * states[targets].sum(axis=1))
        weights /= weights.sum()
        for target, probability in zip(targets, weights):
            transition[index, target] += probability
    return transition


@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
def test_joint_block_single_block_detailed_balance_and_complete_mixture_stationarity(
        classical):
    model, _ = _model(classical)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    states = _coset_states(model, syndrome)
    K = math.log(9.0)
    posterior = np.exp(-K * states.sum(axis=1)); posterior /= posterior.sum()
    transitions = []
    for logical in model.logical_move_basis:
        generators = np.vstack((logical, model.stabilizer_rows[:2]))
        transition = _joint_transition(states, generators, K)
        flow = posterior[:, None] * transition
        assert np.max(np.abs(flow - flow.T)) <= 1e-13
        assert np.max(np.abs(posterior @ transition - posterior)) <= 1e-13
        transitions.append(transition)
    sweep = sum(transitions) / len(transitions)
    assert np.max(np.abs(posterior @ sweep - posterior)) <= 1e-13


def _worm_coordinate_balance(model, syndrome, p, dmax, bias):
    states = np.asarray([
        [(mask >> bit) & 1 for bit in range(model.num_qubits)]
        for mask in range(1 << model.num_qubits)
    ], dtype=np.uint8)
    residuals = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    defects = residuals.sum(axis=1)
    allowed = np.flatnonzero(defects <= dmax)
    lookup = {_state_key(states[index]): position for position, index in enumerate(allowed)}
    K = math.log((1.0 - p) / p)
    log_weight = -K * states[allowed].sum(axis=1) + bias[defects[allowed]]
    weights = np.exp(log_weight - log_weight.max()); weights /= weights.sum()
    for qubit in range(model.num_qubits):
        incoming = np.zeros(len(allowed))
        for position, state_index in enumerate(allowed):
            state = states[state_index]
            flipped = state.copy(); flipped[qubit] ^= 1
            other = lookup.get(_state_key(flipped))
            if other is None:
                incoming[position] += weights[position]
                continue
            probability = weights[other] / (weights[position] + weights[other])
            incoming[other] += weights[position] * probability
            incoming[position] += weights[position] * (1.0 - probability)
            reverse = weights[position] / (weights[position] + weights[other])
            assert abs(weights[position] * probability - weights[other] * reverse) <= 1e-15
        assert np.max(np.abs(incoming - weights)) <= 1e-13
    d0 = defects[allowed] == 0
    conditional = weights[d0] / weights[d0].sum()
    hard = np.exp(-K * states[allowed[d0]].sum(axis=1)); hard /= hard.sum()
    assert np.max(np.abs(conditional - hard)) <= 1e-13


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
def test_fixed_bias_worm_extended_stationarity_and_d0_conditional_posterior(
        classical, p, nonzero):
    model, _ = _model(classical)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    bias = np.linspace(-0.3, 0.4, 5)
    _worm_coordinate_balance(model, syndrome, p, dmax=4, bias=bias)


@pytest.mark.parametrize("method", ["RC8-QC1", "RC8-QC4"])
def test_hardcoset_reference_numba_states_weights_labels_and_counters_match(method):
    model, frame = _model([[1, 1, 1]])
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8); epsilon[[0, 2]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    config = HardCosetConfig(method, 0.10, 4, 8)
    catalog = build_logical_proposal_catalog(model, frame)
    results = [
        run_hardcoset_trajectory(
            model, frame, syndrome, config, _seed(method), epsilon,
            engine=engine, catalog=catalog,
        )
        for engine in ("reference", "numba")
    ]
    assert canonical_global_trajectory_digest(results[0]) == canonical_global_trajectory_digest(results[1])
    assert not results[0]["measurement_residual_weights"].any()


@pytest.mark.parametrize("method", ["RC8-J08", "RC8-J12", "RC8-J16"])
def test_real_m3_joint_reference_numba_transcripts_match(method):
    _, _, H = load_frozen_code(REGISTRY_PATH, "m03_c00")
    model, frame = build_model(H)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    config = HardCosetConfig(method, 0.10, 1, 8)
    catalog = build_logical_proposal_catalog(model, frame)
    joint = build_joint_blocks(model, frame, catalog, config.joint_block_size)
    results = [
        run_hardcoset_trajectory(
            model, frame, syndrome, config, _seed(method), initial,
            engine=engine, catalog=catalog, joint=joint,
        )
        for engine in ("reference", "numba")
    ]
    assert (
        canonical_global_trajectory_digest(results[0])
        == canonical_global_trajectory_digest(results[1])
    )


def test_fixed_bias_worm_reference_numba_fixed_clock_and_excursions_match():
    model, frame = _model([[1, 1, 1]])
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8); epsilon[[0, 2]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    config = DefectTraceConfig("DT16", 0.10, 4, 8)
    bias = np.linspace(-0.2, 0.3, 17)
    results = [
        run_defect_trace_trajectory(
            model, frame, syndrome, config, _seed("DT16"), epsilon,
            bias, "5" * 64, engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    assert canonical_global_trajectory_digest(results[0]) == canonical_global_trajectory_digest(results[1])
    assert np.array_equal(results[0]["fixed_clock_d0_mask"], results[0]["measurement_defect_counts"] == 0)


def test_bias_tuning_reference_numba_is_bit_identical_and_seed_isolated():
    model, _ = _model([[1, 1, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    config = DefectTraceConfig("DT16", 0.10, 4, 8)
    identities = [_seed("DT16", "TUNE", index, "bias") for index in range(8)]
    reference = tune_defect_bias(model, syndrome, config, identities, engine="reference")
    accelerated = tune_defect_bias(model, syndrome, config, identities, engine="numba")
    for field in (
            "bias", "bias_trace", "tuning_histogram", "tuning_final_states_packed",
            "tuning_final_residuals", "tuning_final_defects", "gammas"):
        assert np.array_equal(reference[field], accelerated[field]), field
    assert reference["bias_sha256"] == accelerated["bias_sha256"]
    assert identities[0].seed("tuning") != identities[0].seed("burn")
    assert identities[0].seed("burn") != identities[0].seed("measurement")


@pytest.mark.parametrize(
    "classical",
    [
        np.array([[1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero", [False, True])
def test_character_u_statistic_collision_and_d2_match_exact_posterior(classical, p, nonzero):
    model, frame = _model(classical)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    states = _coset_states(model, syndrome)
    K = math.log((1.0 - p) / p)
    weights = np.exp(-K * states.sum(axis=1)); weights /= weights.sum()
    labels = np.asarray([
        sum(int(bit) << index for index, bit in enumerate(frame.label_of(state)))
        for state in states
    ], dtype=np.uint64)
    masses = np.zeros(1 << model.k)
    for label, weight in zip(labels, weights):
        masses[int(label)] += weight
    exact_qtop = (float(np.dot(masses, masses)) - 2.0 ** (-model.k)) / (1.0 - 2.0 ** (-model.k))
    chars = frozen_character_set(model.k, 123)
    exact_means = (weights[:, None] * character_values(labels, chars.masks)).sum(axis=0)
    chain_means = np.repeat(exact_means[None, :], 16, axis=0)
    estimate = character_qtop_estimate(chars, chain_means)
    assert estimate["q_top"] == pytest.approx(exact_qtop, abs=1e-13)
    assert estimate["q_top_total_se"] == pytest.approx(0.0, abs=1e-15)
    d2 = character_d2_estimate(chars, chain_means, chain_means)
    assert d2["d2_norm"] == pytest.approx(0.0, abs=1e-15)


def test_character_trajectory_jackknife_and_finite_population_se_are_conservative_components():
    masks = np.asarray([1, 2, 4, 3, 5], dtype=np.uint64)
    chars = CharacterSet(masks, np.asarray([0, 1, 2], dtype=np.int32), "sampled", 3, 9, "x")
    means = np.asarray([
        [0.8, 0.5, 0.2, 0.1, -0.5],
        [0.7, 0.4, 0.1, 0.2, -0.4],
        [0.9, 0.6, 0.3, 0.0, -0.6],
        [0.6, 0.3, 0.0, 0.3, -0.3],
    ])
    result = character_qtop_estimate(chars, means)
    assert result["q_top_trajectory_se"] > 0
    assert result["q_top_character_se"] > 0
    assert result["q_top_total_se"] == pytest.approx(math.hypot(
        result["q_top_trajectory_se"], result["q_top_character_se"],
    ))
    d2 = character_d2_estimate(chars, means, means[::-1])
    assert d2["per_character_d2"].shape == masks.shape
    assert d2["delete_one_d2"].shape == (2 * means.shape[0],)


def test_raw_label_collision_is_diagnostic_and_unsafe_float_bits_fail_closed():
    traces = [np.asarray([0, 0, 1, 1], dtype=np.uint64) for _ in range(4)]
    diagnostic = label_collision_diagnostic(traces, 2)
    assert diagnostic["collision_mass"] == pytest.approx(0.5)
    assert diagnostic["q_top"] == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="exact uint64 bits"):
        character_values([float(2**63)], [1])
    with pytest.raises(ValueError, match="exact uint64 bits"):
        character_values([0], [1.5])


@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("nonzero", [False, True])
def test_tiny_weighted_model_count_matches_exact_coset_posterior(
        classical, p, nonzero):
    wmc = importlib.import_module(
        "data.expander_code.exp102.validation."
        "007_q0_global_discovery_20260721.wmc_feasibility"
    )
    model, frame = _model(classical)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    result = wmc.solve_cell(model, frame, syndrome, p, 20, 30.0)
    states = _coset_states(model, syndrome)
    K = math.log((1.0 - p) / p)
    probabilities = np.exp(-K * states.sum(axis=1))
    probabilities /= probabilities.sum()
    masses = np.zeros(1 << model.k)
    for state, probability in zip(states, probabilities):
        label = sum(
            int(bit) << index for index, bit in enumerate(frame.label_of(state))
        )
        masses[label] += probability
    collision = float(np.dot(masses, masses))
    q_top = (collision - 2.0 ** (-model.k)) / (1.0 - 2.0 ** (-model.k))
    assert result["status"] == "EXACT"
    assert result["collision_mass"] == pytest.approx(collision, abs=1e-13)
    assert result["q_top"] == pytest.approx(q_top, abs=1e-13)


def test_k64_masks_bit63_signatures_and_random_characters_never_use_int64_boundary():
    _, _, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    assert model.k == 64
    catalog = build_logical_proposal_catalog(model, frame)
    assert catalog.signatures.dtype == np.uint64
    assert any(int(value) & (1 << 63) for value in catalog.signatures)
    chars = frozen_character_set(64, character_seed("a" * 64, "m08_c06"))
    assert chars.masks.dtype == np.uint64
    assert np.uint64(1) << np.uint64(63) in chars.masks
    assert len(np.unique(chars.masks)) == 64 + 4096
    values = character_values(
        [0, np.uint64(1) << np.uint64(63), np.uint64(0xFFFFFFFFFFFFFFFF)],
        [np.uint64(1) << np.uint64(63), np.uint64(0xFFFFFFFFFFFFFFFF)],
    )
    assert np.array_equal(values, [[1, 1], [-1, -1], [-1, 1]])


def test_task_seed_identity_binds_every_required_axis_and_bias_is_mandatory():
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    cell = config["panels"]["HARD2"]["cells"][0]
    hard = global_task_identity(
        registry, config, SOURCE_COMMIT, "screen", "RC8-QC1", "T1",
        cell, "P", 0,
    )
    changed = global_task_identity(
        registry, config, SOURCE_COMMIT, "screen", "RC8-QC1", "T1",
        cell, "U", 0,
    )
    assert hard["seed_identity"] != changed["seed_identity"]
    bias = bias_task_identity(
        registry, config, SOURCE_COMMIT, "screen", "DT16", "T1", cell,
    )
    assert len(bias["tuning_seed_identities"]) == 8
    with pytest.raises(ValueError, match="bias binding"):
        global_task_identity(
            registry, config, SOURCE_COMMIT, "screen", "DT16", "T1",
            cell, "P", 0,
        )


@pytest.fixture(scope="module")
def global_raw_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("q0_global_raw")
    registry = load_registry(REGISTRY_PATH)
    config = load_global_discovery_config(CONFIG_PATH, registry)
    cell = config["panels"]["EASY3"]["cells"][0]
    hard_task = global_task_identity(
        registry, config, SOURCE_COMMIT, "screen", "RC8-QC1", "T1",
        cell, "P", 0,
    )
    hard_path = root / "hard.npz"
    assert run_hard_task(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, hard_task, hard_path,
    ) == "computed"
    bias_task = bias_task_identity(
        registry, config, SOURCE_COMMIT, "screen", "DT16", "T1", cell,
    )
    bias_path = root / "bias.npz"
    assert run_bias_task(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, bias_task, bias_path,
    ) == "computed"
    binding = bias_binding_from_raw(
        bias_path, registry, config, SOURCE_COMMIT,
    )
    defect_task = global_task_identity(
        registry, config, SOURCE_COMMIT, "screen", "DT16", "T1",
        cell, "P", 0, bias_binding=binding,
    )
    defect_path = root / "defect.npz"
    assert run_defect_task(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, defect_task, bias_path,
        defect_path,
    ) == "computed"
    return root, registry, config, hard_path, bias_path, defect_path


def test_global_raw_schemas_are_no_pickle_replayable_and_have_no_population_parents(global_raw_fixture):
    _, registry, config, hard_path, bias_path, defect_path = global_raw_fixture
    with np.load(hard_path, allow_pickle=False) as data:
        assert set(data.files) == HARD_RAW_FIELDS
        assert "parents" not in data.files
        assert not data["measurement_residual_weights"].any()
    with np.load(bias_path, allow_pickle=False) as data:
        assert set(data.files) == BIAS_RAW_FIELDS
        assert "parents" not in data.files
    with np.load(defect_path, allow_pickle=False) as data:
        assert set(data.files) == DEFECT_RAW_FIELDS
        assert "parents" not in data.files
        assert np.array_equal(data["fixed_clock_d0_mask"], data["measurement_defect_counts"] == 0)
    validate_hard_raw(hard_path, registry, config, SOURCE_COMMIT)
    validate_bias_raw(bias_path, registry, config, SOURCE_COMMIT)
    validate_defect_raw(defect_path, registry, config, SOURCE_COMMIT, bias_path)


def test_parallel_replay_worker_validates_hard_and_defect_raw(global_raw_fixture):
    root, _, _, hard_path, _, defect_path = global_raw_fixture
    entries = []
    for path, bias_relpath in (
            (hard_path, None), (defect_path, "bias.npz")):
        with np.load(path, allow_pickle=False) as data:
            task = json.loads(str(data["task_json"].item()))
            fingerprint = str(data["task_fingerprint"].item())
        entries.append({
            "task": task,
            "task_fingerprint": fingerprint,
            "output_relpath": path.name,
            "bias_relpath": bias_relpath,
        })
    payloads = [
        (str(root), entry, str(REGISTRY_PATH), str(CONFIG_PATH), SOURCE_COMMIT)
        for entry in entries
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(_parallel_validate_measurement_entry, payloads))
    assert [record["task_fingerprint"] for record in records] == [
        entry["task_fingerprint"] for entry in entries
    ]


@pytest.mark.parametrize(
    "kind,field,mutator,match",
    [
        ("hard", "catalog_sha256", lambda value: np.array("0" * 64), "catalog"),
        ("hard", "measurement_labels", lambda value: value ^ np.uint64(1), "measurement_labels"),
        ("bias", "bias", lambda value: value + 1e-6, "bias"),
        ("defect", "fixed_clock_d0_mask", lambda value: ~value, "fixed_clock_d0_mask"),
    ],
)
def test_global_raw_tampering_is_conflict(global_raw_fixture, tmp_path, kind,
                                          field, mutator, match):
    _, registry, config, hard_path, bias_path, defect_path = global_raw_fixture
    source = {"hard": hard_path, "bias": bias_path, "defect": defect_path}[kind]
    with np.load(source, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays[field] = mutator(arrays[field])
    tampered = tmp_path / f"tampered_{kind}.npz"
    atomic_npz(tampered, **arrays)
    with pytest.raises(ValueError, match=match):
        if kind == "hard":
            validate_hard_raw(tampered, registry, config, SOURCE_COMMIT)
        elif kind == "bias":
            validate_bias_raw(tampered, registry, config, SOURCE_COMMIT)
        else:
            validate_defect_raw(tampered, registry, config, SOURCE_COMMIT, bias_path)
