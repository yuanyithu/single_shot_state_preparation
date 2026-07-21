import json
import math
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.io import atomic_npz, sha256_json
from data.expander_code.exp102.exp102_pipeline.pa_discovery import (
    BASE_METHODS,
    CONFIRMATION_PANEL_SHA256,
    PA_RAW_FIELDS,
    RESOLUTION_PANEL_SHA256,
    _portable_raw_evidence,
    _require_float_replay,
    load_pa_discovery_config,
    pa_task_identity,
    run_pa_task,
    validate_pa_raw,
)
from data.expander_code.exp102.exp102_pipeline.q0_pa import (
    PA_SCHEDULE_VERSION,
    PaSeedIdentity,
    Q0PaConfig,
    _mutation_rng_states,
    label_distribution_collision,
    pa_population_gate,
    population_qtop_jackknife,
    run_q0_pa_population,
    systematic_resampling,
    theta_schedule_q32,
    validate_hard_coset_basis,
    weighted_label_distribution,
)
from data.expander_code.exp102.exp102_pipeline.q0_pt import (
    Q32_ONE,
    ladder_x_q32_sha256,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_pa.discovery.v1.json"
SOURCE_COMMIT = "1" * 40


def _model(classical):
    return build_model(np.asarray(classical, dtype=np.uint8))


def _pa_config(p, particles, steps=6, sweeps=0, kernel="coordinate"):
    schedule = theta_schedule_q32(p, steps)
    return Q0PaConfig(
        p_target=p,
        num_particles=particles,
        num_anneal_steps=steps,
        rejuvenation_sweeps=sweeps,
        logical_kernel=kernel,
        schedule_q32=schedule,
        schedule_sha256=ladder_x_q32_sha256(schedule),
    )


def _seed_identity(namespace="test", population=0):
    return PaSeedIdentity(
        source_commit=SOURCE_COMMIT,
        config_sha256="2" * 64,
        cell_fingerprint="3" * 64,
        population_index=population,
        trajectory_namespace=namespace,
    )


def _coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    dimension = generators.shape[0]
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << dimension, axis=0)
    for coefficient in range(1 << dimension):
        for bit in range(dimension):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= generators[bit]
    return states


def test_frozen_config_contains_all_schedules_panels_and_methods():
    registry = load_registry(REGISTRY_PATH)
    config = load_pa_discovery_config(CONFIG_PATH, registry)
    assert config["schedule_version"] == PA_SCHEDULE_VERSION
    assert len(config["schedules"]) == 21
    assert [value["method_id"] for value in config["base_methods"]] == [
        value["method_id"] for value in BASE_METHODS
    ]
    assert sha256_json(config["confirmation"]["cells"]) == CONFIRMATION_PANEL_SHA256
    assert sha256_json(config["resolution"]["cells"]) == RESOLUTION_PANEL_SHA256
    assert len(config["confirmation"]["cells"]) == 17
    assert len(config["resolution"]["cells"]) == 6


@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
@pytest.mark.parametrize("steps", [4, 96, 192, 384])
def test_theta_schedule_is_strict_q32_and_formula_locked(p, steps):
    values = theta_schedule_q32(p, steps)
    assert len(values) == steps + 1
    assert values[0] == 0 and values[-1] == Q32_ONE
    assert all(left < right for left, right in zip(values, values[1:]))
    config = Q0PaConfig(
        p, 8, steps, 0, "coordinate", values, ladder_x_q32_sha256(values),
    )
    assert config.schedule_q32 == values
    tampered = list(values)
    tampered[1] += 1
    with pytest.raises(ValueError, match="theta formula"):
        Q0PaConfig(
            p, 8, steps, 0, "coordinate", tuple(tampered),
            ladder_x_q32_sha256(tuple(tampered)),
        )


@pytest.mark.parametrize(
    "classical",
    [
        np.array([[1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_all_coset_particles_match_exact_weights_sector_posterior_qtop_and_logz(
    classical, nonzero, p,
):
    model, frame = _model(classical)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        epsilon[[0, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    states = _coset_states(model, syndrome)
    assert states.shape[0] == 128
    config = _pa_config(p, states.shape[0], steps=7, sweeps=0)
    result = run_q0_pa_population(
        model, frame, syndrome, config, _seed_identity("exact"),
        engine="reference", initial_states=states, resampling_enabled=False,
    )
    labels = np.asarray([int(sum(int(bit) << i for i, bit in enumerate(frame.label_of(state))))
                         for state in states], dtype=np.uint64)
    for stage, K in enumerate(result["ladder_K"]):
        exact = np.exp(-float(K) * states.sum(axis=1))
        exact /= exact.sum()
        if stage == 0:
            stored = np.full(states.shape[0], 1.0 / states.shape[0])
        else:
            stored = result["stage_post_weights"][stage - 1]
        assert np.allclose(stored, exact, rtol=2e-15, atol=1e-16)
    final = result["final_weights"]
    distribution = weighted_label_distribution(labels, final)
    collision = label_distribution_collision(distribution, distribution)
    uniform = 2.0 ** (-model.k)
    qtop = (collision - uniform) / (1.0 - uniform)
    masses = np.zeros(1 << model.k)
    for label, weight in zip(labels, final):
        masses[int(label)] += weight
    expected_qtop = (float(np.dot(masses, masses)) - uniform) / (1.0 - uniform)
    assert qtop == pytest.approx(expected_qtop, abs=1e-14)
    exact_log_z = math.log(float(np.exp(
        -result["ladder_K"][-1] * states.sum(axis=1)
    ).sum()))
    assert result["log_z"][-1] == pytest.approx(exact_log_z, abs=3e-14)


def test_k0_affine_coordinate_map_is_bijective_and_uniform():
    model, _ = _model([[1, 1, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    dimension = validate_hard_coset_basis(model)
    states = _coset_states(model, syndrome)
    assert dimension == 7
    assert states.shape == (1 << dimension, model.num_qubits)
    assert np.unique(np.packbits(states, axis=1), axis=0).shape[0] == 1 << dimension
    residual = (model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2).T
    assert not residual.any()


def _heatbath_transition(energies, K):
    weights = np.exp(-K * np.asarray(energies, dtype=np.float64))
    return weights / weights.sum()


def test_coordinate_and_overlapping_block_heatbath_obey_detailed_balance():
    model, _ = _model([[1, 1, 1]])
    assert any(
        np.any(model.logical_move_basis[a] & model.logical_move_basis[b])
        for a in range(model.k) for b in range(a)
    )
    base = np.zeros(model.num_qubits, dtype=np.uint8)
    K = math.log(9.0)

    move = model.logical_move_basis[0]
    pair = np.vstack((base, base ^ move))
    pi = _heatbath_transition(pair.sum(axis=1), K)
    transition = np.tile(pi, (2, 1))
    assert np.max(np.abs(pi[:, None] * transition - pi[None, :] * transition.T)) <= 1e-15
    assert np.max(np.abs(pi @ transition - pi)) <= 1e-15

    block_states = []
    for mask in range(1 << model.k):
        state = base.copy()
        for bit in range(model.k):
            if (mask >> bit) & 1:
                state ^= model.logical_move_basis[bit]
        block_states.append(state)
    block_states = np.asarray(block_states)
    pi = _heatbath_transition(block_states.sum(axis=1), K)
    transition = np.tile(pi, (pi.size, 1))
    balance_error = np.max(
        np.abs(pi[:, None] * transition - pi[None, :] * transition.T)
    )
    stationarity_error = np.max(np.abs(pi @ transition - pi))
    assert balance_error <= 1e-13
    assert stationarity_error <= 1e-13


def test_systematic_resampling_boundaries_are_canonical():
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.array_equal(systematic_resampling(weights, 0.0), [0, 1, 2, 3])
    offset = np.nextafter(0.25, 0.0)
    # IEEE addition rounds two positions onto exact CDF boundaries; side="right"
    # fixes their parent assignment deterministically.
    assert np.array_equal(systematic_resampling(weights, offset), [0, 2, 3, 3])
    uneven = np.array([0.5, 0.25, 0.125, 0.125])
    assert np.array_equal(systematic_resampling(uneven, 0.0), [0, 0, 1, 2])
    with pytest.raises(ValueError, match="outside"):
        systematic_resampling(weights, 0.25)


def test_clone_substreams_fork_by_output_slot_instead_of_copying_parent_rng():
    identity = _seed_identity("clone-fork")
    states = _mutation_rng_states(identity, stage=3, sweeps=2, particles=16)
    flattened = {tuple(value.tolist()) for value in states.reshape(-1, 2)}
    assert len(flattened) == 32
    assert np.array_equal(states, _mutation_rng_states(
        identity, stage=3, sweeps=2, particles=16,
    ))
    other = _mutation_rng_states(_seed_identity("clone-fork", 1), 3, 2, 16)
    assert not np.array_equal(states, other)


@pytest.mark.parametrize("kernel", ["coordinate", "block4"])
def test_reference_and_numba_population_transcripts_are_bit_identical(kernel):
    model, frame = _model([[1, 1, 1]])
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, 2]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    config = _pa_config(0.10, 32, steps=8, sweeps=2, kernel=kernel)
    results = [
        run_q0_pa_population(
            model, frame, syndrome, config, _seed_identity(f"identity-{kernel}"),
            engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    arrays = (
        "final_states", "final_weights", "final_labels", "final_energies",
        "stage_energies", "stage_pre_weights", "stage_post_weights",
        "conditional_ess", "ess_before_decision", "ess_after_decision",
        "max_pre_weight", "resampled", "resampling_offsets", "parents",
        "offspring_counts", "root_ancestry", "mutation_counters",
        "logical_bit_flips", "log_normalizer_increments", "log_z",
        "family_masses",
    )
    for field in arrays:
        assert np.array_equal(results[0][field], results[1][field]), field
    for field in (
            "family_ess", "distinct_initial_families", "max_family_mass",
            "max_hard_coset_residual", "affine_dimension", "population_seed"):
        assert results[0][field] == results[1][field], field


def test_ancestry_genealogy_and_population_gate_recompute():
    model, frame = _model([[1, 1, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    config = _pa_config(0.04, 64, steps=12, sweeps=1, kernel="block4")
    result = run_q0_pa_population(
        model, frame, syndrome, config, _seed_identity("genealogy"), engine="numba",
    )
    roots = np.arange(config.num_particles)
    for stage in range(config.num_anneal_steps):
        roots = roots[result["parents"][stage]]
        assert np.array_equal(result["root_ancestry"][stage + 1], roots)
        assert np.array_equal(
            result["offspring_counts"][stage],
            np.bincount(result["parents"][stage], minlength=config.num_particles),
        )
    masses = np.bincount(
        roots, weights=result["final_weights"], minlength=config.num_particles,
    )
    assert np.array_equal(result["family_masses"], masses)
    assert result["family_ess"] == pytest.approx(
        1.0 / np.dot(masses, masses), abs=1e-14,
    )
    valid, failures = pa_population_gate(result)
    assert valid == (not failures)


def test_population_collision_u_statistic_and_delete_one_jackknife():
    distributions = [
        (np.array([0, 1], dtype=np.uint64), np.array([0.8 - 0.02 * i, 0.2 + 0.02 * i]))
        for i in range(8)
    ]
    result = population_qtop_jackknife(distributions, k=1)
    pair_values = [
        label_distribution_collision(distributions[a], distributions[b])
        for a in range(8) for b in range(a + 1, 8)
    ]
    collision = float(np.mean(pair_values))
    assert result["pair_count"] == 28
    assert result["collision_mass"] == pytest.approx(collision)
    assert result["q_top"] == pytest.approx((collision - 0.5) / 0.5)
    delete = []
    for omitted in range(8):
        kept = [i for i in range(8) if i != omitted]
        values = [
            label_distribution_collision(distributions[a], distributions[b])
            for position, a in enumerate(kept) for b in kept[position + 1:]
        ]
        delete.append((float(np.mean(values)) - 0.5) / 0.5)
    delete = np.asarray(delete)
    expected_se = math.sqrt(7.0 / 8.0 * np.sum((delete - delete.mean()) ** 2))
    assert np.array_equal(result["delete_one_q_top"], delete)
    assert result["q_top_mcse"] == pytest.approx(expected_se)


@pytest.fixture(scope="module")
def valid_pa_raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("pa_raw")
    registry = load_registry(REGISTRY_PATH)
    config = load_pa_discovery_config(CONFIG_PATH, registry)
    task = pa_task_identity(
        registry, config, SOURCE_COMMIT, "hard_screen", "B96-1",
        config["hard_screen"]["cells"][0], 0,
    )
    path = root / "population.npz"
    assert run_pa_task(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, task, path,
    ) == "computed"
    record = validate_pa_raw(path, registry, config, SOURCE_COMMIT)
    return path, registry, config, record


def test_pa_raw_is_no_pickle_self_validating_and_discovery_only(valid_pa_raw):
    path, _, _, record = valid_pa_raw
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == PA_RAW_FIELDS
        assert data["raw_version"].item() == "exp102.q0_pa.raw.v1"
        assert data["stage"].item() == "hard_screen"
    assert len(record["population_digest"]) == 64


def test_pa_analyzer_accepts_bounded_cross_libm_rounding(valid_pa_raw, tmp_path):
    path, registry, config, _ = valid_pa_raw
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    ladder_p = arrays["ladder_p"].copy()
    ladder_p[1:-1] = np.nextafter(ladder_p[1:-1], np.inf)
    arrays["ladder_p"] = ladder_p
    rounded = tmp_path / "rounded_ladder_p.npz"
    atomic_npz(rounded, **arrays)
    validate_pa_raw(rounded, registry, config, SOURCE_COMMIT)

    expected = np.array([0.4, 0.08, 100.0], dtype=np.float64)
    replayed = expected.copy()
    for _ in range(8):
        replayed = np.nextafter(replayed, np.inf)
    _require_float_replay("cross-libm", replayed, expected, max_ulps=8)
    with pytest.raises(ValueError, match="cross-libm"):
        _require_float_replay(
            "cross-libm", expected + np.array([1e-10, 0.0, 0.0]), expected,
            max_ulps=8,
        )


def test_pa_report_raw_evidence_uses_portable_paths(tmp_path):
    raw = tmp_path / "stage/node/population.npz"
    raw.parent.mkdir(parents=True)
    raw.touch()
    evidence = _portable_raw_evidence([
        {"path": str(raw.resolve()), "sha256": "a" * 64},
    ], tmp_path)
    assert evidence == [{
        "path": "stage/node/population.npz", "sha256": "a" * 64,
    }]
    with pytest.raises(ValueError, match="outside"):
        _portable_raw_evidence([
            {"path": str(tmp_path.parent / "outside.npz"), "sha256": "b" * 64},
        ], tmp_path)


@pytest.mark.parametrize(
    "field,mutator,match",
    [
        ("schedule_q32", lambda value: np.where(
            np.arange(value.size) == 1, value + np.uint64(1), value
        ), "schedule"),
        ("ladder_p", lambda value: value * (1.0 + 1e-10), "ladder_p"),
        ("final_weights", lambda value: value * np.linspace(0.9, 1.1, value.size), "weights"),
        ("root_ancestry", lambda value: np.zeros_like(value), "root_ancestry|family"),
        ("root_ancestry", lambda value: np.vstack((
            np.zeros_like(value[:1]), value[1:]
        )), "initial root_ancestry"),
        ("uniform_seed", lambda value: np.array(value.item() + 1, dtype=value.dtype), "seed"),
        ("source_commit", lambda value: np.array("4" * 40), "source"),
        ("discovery_config_sha256", lambda value: np.array("5" * 64), "identity"),
    ],
)
def test_pa_analyzer_rejects_tampered_schedule_weights_ancestry_seed_and_identity(
    valid_pa_raw, tmp_path, field, mutator, match,
):
    path, registry, config, _ = valid_pa_raw
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays[field] = mutator(arrays[field])
    tampered = tmp_path / f"tampered_{field}.npz"
    atomic_npz(tampered, **arrays)
    with pytest.raises((ValueError, OverflowError), match=match):
        validate_pa_raw(tampered, registry, config, SOURCE_COMMIT)
