import inspect
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import data.expander_code.exp102.exp102_pipeline.q0_map_mixture as map_mixture
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    MAP_METHOD_ID,
    MAP_RAW_VERSION,
    MapMixtureConflictError,
    MapMixtureConfig,
    MapMixtureSeedIdentity,
    _label_uint64,
    build_affine_coordinate_system,
    build_map_mixture_proposal,
    build_milp_map_anchors,
    enumerate_affine_states,
    estimate_proposal_overlap,
    independence_log_acceptance,
    independence_transition_matrix,
    run_map_mixture_trajectory,
    validate_map_anchor_catalog,
    validate_map_mixture_proposal,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model


SOURCE_COMMIT = "1" * 40


def _model(classical):
    return build_model(np.asarray(classical, dtype=np.uint8))


def _syndrome(model, nonzero):
    if not nonzero:
        return np.zeros(model.num_checks, dtype=np.uint8)
    witness = np.zeros(model.num_qubits, dtype=np.uint8)
    witness[[0, model.num_qubits - 1]] = 1
    return (
        model.H_check.astype(np.int64) @ witness.astype(np.int64) % 2
    ).astype(np.uint8)


def _seed(family="P"):
    return MapMixtureSeedIdentity(
        source_commit=SOURCE_COMMIT,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        init_family=family,
        trajectory_index=0,
        trajectory_namespace="unit_test",
    )


def _catalog_and_proposal(model, syndrome, p, max_anchors=4):
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, p, max_anchors=max_anchors,
    )
    proposal = build_map_mixture_proposal(model, catalog)
    return catalog, proposal


def test_map_runner_rejects_a_foreign_same_dimension_frame_before_artifacts():
    model, _ = _model([[1, 1, 0], [0, 1, 1]])
    _, foreign = _model([[1, 1, 1], [1, 0, 1]])
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    with pytest.raises(MapMixtureConflictError, match="observable frame"):
        run_map_mixture_trajectory(
            model, foreign, syndrome, MapMixtureConfig(0.10, 8, 8),
            _seed(), initial,
        )


def test_map_label_uint64_preserves_k64_bit63():
    registry = "data/expander_code/exp102/registry/registry.json"
    _, _, H = load_frozen_code(registry, "m08_c06")
    model, frame = build_model(H)
    assert model.k == 64
    assert _label_uint64(frame, model.logical_move_basis[63]) == (
        np.uint64(1) << np.uint64(63)
    )


def test_only_verified_artifact_entry_can_replay_a_foreign_solver(monkeypatch):
    model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    catalog, proposal = _catalog_and_proposal(
        model, syndrome, 0.10, max_anchors=8,
    )
    config = MapMixtureConfig(0.10, 8, 8, max_anchors=8)
    initial = catalog.anchors[0]
    monkeypatch.setattr(
        map_mixture, "_solver_identity",
        lambda: "numpy=9.9;scipy=9.9;highs=9.9.9",
    )
    with pytest.raises(MapMixtureConflictError, match="solver identity"):
        run_map_mixture_trajectory(
            model, frame, syndrome, config, _seed(), initial,
            anchor_catalog=catalog, proposal=proposal,
        )
    assert "frozen_artifact_replay" not in inspect.signature(
        run_map_mixture_trajectory,
    ).parameters
    artifact = SimpleNamespace(
        descriptor={
            "anchor_sha256": catalog.anchor_sha256,
            "anchor_solver_identity": catalog.solver_identity,
            "requested_max_anchors": catalog.requested_max_anchors,
            "anchor_count": catalog.size,
            "coordinate_sha256": proposal.coordinates.coordinate_sha256,
            "proposal_sha256": proposal.proposal_sha256,
        },
        catalog=catalog,
        proposal=proposal,
    )
    result = map_mixture._run_map_mixture_trajectory_from_verified_artifact(
        model, frame, syndrome, config, _seed(), initial,
        artifact=artifact,
    )
    assert not result["measurement_residual_weights"].any()
    with pytest.raises(MapMixtureConflictError, match="verified artifact"):
        map_mixture._run_map_mixture_trajectory_from_verified_artifact(
            model, frame, syndrome, config, _seed(), initial,
            artifact=SimpleNamespace(catalog=catalog, proposal=proposal),
        )

    rewritten = replace(
        catalog, solver_identity="numpy=8.8;scipy=8.8;highs=8.8.8",
    )
    rewritten_artifact = SimpleNamespace(
        descriptor={
            **artifact.descriptor,
            "anchor_solver_identity": rewritten.solver_identity,
        },
        catalog=rewritten,
        proposal=proposal,
    )
    with pytest.raises(MapMixtureConflictError, match="SHA"):
        map_mixture._run_map_mixture_trajectory_from_verified_artifact(
            model, frame, syndrome, config, _seed(), initial,
            artifact=rewritten_artifact,
        )


@pytest.mark.parametrize(
    "classical",
    [
        np.array([[1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize("nonzero", [False, True])
def test_milp_anchor_catalog_is_truth_free_replayable_and_exact_on_tiny_codes(
    classical, nonzero,
):
    model, _ = _model(classical)
    syndrome = _syndrome(model, nonzero)
    first = build_milp_map_anchors(
        model.H_check, syndrome, 0.04, max_anchors=4,
    )
    second = build_milp_map_anchors(
        model.H_check, syndrome, 0.04, max_anchors=4,
    )
    assert first.anchor_sha256 == second.anchor_sha256
    assert first.tie_break_seeds == second.tie_break_seeds
    assert np.array_equal(first.anchors, second.anchors)
    assert first.solver_identity == second.solver_identity
    assert "highs=" in first.solver_identity
    assert len(first.anchor_state_sha256) == first.size
    assert len(first.objective_sha256) == first.size
    assert "epsilon" not in inspect.signature(build_milp_map_anchors).parameters

    coordinates = build_affine_coordinate_system(model, first.anchors[0])
    _, states = enumerate_affine_states(coordinates)
    residuals = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    assert not residuals.any()
    exact_minimum = int(states.sum(axis=1).min())
    assert first.optimum_weight == exact_minimum
    assert np.all(first.anchors.sum(axis=1) == exact_minimum)
    assert np.unique(
        np.packbits(first.anchors, axis=1, bitorder="little"), axis=0,
    ).shape[0] == first.size


@pytest.mark.parametrize(
    "classical",
    [
        np.array([[1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
)
def test_affine_coordinates_are_a_bijection(classical):
    model, _ = _model(classical)
    syndrome = _syndrome(model, True)
    catalog, proposal = _catalog_and_proposal(
        model, syndrome, 0.10, max_anchors=8,
    )
    coordinates, states = enumerate_affine_states(proposal.coordinates)
    assert proposal.coordinates.dimension == 7
    assert np.unique(
        np.packbits(states, axis=1, bitorder="little"), axis=0,
    ).shape[0] == 128
    for coordinate, state in zip(coordinates, states):
        assert np.array_equal(
            proposal.coordinates.coordinates_of_state(state), coordinate,
        )
    for anchor in catalog.anchors:
        recovered = proposal.coordinates.state_from_coordinates(
            proposal.coordinates.coordinates_of_state(anchor),
        )
        assert np.array_equal(recovered, anchor)


@pytest.mark.parametrize(
    "classical",
    [
        np.array([[1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_full_proposal_and_imh_transition_obey_normalization_db_and_stationarity(
    classical, nonzero, p,
):
    model, _ = _model(classical)
    syndrome = _syndrome(model, nonzero)
    _, proposal = _catalog_and_proposal(model, syndrome, p)
    coordinates, states = enumerate_affine_states(proposal.coordinates)
    log_q = np.asarray([
        proposal.log_probability_coordinates(row) for row in coordinates
    ])
    q = np.exp(log_q)
    assert np.all(q > 0.0)
    assert float(q.sum()) == pytest.approx(1.0, abs=2e-14)

    unnormalized = (p / (1.0 - p)) ** states.sum(axis=1)
    target = unnormalized / unnormalized.sum()
    transition = independence_transition_matrix(target, q / q.sum())
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 2e-15
    detailed_balance = target[:, None] * transition
    assert np.max(np.abs(detailed_balance - detailed_balance.T)) <= 1e-13
    assert np.max(np.abs(target @ transition - target)) <= 1e-13


def test_independence_acceptance_formula_matches_direct_target_and_q_ratio():
    model, _ = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    _, proposal = _catalog_and_proposal(model, syndrome, 0.04)
    coordinates, states = enumerate_affine_states(proposal.coordinates)
    current, proposed = 3, 101
    q_current = proposal.log_probability_coordinates(coordinates[current])
    q_proposed = proposal.log_probability_coordinates(coordinates[proposed])
    stored = independence_log_acceptance(
        0.04, states[current].sum(), states[proposed].sum(),
        q_current, q_proposed,
    )
    direct = min(
        0.0,
        (states[proposed].sum() - states[current].sum()) * math.log(0.04 / 0.96)
        + q_current - q_proposed,
    )
    assert stored == direct


def test_trajectory_raw_retains_replayable_logq_and_acceptance_transcript():
    model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    catalog, proposal = _catalog_and_proposal(
        model, syndrome, 0.10, max_anchors=8,
    )
    config = MapMixtureConfig(p=0.10, burn_steps=16, measurement_steps=32)
    result = run_map_mixture_trajectory(
        model, frame, syndrome, config, _seed(), catalog.anchors[-1],
        anchor_catalog=catalog, proposal=proposal,
    )
    assert MAP_RAW_VERSION == "exp102.q0_map_mixture.raw.v2"
    assert result["raw_version"] == MAP_RAW_VERSION
    assert result["method_id"] == MAP_METHOD_ID
    assert result["proposal_sha256"] == proposal.proposal_sha256
    assert result["measurement_states_packed"].shape == (32, 2)
    assert result["measurement_proposal_coordinates_packed"].shape == (32, 1)
    assert not result["measurement_residual_weights"].any()
    assert int(result["measurement_attempts"]) == 32
    assert int(result["measurement_accepts"]) == int(
        result["measurement_accepted"].sum()
    )
    assert int(result["burn_state_changes"]) == int(
        result["burn_state_changed"].sum()
    )
    assert int(result["measurement_state_changes"]) == int(
        result["measurement_state_changed"].sum()
    )

    proposed_coordinates = np.unpackbits(
        result["measurement_proposal_coordinates_packed"], axis=1,
        count=proposal.coordinates.dimension, bitorder="little",
    )
    proposed_states = np.unpackbits(
        result["measurement_proposal_states_packed"], axis=1,
        count=model.num_qubits, bitorder="little",
    )
    stored_states = np.unpackbits(
        result["measurement_states_packed"], axis=1,
        count=model.num_qubits, bitorder="little",
    )
    pre_step_states = np.vstack((
        np.unpackbits(
            result["burn_state_packed"], count=model.num_qubits,
            bitorder="little",
        ),
        stored_states[:-1],
    ))
    expected_state_changed = (
        result["measurement_accepted"].astype(np.bool_)
        & np.any(proposed_states != pre_step_states, axis=1)
    )
    assert np.array_equal(
        result["measurement_state_changed"], expected_state_changed,
    )
    assert np.array_equal(
        proposed_states.sum(axis=1), result["measurement_proposal_weights"],
    )
    for index, coordinate in enumerate(proposed_coordinates):
        assert result["measurement_proposal_log_q"][index] == pytest.approx(
            proposal.log_probability_coordinates(coordinate), abs=1e-14,
        )
        expected = independence_log_acceptance(
            config.p,
            result["measurement_weights"][index - 1]
            if index else np.unpackbits(
                result["burn_state_packed"], count=model.num_qubits,
                bitorder="little",
            ).sum(),
            result["measurement_proposal_weights"][index],
            result["measurement_current_log_q_before"][index],
            result["measurement_proposal_log_q"][index],
        )
        assert result["measurement_log_acceptance"][index] == pytest.approx(
            expected, abs=1e-14,
        )
        uniform = result["measurement_accept_uniform"][index]
        expected_accept = uniform == 0.0 or math.log(uniform) < expected
        assert bool(result["measurement_accepted"][index]) == expected_accept


def test_accepted_self_loop_is_not_counted_as_state_transport():
    class SelfLoopProposal:
        @staticmethod
        def log_probability_coordinates(_coordinate):
            return 0.0

        @staticmethod
        def sample(_rng):
            return {
                "state": np.asarray([0], dtype=np.uint8),
                "coordinate": np.asarray([0], dtype=np.uint8),
                "log_q": 0.0,
                "anchor_index": 0,
                "component_index": 0,
            }

    class ZeroFrame:
        k = 1

        @staticmethod
        def label_of(_state):
            return np.asarray([0], dtype=np.uint8)

    class FixedRng:
        @staticmethod
        def random():
            return 0.5

    _, _, _, transcript = map_mixture._run_imh_stage(
        SelfLoopProposal(), ZeroFrame(), 0.10,
        np.asarray([0], dtype=np.uint8),
        np.asarray([0], dtype=np.uint8), FixedRng(), 4,
    )
    assert transcript["accepted"].tolist() == [1, 1, 1, 1]
    assert transcript["state_changed"].tolist() == [0, 0, 0, 0]


def test_invalid_non_full_support_components_are_rejected():
    model, _ = _model([[1, 1, 1]])
    syndrome = _syndrome(model, False)
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, 0.04, max_anchors=2,
    )
    with pytest.raises(ValueError, match="strictly inside"):
        build_map_mixture_proposal(
            model, catalog,
            theta_stabilizer=(0.0,),
            theta_logical=(0.5,),
            component_weights=(1.0,),
        )


def test_overlap_diagnostic_remains_finite_with_defensive_heavy_tail_draws():
    model, _ = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    _, proposal = _catalog_and_proposal(model, syndrome, 0.04)
    result = estimate_proposal_overlap(proposal, 0.04, 128, 987654321)
    assert set(result) == {
        "num_samples", "importance_ess", "importance_ess_fraction",
        "max_normalized_weight", "top10_normalized_weight",
        "weighted_mean_physical_weight", "stationary_imh_acceptance",
        "minimum_sampled_physical_weight",
    }
    assert all(
        math.isfinite(float(value)) for value in result.values()
    )
    assert 0.0 < result["stationary_imh_acceptance"] <= 1.0


@pytest.mark.parametrize("field", ["theta_stabilizer", "theta_logical", "component_weights"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_proposal_parameters_fail_closed(field, bad):
    model, _ = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, 0.04, max_anchors=8,
    )
    values = {
        "theta_stabilizer": list((0.1,) * 6),
        "theta_logical": list((0.2,) * 6),
        "component_weights": list((1.0 / 6.0,) * 6),
    }
    values[field][0] = bad
    with pytest.raises(ValueError):
        build_map_mixture_proposal(model, catalog, **values)


@pytest.mark.parametrize("bad", [np.uint16(256), np.uint16(257), -1, 2])
def test_binary_inputs_are_checked_before_uint8_conversion(bad):
    model, _ = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    H = model.H_check.astype(np.int64)
    H[0, 0] = bad
    with pytest.raises(ValueError, match="zero and one"):
        build_milp_map_anchors(H, syndrome, 0.04, max_anchors=8)


def test_catalog_and_proposal_arrays_are_immutable_and_hashes_are_replayed():
    model, _ = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, 0.04, max_anchors=8,
    )
    proposal = build_map_mixture_proposal(model, catalog)
    validate_map_anchor_catalog(model.H_check, syndrome, 0.04, catalog)
    validate_map_mixture_proposal(model, syndrome, 0.04, catalog, proposal)
    with pytest.raises(ValueError):
        catalog.anchors[0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError):
        proposal.theta_stabilizer[0] = np.nan

    bad_catalog = replace(catalog, anchor_sha256="0" * 64)
    with pytest.raises(MapMixtureConflictError, match="SHA"):
        validate_map_anchor_catalog(
            model.H_check, syndrome, 0.04, bad_catalog,
        )
    bad_theta = proposal.theta_stabilizer.copy()
    bad_theta[0] = np.nan
    object.__setattr__(proposal, "theta_stabilizer", bad_theta)
    with pytest.raises(MapMixtureConflictError, match="proposal"):
        validate_map_mixture_proposal(
            model, syndrome, 0.04, catalog, proposal,
        )


def test_canonical_run_rejects_wrong_p_syndrome_and_noncanonical_anchor_count():
    model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome(model, True)
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, 0.04, max_anchors=8,
    )
    proposal = build_map_mixture_proposal(model, catalog)
    config = MapMixtureConfig(0.04, 8, 8)
    with pytest.raises(MapMixtureConflictError):
        validate_map_mixture_proposal(
            model, syndrome, 0.10, catalog, proposal,
        )
    wrong_syndrome = syndrome.copy()
    wrong_syndrome[0] ^= np.uint8(1)
    with pytest.raises(MapMixtureConflictError):
        validate_map_mixture_proposal(
            model, wrong_syndrome, 0.04, catalog, proposal,
        )
    short_catalog = build_milp_map_anchors(
        model.H_check, syndrome, 0.04, max_anchors=4,
    )
    with pytest.raises(MapMixtureConflictError):
        run_map_mixture_trajectory(
            model, frame, syndrome, config, _seed(), catalog.anchors[0],
            anchor_catalog=short_catalog,
            proposal=build_map_mixture_proposal(model, short_catalog),
        )


@pytest.mark.parametrize("kwargs", [
    {"burn_steps": 8.5, "measurement_steps": 8},
    {"burn_steps": 8, "measurement_steps": 8.5},
    {"burn_steps": True, "measurement_steps": 8},
    {"burn_steps": 8, "measurement_steps": 8, "max_anchors": 4},
])
def test_map_config_rejects_noncanonical_integer_fields(kwargs):
    with pytest.raises(ValueError):
        MapMixtureConfig(p=0.04, **kwargs)
