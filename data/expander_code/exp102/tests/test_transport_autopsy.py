from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.discovery import load_discovery_config
from data.expander_code.exp102.exp102_pipeline.ladders import (
    make_pt_candidate,
    make_uniform_ladder,
    q0_config_from_candidate,
)
from data.expander_code.exp102.exp102_pipeline.q0_pt import (
    expected_swap_attempts,
    run_q0_pt_instance,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.transport_autopsy import (
    AUTOPSY_RAW_VERSION,
    AUTOPSY_VERSION,
    PARENT_SOURCE_COMMIT,
    _run_trace_instance,
    classify_transport,
    load_autopsy_config,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
DISCOVERY_CONFIG_PATH = EXP102_ROOT / "config/discovery.v2.json"
AUTOPSY_CONFIG_PATH = EXP102_ROOT / "config/transport_autopsy.v1.json"


def _small_model():
    classical = np.array([[1, 1, 1]], dtype=np.uint8)
    model, frame = build_model(classical)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, 2]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    return model, frame, syndrome


def test_autopsy_config_binds_exact_four_parent_raw_identities():
    registry = load_registry(REGISTRY_PATH)
    discovery = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    config = load_autopsy_config(AUTOPSY_CONFIG_PATH, registry, discovery)
    assert config["autopsy_version"] == AUTOPSY_VERSION
    assert config["raw_version"] == AUTOPSY_RAW_VERSION
    assert config["parent_source_commit"] == PARENT_SOURCE_COMMIT
    assert [(row["ladder_id"], row["cell"]["code_id"]) for row in config["parents"]] == [
        ("D0", "m06_c00"), ("D0", "m08_c06"),
        ("D4", "m06_c00"), ("D4", "m08_c06"),
    ]
    assert all(len(row["parent_raw_sha256"]) == 64 for row in config["parents"])
    assert all(len(row["instance_seeds"]) == 4 for row in config["parents"])


def test_trace_instrumentation_consumes_no_randomness_and_replays_pt_core():
    model, frame, syndrome = _small_model()
    candidate = make_pt_candidate(
        make_uniform_ladder("trace", 0.475, 6),
        burn_rounds=2,
        measurement_rounds=17,
        swap_sweeps_per_round=3,
    )
    initial_label = np.uint64(3)
    seed = 987654321
    traced = _run_trace_instance(
        model, frame, syndrome, 0.10, candidate, seed, initial_label,
    )
    ordinary = run_q0_pt_instance(
        model, frame, syndrome, 0.10, q0_config_from_candidate(candidate),
        seed, initial_label, engine="numba",
    )
    for field in (
            "labels", "swap_attempts", "swap_accepts", "logical_attempts",
            "logical_accepts", "hot_touches_per_replica",
            "hot_updated_visits_per_replica", "uncertified_round_trips_per_replica",
            "round_trips_per_replica", "sector_changing_round_trips_per_replica"):
        assert np.array_equal(traced[field], ordinary[field]), field
    for field in (
            "hot_touches", "hot_updated_visits", "uncertified_round_trips",
            "round_trips", "sector_changing_round_trips",
            "max_hard_coset_residual"):
        assert traced[field] == ordinary[field], field

    temperatures = candidate["num_temperatures"]
    for round_index, permutation in enumerate(traced["replica_at_rung_by_round"]):
        assert np.array_equal(np.sort(permutation), np.arange(temperatures))
        cold_replica = int(permutation[0])
        assert traced["labels"][round_index] == traced["replica_label_by_round"][
            round_index, cold_replica
        ]
    expected_measurement_attempts = expected_swap_attempts(
        temperatures, candidate["measurement_rounds"],
        candidate["swap_sweeps_per_round"],
    )
    conditional = traced["edge_attempts_by_phase_direction"]
    assert np.array_equal(conditional[:, 0].sum(axis=0), expected_measurement_attempts)
    assert np.array_equal(conditional[:, 1].sum(axis=0), expected_measurement_attempts)
    assert np.all(traced["round_min_rung_by_replica"] <= traced["round_max_rung_by_replica"])
    phase = traced["replica_phase_by_round"]
    direction = traced["direction_by_round"]
    assert np.array_equal(direction[phase == 1], np.ones(np.count_nonzero(phase == 1), dtype=np.int8))
    assert np.array_equal(direction[(phase == 2) | (phase == 3)],
                          -np.ones(np.count_nonzero((phase == 2) | (phase == 3)), dtype=np.int8))


def _classification_result(outbound_attempts=400, outbound_accepts=200,
                           hot_updates=0, frontier=2, inbound_attempts=400,
                           inbound_accepts=200, temperatures=5):
    edges = temperatures - 1
    attempts = np.zeros((4, 2, edges), dtype=np.int64)
    accepts = np.zeros_like(attempts)
    attempts[1, 0] = outbound_attempts
    accepts[1, 0] = outbound_accepts
    attempts[3, 1] = inbound_attempts
    accepts[3, 1] = inbound_accepts
    return {
        "swap_attempts": np.full(edges, 1000, dtype=np.int64),
        "swap_accepts": np.full(edges, 500, dtype=np.int64),
        "hot_updated_visits": hot_updates,
        "frontier_max_rung": np.array([frontier], dtype=np.int64),
        "edge_attempts_by_phase_direction": attempts,
        "edge_accepts_by_phase_direction": accepts,
    }


def _classification_config():
    return {"conditional_min_attempts": 200, "conditional_min_edge_rate": 0.05}


def test_autopsy_classifier_obeys_frozen_precedence_and_attempt_floor():
    result = _classification_result(outbound_attempts=199)
    assert classify_transport([result], _classification_config())[0] == "INCONCLUSIVE"

    result = _classification_result(outbound_accepts=1)
    assert classify_transport([result], _classification_config())[0] == (
        "CONDITIONAL_EDGE_BOTTLENECK"
    )

    result = _classification_result(hot_updates=0, frontier=3)
    assert classify_transport([result], _classification_config())[0] == (
        "GLOBAL_DIFFUSION_OR_RELAXATION_LIMITED"
    )

    result = _classification_result(
        hot_updates=2, frontier=4, inbound_attempts=400, inbound_accepts=1,
    )
    assert classify_transport([result], _classification_config())[0] == (
        "POST_HOT_HYSTERESIS"
    )

    result = _classification_result(
        hot_updates=2, frontier=4, inbound_attempts=199, inbound_accepts=100,
    )
    assert classify_transport([result], _classification_config())[0] == "INCONCLUSIVE"
