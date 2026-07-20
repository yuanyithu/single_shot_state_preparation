import hashlib

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.ladders import (
    ladder_fingerprint,
    make_piecewise_density_ladder,
    make_uniform_ladder,
    piecewise_density_ladder_q32,
    uniform_ladder_q32,
    validate_ladder_record,
)
from data.expander_code.exp102.exp102_pipeline.q0_pt import (
    Q32_ONE,
    Q0PtConfig,
    _mark_hot_local_update,
    _record_transport_endpoints,
    coupling_ladder,
    expected_swap_attempts,
    ladder_x_q32_sha256,
    run_q0_pt_instance,
    validate_ladder_x_q32,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


@pytest.mark.parametrize("temperatures", [2, 3, 88, 104, 128])
def test_uniform_q32_ladder_is_canonical(temperatures):
    values = uniform_ladder_q32(temperatures)
    assert len(values) == temperatures
    assert values[0] == 0 and values[-1] == Q32_ONE
    assert all(left < right for left, right in zip(values, values[1:]))
    assert len(ladder_x_q32_sha256(values)) == 64


def test_q32_validation_accepts_tiny_intervals_and_rejects_bad_knots():
    tiny = (0, 1, 2, Q32_ONE)
    assert validate_ladder_x_q32(tiny, 4) == tiny
    with pytest.raises(ValueError, match="endpoints"):
        validate_ladder_x_q32((1, 2, Q32_ONE), 3)
    with pytest.raises(ValueError, match="strictly"):
        validate_ladder_x_q32((0, 2, 2, Q32_ONE), 4)
    with pytest.raises(ValueError, match="integers"):
        validate_ladder_x_q32((0, 0.5, Q32_ONE), 3)
    with pytest.raises(ValueError, match="length"):
        validate_ladder_x_q32((0, Q32_ONE), 3)


def test_piecewise_density_ladder_round_trips_through_generation_identity():
    record = make_piecewise_density_ladder("D0", 0.45, 88, 6)
    assert validate_ladder_record(record) == record
    assert record["ladder_x_q32"] == list(
        piecewise_density_ladder_q32(0.45, 88, 6)
    )
    assert len(ladder_fingerprint(record)) == 64

    tampered = {**record, "ladder_x_q32": list(record["ladder_x_q32"])}
    tampered["ladder_x_q32"][20] += 1
    with pytest.raises(ValueError, match="SHA256"):
        validate_ladder_record(tampered)

    tampered = {**record, "ladder_generation": dict(record["ladder_generation"])}
    tampered["ladder_generation"]["density"] = 5.0
    with pytest.raises(ValueError, match="generation parameters"):
        validate_ladder_record(tampered)


def test_explicit_q32_coupling_ladder_has_exact_probability_endpoints():
    knots = uniform_ladder_q32(7)
    K, probabilities = coupling_ladder(0.04, 0.49, 7, 1.0, knots)
    assert probabilities[0] == pytest.approx(0.04)
    assert probabilities[-1] == pytest.approx(0.49)
    assert np.all(np.diff(K) < 0.0)
    assert np.all(np.diff(probabilities) > 0.0)


def _small_model():
    from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101

    load_exp101()
    from exp101_certified_src.graphs import cycle_parity_check_matrix

    model, frame = build_model(cycle_parity_check_matrix(3))
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, 2]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    return model, frame, syndrome


@pytest.mark.parametrize(
    "temperatures,swap_sweeps,burn_rounds",
    [(4, 1, 0), (5, 2, 1), (4, 3, 2), (5, 4, 3)],
)
def test_multiswap_numba_matches_reference_and_exact_attempts(
    temperatures, swap_sweeps, burn_rounds,
):
    model, frame, syndrome = _small_model()
    config = Q0PtConfig(
        p_hot=0.475,
        num_temperatures=temperatures,
        gamma=1.0,
        burn_rounds=burn_rounds,
        measurement_rounds=11,
        ladder_x_q32=uniform_ladder_q32(temperatures),
        swap_sweeps_per_round=swap_sweeps,
    )
    results = [
        run_q0_pt_instance(
            model, frame, syndrome, 0.1, config, 987, np.uint64(1), engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    arrays = (
        "labels", "ladder_K", "ladder_p", "swap_attempts", "swap_accepts",
        "logical_attempts", "logical_accepts", "hot_arrival_labels",
        "hot_departure_labels", "hot_touches_per_replica",
        "hot_updated_visits_per_replica", "uncertified_round_trips_per_replica",
        "round_trips_per_replica", "sector_changing_round_trips_per_replica",
        "final_replica_at_rung", "final_transport_phase",
    )
    scalars = (
        "hot_touches", "hot_updated_visits", "uncertified_round_trips",
        "round_trips", "sector_changing_round_trips", "max_hard_coset_residual",
    )
    for field in arrays:
        assert np.array_equal(results[0][field], results[1][field]), field
    for field in scalars:
        assert results[0][field] == results[1][field], field
    expected = expected_swap_attempts(
        temperatures, burn_rounds + 11, swap_sweeps,
    )
    assert np.array_equal(results[0]["swap_attempts"], expected)
    assert np.all(
        results[0]["logical_attempts"]
        == (burn_rounds + 11) * config.sweeps_per_round * config.logical_move_repeat
    )


def test_s1_preserves_v1_trajectory_and_random_protocol_digest():
    model, frame, syndrome = _small_model()
    config = Q0PtConfig(0.475, 5, 1.5, 5, 17, 2, 2, swap_sweeps_per_round=1)
    result = run_q0_pt_instance(
        model, frame, syndrome, 0.1, config, 123, np.uint64(1), engine="reference",
    )
    expected = {
        "labels": "b707241545a346265aab1ffb32ff64b55bf8f8dc1b56a46ef33ce3d15db11d33",
        "swap_attempts": "4b95add3dcfa1cb8308ae98cacf0e19326e976c911ff174164977fbd13dacfca",
        "swap_accepts": "12db315d254f168ddaa087adcd93492ef3f3e4647b28b2a8ffb5f145ed6d53f2",
        "logical_attempts": "7a0c560cb50f2c5a32e51ee991114ad2982b09a26a754fea0e7c9d2a8dde78c4",
        "logical_accepts": "ed1e9c083de120a6ed4db4df768ca1080788365b1bc85ffd4c24f6e8e6b8abb8",
    }
    for field, digest in expected.items():
        assert hashlib.sha256(np.asarray(result[field]).tobytes()).hexdigest() == digest


def _transport_arrays(num_replicas=4):
    return {
        "phase": np.zeros(num_replicas, dtype=np.int8),
        "labels": np.zeros(num_replicas, dtype=np.uint64),
        "arrival": np.zeros(num_replicas, dtype=np.uint64),
        "touch": np.zeros(num_replicas, dtype=np.int64),
        "updated": np.zeros(num_replicas, dtype=np.int64),
        "uncertified": np.zeros(num_replicas, dtype=np.int64),
        "certified": np.zeros(num_replicas, dtype=np.int64),
        "changing": np.zeros(num_replicas, dtype=np.int64),
        "updated_label": np.zeros(num_replicas, dtype=np.uint64),
    }


def _endpoint(replica_at, state):
    _record_transport_endpoints(
        np.asarray(replica_at), state["phase"], state["labels"], state["arrival"],
        state["touch"], state["uncertified"], state["certified"], state["changing"],
    )


def _hot_update(replica_at, state):
    _mark_hot_local_update(
        np.asarray(replica_at), state["phase"], state["labels"], state["updated"],
        state["updated_label"],
    )


def test_transport_requires_a_later_hot_local_update_and_deduplicates_endpoints():
    state = _transport_arrays()
    state["phase"][0] = 1

    _endpoint([3, 1, 2, 0], state)
    _endpoint([3, 1, 2, 0], state)
    assert state["touch"][0] == 1
    _endpoint([0, 1, 2, 3], state)
    assert state["uncertified"][0] == 1
    assert state["certified"][0] == 0

    _endpoint([3, 1, 2, 0], state)
    state["labels"][0] ^= np.uint64(1)
    _hot_update([3, 1, 2, 0], state)
    _endpoint([0, 1, 2, 3], state)
    assert state["updated"][0] == 1
    assert state["certified"][0] == 1
    assert state["changing"][0] == 1


def test_transport_excludes_two_net_cancelling_hot_flips():
    state = _transport_arrays()
    state["phase"][0] = 1
    _endpoint([3, 1, 2, 0], state)
    state["labels"][0] ^= np.uint64(1)
    state["labels"][0] ^= np.uint64(1)
    _hot_update([3, 1, 2, 0], state)
    _endpoint([0, 1, 2, 3], state)
    assert state["certified"][0] == 1
    assert state["changing"][0] == 0


def test_uniform_ladder_record_cannot_hide_mutated_generation():
    record = make_uniform_ladder("uniform_.45_R64", 0.45, 64)
    assert validate_ladder_record(record) == record
    altered = {**record, "ladder_generation": {"algorithm": "uniform_q32.v1", "x": 1}}
    with pytest.raises(ValueError, match="generation fields"):
        validate_ladder_record(altered)
