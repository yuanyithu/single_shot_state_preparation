import numpy as np
import pytest

from data.expander_code.exp103.exp103_pipeline.crossing import (
    classify_final_crossing,
    simultaneous_bootstrap,
    wilson_interval,
)


P_VALUES = np.arange(2, 15, dtype=np.float64) / 100.0


def _bands(values, half_width=0.01):
    values = np.asarray(values, dtype=np.float64)
    return values - half_width, values + half_width


def _classify(endpoint, adjacent=None, half_width=0.01):
    endpoint = np.asarray(endpoint, dtype=np.float64)
    if adjacent is None:
        adjacent = np.zeros((5, 13), dtype=np.float64)
    adjacent = np.asarray(adjacent, dtype=np.float64)
    endpoint_low, endpoint_high = _bands(endpoint, half_width)
    adjacent_low, adjacent_high = _bands(adjacent, half_width)
    return classify_final_crossing(
        P_VALUES, endpoint, endpoint_low, endpoint_high,
        adjacent, adjacent_low, adjacent_high,
    )


def test_single_certified_crossing_with_compatible_triple_is_resolved():
    endpoint = np.asarray([-0.05] * 6 + [0.05] * 7)
    adjacent = np.zeros((5, 13))
    adjacent[0] = endpoint
    adjacent[1] = endpoint
    result = _classify(endpoint, adjacent)
    assert result == {
        "status": "EXP103_DECODER_CROSSING_RESOLVED",
        "bracket": (0.07, 0.08),
        "compatible_triple": [3, 4, 5],
        "correct_reversals": 1,
        "reverse_reversals": 0,
    }


def test_certified_endpoint_without_multisize_consistency_is_pairwise_only():
    endpoint = np.asarray([-0.05] * 6 + [0.05] * 7)
    result = _classify(endpoint)
    assert result["status"] == "EXP103_PAIRWISE_BRACKET_ONLY"
    assert result["bracket"] == (0.07, 0.08)
    assert result["compatible_triple"] is None


@pytest.mark.parametrize(
    "endpoint",
    [
        np.full(13, -0.05),
        np.asarray([0.05] * 6 + [-0.05] * 7),
        np.asarray([-0.05] * 13),  # A crossing outside the window is not extrapolated.
    ],
)
def test_no_correct_direction_reversal_is_no_crossing(endpoint):
    result = _classify(endpoint)
    assert result["status"] == "EXP103_NO_CORRECT_CROSSING_IN_WINDOW"
    assert result["bracket"] is None


def test_multiple_reversals_are_inconclusive():
    endpoint = np.asarray([-0.05] * 3 + [0.05] * 3 + [-0.05] * 3 + [0.05] * 4)
    result = _classify(endpoint)
    assert result["status"] == "EXP103_DECODER_CROSSING_INCONCLUSIVE"
    assert result["correct_reversals"] == 2
    assert result["reverse_reversals"] == 1
    assert result["bracket"] is None


def test_point_reversal_without_simultaneous_sign_exclusion_is_inconclusive():
    endpoint = np.asarray([-0.005] * 6 + [0.005] * 7)
    result = _classify(endpoint, half_width=0.01)
    assert result["status"] == "EXP103_DECODER_CROSSING_INCONCLUSIVE"
    assert result["bracket"] is None


def test_exact_zero_between_signs_does_not_create_a_nonadjacent_bracket():
    endpoint = np.asarray([-0.05] * 6 + [0.0] + [0.05] * 6)
    result = _classify(endpoint)
    assert result["status"] == "EXP103_DECODER_CROSSING_INCONCLUSIVE"
    assert result["correct_reversals"] == 1
    assert result["bracket"] is None


def test_secondary_adjacent_curves_cannot_promote_endpoint_without_reversal():
    endpoint = np.full(13, -0.05)
    adjacent = np.zeros((5, 13))
    adjacent[0] = np.asarray([-0.05] * 6 + [0.05] * 7)
    adjacent[1] = adjacent[0]
    result = _classify(endpoint, adjacent)
    assert result["status"] == "EXP103_NO_CORRECT_CROSSING_IN_WINDOW"


@pytest.mark.parametrize("failures", [0, 1, 5000, 9999, 10000])
def test_wilson_interval_is_finite_ordered_and_contains_observed_rate(failures):
    low, high = wilson_interval(failures, 10_000)
    rate = failures / 10_000
    assert 0.0 <= low <= rate <= high <= 1.0


def test_wilson_interval_rejects_unfrozen_confidence_level():
    with pytest.raises(ValueError):
        wilson_interval(1, 10_000, confidence=0.90)


def test_bootstrap_rerun_is_bit_deterministic(frozen_config, clone_payload):
    test_config = clone_payload(frozen_config)
    test_config["bootstrap"]["replicates"] = 128
    failures = np.arange(6 * 8 * 13, dtype=np.int64).reshape(6, 8, 13) + 100
    trials = np.full((6, 8, 13), 10_000, dtype=np.int64)
    first = simultaneous_bootstrap(
        failures, trials, (0, 1, 2), test_config, "determinism_oracle",
    )
    second = simultaneous_bootstrap(
        failures, trials, (0, 1, 2), test_config, "determinism_oracle",
    )
    assert first.keys() == second.keys()
    for field in first:
        if isinstance(first[field], np.ndarray):
            assert np.array_equal(first[field], second[field])
        else:
            assert first[field] == second[field]
