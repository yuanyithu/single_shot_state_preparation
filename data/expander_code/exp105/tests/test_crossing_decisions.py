"""Oracles for the certification rule, the bracket and the crossing location."""

import numpy as np
import pytest

from data.expander_code.exp105.exp105_pipeline.crossing import (
    CERTIFIED,
    NOT_CERTIFIED,
    classify_crossing,
    cluster_bootstrap,
    crossing_location,
    wilson_interval,
)


P = np.asarray([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])


def band(delta, half_width):
    delta = np.asarray(delta, dtype=np.float64)
    return delta, delta - half_width, delta + half_width


def test_certified_crossing_reports_the_tightest_bracket():
    delta, low, high = band([-0.3, -0.25, -0.2, -0.15, -0.1, 0.1, 0.2, 0.3, 0.4], 0.02)
    decision = classify_crossing(P, delta, low, high)
    assert decision["status"] == CERTIFIED
    assert decision["bracket"] == (0.06, 0.07)
    assert decision["bracket_indices"] == (4, 5)


def test_bracket_does_not_need_adjacent_grid_points():
    """The point nearest the crossing is exactly the one that cannot certify."""
    delta, low, high = band([-0.3, -0.2, -0.1, -0.005, 0.004, 0.1, 0.2, 0.3, 0.4], 0.02)
    decision = classify_crossing(P, delta, low, high)
    assert decision["status"] == CERTIFIED
    # p=0.05 and p=0.06 both sit inside the band, so the bracket widens.
    assert decision["bracket"] == (0.04, 0.07)


def test_a_band_that_contains_zero_everywhere_certifies_nothing():
    delta, low, high = band([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 0.6)
    decision = classify_crossing(P, delta, low, high)
    assert decision["status"] == NOT_CERTIFIED
    assert np.isnan(decision["bracket"][0]) and np.isnan(decision["bracket"][1])


def test_an_always_positive_contrast_is_not_a_crossing():
    delta, low, high = band([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], 0.02)
    decision = classify_crossing(P, delta, low, high)
    assert decision["status"] == NOT_CERTIFIED
    assert decision["certified_negative_p"] == []
    assert len(decision["certified_positive_p"]) == len(P)


def test_a_positive_to_negative_reversal_is_the_wrong_direction():
    delta, low, high = band([0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5], 0.02)
    decision = classify_crossing(P, delta, low, high)
    assert decision["status"] == NOT_CERTIFIED


def test_positive_before_negative_does_not_certify_a_crossing():
    """A certified positive point only counts if some certified negative precedes it."""
    delta, low, high = band([0.3, 0.2, 0.1, 0.05, -0.2, -0.3, -0.4, -0.5, -0.6], 0.02)
    decision = classify_crossing(P, delta, low, high)
    assert decision["status"] == NOT_CERTIFIED


def test_nan_contrasts_are_never_certified():
    delta = np.full(len(P), np.nan)
    decision = classify_crossing(P, delta, delta, delta)
    assert decision["status"] == NOT_CERTIFIED
    assert decision["certified_negative_p"] == []
    assert decision["certified_positive_p"] == []


def test_crossing_location_interpolates_linearly():
    delta, low, high = band([-0.3, -0.25, -0.2, -0.15, -0.1, 0.1, 0.2, 0.3, 0.4], 0.02)
    decision = classify_crossing(P, delta, low, high)
    replicates = np.tile(delta, (200, 1))
    location = crossing_location(P, delta, replicates, decision)
    # -0.1 at 0.06 and +0.1 at 0.07 puts the zero exactly halfway.
    assert location["p_cross"] == pytest.approx(0.065)
    assert location["p_cross_low"] == pytest.approx(0.065)
    assert location["p_cross_high"] == pytest.approx(0.065)
    assert location["defined_fraction"] == 1.0
    assert location["reason"] == ""


def test_crossing_location_is_undefined_without_a_certified_bracket():
    delta, low, high = band([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 0.6)
    decision = classify_crossing(P, delta, low, high)
    location = crossing_location(P, delta, np.tile(delta, (10, 1)), decision)
    assert np.isnan(location["p_cross"])
    assert location["reason"] == "no_certified_bracket"


def test_crossing_location_flags_unstable_replicates():
    delta, low, high = band([-0.3, -0.25, -0.2, -0.15, -0.1, 0.1, 0.2, 0.3, 0.4], 0.02)
    decision = classify_crossing(P, delta, low, high)
    replicates = np.tile(delta, (100, 1))
    # Most replicates never cross inside the frozen sub-grid.
    replicates[:90] = np.linspace(-1.0, -0.1, len(P))
    location = crossing_location(P, delta, replicates, decision)
    assert location["reason"] == "insufficient_defined_replicates"
    assert np.isnan(location["p_cross_low"])
    assert location["defined_fraction"] == pytest.approx(0.10)


def test_cluster_bootstrap_recovers_the_pooled_means(frozen_config):
    trials = int(frozen_config["trials_per_code_p"]["3"])
    n_p = len(frozen_config["p_tokens"])
    rng = np.random.Generator(np.random.PCG64(1))
    failures_by_m = []
    for m_index in range(6):
        rate = 0.2 + 0.1 * m_index
        failures_by_m.append(rng.binomial(trials, rate, size=(400, n_p)))
    result = cluster_bootstrap(
        failures_by_m, [trials] * len(failures_by_m), frozen_config, "unit_test",
    )
    for m_index, failures in enumerate(failures_by_m):
        expected = failures.sum(axis=0) / (failures.shape[0] * trials)
        assert np.allclose(result["point"][m_index], expected)
    assert np.allclose(
        result["endpoint"], result["point"][-1] - result["point"][0],
    )
    assert result["half_width"] > 0.0
    assert result["endpoint_replicates"].shape == (
        int(frozen_config["bootstrap"]["replicates"]), n_p,
    )


def test_cluster_bootstrap_is_reproducible_for_a_given_label(frozen_config):
    trials = int(frozen_config["trials_per_code_p"]["3"])
    rng = np.random.Generator(np.random.PCG64(2))
    failures_by_m = [
        rng.binomial(trials, 0.3, size=(200, len(frozen_config["p_tokens"])))
        for _ in range(6)
    ]
    per_m = [trials] * len(failures_by_m)
    first = cluster_bootstrap(failures_by_m, per_m, frozen_config, "same")
    second = cluster_bootstrap(failures_by_m, per_m, frozen_config, "same")
    other = cluster_bootstrap(failures_by_m, per_m, frozen_config, "different")
    assert first["half_width"] == second["half_width"]
    assert np.array_equal(first["endpoint_replicates"], second["endpoint_replicates"])
    # The label enters the bootstrap seed, so a different label must draw a
    # different set of replicates. The half-width alone is a coarse scalar and
    # can collide by chance, so the replicate arrays are what is compared.
    assert not np.array_equal(
        first["endpoint_replicates"], other["endpoint_replicates"],
    )


def test_more_codes_narrow_the_simultaneous_band(frozen_config):
    """The band must respond to the number of codes, which is what exp105 buys."""
    trials = int(frozen_config["trials_per_code_p"]["3"])
    n_p = len(frozen_config["p_tokens"])
    rng = np.random.Generator(np.random.PCG64(9))
    widths = []
    for size in (50, 800):
        failures_by_m = [
            rng.binomial(trials, 0.4, size=(size, n_p)) for _ in range(6)
        ]
        widths.append(
            cluster_bootstrap(
                failures_by_m, [trials] * len(failures_by_m),
                frozen_config, "scaling",
            )["half_width"]
        )
    assert widths[1] < widths[0] / 3.0


def test_wilson_interval_brackets_the_point_estimate():
    for failures, trials in ((0, 4), (1, 4), (4, 4), (37, 100)):
        low, high = wilson_interval(failures, trials)
        assert 0.0 <= low <= failures / trials <= high <= 1.0
    assert all(np.isnan(value) for value in wilson_interval(0, 0))


def test_cluster_bootstrap_handles_unequal_panels(frozen_config):
    """exp105's whole allocation rule depends on this working."""
    n_p = len(frozen_config["p_tokens"])
    rng = np.random.Generator(np.random.PCG64(17))
    failures_by_m = [
        rng.binomial(4, 0.30, size=(900, n_p)),
        rng.binomial(3, 0.35, size=(120, n_p)),
    ]
    result = cluster_bootstrap(failures_by_m, [4, 3], frozen_config, "unequal")
    for index, (failures, trials) in enumerate(zip(failures_by_m, [4, 3])):
        expected = failures.sum(axis=0) / (failures.shape[0] * trials)
        assert np.allclose(result["point"][index], expected)
    assert result["half_width"] > 0.0


def test_cluster_bootstrap_rejects_a_missing_trial_count(frozen_config):
    n_p = len(frozen_config["p_tokens"])
    failures_by_m = [np.zeros((10, n_p), dtype=np.int64)] * 2
    with pytest.raises(ValueError, match="trials per code"):
        cluster_bootstrap(failures_by_m, [4], frozen_config, "mismatched")
