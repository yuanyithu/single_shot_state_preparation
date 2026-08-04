"""Exact algebra checks for iid cross-fitted importance collision estimates."""

import math

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_crossfit_importance import (
    CrossfitImportanceError,
    crossfit_collision_ratio,
    crossfit_distribution_distance,
)


def test_repeated_exact_block_masses_recover_sector_purity_and_bit63_label():
    label_zero = np.uint64(0)
    label_one = np.uint64(1) << np.uint64(63)
    # Every block has unnormalized sector masses (3, 1), so all cross-block
    # products equal the exact purity (3**2 + 1**2) / 4**2.
    labels = np.asarray(
        [label_zero, label_zero, label_zero, label_one] * 4,
        dtype=np.uint64,
    )
    result = crossfit_collision_ratio(
        labels, np.zeros(labels.size), block_count=4, logical_dimension=64,
    )
    expected = 10.0 / 16.0
    assert result.purity == pytest.approx(expected, abs=1e-15)
    assert result.q_top == pytest.approx(expected, abs=1e-15)
    assert result.purity_jackknife_se == pytest.approx(0.0, abs=1e-15)
    assert result.q_top_jackknife_se == pytest.approx(0.0, abs=1e-15)
    assert result.label_count == 2
    assert np.all(result.block_effective_sample_sizes == 4.0)


def test_log_weight_shift_does_not_change_the_crossfit_ratio():
    labels = np.asarray([0, 1, 0, 2, 1, 2, 0, 1, 2, 2, 1, 0], dtype=np.uint64)
    log_weights = np.asarray([-5.0, -2.0, -4.0, -1.0] * 3)
    left = crossfit_collision_ratio(
        labels, log_weights, block_count=3, logical_dimension=3,
    )
    right = crossfit_collision_ratio(
        labels, log_weights + 123.0, block_count=3, logical_dimension=3,
    )
    assert left.purity == pytest.approx(right.purity, abs=1e-14)
    assert left.q_top == pytest.approx(right.q_top, abs=1e-14)
    assert math.isfinite(left.purity_jackknife_se)


def test_crossfit_distribution_distance_recovers_identical_and_opposite_two_sector_laws():
    same = np.asarray([0, 0, 0, 1] * 4, dtype=np.uint64)
    opposite = np.asarray([0, 1, 1, 1] * 4, dtype=np.uint64)
    identical = crossfit_distribution_distance(
        same, np.zeros(same.size), same, np.zeros(same.size),
        block_count=4, logical_dimension=1,
    )
    assert identical.d2_norm == pytest.approx(0.0, abs=1e-15)
    assert identical.d2_norm_jackknife_se == pytest.approx(0.0, abs=1e-15)

    different = crossfit_distribution_distance(
        same, np.zeros(same.size), opposite, np.zeros(opposite.size),
        block_count=4, logical_dimension=1,
    )
    # The sector laws are (3/4,1/4) and (1/4,3/4): squared L2 distance
    # is 1/2 and the k=1 normalization is also 1/2.
    assert different.d2_norm == pytest.approx(1.0, abs=1e-15)
    assert different.d2_norm_jackknife_se == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    "labels,logs,blocks",
    [
        (np.asarray([], dtype=np.uint64), np.asarray([], dtype=np.float64), 3),
        (np.asarray([0, 1, 2], dtype=np.int64), np.zeros(3), 3),
        (np.asarray([0, 1, 2], dtype=np.uint64), np.asarray([0.0, np.nan, 0.0]), 3),
        (np.asarray([0, 1, 2, 3], dtype=np.uint64), np.zeros(4), 3),
    ],
)
def test_crossfit_rejects_invalid_inputs(labels, logs, blocks):
    with pytest.raises(CrossfitImportanceError):
        crossfit_collision_ratio(labels, logs, block_count=blocks, logical_dimension=3)
