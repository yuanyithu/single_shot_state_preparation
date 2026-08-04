"""Exact rational checks for collapsed-B interval-envelope primitives."""

from fractions import Fraction
import itertools

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_collapsed_tail_bound import (
    classical_coset_mass_interval,
    partial_rows_scaled_upper,
    prefix_max_upper_tables,
    purity_interval_from_sector_tail,
    q_top_interval_from_purity,
    row_prefix_partition_upper,
    scaled_state_weight_interval,
)


SMALL_H = (
    np.asarray([[1, 1, 1]], dtype=np.uint8),
    np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
)


def _mask(bits):
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _exact_classical_mass(H, numerator, denominator):
    H = np.asarray(H, dtype=np.uint8)
    r, n = H.shape
    result = [Fraction(0) for _ in range(1 << r)]
    for bits in itertools.product((0, 1), repeat=n):
        vector = np.asarray(bits, dtype=np.uint8)
        syndrome = (H.astype(np.int64) @ vector.astype(np.int64) % 2).astype(np.uint8)
        weight = int(vector.sum())
        result[_mask(syndrome)] += Fraction(
            numerator ** weight * (denominator - numerator) ** (n - weight),
            denominator ** n,
        )
    return result


def _all_B(rank):
    for value in range(1 << (rank * rank)):
        result = np.zeros((rank, rank), dtype=np.uint8)
        for row in range(rank):
            for column in range(rank):
                result[row, column] = (value >> (row * rank + column)) & 1
        yield result


def _exact_collapsed_weight(H, syndrome, B, numerator, denominator, mass):
    H = np.asarray(H, dtype=np.uint8)
    r, n = H.shape
    values = syndrome ^ ((B.astype(np.int64) @ H.astype(np.int64)) % 2).astype(np.uint8)
    result = Fraction(
        numerator ** int(B.sum()) * (denominator - numerator) ** (r * r - int(B.sum())),
        denominator ** (r * r),
    )
    for column in range(n):
        result *= mass[_mask(values[:, column])]
    return result


@pytest.mark.parametrize("H", SMALL_H)
@pytest.mark.parametrize("numerator,denominator", [(1, 25), (1, 10)])
def test_interval_classical_mass_contains_exact_rational_values(H, numerator, denominator):
    lower, upper = classical_coset_mass_interval(H, numerator, denominator)
    exact = _exact_classical_mass(H, numerator, denominator)
    for lower_value, upper_value, exact_value in zip(lower, upper, exact, strict=True):
        assert Fraction.from_float(float(lower_value)) <= exact_value
        assert exact_value <= Fraction.from_float(float(upper_value))


def test_partial_row_envelopes_contain_exact_collapsed_mass():
    H = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    numerator, denominator = 1, 25
    syndrome = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    lower, upper = classical_coset_mass_interval(H, numerator, denominator)
    exact_mass = _exact_classical_mass(H, numerator, denominator)
    scale_lower = float(np.max(lower))
    scale = Fraction.from_float(scale_lower)
    tables = prefix_max_upper_tables(upper, (0, 1, 2))
    target = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    exact_total = sum(
        (_exact_collapsed_weight(H, syndrome, B, numerator, denominator, exact_mass)
         for B in _all_B(H.shape[0])),
        Fraction(0),
    ) / scale ** H.shape[1]

    state_lower, state_upper, _ = scaled_state_weight_interval(
        H, syndrome, target, lower, upper, scale_lower, numerator, denominator,
    )
    exact_target = _exact_collapsed_weight(
        H, syndrome, target, numerator, denominator, exact_mass,
    ) / scale ** H.shape[1]
    assert Fraction.from_float(state_lower) <= exact_target
    assert exact_target <= Fraction.from_float(state_upper)

    for depth in range(H.shape[0] + 1):
        expected = sum(
            (_exact_collapsed_weight(H, syndrome, B, numerator, denominator, exact_mass)
             for B in _all_B(H.shape[0])
             if np.array_equal(B[:depth], target[:depth])),
            Fraction(0),
        ) / scale ** H.shape[1]
        upper_bound = partial_rows_scaled_upper(
            H, syndrome, target[:depth], tables[depth], scale_lower,
            numerator, denominator,
        )
        assert expected <= Fraction.from_float(upper_bound)

        partition_upper, _, _ = row_prefix_partition_upper(
            H, syndrome, depth, tables[depth], scale_lower,
            numerator, denominator, width_cap=12,
        )
        assert exact_total <= Fraction.from_float(partition_upper)


def test_b_tail_purity_bound_keeps_the_same_sector_cross_term():
    retained = np.asarray([5.0, 4.0])
    tail = np.asarray([1.0, 0.0])
    actual = float(np.dot(retained + tail, retained + tail) / (retained + tail).sum() ** 2)
    lower, upper = purity_interval_from_sector_tail(retained, float(tail.sum()))
    assert lower <= actual <= upper

    # This tempting sector-disjoint expression is false for a B-tail: its
    # omitted mode can land in the already retained first logical sector.
    invalid_without_cross_term = (
        float(np.dot(retained, retained)) + float(tail.sum()) ** 2
    ) / retained.sum() ** 2
    assert actual > invalid_without_cross_term
    q_lower, q_upper = q_top_interval_from_purity(lower, upper, logical_dimension=3)
    assert q_lower <= (8.0 * actual - 1.0) / 7.0 <= q_upper


def test_q_top_tail_interval_supports_the_uint64_logical_boundary():
    purity = 0.5
    lower, upper = q_top_interval_from_purity(purity, purity, logical_dimension=64)
    exact = ((1 << 64) * Fraction.from_float(purity) - 1) / ((1 << 64) - 1)
    assert lower <= float(exact) <= upper
