"""Conservative collapsed-HGP mass envelopes for q=0 feasibility checks.

The routines here bound the exact B-marginal after the A columns have been
integrated out.  They are deliberately not an estimator: a useful q_top
result still needs a certified treatment of retained logical-character mass
and of every omitted B branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


class CollapsedTailBoundError(ValueError):
    """Raised when an interval-envelope input is not a valid HGP object."""


@dataclass(frozen=True)
class ProbabilityInterval:
    """Outward-rounded endpoints for one exact rational Bernoulli parameter."""

    lower: float
    upper: float


@dataclass(frozen=True)
class BinaryFactor:
    """A nonnegative binary factor stored in the explicit order of ``scope``."""

    scope: tuple[int, ...]
    values: np.ndarray


def _require(condition, message):
    if not condition:
        raise CollapsedTailBoundError(message)


def _round_down_scalar(value):
    return max(0.0, math.nextafter(float(value), -math.inf))


def _round_up_scalar(value):
    return math.nextafter(float(value), math.inf)


def _round_down_inplace(values):
    np.nextafter(values, -np.inf, out=values)
    np.maximum(values, 0.0, out=values)
    return values


def _round_up_inplace(values):
    np.nextafter(values, np.inf, out=values)
    return values


def rational_probability_interval(numerator, denominator):
    """Return an outward float interval for ``numerator / denominator``."""
    numerator = int(numerator)
    denominator = int(denominator)
    _require(0 < numerator < denominator, "Bernoulli probability must lie in (0, 1)")
    nearest = float(numerator / denominator)
    return ProbabilityInterval(
        lower=math.nextafter(nearest, -math.inf),
        upper=math.nextafter(nearest, math.inf),
    )


def _gf2_rank(matrix):
    matrix = np.asarray(matrix, dtype=np.uint8)
    _require(matrix.ndim == 2, "H must be two-dimensional")
    rows = []
    for row in matrix:
        value = 0
        for column, bit in enumerate(row):
            _require(int(bit) in (0, 1), "H must be binary")
            value |= int(bit) << column
        rows.append(value)
    rank = 0
    for column in range(matrix.shape[1]):
        pivot = next((index for index in range(rank, len(rows))
                      if (rows[index] >> column) & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for index in range(len(rows)):
            if index != rank and ((rows[index] >> column) & 1):
                rows[index] ^= rows[rank]
        rank += 1
        if rank == matrix.shape[0]:
            break
    return rank


def _validated_h(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    _require(H.ndim == 2 and H.shape[0] > 0 and H.shape[1] > 0,
             "H must have positive dimensions")
    _require(H.shape[0] <= 24, "interval mass table is capped at rank 24")
    _require(not np.any((H != 0) & (H != 1)), "H must be binary")
    _require(_gf2_rank(H) == H.shape[0], "H must have full row rank")
    _require(np.all(H.sum(axis=0) > 0), "every H column must be nonzero")
    return H


def _column_masks(H):
    masks = np.zeros(H.shape[1], dtype=np.uint32)
    for column in range(H.shape[1]):
        value = 0
        for row in np.flatnonzero(H[:, column]):
            value |= 1 << int(row)
        masks[column] = value
    return masks


def classical_coset_mass_interval(H, numerator, denominator):
    """Bound every ``Pr[H a=s]`` using directed float interval recurrence.

    ``numerator / denominator`` is supplied as an exact rational so a decimal
    parameter such as ``.04`` does not silently become an unspecified binary
    input.  The returned arrays enclose the exact rational probability of each
    classical syndrome.
    """
    H = _validated_h(H)
    numerator = int(numerator)
    denominator = int(denominator)
    _require(0 < numerator < denominator, "Bernoulli probability must lie in (0, 1)")
    p = rational_probability_interval(numerator, denominator)
    keep = rational_probability_interval(denominator - numerator, denominator)
    size = 1 << int(H.shape[0])
    indices = np.arange(size, dtype=np.uint32)
    partner_indices = np.empty(size, dtype=np.uint32)
    lower = np.zeros(size, dtype=np.float64)
    upper = np.zeros(size, dtype=np.float64)
    scratch_lower = np.empty(size, dtype=np.float64)
    scratch_upper = np.empty(size, dtype=np.float64)
    partner = np.empty(size, dtype=np.float64)
    lower[0] = 1.0
    upper[0] = 1.0
    for mask in _column_masks(H):
        np.bitwise_xor(indices, np.uint32(mask), out=partner_indices)

        np.multiply(lower, keep.lower, out=scratch_lower)
        _round_down_inplace(scratch_lower)
        np.take(lower, partner_indices, out=partner)
        np.multiply(partner, p.lower, out=partner)
        _round_down_inplace(partner)
        np.add(scratch_lower, partner, out=scratch_lower)
        _round_down_inplace(scratch_lower)

        np.multiply(upper, keep.upper, out=scratch_upper)
        _round_up_inplace(scratch_upper)
        np.take(upper, partner_indices, out=partner)
        np.multiply(partner, p.upper, out=partner)
        _round_up_inplace(partner)
        np.add(scratch_upper, partner, out=scratch_upper)
        _round_up_inplace(scratch_upper)

        lower, scratch_lower = scratch_lower, lower
        upper, scratch_upper = scratch_upper, upper
    _require(np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)),
             "interval mass recurrence produced a non-finite value")
    _require(np.all(lower >= 0.0) and np.all(upper > 0.0),
             "interval mass recurrence produced an invalid probability")
    _require(np.all(lower <= upper), "interval mass recurrence inverted an interval")
    return lower, upper


def prefix_max_upper_tables(upper_mass, depths):
    """Return max-mass tables after fixing the lowest-indexed syndrome bits."""
    upper_mass = np.asarray(upper_mass, dtype=np.float64)
    _require(upper_mass.ndim == 1 and upper_mass.size > 0,
             "upper mass table must be one-dimensional")
    rank = int(math.log2(upper_mass.size))
    _require((1 << rank) == upper_mass.size, "upper mass length must be a power of two")
    requested = {int(depth) for depth in depths}
    _require(all(0 <= depth <= rank for depth in requested),
             "requested prefix depth is outside the mass-table rank")
    current = upper_mass.copy()
    result = {}
    for depth in range(rank, -1, -1):
        if depth in requested:
            result[depth] = current.copy()
        if depth:
            half = 1 << (depth - 1)
            current = np.maximum(current[:half], current[half:])
    return result


def _prefix_mask(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    _require(not np.any((bits != 0) & (bits != 1)), "prefix bits must be binary")
    value = 0
    for index, bit in enumerate(bits):
        value |= int(bit) << index
    return value


def syndrome_masks_for_B(H, syndrome, B):
    """Return packed column syndromes for ``Y xor B H`` in little-endian rows."""
    H = _validated_h(H)
    r, n = H.shape
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)
    _require(syndrome.shape == (r, n), "HGP syndrome has the wrong shape")
    _require(B.shape == (r, r), "collapsed B matrix has the wrong shape")
    _require(not np.any((syndrome != 0) & (syndrome != 1)), "syndrome must be binary")
    _require(not np.any((B != 0) & (B != 1)), "B must be binary")
    values = syndrome ^ ((B.astype(np.int64) @ H.astype(np.int64)) % 2).astype(np.uint8)
    return np.asarray([_prefix_mask(values[:, column]) for column in range(n)], dtype=np.uint32)


def bernoulli_assignment_interval(one_count, zero_count, numerator, denominator):
    """Bound the exact iid probability of one fixed binary assignment."""
    one_count = int(one_count)
    zero_count = int(zero_count)
    _require(one_count >= 0 and zero_count >= 0, "assignment counts must be nonnegative")
    p = rational_probability_interval(numerator, denominator)
    keep = rational_probability_interval(int(denominator) - int(numerator), denominator)
    lower = 1.0
    upper = 1.0
    for _ in range(one_count):
        lower = _round_down_scalar(lower * p.lower)
        upper = _round_up_scalar(upper * p.upper)
    for _ in range(zero_count):
        lower = _round_down_scalar(lower * keep.lower)
        upper = _round_up_scalar(upper * keep.upper)
    return lower, upper


def scaled_state_weight_interval(H, syndrome, B, lower_mass, upper_mass,
                                 scale_lower, numerator, denominator):
    """Bound one collapsed B weight divided by ``scale_lower**n``."""
    H = _validated_h(H)
    r, n = H.shape
    lower_mass = np.asarray(lower_mass, dtype=np.float64)
    upper_mass = np.asarray(upper_mass, dtype=np.float64)
    _require(lower_mass.shape == upper_mass.shape == (1 << r,),
             "mass interval has the wrong shape")
    scale_lower = float(scale_lower)
    _require(math.isfinite(scale_lower) and scale_lower > 0.0,
             "scale lower endpoint must be positive")
    masks = syndrome_masks_for_B(H, syndrome, B)
    one_count = int(np.asarray(B, dtype=np.uint8).sum())
    lower, upper = bernoulli_assignment_interval(
        one_count, r * r - one_count, numerator, denominator,
    )
    for mask in masks:
        lower_factor = _round_down_scalar(float(lower_mass[int(mask)]) / scale_lower)
        upper_factor = _round_up_scalar(float(upper_mass[int(mask)]) / scale_lower)
        lower = _round_down_scalar(lower * lower_factor)
        upper = _round_up_scalar(upper * upper_factor)
    return lower, upper, masks


def partial_rows_scaled_upper(H, syndrome, B_prefix, prefix_upper_mass,
                              scale_lower, numerator, denominator):
    """Upper-bound the scaled mass of every B completion of fixed leading rows.

    The B prior of unassigned rows sums to one.  For each A-column likelihood
    factor we maximize over its remaining syndrome bits independently; this is
    conservative even though different factors share those B variables.
    """
    H = _validated_h(H)
    r, n = H.shape
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    B_prefix = np.asarray(B_prefix, dtype=np.uint8)
    _require(syndrome.shape == (r, n), "HGP syndrome has the wrong shape")
    _require(B_prefix.ndim == 2 and B_prefix.shape[1] == r,
             "B prefix has the wrong shape")
    depth = int(B_prefix.shape[0])
    _require(0 <= depth <= r, "B prefix has too many rows")
    _require(not np.any((B_prefix != 0) & (B_prefix != 1)), "B prefix must be binary")
    table = np.asarray(prefix_upper_mass, dtype=np.float64)
    _require(table.shape == (1 << depth,), "prefix max table has the wrong shape")
    scale_lower = float(scale_lower)
    _require(math.isfinite(scale_lower) and scale_lower > 0.0,
             "scale lower endpoint must be positive")
    one_count = int(B_prefix.sum())
    _, upper = bernoulli_assignment_interval(
        one_count, depth * r - one_count, numerator, denominator,
    )
    if depth == 0:
        prefix_values = np.zeros((0, n), dtype=np.uint8)
    else:
        prefix_values = syndrome[:depth] ^ (
            (B_prefix.astype(np.int64) @ H.astype(np.int64)) % 2
        ).astype(np.uint8)
    for column in range(n):
        mask = _prefix_mask(prefix_values[:, column])
        factor = _round_up_scalar(float(table[mask]) / scale_lower)
        upper = _round_up_scalar(upper * factor)
    return upper


def binary_elimination_plan(scopes, variable_count):
    """Return deterministic min-fill order and the largest induced width."""
    variable_count = int(variable_count)
    _require(variable_count >= 0, "variable count must be nonnegative")
    active = [set(int(value) for value in scope) for scope in scopes if scope]
    remaining = set(range(variable_count))
    order = []
    maximum_width = 0
    while remaining:
        choices = []
        for variable in sorted(remaining):
            touching = [scope for scope in active if variable in scope]
            union = set().union(*touching) if touching else {variable}
            choices.append((len(union) - 1, variable, touching, union))
        _, variable, touching, union = min(choices, key=lambda item: (item[0], item[1]))
        maximum_width = max(maximum_width, len(union) - 1)
        active = [scope for scope in active if variable not in scope]
        reduced = union - {variable}
        if reduced:
            active.append(reduced)
        remaining.remove(variable)
        order.append(variable)
    return tuple(order), int(maximum_width)


def _combined_factor_upper(factors):
    scope = tuple(sorted({variable for factor in factors for variable in factor.scope}))
    values = np.ones((2,) * len(scope), dtype=np.float64)
    positions = {variable: index for index, variable in enumerate(scope)}
    for factor in factors:
        expected = (2,) * len(factor.scope)
        _require(factor.values.shape == expected,
                 "binary factor shape does not match its scope")
        shape = [1] * len(scope)
        for index, variable in enumerate(factor.scope):
            shape[positions[variable]] = factor.values.shape[index]
        np.multiply(values, factor.values.reshape(shape), out=values)
        _round_up_inplace(values)
    return scope, values


def binary_factor_sum_upper(factors, variable_count, width_cap):
    """Contract nonnegative binary upper factors with directed rounding."""
    factors = [BinaryFactor(tuple(factor.scope), np.asarray(factor.values, dtype=np.float64))
               for factor in factors]
    _require(all(np.all(np.isfinite(factor.values)) and np.all(factor.values >= 0.0)
                 for factor in factors), "binary upper factor is invalid")
    order, width = binary_elimination_plan([factor.scope for factor in factors], variable_count)
    width_cap = int(width_cap)
    _require(width <= width_cap,
             f"binary factor upper bound needs induced width {width}, cap is {width_cap}")
    active = list(factors)
    for variable in order:
        touching = [factor for factor in active if variable in factor.scope]
        active = [factor for factor in active if variable not in factor.scope]
        if not touching:
            # An unconstrained binary variable contributes its two unit states.
            touching = [BinaryFactor((variable,), np.ones(2, dtype=np.float64))]
        scope, values = _combined_factor_upper(touching)
        axis = scope.index(variable)
        reduced = np.take(values, 0, axis=axis) + np.take(values, 1, axis=axis)
        if np.ndim(reduced) == 0:
            reduced = np.asarray(_round_up_scalar(reduced), dtype=np.float64)
        else:
            _round_up_inplace(reduced)
        reduced_scope = tuple(entry for entry in scope if entry != variable)
        active.append(BinaryFactor(reduced_scope, reduced))
    _, values = _combined_factor_upper(active)
    _require(values.shape == (), "factor elimination did not eliminate every variable")
    return float(values), order, width


def row_prefix_factor_scopes(H, depth):
    """Scopes of the factorized leading-row envelope at one fixed depth."""
    H = _validated_h(H)
    depth = int(depth)
    _require(0 <= depth <= H.shape[0], "row-prefix depth is outside H")
    r, n = H.shape
    scopes = [tuple([variable]) for variable in range(depth * r)]
    for column in range(n):
        neighbors = tuple(int(value) for value in np.flatnonzero(H[:, column]))
        scopes.append(tuple(
            row * r + neighbor
            for row in range(depth)
            for neighbor in neighbors
        ))
    return tuple(scopes)


def row_prefix_partition_upper(H, syndrome, depth, prefix_upper_mass,
                               scale_lower, numerator, denominator, width_cap):
    """Bound the full scaled B partition function after ``depth`` leading rows.

    The contraction is exact for the factorized envelope, not for the true
    posterior.  It is a feasibility diagnostic for whether that envelope is
    tight enough to become a useful branch-and-bound node bound.
    """
    H = _validated_h(H)
    r, n = H.shape
    depth = int(depth)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    _require(syndrome.shape == (r, n), "HGP syndrome has the wrong shape")
    table = np.asarray(prefix_upper_mass, dtype=np.float64)
    _require(table.shape == (1 << depth,), "prefix max table has the wrong shape")
    scale_lower = float(scale_lower)
    _require(math.isfinite(scale_lower) and scale_lower > 0.0,
             "scale lower endpoint must be positive")
    p = rational_probability_interval(numerator, denominator)
    keep = rational_probability_interval(int(denominator) - int(numerator), denominator)
    factors = [BinaryFactor((variable,), np.asarray([keep.upper, p.upper], dtype=np.float64))
               for variable in range(depth * r)]
    for column in range(n):
        neighbors = tuple(int(value) for value in np.flatnonzero(H[:, column]))
        scope = tuple(
            row * r + neighbor
            for row in range(depth)
            for neighbor in neighbors
        )
        values = np.empty((2,) * len(scope), dtype=np.float64)
        for assignment in np.ndindex(values.shape):
            mask = 0
            for row in range(depth):
                start = row * len(neighbors)
                parity = 0
                for local in range(len(neighbors)):
                    parity ^= int(assignment[start + local])
                mask |= (int(syndrome[row, column]) ^ parity) << row
            values[assignment] = _round_up_scalar(float(table[mask]) / scale_lower)
        factors.append(BinaryFactor(scope, values))
    return binary_factor_sum_upper(factors, depth * r, width_cap)


def purity_interval_from_sector_tail(retained_sector_masses, tail_upper,
                                     *, tail_is_sector_disjoint=False):
    """Bound posterior purity from retained sector masses and omitted mass.

    A B-tail is normally *not* sector-disjoint: omitted B modes may contribute
    to an already retained logical sector.  The default therefore keeps the
    necessary ``2 U max_i a_i`` cross term in the upper bound.  Callers may set
    ``tail_is_sector_disjoint`` only when they have explicitly partitioned by
    logical sector rather than by latent B coordinates.
    """
    retained = np.asarray(retained_sector_masses, dtype=np.float64)
    _require(retained.ndim == 1 and retained.size > 0,
             "retained sector masses must be a nonempty vector")
    _require(np.all(np.isfinite(retained)) and np.all(retained >= 0.0),
             "retained sector masses must be finite and nonnegative")
    tail_upper = float(tail_upper)
    _require(math.isfinite(tail_upper) and tail_upper >= 0.0,
             "tail upper bound must be finite and nonnegative")
    retained_exact = [Fraction.from_float(float(value)) for value in retained]
    tail_exact = Fraction.from_float(tail_upper)
    total = sum(retained_exact, Fraction(0))
    _require(total > 0, "retained sector masses must have positive total")
    square_sum = sum((value * value for value in retained_exact), Fraction(0))
    lower_exact = square_sum / (total + tail_exact) ** 2
    numerator_upper = square_sum + tail_exact ** 2
    if not tail_is_sector_disjoint:
        numerator_upper += 2 * tail_exact * max(retained_exact)
    upper_exact = min(Fraction(1), numerator_upper / total ** 2)
    lower = 0.0 if lower_exact == 0 else math.nextafter(float(lower_exact), -math.inf)
    upper = 1.0 if upper_exact == 1 else math.nextafter(float(upper_exact), math.inf)
    return lower, upper


def q_top_interval_from_purity(purity_lower, purity_upper, logical_dimension):
    """Map a purity interval through the exact normalized q_top relation."""
    purity_lower = float(purity_lower)
    purity_upper = float(purity_upper)
    logical_dimension = int(logical_dimension)
    _require(1 <= logical_dimension <= 64,
             "logical dimension must lie in the supported uint64 range")
    _require(math.isfinite(purity_lower) and math.isfinite(purity_upper)
             and purity_lower <= purity_upper,
             "purity interval is invalid")
    sector_count = 1 << logical_dimension
    lower_exact = (
        sector_count * Fraction.from_float(purity_lower) - 1
    ) / (sector_count - 1)
    upper_exact = (
        sector_count * Fraction.from_float(purity_upper) - 1
    ) / (sector_count - 1)
    lower = math.nextafter(float(lower_exact), -math.inf)
    upper = math.nextafter(float(upper_exact), math.inf)
    return lower, upper
