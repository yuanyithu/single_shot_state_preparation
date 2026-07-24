"""Exact full-B-row Gibbs updates using deterministic variable elimination."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .q0_hgp_full_column_gibbs import (
    FullColumnGibbsConflictError,
    collapsed_a_syndromes,
)


FULL_ROW_GIBBS_VERSION = "exp102.q0_hgp_full_row_elimination.v0"
FULL_ROW_GIBBS_METHOD_ID = "RFRG-R24-VE12"


def _matrix_sha256(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(H.shape, dtype=">u8").tobytes())
    digest.update(H.tobytes())
    return digest.hexdigest()


def _factor_scopes(H):
    rows, columns = H.shape
    unary = [(variable,) for variable in range(rows)]
    parity = [
        tuple(int(value) for value in np.flatnonzero(H[:, column]))
        for column in range(columns)
    ]
    if any(not scope for scope in parity):
        raise ValueError("full-row Gibbs H contains an empty column")
    return tuple(unary + parity)


def _interaction_graph(variable_count, scopes):
    adjacency = [set() for _ in range(variable_count)]
    for scope in scopes:
        for left in scope:
            adjacency[left].update(value for value in scope if value != left)
    return adjacency


def _min_fill_order(variable_count, scopes):
    graph = _interaction_graph(variable_count, scopes)
    remaining = set(range(variable_count))
    order = []
    widths = []
    while remaining:
        candidates = []
        for variable in sorted(remaining):
            neighbors = sorted(graph[variable] & remaining)
            missing = sum(
                right not in graph[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1:]
            )
            candidates.append(((missing, len(neighbors), variable), variable, neighbors))
        _, variable, neighbors = min(candidates)
        order.append(variable)
        widths.append(len(neighbors))
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1:]:
                graph[left].add(right)
                graph[right].add(left)
        remaining.remove(variable)
    return tuple(order), tuple(widths)


@dataclass(frozen=True)
class FullRowEliminationPlan:
    variable_count: int
    factor_scopes: tuple
    elimination_order: tuple
    width_by_step: tuple
    h_sha256: str
    plan_sha256: str

    @property
    def induced_width(self):
        return max(self.width_by_step, default=0)

    @property
    def largest_factor_entries(self):
        return 1 << (self.induced_width + 1)

    def validate(self, H):
        H = np.asarray(H, dtype=np.uint8)
        if (
            H.ndim != 2
            or H.shape[0] != self.variable_count
            or _matrix_sha256(H) != self.h_sha256
            or _factor_scopes(H) != self.factor_scopes
        ):
            raise FullColumnGibbsConflictError("full-row elimination plan binding changed")
        return True

    def as_dict(self):
        return {
            "elimination_order": list(self.elimination_order),
            "factor_scopes": [list(scope) for scope in self.factor_scopes],
            "h_sha256": self.h_sha256,
            "induced_width": self.induced_width,
            "largest_factor_entries": self.largest_factor_entries,
            "plan_sha256": self.plan_sha256,
            "variable_count": self.variable_count,
            "version": FULL_ROW_GIBBS_VERSION,
            "width_by_step": list(self.width_by_step),
        }


def build_full_row_elimination_plan(H):
    H = np.ascontiguousarray(H, dtype=np.uint8)
    if (
        H.ndim != 2
        or not 1 <= H.shape[0] <= 24
        or H.shape[1] > 32
        or not np.all((H == 0) | (H == 1))
    ):
        raise ValueError("full-row Gibbs H is outside the supported binary shape")
    scopes = _factor_scopes(H)
    order, widths = _min_fill_order(H.shape[0], scopes)
    digest = hashlib.sha256()
    digest.update(_matrix_sha256(H).encode("ascii"))
    digest.update(np.asarray(order, dtype=">u4").tobytes())
    for scope in scopes:
        digest.update(np.asarray([len(scope), *scope], dtype=">u4").tobytes())
    return FullRowEliminationPlan(
        variable_count=H.shape[0],
        factor_scopes=scopes,
        elimination_order=order,
        width_by_step=widths,
        h_sha256=_matrix_sha256(H),
        plan_sha256=digest.hexdigest(),
    )


def _h_column_masks(H):
    masks = np.zeros(H.shape[1], dtype=np.uint32)
    for column in range(H.shape[1]):
        for row in np.flatnonzero(H[:, column]):
            masks[column] |= np.uint32(1) << np.uint32(row)
    return masks


def _current_row_mask(b_columns, row_index):
    row_bit = np.uint32(1) << np.uint32(row_index)
    value = np.uint32(0)
    for column, column_mask in enumerate(b_columns):
        if np.uint32(column_mask) & row_bit:
            value |= np.uint32(1) << np.uint32(column)
    return value


def _row_factor_tables(H, b_columns, a_syndromes, row_index, log_mass,
                       log_odds, plan):
    plan.validate(H)
    rows, columns = H.shape
    if (
        np.asarray(b_columns).shape != (rows,)
        or np.asarray(a_syndromes).shape != (columns,)
        or np.asarray(log_mass).shape != (1 << rows,)
        or not np.all(np.isfinite(log_mass))
        or not 0 <= int(row_index) < rows
        or not math.isfinite(float(log_odds))
    ):
        raise ValueError("full-row conditional input changed")
    old_row = _current_row_mask(b_columns, row_index)
    row_bit = np.uint32(1) << np.uint32(row_index)
    h_masks = _h_column_masks(H)
    factors = []
    for variable in range(rows):
        factors.append(((variable,), np.asarray([0.0, float(log_odds)])))
    for factor in range(columns):
        scope = plan.factor_scopes[rows + factor]
        old_parity = int(old_row & h_masks[factor]).bit_count() & 1
        base = np.uint32(a_syndromes[factor])
        if old_parity:
            base ^= row_bit
        values = np.empty((2,) * len(scope), dtype=np.float64)
        for category in range(1 << len(scope)):
            syndrome = base ^ (row_bit if category.bit_count() & 1 else np.uint32(0))
            assignment = tuple((category >> bit) & 1 for bit in range(len(scope)))
            values[assignment] = float(log_mass[int(syndrome)])
        factors.append((scope, values))
    return factors


def _expand_factor(values, scope, union_scope):
    shape = [1] * len(union_scope)
    for variable in scope:
        shape[union_scope.index(variable)] = 2
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _logsumexp(values, axis):
    maximum = np.max(values, axis=axis)
    expanded = np.expand_dims(maximum, axis=axis)
    return maximum + np.log(np.exp(values - expanded).sum(axis=axis))


def eliminate_full_row_conditional(H, b_columns, a_syndromes, row_index,
                                   log_mass, log_odds, *, plan=None):
    """Return log normalizer and backward conditionals for one row heatbath."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    plan = build_full_row_elimination_plan(H) if plan is None else plan
    factors = _row_factor_tables(
        H, b_columns, a_syndromes, row_index, log_mass, log_odds, plan,
    )
    records = []
    for variable in plan.elimination_order:
        bucket_indices = [
            index for index, (scope, _) in enumerate(factors)
            if variable in scope
        ]
        if not bucket_indices:
            raise FullColumnGibbsConflictError("full-row elimination bucket vanished")
        union_scope = tuple(sorted(set().union(*(
            factors[index][0] for index in bucket_indices
        ))))
        joint = np.zeros((2,) * len(union_scope), dtype=np.float64)
        for index in bucket_indices:
            scope, values = factors[index]
            joint += _expand_factor(values, scope, union_scope)
        axis = union_scope.index(variable)
        message_scope = tuple(value for value in union_scope if value != variable)
        message = _logsumexp(joint, axis)
        conditional = joint - _expand_factor(message, message_scope, union_scope)
        records.append((variable, union_scope, conditional))
        factors = [
            factor for index, factor in enumerate(factors)
            if index not in bucket_indices
        ]
        factors.append((message_scope, np.asarray(message, dtype=np.float64)))
    if any(scope for scope, _ in factors):
        raise FullColumnGibbsConflictError("full-row elimination left live variables")
    log_normalizer = sum(float(np.asarray(values)) for _, values in factors)
    if not math.isfinite(log_normalizer):
        raise FullColumnGibbsConflictError("full-row normalizer is non-finite")
    return log_normalizer, tuple(records)


def _eliminate_additive_statistics(factors, elimination_order,
                                   statistic_tables):
    """Integrate additive observables with the same exact elimination."""
    working = [
        (scope, np.asarray(values, dtype=np.float64),
         np.asarray(statistics, dtype=np.float64))
        for (scope, values), statistics in zip(factors, statistic_tables)
    ]
    statistic_count = statistic_tables[0].shape[-1]
    for variable in elimination_order:
        bucket_indices = [
            index for index, (scope, _, _) in enumerate(working)
            if variable in scope
        ]
        if not bucket_indices:
            raise FullColumnGibbsConflictError(
                "full-row statistic bucket vanished"
            )
        union_scope = tuple(sorted(set().union(*(
            working[index][0] for index in bucket_indices
        ))))
        joint_log = np.zeros((2,) * len(union_scope), dtype=np.float64)
        joint_statistics = np.zeros(
            (2,) * len(union_scope) + (statistic_count,), dtype=np.float64,
        )
        for index in bucket_indices:
            scope, values, statistics = working[index]
            joint_log += _expand_factor(values, scope, union_scope)
            expanded_shape = [1] * len(union_scope) + [statistic_count]
            for item in scope:
                expanded_shape[union_scope.index(item)] = 2
            joint_statistics += statistics.reshape(expanded_shape)
        axis = union_scope.index(variable)
        message_scope = tuple(item for item in union_scope if item != variable)
        message_log = _logsumexp(joint_log, axis)
        log_probabilities = joint_log - _expand_factor(
            message_log, message_scope, union_scope,
        )
        message_statistics = np.sum(
            np.exp(log_probabilities)[..., None] * joint_statistics, axis=axis,
        )
        working = [
            factor for index, factor in enumerate(working)
            if index not in bucket_indices
        ]
        working.append((message_scope, message_log, message_statistics))
    if any(scope for scope, _, _ in working):
        raise FullColumnGibbsConflictError(
            "full-row statistic elimination left live variables"
        )
    log_normalizer = sum(float(np.asarray(values)) for _, values, _ in working)
    expectations = np.sum(
        np.stack([
            np.asarray(statistics, dtype=np.float64).reshape(statistic_count)
            for _, _, statistics in working
        ]),
        axis=0,
    )
    return log_normalizer, expectations


def full_row_conditional_statistics(H, b_columns, a_syndromes, row_index,
                                    log_mass, log_odds, *, plan=None):
    """Return exact entropy and movement statistics for a row conditional."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    plan = build_full_row_elimination_plan(H) if plan is None else plan
    factors = _row_factor_tables(
        H, b_columns, a_syndromes, row_index, log_mass, log_odds, plan,
    )
    old_row = int(_current_row_mask(b_columns, row_index))
    statistic_tables = []
    for factor_index, (scope, values) in enumerate(factors):
        statistics = np.zeros(values.shape + (3,), dtype=np.float64)
        statistics[..., 0] = values
        if factor_index < H.shape[0]:
            variable = scope[0]
            old_bit = (old_row >> variable) & 1
            statistics[:, 1] = (old_bit, 1 - old_bit)
            statistics[:, 2] = (0.0, 1.0)
        statistic_tables.append(statistics)
    log_normalizer, expectations = _eliminate_additive_statistics(
        factors, plan.elimination_order, statistic_tables,
    )
    record_log_normalizer, records = eliminate_full_row_conditional(
        H, b_columns, a_syndromes, row_index, log_mass, log_odds, plan=plan,
    )
    if abs(log_normalizer - record_log_normalizer) > 1e-11:
        raise FullColumnGibbsConflictError(
            "full-row statistic normalizer changed"
        )
    old_assignment = tuple(
        (old_row >> variable) & 1 for variable in range(H.shape[0])
    )
    self_log_probability = _assignment_log_probability(
        records, old_assignment,
    )
    entropy_nats = log_normalizer - float(expectations[0])
    result = {
        "entropy_bits": entropy_nats / math.log(2.0),
        "entropy_nats": entropy_nats,
        "expected_hamming_change": float(expectations[1]),
        "expected_row_weight": float(expectations[2]),
        "log_normalizer": log_normalizer,
        "self_log_probability": self_log_probability,
        "self_probability": math.exp(self_log_probability),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FullColumnGibbsConflictError(
            "full-row conditional statistics are non-finite"
        )
    return result


def _assignment_log_probability(records, assignment):
    assignment = tuple(int(value) for value in assignment)
    value = 0.0
    for variable, scope, conditional in reversed(records):
        index = tuple(assignment[item] for item in scope)
        value += float(conditional[index])
    return value


def sample_full_row_mask(records, rng):
    """Backward-sample a row mask from frozen elimination conditionals."""
    assignment = {}
    for variable, scope, conditional in reversed(records):
        index0 = tuple(0 if item == variable else assignment[item] for item in scope)
        index1 = tuple(1 if item == variable else assignment[item] for item in scope)
        logits = np.asarray([
            conditional[index0], conditional[index1],
        ], dtype=np.float64)
        maximum = float(logits.max())
        weights = np.exp(logits - maximum)
        threshold = float(rng.random()) * float(weights.sum())
        assignment[variable] = int(threshold >= float(weights[0]))
    mask = np.uint32(0)
    for variable, value in assignment.items():
        if value:
            mask |= np.uint32(1) << np.uint32(variable)
    return mask


def full_row_conditional_log_weights(H, b_columns, a_syndromes, row_index,
                                     log_mass, log_odds):
    """Enumerate row log weights for small-code exact oracles."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    rows, columns = H.shape
    if rows > 16:
        raise ValueError("full-row enumeration is restricted to small codes")
    old_row = _current_row_mask(b_columns, row_index)
    row_bit = np.uint32(1) << np.uint32(row_index)
    h_masks = _h_column_masks(H)
    base = np.asarray(a_syndromes, dtype=np.uint32).copy()
    for factor in range(columns):
        if int(old_row & h_masks[factor]).bit_count() & 1:
            base[factor] ^= row_bit
    result = np.empty(1 << rows, dtype=np.float64)
    for candidate in range(result.size):
        value = candidate.bit_count() * float(log_odds)
        for factor in range(columns):
            syndrome = base[factor]
            if int(candidate & int(h_masks[factor])).bit_count() & 1:
                syndrome ^= row_bit
            value += float(log_mass[int(syndrome)])
        result[candidate] = value
    return result


def full_row_elimination_gibbs_update(
        b_columns, a_syndromes, H, syndrome, row_index, log_mass, log_odds,
        rng, *, plan=None):
    """Heatbath one complete B row and update cached A syndromes exactly."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    a_syndromes = np.asarray(a_syndromes, dtype=np.uint32)
    expected = collapsed_a_syndromes(H, syndrome, b_columns)
    if not np.array_equal(a_syndromes, expected):
        raise FullColumnGibbsConflictError("full-row cached syndromes changed")
    plan = build_full_row_elimination_plan(H) if plan is None else plan
    _, records = eliminate_full_row_conditional(
        H, b_columns, a_syndromes, row_index, log_mass, log_odds, plan=plan,
    )
    old = _current_row_mask(b_columns, row_index)
    new = sample_full_row_mask(records, rng)
    delta = old ^ new
    if delta:
        row_bit = np.uint32(1) << np.uint32(row_index)
        h_masks = _h_column_masks(H)
        for column in range(H.shape[0]):
            if delta & (np.uint32(1) << np.uint32(column)):
                b_columns[column] ^= row_bit
        for factor in range(H.shape[1]):
            if int(delta & h_masks[factor]).bit_count() & 1:
                a_syndromes[factor] ^= row_bit
    if not np.array_equal(
        a_syndromes, collapsed_a_syndromes(H, syndrome, b_columns),
    ):
        raise FullColumnGibbsConflictError("full-row update transcript changed")
    return bool(delta), int(int(delta).bit_count()), old, new
