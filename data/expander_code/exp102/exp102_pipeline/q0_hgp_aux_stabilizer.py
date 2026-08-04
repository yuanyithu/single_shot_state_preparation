"""Exact auxiliary-coordinate stabilizer heatbaths for q=0 HGP states.

For ``H A + B H = Y``, choose an A-row ``i`` and a binary vector ``v`` of
length ``rank(H)``.  The joint update

``A[i, :] <- A[i, :] xor v H``
``B      <- B      xor H[:, i] v``

stays in the hard coset.  Its conditional has only the degree-three factor
graph of H, so min-fill variable elimination samples it exactly.  These
updates are stabilizer directions; their role is to make coordinated B moves
that a collapsed B-row heatbath cannot express in one step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .q0_hgp_full_row_gibbs_v0 import (
    FullRowEliminationPlan,
    _as_binary_matrix,
    _bucket_assignment_index,
    _stable_logaddexp,
    build_full_row_elimination_plan,
    compile_full_row_elimination_plan,
)


AUX_STABILIZER_VERSION = "exp102.q0_hgp_aux_stabilizer.v0"
AUX_STABILIZER_KERNEL = "exact_auxiliary_A_row_stabilizer_heatbath.v1"


class AuxiliaryStabilizerConflictError(ValueError):
    """Raised when an auxiliary stabilizer update loses its algebraic contract."""


@dataclass(frozen=True)
class _AuxiliaryTrace:
    compiled: object
    buckets: tuple[np.ndarray, ...]
    log_normalizer: float


def _validate_state_matrices(H, A, B):
    matrix = _as_binary_matrix(H)
    rows, columns = matrix.shape
    left = np.asarray(A, dtype=np.uint8)
    right = np.asarray(B, dtype=np.uint8)
    if left.shape != (columns, columns) or right.shape != (rows, rows):
        raise ValueError("auxiliary stabilizer state has the wrong HGP block shapes")
    if not np.all((left == 0) | (left == 1)) or not np.all((right == 0) | (right == 1)):
        raise ValueError("auxiliary stabilizer state must be binary")
    return matrix, np.ascontiguousarray(left), np.ascontiguousarray(right)


def auxiliary_stabilizer_delta(H, a_row, assignment):
    """Return the hard-coset-preserving delta for one A-row block assignment."""
    matrix = _as_binary_matrix(H)
    rows, columns = matrix.shape
    index = int(a_row)
    value = int(assignment)
    if not 0 <= index < columns or not 0 <= value < (1 << rows):
        raise ValueError("auxiliary stabilizer row or assignment is invalid")
    vector = np.asarray([(value >> bit) & 1 for bit in range(rows)], dtype=np.uint8)
    delta_a = np.zeros((columns, columns), dtype=np.uint8)
    delta_a[index] = (vector.astype(np.int64) @ matrix.astype(np.int64) % 2).astype(np.uint8)
    delta_b = (
        matrix[:, index:index + 1].astype(np.int64) @ vector[None, :].astype(np.int64) % 2
    ).astype(np.uint8)
    return delta_a, delta_b


def _initial_values(matrix, A, B, a_row, plan, log_odds):
    """Build exact unary/factor log weights for one auxiliary block."""
    rows, columns = matrix.shape
    column = matrix[:, int(a_row)]
    values = []
    for variable in range(rows):
        zero_weight = int(B[column.astype(bool), variable].sum())
        one_weight = int((1 - B[column.astype(bool), variable]).sum())
        values.append(np.asarray(
            (float(log_odds) * zero_weight, float(log_odds) * one_weight), dtype=np.float64,
        ))
    for output in range(columns):
        scope = plan.factor_scopes[output]
        factor = np.empty(1 << len(scope), dtype=np.float64)
        current = int(A[int(a_row), output])
        for assignment in range(factor.size):
            factor[assignment] = float(log_odds) * (current ^ (assignment.bit_count() & 1))
        values.append(factor)
    return values


def _trace(matrix, A, B, a_row, plan, compiled, log_odds):
    values = _initial_values(matrix, A, B, a_row, plan, log_odds)
    if len(values) != compiled.initial_factor_count:
        raise AuxiliaryStabilizerConflictError("auxiliary factor count drifted")
    values.extend([None] * len(compiled.steps))
    buckets = []
    for step in compiled.steps:
        table = np.zeros(1 << len(step.scope), dtype=np.float64)
        for table_index in range(table.size):
            value = 0.0
            for factor_id, projection in zip(step.source_factor_ids, step.source_projection_indices):
                source = values[factor_id]
                if source is None:
                    raise AuxiliaryStabilizerConflictError("auxiliary elimination source vanished")
                value += float(source[int(projection[table_index])])
            table[table_index] = value
        if not np.all(np.isfinite(table)):
            raise AuxiliaryStabilizerConflictError("auxiliary factor table is non-finite")
        output = np.empty(step.zero_indices.size, dtype=np.float64)
        for output_index in range(output.size):
            output[output_index] = _stable_logaddexp(
                float(table[int(step.zero_indices[output_index])]),
                float(table[int(step.one_indices[output_index])]),
            )
        values[step.output_factor_id] = output
        buckets.append(table)
    log_normalizer = 0.0
    for factor_id in compiled.terminal_factor_ids:
        terminal = values[factor_id]
        if terminal is None or terminal.shape != (1,):
            raise AuxiliaryStabilizerConflictError("auxiliary terminal factor is invalid")
        log_normalizer += float(terminal[0])
    if not math.isfinite(log_normalizer):
        raise AuxiliaryStabilizerConflictError("auxiliary normalizer is non-finite")
    return _AuxiliaryTrace(compiled=compiled, buckets=tuple(buckets), log_normalizer=log_normalizer)


def _assignment_log_probability(trace, assignment):
    result = 0.0
    for step_index in range(len(trace.compiled.steps) - 1, -1, -1):
        step = trace.compiled.steps[step_index]
        table = trace.buckets[step_index]
        zero = _bucket_assignment_index(step.scope, step.variable, assignment, 0)
        one = _bucket_assignment_index(step.scope, step.variable, assignment, 1)
        normalizer = _stable_logaddexp(float(table[zero]), float(table[one]))
        result += float(table[one if ((int(assignment) >> step.variable) & 1) else zero]) - normalizer
    return result


def _sample_assignment(trace, rng):
    assignment = 0
    for step_index in range(len(trace.compiled.steps) - 1, -1, -1):
        step = trace.compiled.steps[step_index]
        table = trace.buckets[step_index]
        zero = _bucket_assignment_index(step.scope, step.variable, assignment, 0)
        one = _bucket_assignment_index(step.scope, step.variable, assignment, 1)
        log_zero = float(table[zero])
        log_one = float(table[one])
        probability_one = math.exp(log_one - _stable_logaddexp(log_zero, log_one))
        if rng.random() < probability_one:
            assignment |= 1 << step.variable
    return assignment


def auxiliary_stabilizer_conditional_probabilities(H, A, B, a_row, p):
    """Enumerate a small auxiliary block conditional for exact-oracle tests."""
    matrix, left, right = _validate_state_matrices(H, A, B)
    rows, _ = matrix.shape
    if rows > 16:
        raise ValueError("auxiliary conditional enumeration is restricted to r<=16")
    p = float(p)
    if not 0.0 < p < 0.5:
        raise ValueError("auxiliary p must lie in (0, .5)")
    plan = build_full_row_elimination_plan(matrix)
    compiled = compile_full_row_elimination_plan(plan)
    trace = _trace(matrix, left, right, a_row, plan, compiled, math.log(p / (1.0 - p)))
    values = np.asarray([
        math.exp(_assignment_log_probability(trace, assignment))
        for assignment in range(1 << rows)
    ], dtype=np.float64)
    values /= values.sum(dtype=np.float64)
    return values, trace.log_normalizer


def brute_force_auxiliary_stabilizer_conditional(H, A, B, a_row, p):
    """Direct energy enumeration for a small exact oracle."""
    matrix, left, right = _validate_state_matrices(H, A, B)
    rows, _ = matrix.shape
    if rows > 16:
        raise ValueError("auxiliary brute-force oracle is restricted to r<=16")
    p = float(p)
    if not 0.0 < p < 0.5:
        raise ValueError("auxiliary p must lie in (0, .5)")
    log_odds = math.log(p / (1.0 - p))
    log_weights = np.empty(1 << rows, dtype=np.float64)
    for assignment in range(log_weights.size):
        delta_a, delta_b = auxiliary_stabilizer_delta(matrix, a_row, assignment)
        log_weights[assignment] = log_odds * int((left ^ delta_a).sum() + (right ^ delta_b).sum())
    maximum = float(log_weights.max())
    values = np.exp(log_weights - maximum)
    normalizer = maximum + math.log(float(values.sum(dtype=np.float64)))
    values /= values.sum(dtype=np.float64)
    return values, normalizer


def auxiliary_stabilizer_row_heatbath(H, A, B, a_row, p, rng, *, plan=None, compiled=None):
    """Sample and apply one exact full auxiliary stabilizer block."""
    matrix, left, right = _validate_state_matrices(H, A, B)
    p = float(p)
    if not 0.0 < p < 0.5:
        raise ValueError("auxiliary p must lie in (0, .5)")
    if plan is None:
        plan = build_full_row_elimination_plan(matrix)
    if not isinstance(plan, FullRowEliminationPlan):
        raise TypeError("auxiliary plan has the wrong type")
    if compiled is None:
        compiled = compile_full_row_elimination_plan(plan)
    trace = _trace(matrix, left, right, a_row, plan, compiled, math.log(p / (1.0 - p)))
    assignment = _sample_assignment(trace, rng)
    delta_a, delta_b = auxiliary_stabilizer_delta(matrix, a_row, assignment)
    return np.ascontiguousarray(left ^ delta_a), np.ascontiguousarray(right ^ delta_b), assignment


def auxiliary_stabilizer_sweep(H, A, B, p, rng, *, plan=None, row_order=None):
    """Apply a deterministic or pre-frozen sequence of exact A-row blocks."""
    matrix, left, right = _validate_state_matrices(H, A, B)
    _, columns = matrix.shape
    if row_order is None:
        row_order = tuple(range(columns))
    row_order = tuple(int(value) for value in row_order)
    if any(not 0 <= value < columns for value in row_order):
        raise ValueError("auxiliary row order is invalid")
    if plan is None:
        plan = build_full_row_elimination_plan(matrix)
    compiled = compile_full_row_elimination_plan(plan)
    assignments = np.zeros(len(row_order), dtype=np.uint32)
    for offset, a_row in enumerate(row_order):
        left, right, assignment = auxiliary_stabilizer_row_heatbath(
            matrix, left, right, a_row, p, rng, plan=plan, compiled=compiled,
        )
        assignments[offset] = assignment
    return left, right, assignments
