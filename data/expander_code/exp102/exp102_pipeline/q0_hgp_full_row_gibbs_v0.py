"""Exact full-row Gibbs updates for the collapsed q=0 HGP posterior.

For an HGP built from a full-row-rank classical matrix ``H`` and fixed hard
syndrome ``Y``, integrating out ``A`` gives

``pi(B | Y) proportional to odds**|B| * product_j M(Y_j xor (B H)_j)``,

where ``M(s)`` is the iid-Bernoulli classical coset mass.  Holding every row
of ``B`` except row ``i`` fixed leaves a binary parity-factor graph in the
``r`` entries of that row.  Each factor has the support of one column of
``H`` (degree three for the registered codes), so variable elimination gives
an exact full-row heatbath without a logical proposal catalogue.

This is deliberately a new, local-only diagnostic kernel.  It neither reuses
raw data from prior q=0 searches nor has any formal-production authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .q0_global import GlobalConflictError, validate_observable_frame
from .q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _pack_state,
    _qubit_signatures,
    _reference_sample_full_state,
    _section_and_kernel_masks,
    _state_label,
    build_classical_coset_mass,
    validate_hgp_wiring,
)
from .seeds import derive_seed

try:  # The reference path remains importable when Numba is unavailable.
    from .q0_hgp_collapsed import _hc_pack, _hc_random, _hc_sample_full_state
except ImportError:  # pragma: no cover
    _hc_pack = _hc_random = _hc_sample_full_state = None

try:  # pragma: no cover - availability is exercised by transcript tests.
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


FULL_ROW_GIBBS_VERSION = "exp102.q0_hgp_full_row_gibbs.v0"
FULL_ROW_GIBBS_RAW_VERSION = "exp102.q0_hgp_full_row_gibbs.raw.v0"
FULL_ROW_GIBBS_KERNEL = "exact_collapsed_full_row_variable_elimination.v1"
FULL_ROW_PLAN_VERSION = "exp102.q0_hgp_full_row_plan.v1"
FULL_ROW_METHOD_ID = "FRG-VE1"
FULL_ROW_L_START_RULE = "planted_xor_minimum_energy_reduced_logical_1to3.v1"
FULL_ROW_COUNTER_NAMES = (
    "full_row_updates",
    "row_changes",
    "row_changed_bits",
    "a_column_draws",
    "a_column_changes",
)


class FullRowGibbsConflictError(ValueError):
    """A full-row Gibbs identity, algebra, or numerical invariant failed."""


def select_low_energy_logical_start(epsilon, model, frame):
    """Build a legal, logically separated L start without reading chain output."""
    from .q0_global import reduce_logical_basis

    epsilon = np.asarray(epsilon, dtype=np.uint8)
    if epsilon.shape != (model.num_qubits,) or not np.all((epsilon == 0) | (epsilon == 1)):
        raise ValueError("full-row Gibbs planted state is invalid")
    reduced = reduce_logical_basis(model.logical_move_basis)
    reduced_residual = (
        model.H_check.astype(np.int64) @ reduced.T.astype(np.int64) % 2
    ).astype(np.uint8)
    if reduced_residual.any():
        raise FullRowGibbsConflictError("full-row reduced logical basis left the kernel")
    seen = set()
    selected = None
    candidate_count = 0
    for order in (1, 2, 3):
        for combination in itertools.combinations(range(reduced.shape[0]), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in seen:
                continue
            seen.add(packed)
            signature = int(_state_label(frame, move))
            if signature == 0:
                raise FullRowGibbsConflictError("full-row L candidate has zero signature")
            # Kernel membership is already checked for the reduced basis;
            # linearity then certifies every XOR combination below.
            candidate_count += 1
            start = epsilon ^ move
            key = (int(start.sum()), int(move.sum()), signature, packed)
            if selected is None or key < selected[0]:
                selected = (key, np.ascontiguousarray(move, dtype=np.uint8))
    if selected is None:
        raise FullRowGibbsConflictError("full-row Gibbs has no nontrivial L start")
    key, move = selected
    start = np.ascontiguousarray(epsilon ^ move, dtype=np.uint8)
    if _state_label(frame, start) == _state_label(frame, epsilon):
        raise FullRowGibbsConflictError("full-row L start lost logical separation")
    return start, {
        "rule": FULL_ROW_L_START_RULE,
        "candidate_orders": [1, 2, 3],
        "candidate_count": candidate_count,
        "selected_absolute_weight": int(key[0]),
        "selected_move_weight": int(key[1]),
        "selected_signature": int(key[2]),
        "selected_move_sha256": hashlib.sha256(move.tobytes()).hexdigest(),
    }


def _strict_sha256(value, name):
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _strict_commit(value):
    value = str(value)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source_commit must be a lowercase full Git SHA")
    return value


def _as_binary_matrix(H):
    matrix = np.asarray(H, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("full-row Gibbs H must be a nonempty matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("full-row Gibbs H must be binary")
    rows, columns = matrix.shape
    if rows > 24 or columns > 32:
        raise ValueError("full-row Gibbs supports classical dimensions r<=24, n<=32")
    return np.ascontiguousarray(matrix)


def h_matrix_sha256(H):
    """Hash matrix bytes together with shape, avoiding an ambiguous byte stream."""
    matrix = _as_binary_matrix(H)
    return sha256_json({
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "bits_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    })


@dataclass(frozen=True)
class FullRowEliminationPlan:
    """Deterministic min-fill structure used for every disorder on one code."""

    matrix_sha256: str
    rows: int
    columns: int
    factor_scopes: tuple[tuple[int, ...], ...]
    order: tuple[int, ...]
    widths: tuple[int, ...]
    bucket_scopes: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "matrix_sha256", _strict_sha256(
            self.matrix_sha256, "matrix_sha256",
        ))
        for name in ("rows", "columns"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"full-row plan {name} is invalid")
            object.__setattr__(self, name, int(value))
        if self.rows > 24 or self.columns > 32:
            raise ValueError("full-row plan dimensions exceed the supported cap")
        normalized_factors = tuple(
            tuple(int(variable) for variable in scope) for scope in self.factor_scopes
        )
        if len(normalized_factors) != self.columns:
            raise ValueError("full-row plan factor count does not match H columns")
        for scope in normalized_factors:
            if tuple(sorted(set(scope))) != scope or any(
                    variable < 0 or variable >= self.rows for variable in scope):
                raise ValueError("full-row plan factor scope is invalid")
        object.__setattr__(self, "factor_scopes", normalized_factors)
        order = tuple(int(variable) for variable in self.order)
        widths = tuple(int(width) for width in self.widths)
        buckets = tuple(tuple(int(variable) for variable in scope) for scope in self.bucket_scopes)
        if sorted(order) != list(range(self.rows)):
            raise ValueError("full-row plan elimination order is not a permutation")
        if len(widths) != self.rows or len(buckets) != self.rows:
            raise ValueError("full-row plan elimination metadata has the wrong length")
        for variable, width, scope in zip(order, widths, buckets):
            if width < 0 or len(scope) != width + 1 or variable not in scope:
                raise ValueError("full-row plan bucket metadata is invalid")
            if tuple(sorted(set(scope))) != scope or any(
                    entry < 0 or entry >= self.rows for entry in scope):
                raise ValueError("full-row plan bucket scope is invalid")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "bucket_scopes", buckets)

    @property
    def max_width(self):
        return max(self.widths, default=0)

    @property
    def max_table_entries(self):
        return 1 << (self.max_width + 1)

    @property
    def structural_table_cells(self):
        """Cells summed into pre-elimination buckets during one row update."""
        return int(sum(1 << len(scope) for scope in self.bucket_scopes))

    @property
    def output_table_cells(self):
        return int(sum(1 << (len(scope) - 1) for scope in self.bucket_scopes))

    def as_dict(self):
        return {
            "plan_version": FULL_ROW_PLAN_VERSION,
            "tie_break": "min_fill_then_degree_then_variable_index",
            "matrix_sha256": self.matrix_sha256,
            "rows": self.rows,
            "columns": self.columns,
            "factor_scopes": [list(scope) for scope in self.factor_scopes],
            "order": list(self.order),
            "widths": list(self.widths),
            "bucket_scopes": [list(scope) for scope in self.bucket_scopes],
            "max_width": self.max_width,
            "max_table_entries": self.max_table_entries,
            "structural_table_cells": self.structural_table_cells,
            "output_table_cells": self.output_table_cells,
        }

    @property
    def sha256(self):
        return sha256_json(self.as_dict())


def build_full_row_elimination_plan(H):
    """Freeze a min-fill plan using only the binary classical matrix ``H``."""
    matrix = _as_binary_matrix(H)
    rows, columns = matrix.shape
    factor_scopes = tuple(
        tuple(int(value) for value in np.flatnonzero(matrix[:, column]))
        for column in range(columns)
    )
    adjacency = [set() for _ in range(rows)]
    for scope in factor_scopes:
        for left_index, left in enumerate(scope):
            for right in scope[left_index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
    remaining = set(range(rows))
    order = []
    widths = []
    bucket_scopes = []
    while remaining:
        choices = []
        for variable in sorted(remaining):
            neighbors = sorted(adjacency[variable] & remaining)
            fill_count = 0
            for left_index, left in enumerate(neighbors):
                for right in neighbors[left_index + 1:]:
                    if right not in adjacency[left]:
                        fill_count += 1
            choices.append((fill_count, len(neighbors), variable, neighbors))
        _, _, variable, neighbors = min(choices, key=lambda item: item[:3])
        order.append(variable)
        widths.append(len(neighbors))
        bucket_scopes.append(tuple(sorted((variable, *neighbors))))
        for left_index, left in enumerate(neighbors):
            adjacency[left].discard(variable)
            for right in neighbors[left_index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        adjacency[variable].clear()
        remaining.remove(variable)
    return FullRowEliminationPlan(
        matrix_sha256=h_matrix_sha256(matrix),
        rows=rows,
        columns=columns,
        factor_scopes=factor_scopes,
        order=tuple(order),
        widths=tuple(widths),
        bucket_scopes=tuple(bucket_scopes),
    )


def _validate_plan(H, plan):
    matrix = _as_binary_matrix(H)
    if not isinstance(plan, FullRowEliminationPlan):
        raise TypeError("full-row Gibbs plan has the wrong type")
    if (plan.rows, plan.columns) != matrix.shape or plan.matrix_sha256 != h_matrix_sha256(matrix):
        raise FullRowGibbsConflictError("full-row Gibbs plan does not match H")
    expected = build_full_row_elimination_plan(matrix)
    if plan.as_dict() != expected.as_dict():
        raise FullRowGibbsConflictError("full-row Gibbs plan is not canonical for H")
    return matrix


@dataclass(frozen=True)
class _CompiledEliminationStep:
    variable: int
    scope: tuple[int, ...]
    source_factor_ids: tuple[int, ...]
    source_projection_indices: tuple[np.ndarray, ...]
    zero_indices: np.ndarray
    one_indices: np.ndarray
    output_factor_id: int


@dataclass(frozen=True)
class _CompiledEliminationPlan:
    plan: FullRowEliminationPlan
    steps: tuple[_CompiledEliminationStep, ...]
    terminal_factor_ids: tuple[int, ...]
    initial_factor_count: int


def _projection_indices(scope, factor_scope):
    table_size = 1 << len(scope)
    assignments = np.arange(table_size, dtype=np.uint32)
    positions = {variable: position for position, variable in enumerate(scope)}
    result = np.zeros(table_size, dtype=np.intp)
    for local_position, variable in enumerate(factor_scope):
        if variable not in positions:
            raise FullRowGibbsConflictError("factor escaped its elimination bucket")
        result |= (((assignments >> np.uint32(positions[variable])) & np.uint32(1))
                   .astype(np.intp) << local_position)
    return result


def _elimination_indices(scope, variable):
    output_scope = tuple(entry for entry in scope if entry != variable)
    position = scope.index(variable)
    positions = {entry: index for index, entry in enumerate(scope)}
    zero = np.zeros(1 << len(output_scope), dtype=np.intp)
    for assignment in range(zero.size):
        index = 0
        for local_position, entry in enumerate(output_scope):
            if (assignment >> local_position) & 1:
                index |= 1 << positions[entry]
        zero[assignment] = index
    one = zero | np.intp(1 << position)
    return zero, one


def compile_full_row_elimination_plan(plan):
    """Compile structure-only table projections; no disorder values enter here."""
    if not isinstance(plan, FullRowEliminationPlan):
        raise TypeError("full-row Gibbs plan has the wrong type")
    scopes = [tuple((variable,)) for variable in range(plan.rows)]
    scopes.extend(plan.factor_scopes)
    active = list(range(len(scopes)))
    steps = []
    for step_index, (variable, expected_scope) in enumerate(zip(
            plan.order, plan.bucket_scopes)):
        source_ids = tuple(factor_id for factor_id in active if variable in scopes[factor_id])
        if not source_ids:
            raise FullRowGibbsConflictError("full-row plan lost a unary factor")
        combined_scope = tuple(sorted({
            entry for factor_id in source_ids for entry in scopes[factor_id]
        }))
        if combined_scope != expected_scope:
            raise FullRowGibbsConflictError("full-row plan bucket disagrees with factor graph")
        source_maps = tuple(
            _projection_indices(combined_scope, scopes[factor_id])
            for factor_id in source_ids
        )
        zero, one = _elimination_indices(combined_scope, variable)
        output_id = len(scopes)
        scopes.append(tuple(entry for entry in combined_scope if entry != variable))
        selected = set(source_ids)
        active = [factor_id for factor_id in active if factor_id not in selected]
        active.append(output_id)
        steps.append(_CompiledEliminationStep(
            variable=variable,
            scope=combined_scope,
            source_factor_ids=source_ids,
            source_projection_indices=source_maps,
            zero_indices=zero,
            one_indices=one,
            output_factor_id=output_id,
        ))
    return _CompiledEliminationPlan(
        plan=plan,
        steps=tuple(steps),
        terminal_factor_ids=tuple(active),
        initial_factor_count=plan.rows + plan.columns,
    )


@dataclass(frozen=True)
class _NumbaEliminationArrays:
    """Packed structural tables consumed by the accelerated reference twin."""

    column_scope_variables: np.ndarray
    column_scope_lengths: np.ndarray
    scope_lengths: np.ndarray
    scope_variables: np.ndarray
    elimination_variables: np.ndarray
    source_starts: np.ndarray
    source_factor_ids: np.ndarray
    projection_maps: np.ndarray
    zero_indices: np.ndarray
    one_indices: np.ndarray
    output_factor_ids: np.ndarray
    max_table_entries: int
    total_factor_count: int


def _numba_elimination_arrays(compiled):
    """Convert the immutable plan into rectangular arrays without target data."""
    plan = compiled.plan
    max_scope = max((len(step.scope) for step in compiled.steps), default=1)
    max_table_entries = 1 << max_scope
    max_column_scope = max((len(scope) for scope in plan.factor_scopes), default=0)
    column_scope_variables = np.full(
        (plan.columns, max(1, max_column_scope)), -1, dtype=np.int32,
    )
    column_scope_lengths = np.zeros(plan.columns, dtype=np.int32)
    for column, scope in enumerate(plan.factor_scopes):
        column_scope_lengths[column] = len(scope)
        for position, variable in enumerate(scope):
            column_scope_variables[column, position] = variable
    scope_lengths = np.asarray(
        [len(step.scope) for step in compiled.steps], dtype=np.int32,
    )
    elimination_variables = np.asarray(
        [step.variable for step in compiled.steps], dtype=np.int32,
    )
    scope_variables = np.full((plan.rows, max_scope), -1, dtype=np.int32)
    zero_indices = np.full((plan.rows, max_table_entries), -1, dtype=np.int32)
    one_indices = np.full((plan.rows, max_table_entries), -1, dtype=np.int32)
    output_factor_ids = np.empty(plan.rows, dtype=np.int32)
    source_starts = np.zeros(plan.rows + 1, dtype=np.int32)
    source_count = sum(len(step.source_factor_ids) for step in compiled.steps)
    source_factor_ids = np.empty(source_count, dtype=np.int32)
    projection_maps = np.zeros((source_count, max_table_entries), dtype=np.int32)
    source_offset = 0
    for step_index, step in enumerate(compiled.steps):
        scope_variables[step_index, :len(step.scope)] = step.scope
        zero_indices[step_index, :step.zero_indices.size] = step.zero_indices
        one_indices[step_index, :step.one_indices.size] = step.one_indices
        output_factor_ids[step_index] = step.output_factor_id
        source_starts[step_index] = source_offset
        for factor_id, projection in zip(
                step.source_factor_ids, step.source_projection_indices):
            source_factor_ids[source_offset] = factor_id
            projection_maps[source_offset, :projection.size] = projection
            source_offset += 1
    source_starts[-1] = source_offset
    if source_offset != source_count:
        raise AssertionError("full-row Gibbs source-plan packing drifted")
    return _NumbaEliminationArrays(
        column_scope_variables=np.ascontiguousarray(column_scope_variables),
        column_scope_lengths=np.ascontiguousarray(column_scope_lengths),
        scope_lengths=np.ascontiguousarray(scope_lengths),
        scope_variables=np.ascontiguousarray(scope_variables),
        elimination_variables=np.ascontiguousarray(elimination_variables),
        source_starts=np.ascontiguousarray(source_starts),
        source_factor_ids=np.ascontiguousarray(source_factor_ids),
        projection_maps=np.ascontiguousarray(projection_maps),
        zero_indices=np.ascontiguousarray(zero_indices),
        one_indices=np.ascontiguousarray(one_indices),
        output_factor_ids=np.ascontiguousarray(output_factor_ids),
        max_table_entries=max_table_entries,
        total_factor_count=compiled.initial_factor_count + len(compiled.steps),
    )


if njit is not None and _hc_random is not None and _hc_sample_full_state is not None:
    @njit(cache=True, inline="always")
    def _frg_nb_logaddexp(left, right):
        if left > right:
            return left + math.log1p(math.exp(right - left))
        return right + math.log1p(math.exp(left - right))


    @njit(cache=True)
    def _frg_nb_full_sweep(b_columns, a_syndromes, H, log_mass, log_odds,
                           column_scope_variables, column_scope_lengths,
                           scope_lengths, scope_variables, elimination_variables,
                           source_starts,
                           source_factor_ids, projection_maps, zero_indices,
                           one_indices, output_factor_ids, values, buckets,
                           base_syndromes, rng_state, counters):
        rows = b_columns.size
        columns = a_syndromes.size
        for row_index in range(rows):
            row_mask = np.uint32(1) << np.uint32(row_index)
            old_assignment = 0
            for variable in range(rows):
                if b_columns[variable] & row_mask:
                    old_assignment |= 1 << variable
            for column in range(columns):
                parity = 0
                for position in range(column_scope_lengths[column]):
                    variable = column_scope_variables[column, position]
                    parity ^= (old_assignment >> variable) & 1
                base_syndromes[column] = a_syndromes[column]
                if parity:
                    base_syndromes[column] ^= row_mask
            for variable in range(rows):
                values[variable, 0] = 0.0
                values[variable, 1] = log_odds
            for column in range(columns):
                factor_id = rows + column
                factor_size = 1 << column_scope_lengths[column]
                for assignment in range(factor_size):
                    parity = 0
                    temporary = assignment
                    while temporary:
                        parity ^= temporary & 1
                        temporary >>= 1
                    syndrome = int(base_syndromes[column])
                    if parity:
                        syndrome ^= 1 << row_index
                    values[factor_id, assignment] = log_mass[syndrome]
            for step in range(rows):
                table_size = 1 << scope_lengths[step]
                source_start = source_starts[step]
                source_stop = source_starts[step + 1]
                for table_index in range(table_size):
                    value = 0.0
                    for source_index in range(source_start, source_stop):
                        factor_id = source_factor_ids[source_index]
                        projection = projection_maps[source_index, table_index]
                        value += values[factor_id, projection]
                    buckets[step, table_index] = value
                output_id = output_factor_ids[step]
                output_size = table_size >> 1
                for output_index in range(output_size):
                    left = buckets[step, zero_indices[step, output_index]]
                    right = buckets[step, one_indices[step, output_index]]
                    values[output_id, output_index] = _frg_nb_logaddexp(left, right)
            new_assignment = 0
            for step in range(rows - 1, -1, -1):
                variable = elimination_variables[step]
                zero = 0
                one = 0
                for position in range(scope_lengths[step]):
                    entry = scope_variables[step, position]
                    if entry == variable:
                        one |= 1 << position
                    elif (new_assignment >> entry) & 1:
                        zero |= 1 << position
                        one |= 1 << position
                log_zero = buckets[step, zero]
                log_one = buckets[step, one]
                probability_one = math.exp(log_one - _frg_nb_logaddexp(log_zero, log_one))
                if _hc_random(rng_state) < probability_one:
                    new_assignment |= 1 << variable
            delta = old_assignment ^ new_assignment
            counters[0] += 1
            if delta:
                counters[1] += 1
                temporary = delta
                while temporary:
                    counters[2] += 1
                    temporary &= temporary - 1
                for variable in range(rows):
                    if (delta >> variable) & 1:
                        b_columns[variable] ^= row_mask
                        for column in range(columns):
                            if H[variable, column]:
                                a_syndromes[column] ^= row_mask


    @njit(cache=True)
    def _run_full_row_numba_stage(state, b_columns, a_syndromes, rng_state,
                                  sweeps, record, H, section_masks,
                                  kernel_combinations, log_mass, log_odds,
                                  odds_powers, qubit_signatures,
                                  column_scope_variables,
                                  column_scope_lengths, scope_lengths,
                                  scope_variables, elimination_variables,
                                  source_starts,
                                  source_factor_ids, projection_maps,
                                  zero_indices, one_indices,
                                  output_factor_ids, total_factor_count,
                                  max_table_entries):
        rows, columns = H.shape
        packed = np.empty(
            (sweeps if record else 0, (state.size + 7) // 8), dtype=np.uint8,
        )
        b_trace = np.empty((sweeps if record else 0, rows), dtype=np.uint32)
        a_trace = np.empty((sweeps if record else 0, columns), dtype=np.uint32)
        labels = np.empty(sweeps, dtype=np.uint64)
        weights = np.empty(sweeps, dtype=np.int32)
        counters = np.zeros(5, dtype=np.int64)
        previous_state = state.copy()
        values = np.empty((total_factor_count, max_table_entries), dtype=np.float64)
        buckets = np.empty((rows, max_table_entries), dtype=np.float64)
        base_syndromes = np.empty(columns, dtype=np.uint32)
        candidate_weights = np.empty(kernel_combinations.size, dtype=np.float64)
        for sweep in range(sweeps):
            _frg_nb_full_sweep(
                b_columns, a_syndromes, H, log_mass, log_odds,
                column_scope_variables, column_scope_lengths, scope_lengths,
                scope_variables, elimination_variables, source_starts, source_factor_ids,
                projection_maps, zero_indices, one_indices, output_factor_ids,
                values, buckets, base_syndromes, rng_state, counters,
            )
            previous_state[:] = state
            label, weight = _hc_sample_full_state(
                state, previous_state, b_columns, a_syndromes, rng_state,
                columns, rows, section_masks, kernel_combinations, odds_powers,
                qubit_signatures, candidate_weights, counters,
            )
            labels[sweep] = label
            weights[sweep] = weight
            if record:
                _hc_pack(state, packed[sweep])
                b_trace[sweep] = b_columns
                a_trace[sweep] = a_syndromes
        return state, b_columns, a_syndromes, packed, b_trace, a_trace, labels, weights, counters
else:  # pragma: no cover
    _run_full_row_numba_stage = None


def _row_base_syndromes(H, b_columns, a_syndromes, row_index, factor_scopes):
    rows, columns = H.shape
    if not 0 <= int(row_index) < rows:
        raise ValueError("full-row Gibbs row index is invalid")
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    a_syndromes = np.asarray(a_syndromes, dtype=np.uint32)
    if b_columns.shape != (rows,) or a_syndromes.shape != (columns,):
        raise ValueError("full-row Gibbs collapsed state has the wrong shape")
    row_mask = np.uint32(1 << int(row_index))
    old_assignment = 0
    for variable in range(rows):
        if b_columns[variable] & row_mask:
            old_assignment |= 1 << variable
    base = a_syndromes.copy()
    for column, scope in enumerate(factor_scopes):
        parity = 0
        for variable in scope:
            parity ^= (old_assignment >> variable) & 1
        if parity:
            base[column] ^= row_mask
    return old_assignment, base


def _initial_factor_values(plan, base_syndromes, row_index, log_mass, log_odds,
                           power=1.0):
    """Build one row's factors under an optional scalar energy power.

    Keeping the scale scalar avoids materializing one 2**r mass table per
    tempering rung.  The default is exactly the original cold conditional.
    """
    power = float(power)
    if not math.isfinite(power) or power < 0.0:
        raise FullRowGibbsConflictError("full-row Gibbs power is invalid")
    values = [np.asarray((0.0, power * float(log_odds)), dtype=np.float64)
              for _ in range(plan.rows)]
    row_mask = 1 << int(row_index)
    for column, scope in enumerate(plan.factor_scopes):
        factor = np.empty(1 << len(scope), dtype=np.float64)
        base = int(base_syndromes[column])
        for assignment in range(factor.size):
            syndrome = base ^ (row_mask if assignment.bit_count() & 1 else 0)
            if not 0 <= syndrome < len(log_mass):
                raise FullRowGibbsConflictError("full-row Gibbs syndrome lookup is invalid")
            factor[assignment] = power * float(log_mass[syndrome])
        values.append(factor)
    return values


def _stable_logaddexp(left, right):
    """Scalar form shared with the Numba kernel to preserve draw decisions."""
    if left > right:
        return left + math.log1p(math.exp(right - left))
    return right + math.log1p(math.exp(left - right))


@dataclass(frozen=True)
class _RowEliminationTrace:
    compiled: _CompiledEliminationPlan
    buckets: tuple[np.ndarray, ...]
    log_normalizer: float


def _row_elimination_trace(H, plan, compiled, b_columns, a_syndromes,
                           row_index, log_mass, log_odds, power=1.0):
    _, base = _row_base_syndromes(
        H, b_columns, a_syndromes, row_index, plan.factor_scopes,
    )
    values = _initial_factor_values(
        plan, base, row_index, log_mass, log_odds, power=power,
    )
    if len(values) != compiled.initial_factor_count:
        raise FullRowGibbsConflictError("full-row Gibbs factor count drifted")
    values.extend([None] * len(compiled.steps))
    buckets = []
    for step in compiled.steps:
        table = np.zeros(1 << len(step.scope), dtype=np.float64)
        for table_index in range(table.size):
            value = 0.0
            for factor_id, projection in zip(
                    step.source_factor_ids, step.source_projection_indices):
                source = values[factor_id]
                if source is None:
                    raise FullRowGibbsConflictError("full-row Gibbs elimination source vanished")
                value += float(source[int(projection[table_index])])
            table[table_index] = value
        if not np.all(np.isfinite(table)):
            raise FullRowGibbsConflictError("full-row Gibbs factor table is non-finite")
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
        value = values[factor_id]
        if value is None or value.shape != (1,):
            raise FullRowGibbsConflictError("full-row Gibbs terminal factor is invalid")
        log_normalizer += float(value[0])
    if not math.isfinite(log_normalizer):
        raise FullRowGibbsConflictError("full-row Gibbs row normalizer is non-finite")
    return _RowEliminationTrace(
        compiled=compiled,
        buckets=tuple(buckets),
        log_normalizer=log_normalizer,
    )


def _bucket_assignment_index(scope, variable, assignment, value):
    index = 0
    for position, entry in enumerate(scope):
        bit = value if entry == variable else ((int(assignment) >> entry) & 1)
        index |= int(bit) << position
    return index


def _sample_assignment_from_trace(trace, rng):
    assignment = 0
    for step_index in range(len(trace.compiled.steps) - 1, -1, -1):
        step = trace.compiled.steps[step_index]
        table = trace.buckets[step_index]
        zero = _bucket_assignment_index(step.scope, step.variable, assignment, 0)
        one = _bucket_assignment_index(step.scope, step.variable, assignment, 1)
        log_zero = float(table[zero])
        log_one = float(table[one])
        normalizer = _stable_logaddexp(log_zero, log_one)
        probability_one = math.exp(log_one - normalizer)
        if rng.random() < probability_one:
            assignment |= 1 << step.variable
    return assignment


def _assignment_probability_from_trace(trace, assignment):
    """Exact probability of one assignment under backward VE sampling.

    This is an oracle helper used by tests to verify the backward sampler,
    rather than relying on empirical draws from a small state space.
    """
    log_probability = 0.0
    for step_index in range(len(trace.compiled.steps) - 1, -1, -1):
        step = trace.compiled.steps[step_index]
        table = trace.buckets[step_index]
        zero = _bucket_assignment_index(step.scope, step.variable, assignment, 0)
        one = _bucket_assignment_index(step.scope, step.variable, assignment, 1)
        normalizer = _stable_logaddexp(float(table[zero]), float(table[one]))
        selected = one if ((int(assignment) >> step.variable) & 1) else zero
        log_probability += float(table[selected]) - normalizer
    return math.exp(log_probability)


def _assignment_log_probability_from_trace(trace, assignment):
    log_probability = 0.0
    for step_index in range(len(trace.compiled.steps) - 1, -1, -1):
        step = trace.compiled.steps[step_index]
        table = trace.buckets[step_index]
        zero = _bucket_assignment_index(step.scope, step.variable, assignment, 0)
        one = _bucket_assignment_index(step.scope, step.variable, assignment, 1)
        normalizer = _stable_logaddexp(float(table[zero]), float(table[one]))
        selected = one if ((int(assignment) >> step.variable) & 1) else zero
        log_probability += float(table[selected]) - normalizer
    return log_probability


def full_row_conditional_probabilities(H, plan, b_columns, a_syndromes,
                                       row_index, log_mass, log_odds):
    """Return the exact full-row conditional from VE for small-code oracles."""
    matrix = _validate_plan(H, plan)
    compiled = compile_full_row_elimination_plan(plan)
    log_mass = np.asarray(log_mass, dtype=np.float64)
    if log_mass.ndim != 1 or not np.all(np.isfinite(log_mass)):
        raise ValueError("full-row Gibbs log mass table is invalid")
    trace = _row_elimination_trace(
        matrix, plan, compiled, b_columns, a_syndromes, row_index,
        log_mass, float(log_odds),
    )
    probabilities = np.asarray([
        _assignment_probability_from_trace(trace, assignment)
        for assignment in range(1 << plan.rows)
    ], dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise FullRowGibbsConflictError("full-row Gibbs conditional is invalid")
    probabilities /= probabilities.sum(dtype=np.float64)
    return probabilities, trace.log_normalizer


def full_row_current_assignment_log_probability(H, plan, b_columns,
                                                a_syndromes, row_index,
                                                log_mass, log_odds):
    """Exact log chance that a row heatbath leaves its current row unchanged.

    This is a local diagnostic, not a convergence metric.  It lets a runtime
    probe distinguish an accidental short no-change trace from a conditional
    probability that is numerically pinned at one.
    """
    matrix = _validate_plan(H, plan)
    compiled = compile_full_row_elimination_plan(plan)
    log_mass = np.asarray(log_mass, dtype=np.float64)
    old_assignment, _ = _row_base_syndromes(
        matrix, b_columns, a_syndromes, row_index, plan.factor_scopes,
    )
    trace = _row_elimination_trace(
        matrix, plan, compiled, b_columns, a_syndromes, row_index,
        log_mass, float(log_odds),
    )
    return _assignment_log_probability_from_trace(trace, old_assignment)


def brute_force_full_row_conditional(H, b_columns, a_syndromes, row_index,
                                     log_mass, log_odds):
    """Direct row enumeration, retained only as an exact small-code oracle."""
    matrix = _as_binary_matrix(H)
    rows, _ = matrix.shape
    if rows > 16:
        raise ValueError("brute-force row oracle is restricted to r<=16")
    scopes = tuple(
        tuple(int(value) for value in np.flatnonzero(matrix[:, column]))
        for column in range(matrix.shape[1])
    )
    _, base = _row_base_syndromes(
        matrix, b_columns, a_syndromes, row_index, scopes,
    )
    log_mass = np.asarray(log_mass, dtype=np.float64)
    log_weights = np.empty(1 << rows, dtype=np.float64)
    row_mask = 1 << int(row_index)
    for assignment in range(log_weights.size):
        value = assignment.bit_count() * float(log_odds)
        for column, scope in enumerate(scopes):
            parity = 0
            for variable in scope:
                parity ^= (assignment >> variable) & 1
            value += float(log_mass[int(base[column]) ^ (row_mask if parity else 0)])
        log_weights[assignment] = value
    maximum = float(log_weights.max())
    probabilities = np.exp(log_weights - maximum)
    normalizer = maximum + math.log(float(probabilities.sum(dtype=np.float64)))
    probabilities /= probabilities.sum(dtype=np.float64)
    return probabilities, normalizer


def _full_row_gibbs_update_unchecked(b_columns, a_syndromes, matrix, plan,
                                     compiled, log_mass, log_odds, rng):
    """Inner sweep after the one-time model and plan identity checks."""
    rows, columns = matrix.shape
    b_columns = np.asarray(b_columns, dtype=np.uint32)
    a_syndromes = np.asarray(a_syndromes, dtype=np.uint32)
    if b_columns.shape != (rows,) or a_syndromes.shape != (columns,):
        raise ValueError("full-row Gibbs collapsed state has the wrong shape")
    counters = np.zeros(3, dtype=np.int64)
    for row_index in range(rows):
        old_assignment, _ = _row_base_syndromes(
            matrix, b_columns, a_syndromes, row_index, plan.factor_scopes,
        )
        trace = _row_elimination_trace(
            matrix, plan, compiled, b_columns, a_syndromes, row_index,
            log_mass, log_odds,
        )
        new_assignment = _sample_assignment_from_trace(trace, rng)
        delta = old_assignment ^ new_assignment
        counters[0] += 1
        if delta:
            counters[1] += 1
            counters[2] += delta.bit_count()
            row_mask = np.uint32(1 << row_index)
            for variable in range(rows):
                if (delta >> variable) & 1:
                    b_columns[variable] ^= row_mask
                    for column in np.flatnonzero(matrix[variable]):
                        a_syndromes[int(column)] ^= row_mask
    return counters


def full_row_gibbs_update(b_columns, a_syndromes, H, plan, compiled,
                          log_mass, log_odds, rng):
    """Perform a deterministic ascending sweep of exact full-row heatbaths."""
    matrix = _validate_plan(H, plan)
    if not isinstance(compiled, _CompiledEliminationPlan) or compiled.plan != plan:
        raise FullRowGibbsConflictError("full-row compiled plan does not match plan")
    return _full_row_gibbs_update_unchecked(
        b_columns, a_syndromes, matrix, plan, compiled, log_mass, log_odds, rng,
    )


@dataclass(frozen=True)
class FullRowGibbsConfig:
    """Fixed-clock local diagnostic configuration for the exact row kernel."""

    p: float
    burn_sweeps: int
    measurement_sweeps: int
    method_id: str = FULL_ROW_METHOD_ID
    row_schedule: str = "ascending"

    def __post_init__(self):
        p = float(self.p)
        if not math.isfinite(p) or not 0.0 < p < 0.5:
            raise ValueError("full-row Gibbs p must lie in (0, 0.5)")
        object.__setattr__(self, "p", p)
        for name in ("burn_sweeps", "measurement_sweeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.measurement_sweeps % 8:
            raise ValueError("measurement_sweeps must divide into eight time blocks")
        if self.method_id != FULL_ROW_METHOD_ID:
            raise ValueError("full-row Gibbs method ID is not frozen")
        if self.row_schedule != "ascending":
            raise ValueError("full-row Gibbs row schedule is not frozen")

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": self.p,
            "burn_sweeps": self.burn_sweeps,
            "measurement_sweeps": self.measurement_sweeps,
            "row_schedule": self.row_schedule,
            "kernel": FULL_ROW_GIBBS_KERNEL,
            "a_update": "exact_column_conditional_draw_after_each_B_sweep",
        }


@dataclass(frozen=True)
class FullRowGibbsSeedIdentity:
    """Disjoint identity for one legal P/U/L diagnostic trajectory."""

    source_commit: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    method_id: str
    resource_tier: str
    init_family: str
    trajectory_index: int
    trajectory_namespace: str

    def __post_init__(self):
        object.__setattr__(self, "source_commit", _strict_commit(self.source_commit))
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            object.__setattr__(self, name, _strict_sha256(getattr(self, name), name))
        if self.method_id != FULL_ROW_METHOD_ID:
            raise ValueError("full-row Gibbs seed method ID is invalid")
        if not isinstance(self.resource_tier, str) or not self.resource_tier:
            raise ValueError("full-row Gibbs resource tier is empty")
        if self.init_family not in ("P", "U", "L"):
            raise ValueError("full-row Gibbs initialization family must be P, U, or L")
        if isinstance(self.trajectory_index, bool) or int(self.trajectory_index) < 0:
            raise ValueError("full-row Gibbs trajectory index is invalid")
        object.__setattr__(self, "trajectory_index", int(self.trajectory_index))
        if not isinstance(self.trajectory_namespace, str) or not self.trajectory_namespace:
            raise ValueError("full-row Gibbs trajectory namespace is empty")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            FULL_ROW_GIBBS_VERSION,
            self.trajectory_namespace,
            self.source_commit,
            self.config_sha256,
            self.registry_sha256,
            self.cell_fingerprint,
            self.method_id,
            self.resource_tier,
            self.init_family,
            self.trajectory_index,
            str(stage),
            str(role),
            int(index),
        )

    def as_dict(self):
        return {
            "source_commit": self.source_commit,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_fingerprint": self.cell_fingerprint,
            "method_id": self.method_id,
            "resource_tier": self.resource_tier,
            "init_family": self.init_family,
            "trajectory_index": self.trajectory_index,
            "trajectory_namespace": self.trajectory_namespace,
        }


def _basis_seen(labels, k):
    result = np.zeros((int(k), 2), dtype=np.uint8)
    for label in np.asarray(labels, dtype=np.uint64):
        for bit in range(int(k)):
            result[bit, int((label >> np.uint64(bit)) & np.uint64(1))] = 1
    return result


def _run_reference_stage(state, b_columns, a_syndromes, *, sweeps, record,
                         model, frame, H, plan, compiled, log_mass, log_odds,
                         section_masks, kernel_combinations, odds_powers,
                         qubit_signatures, rng):
    rows, columns = H.shape
    labels = np.empty(sweeps, dtype=np.uint64)
    weights = np.empty(sweeps, dtype=np.int32)
    packed = np.empty(
        (sweeps if record else 0, (model.num_qubits + 7) // 8), dtype=np.uint8,
    )
    b_trace = np.empty((sweeps if record else 0, rows), dtype=np.uint32)
    a_trace = np.empty((sweeps if record else 0, columns), dtype=np.uint32)
    counters = np.zeros(len(FULL_ROW_COUNTER_NAMES), dtype=np.int64)
    current = np.asarray(state, dtype=np.uint8).copy()
    for sweep in range(sweeps):
        counters[:3] += _full_row_gibbs_update_unchecked(
            b_columns, a_syndromes, H, plan, compiled, log_mass, log_odds, rng,
        )
        previous_a = current[:columns * columns].copy()
        current, label, weight = _reference_sample_full_state(
            current, b_columns, a_syndromes, rng, columns, rows,
            section_masks, kernel_combinations,
            odds_powers, qubit_signatures,
        )
        counters[3] += columns
        new_a = current[:columns * columns].reshape(columns, columns)
        old_a = previous_a.reshape(columns, columns)
        counters[4] += sum(
            int(np.any(new_a[:, column] != old_a[:, column]))
            for column in range(columns)
        )
        labels[sweep] = label
        weights[sweep] = weight
        if record:
            packed[sweep] = _pack_state(current)
            b_trace[sweep] = b_columns
            a_trace[sweep] = a_syndromes
    return current, b_columns, a_syndromes, packed, b_trace, a_trace, labels, weights, counters


def run_full_row_gibbs_trajectory(model, frame, H, syndrome, config,
                                  seed_identity, initial_state, *,
                                  engine="reference", mass=None, plan=None):
    """Run one fixed-budget exact-row trajectory from any legal hard-coset start."""
    if engine not in ("reference", "numba"):
        raise ValueError("full-row Gibbs engine must be reference or numba")
    if engine == "numba" and _run_full_row_numba_stage is None:
        raise RuntimeError("Numba is required for the accelerated full-row Gibbs engine")
    if not isinstance(config, FullRowGibbsConfig):
        raise TypeError("full-row Gibbs config has the wrong type")
    if not isinstance(seed_identity, FullRowGibbsSeedIdentity):
        raise TypeError("full-row Gibbs seed identity has the wrong type")
    if config.method_id != seed_identity.method_id:
        raise FullRowGibbsConflictError("full-row Gibbs config/seed method mismatch")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise FullRowGibbsConflictError("full-row Gibbs observable frame mismatch") from exc
    matrix = _as_binary_matrix(H)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    if syndrome.shape != (model.num_checks,) or state.shape != (model.num_qubits,):
        raise ValueError("full-row Gibbs syndrome or state shape mismatch")
    plan = build_full_row_elimination_plan(matrix) if plan is None else plan
    _validate_plan(matrix, plan)
    compiled = compile_full_row_elimination_plan(plan)
    numba_plan = _numba_elimination_arrays(compiled) if engine == "numba" else None
    b_columns, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, matrix)
    mass_engine = "numba" if engine == "numba" else "reference"
    expected_mass = build_classical_coset_mass(matrix, config.p, engine=mass_engine)
    mass = expected_mass if mass is None else np.asarray(mass, dtype=np.float64)
    if (mass.shape != expected_mass.shape or not np.all(np.isfinite(mass))
            or np.any(mass <= 0.0) or not np.array_equal(mass, expected_mass)):
        raise FullRowGibbsConflictError("full-row Gibbs mass table does not match H and p")
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    log_odds = math.log(config.p / (1.0 - config.p))
    odds_powers = np.ones(matrix.shape[1] + 1, dtype=np.float64)
    for index in range(1, odds_powers.size):
        odds_powers[index] = odds_powers[index - 1] * config.p / (1.0 - config.p)
    qubit_signatures = _qubit_signatures(frame)
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    initial = state.copy()
    section_masks, kernel_combinations = _section_and_kernel_masks(matrix)
    if engine == "reference":
        burn = _run_reference_stage(
            state, b_columns, a_syndromes, sweeps=config.burn_sweeps, record=False,
            model=model, frame=frame, H=matrix, plan=plan, compiled=compiled,
            log_mass=log_mass, log_odds=log_odds, section_masks=section_masks,
            kernel_combinations=kernel_combinations, odds_powers=odds_powers,
            qubit_signatures=qubit_signatures,
            rng=PortablePrng(seed_identity.seed("burn")),
        )
    else:
        burn = _run_full_row_numba_stage(
            state, b_columns, a_syndromes,
            PortablePrng(seed_identity.seed("burn")).state_array(),
            config.burn_sweeps, False, matrix, section_masks, kernel_combinations,
            log_mass, log_odds, odds_powers, qubit_signatures,
            numba_plan.column_scope_variables, numba_plan.column_scope_lengths,
            numba_plan.scope_lengths, numba_plan.scope_variables,
            numba_plan.elimination_variables, numba_plan.source_starts,
            numba_plan.source_factor_ids, numba_plan.projection_maps,
            numba_plan.zero_indices, numba_plan.one_indices,
            numba_plan.output_factor_ids, numba_plan.total_factor_count,
            numba_plan.max_table_entries,
        )
    state, b_columns, a_syndromes = burn[0], burn[1], burn[2]
    burn_endpoint = state.copy()
    if engine == "reference":
        measured = _run_reference_stage(
            state, b_columns, a_syndromes, sweeps=config.measurement_sweeps, record=True,
            model=model, frame=frame, H=matrix, plan=plan, compiled=compiled,
            log_mass=log_mass, log_odds=log_odds, section_masks=section_masks,
            kernel_combinations=kernel_combinations, odds_powers=odds_powers,
            qubit_signatures=qubit_signatures,
            rng=PortablePrng(seed_identity.seed("measurement")),
        )
    else:
        measured = _run_full_row_numba_stage(
            state, b_columns, a_syndromes,
            PortablePrng(seed_identity.seed("measurement")).state_array(),
            config.measurement_sweeps, True, matrix, section_masks, kernel_combinations,
            log_mass, log_odds, odds_powers, qubit_signatures,
            numba_plan.column_scope_variables, numba_plan.column_scope_lengths,
            numba_plan.scope_lengths, numba_plan.scope_variables,
            numba_plan.elimination_variables, numba_plan.source_starts,
            numba_plan.source_factor_ids, numba_plan.projection_maps,
            numba_plan.zero_indices, numba_plan.one_indices,
            numba_plan.output_factor_ids, numba_plan.total_factor_count,
            numba_plan.max_table_entries,
        )
    state, packed, b_trace, a_trace, labels, weights = (
        measured[0], measured[3], measured[4], measured[5], measured[6], measured[7]
    )
    unpacked = np.unpackbits(
        packed, axis=1, count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    if residuals.any():
        raise FullRowGibbsConflictError("full-row Gibbs emitted a state outside the hard coset")
    replay_labels = np.asarray([_state_label(frame, row) for row in unpacked], dtype=np.uint64)
    if not np.array_equal(labels, replay_labels) or not np.array_equal(
            weights, unpacked.sum(axis=1)):
        raise FullRowGibbsConflictError("full-row Gibbs cached labels or weights drifted")
    return {
        "raw_version": FULL_ROW_GIBBS_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "plan_json": canonical_json(plan.as_dict()),
        "plan_sha256": plan.sha256,
        "mass_sha256": hashlib.sha256(np.asarray(mass, dtype=">f8").tobytes()).hexdigest(),
        "initial_state_packed": _pack_state(initial),
        "burn_state_packed": _pack_state(burn_endpoint),
        "final_state_packed": _pack_state(state),
        "measurement_states_packed": packed,
        "measurement_b_columns": b_trace,
        "measurement_a_syndromes": a_trace,
        "burn_labels": burn[6],
        "measurement_labels": labels,
        "measurement_weights": weights,
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_sweeps // 8,
        ),
        "burn_counters": burn[8],
        "measurement_counters": measured[8],
        "burn_basis_seen": _basis_seen(burn[6], model.k),
        "initial_label": _state_label(frame, initial),
        "burn_label": _state_label(frame, burn_endpoint),
        "final_label": _state_label(frame, state),
        "engine": engine,
    }
