"""Uniform-anchored replica exchange for the collapsed q=0 HGP posterior.

This is deliberately different from the historical collapsed power ladder.
At an intermediate value ``lambda`` it targets

``pi_lambda(B) proportional to exp(lambda * S(B))``,

where ``S(B)`` is the *complete* collapsed log density: the B-bit odds term
and the integrated A-column likelihood.  Thus lambda zero is exactly uniform
over every B bit.  That endpoint is refreshed exactly, while every positive
rung uses an exact full-row conditional heatbath.  Adjacent replica swaps use
the corresponding complete-score Metropolis ratio.

The endpoint at lambda one is the q=0 posterior after drawing A from its exact
conditional distribution.  The module is a local feasibility implementation;
it has no formal or posterior-reporting authority by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .q0_global import GlobalConflictError, validate_observable_frame
from .q0_hgp_collapsed import (
    _bits_to_mask,
    _initial_collapsed_masks,
    _pack_state,
    _qubit_signatures,
    _reference_sample_full_state,
    _section_and_kernel_masks,
    _state_label,
    build_classical_coset_mass,
    validate_hgp_wiring,
)
from .q0_hgp_full_row_gibbs_v0 import (
    _as_binary_matrix,
    _row_base_syndromes,
    _row_elimination_trace,
    _sample_assignment_from_trace,
    build_full_row_elimination_plan,
    compile_full_row_elimination_plan,
)
from .seeds import derive_seed

try:  # The reference engine remains importable when Numba is unavailable.
    from .q0_hgp_collapsed import (
        _hc_pack,
        _hc_popcount,
        _hc_random,
        _hc_sample_full_state,
    )
    from .q0_hgp_full_row_gibbs_v0 import _frg_nb_logaddexp, _numba_elimination_arrays
except ImportError:  # pragma: no cover
    _hc_pack = _hc_popcount = _hc_random = _hc_sample_full_state = None
    _frg_nb_logaddexp = _numba_elimination_arrays = None

try:  # pragma: no cover - the accelerated path has dedicated transcript tests.
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


UNIFORM_ANCHOR_PT_VERSION = "exp102.q0_hgp_uniform_anchor_pt.v0"
UNIFORM_ANCHOR_PT_RAW_VERSION = "exp102.q0_hgp_uniform_anchor_pt.raw.v0"
UNIFORM_ANCHOR_PT_KERNEL = (
    "uniform_endpoint_full_collapsed_energy_replica_exchange.v1"
)
UNIFORM_ANCHOR_PT_COUNTER_NAMES = (
    "positive_row_updates",
    "positive_row_changes",
    "positive_row_changed_bits",
    "hot_uniform_refreshes",
    "hot_uniform_refresh_changed_bits",
    "cold_a_column_draws",
)


class UniformAnchorPtConflictError(ValueError):
    """A target, algebra, identity, or replay invariant has failed."""


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


def uniform_anchor_lambda_schedule(num_replicas):
    """Return a frozen cosine ladder ordered from exact-uniform to cold.

    Cosine endpoint clustering is independent of all sampler output.  It keeps
    the large score fluctuations near both endpoints from being represented by
    one disproportionately large exchange interval.
    """
    count = int(num_replicas)
    if count < 2:
        raise ValueError("uniform-anchor ladder needs at least two replicas")
    denominator = count - 1
    values = np.asarray(
        [0.5 * (1.0 - math.cos(math.pi * index / denominator))
         for index in range(count)],
        dtype=np.float64,
    )
    values[0], values[-1] = 0.0, 1.0
    if np.any(values[1:] <= values[:-1]):
        raise AssertionError("uniform-anchor ladder lost strict ordering")
    return values


def _float64_sha256(values):
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=">f8").tobytes(order="C")
    ).hexdigest()


@dataclass(frozen=True)
class UniformAnchorReplicaExchangeConfig:
    """Frozen resource and transition parameters for one UARE candidate."""

    p: float
    burn_rounds: int
    measurement_rounds: int
    num_replicas: int
    positive_row_updates_per_round: int = 1
    method_id: str = ""

    def __post_init__(self):
        p = float(self.p)
        if not math.isfinite(p) or not 0.0 < p < 0.5:
            raise ValueError("uniform-anchor p must lie in (0, 0.5)")
        object.__setattr__(self, "p", p)
        for name in (
                "burn_rounds", "measurement_rounds", "num_replicas",
                "positive_row_updates_per_round"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.burn_rounds <= 0 or self.measurement_rounds <= 0:
            raise ValueError("uniform-anchor clocks must be positive")
        if self.measurement_rounds % 8:
            raise ValueError("uniform-anchor measurement must divide into eight blocks")
        if not 2 <= self.num_replicas <= 128:
            raise ValueError("uniform-anchor replicas must lie in [2, 128]")
        if not 1 <= self.positive_row_updates_per_round <= 24:
            raise ValueError("uniform-anchor row updates must lie in [1, 24]")
        expected = (
            f"UARE{self.num_replicas:02d}"
            f"-R{self.positive_row_updates_per_round}"
        )
        if self.method_id and self.method_id != expected:
            raise ValueError("uniform-anchor method ID does not match its parameters")
        object.__setattr__(self, "method_id", expected)

    @property
    def lambda_values(self):
        return uniform_anchor_lambda_schedule(self.num_replicas)

    @property
    def lambda_sha256(self):
        return _float64_sha256(self.lambda_values)

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": self.p,
            "burn_rounds": self.burn_rounds,
            "measurement_rounds": self.measurement_rounds,
            "num_replicas": self.num_replicas,
            "positive_row_updates_per_round": self.positive_row_updates_per_round,
            "lambda_values": self.lambda_values.tolist(),
            "lambda_schedule": "cosine_endpoint_cluster_v1",
            "kernel": UNIFORM_ANCHOR_PT_KERNEL,
            "hot_endpoint": "exact_uniform_B_refresh",
            "tempered_term": "complete_collapsed_log_density",
        }


@dataclass(frozen=True)
class UniformAnchorReplicaExchangeSeedIdentity:
    """Disjoint replayable identity for a P/U/L local trajectory."""

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
        if not isinstance(self.method_id, str) or not self.method_id.startswith("UARE"):
            raise ValueError("uniform-anchor seed method ID is invalid")
        if not isinstance(self.resource_tier, str) or not self.resource_tier:
            raise ValueError("uniform-anchor resource tier is empty")
        if self.init_family not in ("P", "U", "L"):
            raise ValueError("uniform-anchor initial family must be P, U, or L")
        if isinstance(self.trajectory_index, bool) or int(self.trajectory_index) < 0:
            raise ValueError("uniform-anchor trajectory index is invalid")
        object.__setattr__(self, "trajectory_index", int(self.trajectory_index))
        if not isinstance(self.trajectory_namespace, str) or not self.trajectory_namespace:
            raise ValueError("uniform-anchor trajectory namespace is empty")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            UNIFORM_ANCHOR_PT_VERSION,
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


def collapsed_complete_score(b_columns, a_syndromes, log_mass, log_odds):
    """Return the nonconstant complete collapsed log density S(B)."""
    score = 0.0
    for value in np.asarray(b_columns, dtype=np.uint32):
        score += int(value).bit_count() * float(log_odds)
    for value in np.asarray(a_syndromes, dtype=np.uint32):
        score += float(log_mass[int(value)])
    if not math.isfinite(score):
        raise UniformAnchorPtConflictError("collapsed complete score is non-finite")
    return score


def full_energy_swap_log_acceptance(lambda_lower, lambda_upper,
                                    score_lower, score_upper):
    """Metropolis log ratio for a swap of complete-energy replicas."""
    lower = float(lambda_lower)
    upper = float(lambda_upper)
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("uniform-anchor swap lambdas are not ordered")
    result = (lower - upper) * (float(score_upper) - float(score_lower))
    if not math.isfinite(result):
        raise UniformAnchorPtConflictError("uniform-anchor swap ratio is non-finite")
    return result


def _y_columns(syndrome, H):
    rows, columns = H.shape
    values = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    return np.asarray(
        [_bits_to_mask(values[:, column]) for column in range(columns)],
        dtype=np.uint32,
    )


def _recompute_a_syndromes(y_columns, b_columns, factor_neighbors):
    result = np.asarray(y_columns, dtype=np.uint32).copy()
    for b_column, value in enumerate(np.asarray(b_columns, dtype=np.uint32)):
        for factor in factor_neighbors[b_column]:
            result[factor] ^= value
    return result


def _factor_neighbors(H):
    return tuple(
        tuple(int(value) for value in np.flatnonzero(H[row]))
        for row in range(H.shape[0])
    )


def _reference_uniform_b_refresh(b_columns, a_syndromes, y_columns,
                                 factor_neighbors, rng):
    """Draw every B bit iid fair and rebuild its deterministic A syndromes."""
    changed_bits = 0
    rows = int(b_columns.size)
    for column in range(rows):
        old = int(b_columns[column])
        value = 0
        for row in range(rows):
            if rng.random() < 0.5:
                value |= 1 << row
        b_columns[column] = np.uint32(value)
        changed_bits += (old ^ value).bit_count()
    a_syndromes[:] = _recompute_a_syndromes(
        y_columns, b_columns, factor_neighbors,
    )
    return changed_bits


def _reference_one_row_heatbath(b_columns, a_syndromes, H, plan, compiled,
                                factor_neighbors, row_index, log_mass,
                                log_odds, power, rng):
    """Apply one exact scaled-energy row conditional in place."""
    old_assignment, _ = _row_base_syndromes(
        H, b_columns, a_syndromes, row_index, plan.factor_scopes,
    )
    trace = _row_elimination_trace(
        H, plan, compiled, b_columns, a_syndromes, row_index,
        log_mass, log_odds, power=power,
    )
    new_assignment = _sample_assignment_from_trace(trace, rng)
    delta = old_assignment ^ new_assignment
    if not delta:
        return 0, 0
    row_mask = np.uint32(1 << int(row_index))
    for variable in range(H.shape[0]):
        if (delta >> variable) & 1:
            b_columns[variable] ^= row_mask
            for factor in factor_neighbors[variable]:
                a_syndromes[factor] ^= row_mask
    return 1, int(delta).bit_count()


def _reference_stage(b_states, a_states, state, *, rounds, round_offset,
                     config, lambdas, log_mass, log_odds,
                     H, plan, compiled, factor_neighbors, y_columns,
                     rung_rngs, swap_rng, observation_rng, section_masks,
                     kernel_combinations, odds_powers, qubit_signatures,
                     record):
    """Run one fixed-clock phase and optionally retain cold full states."""
    replicas = int(lambdas.size)
    rows = int(H.shape[0])
    columns = int(H.shape[1])
    score = np.empty(replicas, dtype=np.float64)
    for rung in range(replicas):
        score[rung] = collapsed_complete_score(
            b_states[rung], a_states[rung], log_mass, log_odds,
        )
    labels = np.empty(rounds, dtype=np.uint64)
    weights = np.empty(rounds, dtype=np.int32)
    packed = np.empty(
        (rounds if record else 0, (state.size + 7) // 8), dtype=np.uint8,
    )
    b_trace = np.empty((rounds if record else 0, rows), dtype=np.uint32)
    a_trace = np.empty((rounds if record else 0, columns), dtype=np.uint32)
    score_trace = np.empty(rounds, dtype=np.float64)
    b_weight_trace = np.empty(rounds, dtype=np.int32)
    row_counters = np.zeros((replicas, 3), dtype=np.int64)
    hot_changed_bits = np.zeros(rounds, dtype=np.int32)
    swap_attempts = np.zeros(replicas - 1, dtype=np.int64)
    swap_accepts = np.zeros(replicas - 1, dtype=np.int64)
    current = np.asarray(state, dtype=np.uint8).copy()
    for round_local in range(rounds):
        round_index = int(round_offset) + round_local
        hot_changed_bits[round_local] = _reference_uniform_b_refresh(
            b_states[0], a_states[0], y_columns, factor_neighbors, rung_rngs[0],
        )
        score[0] = collapsed_complete_score(
            b_states[0], a_states[0], log_mass, log_odds,
        )
        for rung in range(1, replicas):
            for update in range(config.positive_row_updates_per_round):
                # This schedule is state-independent and visits every B row.
                row_index = (round_index + rung + update) % rows
                changed, changed_bits = _reference_one_row_heatbath(
                    b_states[rung], a_states[rung], H, plan, compiled,
                    factor_neighbors, row_index, log_mass, log_odds,
                    lambdas[rung], rung_rngs[rung],
                )
                row_counters[rung, 0] += 1
                row_counters[rung, 1] += changed
                row_counters[rung, 2] += changed_bits
            score[rung] = collapsed_complete_score(
                b_states[rung], a_states[rung], log_mass, log_odds,
            )
        for lower in range(round_index & 1, replicas - 1, 2):
            upper = lower + 1
            log_acceptance = full_energy_swap_log_acceptance(
                lambdas[lower], lambdas[upper], score[lower], score[upper],
            )
            swap_attempts[lower] += 1
            if log_acceptance >= 0.0 or swap_rng.random() < math.exp(log_acceptance):
                swap_accepts[lower] += 1
                temporary_b = b_states[lower].copy()
                b_states[lower] = b_states[upper]
                b_states[upper] = temporary_b
                temporary_a = a_states[lower].copy()
                a_states[lower] = a_states[upper]
                a_states[upper] = temporary_a
                score[lower], score[upper] = score[upper], score[lower]
        current, label, weight = _reference_sample_full_state(
            current, b_states[-1], a_states[-1], observation_rng, columns, rows,
            section_masks, kernel_combinations, odds_powers, qubit_signatures,
        )
        labels[round_local] = label
        weights[round_local] = weight
        score_trace[round_local] = score[-1]
        b_weight_trace[round_local] = sum(
            int(value).bit_count() for value in b_states[-1]
        )
        if record:
            packed[round_local] = _pack_state(current)
            b_trace[round_local] = b_states[-1]
            a_trace[round_local] = a_states[-1]
    return {
        "state": current,
        "labels": labels,
        "weights": weights,
        "packed": packed,
        "b_trace": b_trace,
        "a_trace": a_trace,
        "score_trace": score_trace,
        "b_weight_trace": b_weight_trace,
        "row_counters": row_counters,
        "hot_changed_bits": hot_changed_bits,
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
    }


if (njit is not None and _hc_pack is not None and _hc_popcount is not None
        and _hc_random is not None and _hc_sample_full_state is not None
        and _frg_nb_logaddexp is not None):
    @njit(cache=True, inline="always")
    def _uare_nb_complete_score(b_columns, a_syndromes, log_mass, log_odds):
        value = 0.0
        for column in range(b_columns.size):
            value += float(_hc_popcount(b_columns[column])) * log_odds
        for factor in range(a_syndromes.size):
            value += log_mass[int(a_syndromes[factor])]
        return value


    @njit(cache=True)
    def _uare_nb_uniform_refresh(b_columns, a_syndromes, y_columns, H, rng_state):
        rows, columns = H.shape
        changed_bits = 0
        for column in range(rows):
            old = b_columns[column]
            value = np.uint32(0)
            for row in range(rows):
                if _hc_random(rng_state) < 0.5:
                    value |= np.uint32(1) << np.uint32(row)
            b_columns[column] = value
            changed_bits += _hc_popcount(old ^ value)
        for factor in range(columns):
            a_syndromes[factor] = y_columns[factor]
        for column in range(rows):
            value = b_columns[column]
            for factor in range(columns):
                if H[column, factor]:
                    a_syndromes[factor] ^= value
        return changed_bits


    @njit(cache=True)
    def _uare_nb_one_row_heatbath(
            b_columns, a_syndromes, H, power, log_mass, log_odds,
            column_scope_variables, column_scope_lengths, scope_lengths,
            scope_variables, elimination_variables, source_starts,
            source_factor_ids, projection_maps, zero_indices, one_indices,
            output_factor_ids, values, buckets, base_syndromes, row_index,
            rng_state):
        rows, columns = H.shape
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
            values[variable, 1] = power * log_odds
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
                values[factor_id, assignment] = power * log_mass[syndrome]
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
            probability_one = math.exp(
                log_one - _frg_nb_logaddexp(log_zero, log_one),
            )
            if _hc_random(rng_state) < probability_one:
                new_assignment |= 1 << variable
        delta = old_assignment ^ new_assignment
        changed_bits = 0
        if delta:
            temporary = delta
            while temporary:
                changed_bits += 1
                temporary &= temporary - 1
            for variable in range(rows):
                if (delta >> variable) & 1:
                    b_columns[variable] ^= row_mask
                    for factor in range(columns):
                        if H[variable, factor]:
                            a_syndromes[factor] ^= row_mask
            return 1, changed_bits
        return 0, changed_bits


    @njit(cache=True)
    def _uare_nb_stage(
            state, b_states, a_states, rung_rng_states, swap_rng_state,
            observation_rng_state, lambdas, rounds, round_offset,
            positive_row_updates_per_round, record, H, y_columns, log_mass,
            log_odds, section_masks, kernel_combinations, odds_powers,
            qubit_signatures, column_scope_variables, column_scope_lengths,
            scope_lengths, scope_variables, elimination_variables, source_starts,
            source_factor_ids, projection_maps, zero_indices, one_indices,
            output_factor_ids, total_factor_count, max_table_entries):
        replicas = lambdas.size
        rows, columns = H.shape
        labels = np.empty(rounds, dtype=np.uint64)
        weights = np.empty(rounds, dtype=np.int32)
        packed = np.empty(
            (rounds if record else 0, (state.size + 7) // 8), dtype=np.uint8,
        )
        b_trace = np.empty((rounds if record else 0, rows), dtype=np.uint32)
        a_trace = np.empty((rounds if record else 0, columns), dtype=np.uint32)
        score_trace = np.empty(rounds, dtype=np.float64)
        b_weight_trace = np.empty(rounds, dtype=np.int32)
        row_counters = np.zeros((replicas, 3), dtype=np.int64)
        hot_changed_bits = np.zeros(rounds, dtype=np.int32)
        swap_attempts = np.zeros(replicas - 1, dtype=np.int64)
        swap_accepts = np.zeros(replicas - 1, dtype=np.int64)
        score = np.empty(replicas, dtype=np.float64)
        values = np.empty((total_factor_count, max_table_entries), dtype=np.float64)
        buckets = np.empty((rows, max_table_entries), dtype=np.float64)
        base_syndromes = np.empty(columns, dtype=np.uint32)
        candidate_weights = np.empty(kernel_combinations.size, dtype=np.float64)
        sample_counters = np.zeros(5, dtype=np.int64)
        previous_state = state.copy()
        for round_local in range(rounds):
            round_index = round_offset + round_local
            hot_changed_bits[round_local] = _uare_nb_uniform_refresh(
                b_states[0], a_states[0], y_columns, H, rung_rng_states[0],
            )
            score[0] = _uare_nb_complete_score(
                b_states[0], a_states[0], log_mass, log_odds,
            )
            for rung in range(1, replicas):
                for update in range(positive_row_updates_per_round):
                    row_index = (round_index + rung + update) % rows
                    changed, changed_bits = _uare_nb_one_row_heatbath(
                        b_states[rung], a_states[rung], H, lambdas[rung],
                        log_mass, log_odds, column_scope_variables,
                        column_scope_lengths, scope_lengths, scope_variables,
                        elimination_variables, source_starts, source_factor_ids,
                        projection_maps, zero_indices, one_indices,
                        output_factor_ids, values, buckets, base_syndromes,
                        row_index, rung_rng_states[rung],
                    )
                    row_counters[rung, 0] += 1
                    row_counters[rung, 1] += changed
                    row_counters[rung, 2] += changed_bits
                score[rung] = _uare_nb_complete_score(
                    b_states[rung], a_states[rung], log_mass, log_odds,
                )
            for lower in range(round_index & 1, replicas - 1, 2):
                upper = lower + 1
                log_acceptance = ((lambdas[lower] - lambdas[upper])
                                  * (score[upper] - score[lower]))
                swap_attempts[lower] += 1
                accept = log_acceptance >= 0.0
                if not accept:
                    accept = _hc_random(swap_rng_state) < math.exp(log_acceptance)
                if accept:
                    swap_accepts[lower] += 1
                    for index in range(rows):
                        temporary_b = b_states[lower, index]
                        b_states[lower, index] = b_states[upper, index]
                        b_states[upper, index] = temporary_b
                    for index in range(columns):
                        temporary_a = a_states[lower, index]
                        a_states[lower, index] = a_states[upper, index]
                        a_states[upper, index] = temporary_a
                    temporary_score = score[lower]
                    score[lower] = score[upper]
                    score[upper] = temporary_score
            previous_state[:] = state
            label, weight = _hc_sample_full_state(
                state, previous_state, b_states[replicas - 1],
                a_states[replicas - 1], observation_rng_state, columns, rows,
                section_masks, kernel_combinations, odds_powers,
                qubit_signatures, candidate_weights, sample_counters,
            )
            labels[round_local] = label
            weights[round_local] = weight
            score_trace[round_local] = score[replicas - 1]
            b_weight = 0
            for index in range(rows):
                b_weight += _hc_popcount(b_states[replicas - 1, index])
            b_weight_trace[round_local] = b_weight
            if record:
                _hc_pack(state, packed[round_local])
                b_trace[round_local] = b_states[replicas - 1]
                a_trace[round_local] = a_states[replicas - 1]
        return (state, labels, weights, packed, b_trace, a_trace, score_trace,
                b_weight_trace, row_counters, hot_changed_bits, swap_attempts,
                swap_accepts, sample_counters[3])
else:  # pragma: no cover
    _uare_nb_stage = None


def run_uniform_anchor_replica_exchange_trajectory(model, frame, H, syndrome,
                                                    config, seed_identity,
                                                    initial_state, *,
                                                    engine="reference",
                                                    mass=None):
    """Run one fixed UARE trajectory from a legal P, U, or L hard-coset state."""
    if engine not in ("reference", "numba"):
        raise ValueError("uniform-anchor engine must be reference or numba")
    if engine == "numba" and _uare_nb_stage is None:
        raise RuntimeError("Numba is required for accelerated uniform-anchor UARE")
    if not isinstance(config, UniformAnchorReplicaExchangeConfig):
        raise TypeError("uniform-anchor config has the wrong type")
    if not isinstance(seed_identity, UniformAnchorReplicaExchangeSeedIdentity):
        raise TypeError("uniform-anchor seed identity has the wrong type")
    if config.method_id != seed_identity.method_id:
        raise UniformAnchorPtConflictError("uniform-anchor config/seed method mismatch")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise UniformAnchorPtConflictError("uniform-anchor observable frame mismatch") from exc
    matrix = _as_binary_matrix(H)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    if syndrome.shape != (model.num_checks,) or state.shape != (model.num_qubits,):
        raise ValueError("uniform-anchor syndrome or state shape mismatch")
    initial = state.copy()
    plan = build_full_row_elimination_plan(matrix)
    compiled = compile_full_row_elimination_plan(plan)
    numba_plan = _numba_elimination_arrays(compiled) if engine == "numba" else None
    b_columns, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, matrix)
    mass_engine = "numba" if engine == "numba" else "reference"
    expected_mass = build_classical_coset_mass(matrix, config.p, engine=mass_engine)
    mass = expected_mass if mass is None else np.asarray(mass, dtype=np.float64)
    if (mass.shape != expected_mass.shape or not np.all(np.isfinite(mass))
            or np.any(mass <= 0.0) or not np.array_equal(mass, expected_mass)):
        raise UniformAnchorPtConflictError("uniform-anchor mass table does not match H and p")
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    log_odds = math.log(config.p / (1.0 - config.p))
    lambdas = config.lambda_values
    rows = int(matrix.shape[0])
    replicas = int(config.num_replicas)
    b_states = np.repeat(b_columns[None, :], replicas, axis=0)
    a_states = np.repeat(a_syndromes[None, :], replicas, axis=0)
    factor_neighbors = _factor_neighbors(matrix)
    y_columns = _y_columns(syndrome, matrix)
    section_masks, kernel_combinations = _section_and_kernel_masks(matrix)
    odds = config.p / (1.0 - config.p)
    odds_powers = np.ones(matrix.shape[1] + 1, dtype=np.float64)
    for index in range(1, odds_powers.size):
        odds_powers[index] = odds_powers[index - 1] * odds
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    signatures = _qubit_signatures(frame)
    if engine == "reference":
        burn = _reference_stage(
            b_states, a_states, state, rounds=config.burn_rounds, round_offset=0,
            config=config, lambdas=lambdas, log_mass=log_mass, log_odds=log_odds,
            H=matrix, plan=plan, compiled=compiled,
            factor_neighbors=factor_neighbors, y_columns=y_columns,
            rung_rngs=[PortablePrng(seed_identity.seed("burn", "rung", rung))
                       for rung in range(replicas)],
            swap_rng=PortablePrng(seed_identity.seed("burn", "swap")),
            observation_rng=PortablePrng(seed_identity.seed("burn", "observation")),
            section_masks=section_masks, kernel_combinations=kernel_combinations,
            odds_powers=odds_powers, qubit_signatures=signatures,
            record=False,
        )
        burn["a_column_draws"] = int(config.burn_rounds * matrix.shape[1])
        burn_endpoint = burn["state"].copy()
        measured = _reference_stage(
            b_states, a_states, burn["state"], rounds=config.measurement_rounds,
            round_offset=config.burn_rounds, config=config, lambdas=lambdas,
            log_mass=log_mass, log_odds=log_odds, H=matrix, plan=plan,
            compiled=compiled, factor_neighbors=factor_neighbors,
            y_columns=y_columns,
            rung_rngs=[PortablePrng(seed_identity.seed("measurement", "rung", rung))
                       for rung in range(replicas)],
            swap_rng=PortablePrng(seed_identity.seed("measurement", "swap")),
            observation_rng=PortablePrng(seed_identity.seed("measurement", "observation")),
            section_masks=section_masks, kernel_combinations=kernel_combinations,
            odds_powers=odds_powers, qubit_signatures=signatures,
            record=True,
        )
        measured["a_column_draws"] = int(config.measurement_rounds * matrix.shape[1])
    else:
        burn_values = _uare_nb_stage(
            state, b_states, a_states,
            np.asarray([
                PortablePrng(seed_identity.seed("burn", "rung", rung)).state_array()
                for rung in range(replicas)
            ], dtype=np.uint64),
            PortablePrng(seed_identity.seed("burn", "swap")).state_array(),
            PortablePrng(seed_identity.seed("burn", "observation")).state_array(),
            np.ascontiguousarray(lambdas), int(config.burn_rounds), 0,
            int(config.positive_row_updates_per_round), False, matrix, y_columns,
            log_mass, float(log_odds), section_masks, kernel_combinations,
            odds_powers, signatures, numba_plan.column_scope_variables,
            numba_plan.column_scope_lengths, numba_plan.scope_lengths,
            numba_plan.scope_variables, numba_plan.elimination_variables,
            numba_plan.source_starts, numba_plan.source_factor_ids,
            numba_plan.projection_maps, numba_plan.zero_indices,
            numba_plan.one_indices, numba_plan.output_factor_ids,
            int(numba_plan.total_factor_count), int(numba_plan.max_table_entries),
        )
        burn = {
            "state": burn_values[0], "labels": burn_values[1],
            "weights": burn_values[2], "packed": burn_values[3],
            "b_trace": burn_values[4], "a_trace": burn_values[5],
            "score_trace": burn_values[6], "b_weight_trace": burn_values[7],
            "row_counters": burn_values[8], "hot_changed_bits": burn_values[9],
            "swap_attempts": burn_values[10], "swap_accepts": burn_values[11],
            "a_column_draws": int(burn_values[12]),
        }
        burn_endpoint = burn["state"].copy()
        measured_values = _uare_nb_stage(
            burn["state"], b_states, a_states,
            np.asarray([
                PortablePrng(seed_identity.seed("measurement", "rung", rung)).state_array()
                for rung in range(replicas)
            ], dtype=np.uint64),
            PortablePrng(seed_identity.seed("measurement", "swap")).state_array(),
            PortablePrng(seed_identity.seed("measurement", "observation")).state_array(),
            np.ascontiguousarray(lambdas), int(config.measurement_rounds),
            int(config.burn_rounds), int(config.positive_row_updates_per_round),
            True, matrix, y_columns, log_mass, float(log_odds), section_masks,
            kernel_combinations, odds_powers, signatures,
            numba_plan.column_scope_variables, numba_plan.column_scope_lengths,
            numba_plan.scope_lengths, numba_plan.scope_variables,
            numba_plan.elimination_variables, numba_plan.source_starts,
            numba_plan.source_factor_ids, numba_plan.projection_maps,
            numba_plan.zero_indices, numba_plan.one_indices,
            numba_plan.output_factor_ids, int(numba_plan.total_factor_count),
            int(numba_plan.max_table_entries),
        )
        measured = {
            "state": measured_values[0], "labels": measured_values[1],
            "weights": measured_values[2], "packed": measured_values[3],
            "b_trace": measured_values[4], "a_trace": measured_values[5],
            "score_trace": measured_values[6],
            "b_weight_trace": measured_values[7],
            "row_counters": measured_values[8],
            "hot_changed_bits": measured_values[9],
            "swap_attempts": measured_values[10],
            "swap_accepts": measured_values[11],
            "a_column_draws": int(measured_values[12]),
        }
    final_state = measured["state"]
    unpacked = np.unpackbits(
        measured["packed"], axis=1, count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    replay_labels = np.asarray(
        [_state_label(frame, row) for row in unpacked], dtype=np.uint64,
    )
    if (residuals.any() or not np.array_equal(measured["labels"], replay_labels)
            or not np.array_equal(measured["weights"], unpacked.sum(axis=1))):
        raise UniformAnchorPtConflictError("uniform-anchor raw replay failed")
    if not np.array_equal(
            measured["a_trace"],
            np.asarray([
                _recompute_a_syndromes(y_columns, row, factor_neighbors)
                for row in measured["b_trace"]
            ], dtype=np.uint32)):
        raise UniformAnchorPtConflictError("uniform-anchor B/A trace drifted")
    return {
        "raw_version": UNIFORM_ANCHOR_PT_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "plan_json": canonical_json(plan.as_dict()),
        "plan_sha256": plan.sha256,
        "initial_state_packed": _pack_state(initial),
        "burn_state_packed": _pack_state(burn_endpoint),
        "final_state_packed": _pack_state(final_state),
        "measurement_states_packed": measured["packed"],
        "measurement_b_columns": measured["b_trace"],
        "measurement_a_syndromes": measured["a_trace"],
        "burn_labels": burn["labels"],
        "measurement_labels": measured["labels"],
        "measurement_weights": measured["weights"],
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_rounds // 8,
        ),
        "burn_complete_scores": burn["score_trace"],
        "measurement_complete_scores": measured["score_trace"],
        "burn_b_weights": burn["b_weight_trace"],
        "measurement_b_weights": measured["b_weight_trace"],
        "burn_row_counters": burn["row_counters"],
        "measurement_row_counters": measured["row_counters"],
        "burn_hot_refresh_changed_bits": burn["hot_changed_bits"],
        "measurement_hot_refresh_changed_bits": measured["hot_changed_bits"],
        "burn_cold_a_column_draws": np.asarray(burn["a_column_draws"], dtype=np.int64),
        "measurement_cold_a_column_draws": np.asarray(
            measured["a_column_draws"], dtype=np.int64,
        ),
        "burn_swap_attempts": burn["swap_attempts"],
        "burn_swap_accepts": burn["swap_accepts"],
        "measurement_swap_attempts": measured["swap_attempts"],
        "measurement_swap_accepts": measured["swap_accepts"],
        "lambda_values": lambdas,
        "lambda_sha256": config.lambda_sha256,
        "mass_sha256": _float64_sha256(mass),
        "initial_label": _state_label(frame, initial),
        "burn_label": _state_label(frame, burn_endpoint),
        "final_label": _state_label(frame, final_state),
        "engine": engine,
    }
