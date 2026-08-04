"""Uniform-anchor replica exchange with exact auxiliary stabilizer blocks.

This is a new local-only q=0 kernel.  It retains the exact-uniform B endpoint
and complete collapsed-energy swaps, then applies a cold-rung A-row/B-block
stabilizer heatbath after every exchange clock.  The auxiliary move is an
exact full-posterior Gibbs block, so it changes coordinated B directions while
preserving the cold marginal used by the replica-exchange ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .q0_global import GlobalConflictError, validate_observable_frame
from .q0_hgp_aux_stabilizer import (
    AUX_STABILIZER_KERNEL,
    auxiliary_stabilizer_row_heatbath,
)
from .q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _pack_state,
    _qubit_signatures,
    _section_and_kernel_masks,
    _state_label,
    build_classical_coset_mass,
    join_hgp_state,
    split_hgp_state,
    validate_hgp_wiring,
)
from .q0_hgp_full_row_gibbs_v0 import (
    _as_binary_matrix,
    build_full_row_elimination_plan,
    compile_full_row_elimination_plan,
)
from .q0_hgp_uniform_anchor_pt import (
    UNIFORM_ANCHOR_PT_KERNEL,
    UniformAnchorReplicaExchangeConfig,
    _factor_neighbors,
    _float64_sha256,
    _numba_elimination_arrays,
    _recompute_a_syndromes,
    _reference_stage,
    _uare_nb_stage,
    _y_columns,
    collapsed_complete_score,
    uniform_anchor_lambda_schedule,
)
from .seeds import derive_seed


AUX_STABILIZER_PT_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.v0"
AUX_STABILIZER_PT_RAW_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.raw.v0"
AUX_STABILIZER_PT_KERNEL = (
    "uniform_anchor_complete_energy_with_cold_auxiliary_stabilizer_heatbath.v1"
)


class AuxiliaryStabilizerPtConflictError(ValueError):
    """Raised for an identity, target, or replay violation in the new kernel."""


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


@dataclass(frozen=True)
class AuxiliaryStabilizerReplicaExchangeConfig:
    """Fixed clocks for a uniform-anchor ladder plus cold stabilizer blocks."""

    p: float
    burn_rounds: int
    measurement_rounds: int
    num_replicas: int
    positive_row_updates_per_round: int = 1
    cold_auxiliary_rows_per_round: int = 1
    method_id: str = ""

    def __post_init__(self):
        p = float(self.p)
        if not math.isfinite(p) or not 0.0 < p < 0.5:
            raise ValueError("auxiliary-stabilizer p must lie in (0, .5)")
        object.__setattr__(self, "p", p)
        for name in (
                "burn_rounds", "measurement_rounds", "num_replicas",
                "positive_row_updates_per_round", "cold_auxiliary_rows_per_round"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.burn_rounds <= 0 or self.measurement_rounds <= 0:
            raise ValueError("auxiliary-stabilizer clocks must be positive")
        if self.measurement_rounds % 8:
            raise ValueError("auxiliary-stabilizer measurement must divide into eight blocks")
        if not 2 <= self.num_replicas <= 128:
            raise ValueError("auxiliary-stabilizer replicas must lie in [2, 128]")
        if not 1 <= self.positive_row_updates_per_round <= 24:
            raise ValueError("auxiliary-stabilizer row updates must lie in [1, 24]")
        if not 1 <= self.cold_auxiliary_rows_per_round <= 32:
            raise ValueError("auxiliary-stabilizer cold rows must lie in [1, 32]")
        expected = (
            f"UASRE{self.num_replicas:02d}"
            f"-R{self.positive_row_updates_per_round}"
            f"-A{self.cold_auxiliary_rows_per_round}"
        )
        if self.method_id and self.method_id != expected:
            raise ValueError("auxiliary-stabilizer method ID does not match its parameters")
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
            "cold_auxiliary_rows_per_round": self.cold_auxiliary_rows_per_round,
            "lambda_values": self.lambda_values.tolist(),
            "lambda_schedule": "cosine_endpoint_cluster_v1",
            "kernel": AUX_STABILIZER_PT_KERNEL,
            "hot_endpoint": "exact_uniform_B_refresh",
            "tempered_term": "complete_collapsed_log_density",
            "cold_block_kernel": AUX_STABILIZER_KERNEL,
            "cold_block_schedule": "post_swap_cyclic_A_rows.v1",
        }


@dataclass(frozen=True)
class AuxiliaryStabilizerReplicaExchangeSeedIdentity:
    """Disjoint seed identity for an immutable P/U/L trajectory."""

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
        if not isinstance(self.method_id, str) or not self.method_id.startswith("UASRE"):
            raise ValueError("auxiliary-stabilizer seed method ID is invalid")
        if not isinstance(self.resource_tier, str) or not self.resource_tier:
            raise ValueError("auxiliary-stabilizer resource tier is empty")
        if self.init_family not in ("P", "U", "L"):
            raise ValueError("auxiliary-stabilizer initial family must be P, U, or L")
        if isinstance(self.trajectory_index, bool) or int(self.trajectory_index) < 0:
            raise ValueError("auxiliary-stabilizer trajectory index is invalid")
        object.__setattr__(self, "trajectory_index", int(self.trajectory_index))
        if not isinstance(self.trajectory_namespace, str) or not self.trajectory_namespace:
            raise ValueError("auxiliary-stabilizer trajectory namespace is empty")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            AUX_STABILIZER_PT_VERSION,
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


def _inner_config(config):
    """Use the proven UARE local/swap kernel without sharing its raw schema."""
    return UniformAnchorReplicaExchangeConfig(
        p=config.p,
        burn_rounds=1,
        measurement_rounds=8,
        num_replicas=config.num_replicas,
        positive_row_updates_per_round=config.positive_row_updates_per_round,
    )


def _phase_reference(state, b_states, a_states, *, phase, rounds, round_offset,
                     config, inner, matrix, syndrome, plan, compiled,
                     factor_neighbors, y_columns, log_mass, log_odds, lambdas,
                     section_masks, kernel_combinations, odds_powers, signatures,
                     seed_identity, record):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rows, columns = matrix.shape
    rung_rngs = [PortablePrng(seed_identity.seed(phase, "rung", rung))
                 for rung in range(config.num_replicas)]
    swap_rng = PortablePrng(seed_identity.seed(phase, "swap"))
    lift_rng = PortablePrng(seed_identity.seed(phase, "cold_lift"))
    auxiliary_rng = PortablePrng(seed_identity.seed(phase, "cold_auxiliary"))
    labels = np.empty(rounds, dtype=np.uint64)
    weights = np.empty(rounds, dtype=np.int32)
    packed = np.empty((rounds if record else 0, (state.size + 7) // 8), dtype=np.uint8)
    b_trace = np.empty((rounds if record else 0, rows), dtype=np.uint32)
    a_trace = np.empty((rounds if record else 0, columns), dtype=np.uint32)
    score_trace = np.empty(rounds, dtype=np.float64)
    b_weight_trace = np.empty(rounds, dtype=np.int32)
    row_counters = np.zeros((config.num_replicas, 3), dtype=np.int64)
    hot_changed_bits = np.zeros(rounds, dtype=np.int32)
    swap_attempts = np.zeros(config.num_replicas - 1, dtype=np.int64)
    swap_accepts = np.zeros(config.num_replicas - 1, dtype=np.int64)
    auxiliary_counters = np.zeros(3, dtype=np.int64)
    auxiliary_assignments = np.zeros(
        (rounds if record else 0, config.cold_auxiliary_rows_per_round), dtype=np.uint32,
    )
    current = np.asarray(state, dtype=np.uint8).copy()
    for local_round in range(rounds):
        round_index = int(round_offset) + local_round
        step = _reference_stage(
            b_states, a_states, current, rounds=1, round_offset=round_index,
            config=inner, lambdas=lambdas, log_mass=log_mass, log_odds=log_odds,
            H=matrix, plan=plan, compiled=compiled, factor_neighbors=factor_neighbors,
            y_columns=y_columns, rung_rngs=rung_rngs, swap_rng=swap_rng,
            observation_rng=lift_rng, section_masks=section_masks,
            kernel_combinations=kernel_combinations, odds_powers=odds_powers,
            qubit_signatures=signatures, record=False,
        )
        current = step["state"]
        row_counters += step["row_counters"]
        hot_changed_bits[local_round] = step["hot_changed_bits"][0]
        swap_attempts += step["swap_attempts"]
        swap_accepts += step["swap_accepts"]
        A, B = split_hgp_state(current, matrix)
        for auxiliary_offset in range(config.cold_auxiliary_rows_per_round):
            a_row = (round_index + auxiliary_offset) % columns
            A, B, assignment = auxiliary_stabilizer_row_heatbath(
                matrix, A, B, a_row, config.p, auxiliary_rng, plan=plan, compiled=compiled,
            )
            auxiliary_counters[0] += 1
            auxiliary_counters[2] += int(assignment).bit_count()
            if assignment:
                auxiliary_counters[1] += 1
            if record:
                auxiliary_assignments[local_round, auxiliary_offset] = assignment
        current = join_hgp_state(A, B)
        b_columns, a_syndromes, _ = _initial_collapsed_masks(current, syndrome, matrix)
        b_states[-1] = b_columns
        a_states[-1] = a_syndromes
        labels[local_round] = _state_label_from_signatures(current, signatures)
        weights[local_round] = int(current.sum())
        score_trace[local_round] = collapsed_complete_score(
            b_columns, a_syndromes, log_mass, log_odds,
        )
        b_weight_trace[local_round] = sum(int(value).bit_count() for value in b_columns)
        if record:
            packed[local_round] = _pack_state(current)
            b_trace[local_round] = b_columns
            a_trace[local_round] = a_syndromes
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
        "auxiliary_counters": auxiliary_counters,
        "auxiliary_assignments": auxiliary_assignments,
        "a_column_draws": int(rounds * columns),
    }


def _state_label_from_signatures(state, signatures):
    label = np.uint64(0)
    for index, bit in enumerate(np.asarray(state, dtype=np.uint8)):
        if bit:
            label ^= np.uint64(signatures[index])
    return label


def _phase_numba(state, b_states, a_states, *, phase, rounds, round_offset,
                 config, inner, matrix, syndrome, plan, compiled, numba_plan,
                 y_columns, log_mass, log_odds, lambdas, section_masks,
                 kernel_combinations, odds_powers, signatures, seed_identity, record):
    if _uare_nb_stage is None:
        raise RuntimeError("Numba is required for the auxiliary-stabilizer accelerated path")
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rows, columns = matrix.shape
    rung_rng_states = np.asarray([
        PortablePrng(seed_identity.seed(phase, "rung", rung)).state_array()
        for rung in range(config.num_replicas)
    ], dtype=np.uint64)
    swap_rng_state = PortablePrng(seed_identity.seed(phase, "swap")).state_array()
    lift_rng_state = PortablePrng(seed_identity.seed(phase, "cold_lift")).state_array()
    auxiliary_rng = PortablePrng(seed_identity.seed(phase, "cold_auxiliary"))
    labels = np.empty(rounds, dtype=np.uint64)
    weights = np.empty(rounds, dtype=np.int32)
    packed = np.empty((rounds if record else 0, (state.size + 7) // 8), dtype=np.uint8)
    b_trace = np.empty((rounds if record else 0, rows), dtype=np.uint32)
    a_trace = np.empty((rounds if record else 0, columns), dtype=np.uint32)
    score_trace = np.empty(rounds, dtype=np.float64)
    b_weight_trace = np.empty(rounds, dtype=np.int32)
    row_counters = np.zeros((config.num_replicas, 3), dtype=np.int64)
    hot_changed_bits = np.zeros(rounds, dtype=np.int32)
    swap_attempts = np.zeros(config.num_replicas - 1, dtype=np.int64)
    swap_accepts = np.zeros(config.num_replicas - 1, dtype=np.int64)
    auxiliary_counters = np.zeros(3, dtype=np.int64)
    auxiliary_assignments = np.zeros(
        (rounds if record else 0, config.cold_auxiliary_rows_per_round), dtype=np.uint32,
    )
    current = np.asarray(state, dtype=np.uint8).copy()
    for local_round in range(rounds):
        round_index = int(round_offset) + local_round
        step = _uare_nb_stage(
            current, b_states, a_states, rung_rng_states, swap_rng_state, lift_rng_state,
            np.ascontiguousarray(lambdas), 1, round_index,
            int(inner.positive_row_updates_per_round), False, matrix, y_columns,
            log_mass, float(log_odds), section_masks, kernel_combinations,
            odds_powers, signatures, numba_plan.column_scope_variables,
            numba_plan.column_scope_lengths, numba_plan.scope_lengths,
            numba_plan.scope_variables, numba_plan.elimination_variables,
            numba_plan.source_starts, numba_plan.source_factor_ids,
            numba_plan.projection_maps, numba_plan.zero_indices,
            numba_plan.one_indices, numba_plan.output_factor_ids,
            int(numba_plan.total_factor_count), int(numba_plan.max_table_entries),
        )
        current = step[0]
        row_counters += step[8]
        hot_changed_bits[local_round] = step[9][0]
        swap_attempts += step[10]
        swap_accepts += step[11]
        A, B = split_hgp_state(current, matrix)
        for auxiliary_offset in range(config.cold_auxiliary_rows_per_round):
            a_row = (round_index + auxiliary_offset) % columns
            A, B, assignment = auxiliary_stabilizer_row_heatbath(
                matrix, A, B, a_row, config.p, auxiliary_rng, plan=plan, compiled=compiled,
            )
            auxiliary_counters[0] += 1
            auxiliary_counters[2] += int(assignment).bit_count()
            if assignment:
                auxiliary_counters[1] += 1
            if record:
                auxiliary_assignments[local_round, auxiliary_offset] = assignment
        current = join_hgp_state(A, B)
        b_columns, a_syndromes, _ = _initial_collapsed_masks(current, syndrome, matrix)
        b_states[-1] = b_columns
        a_states[-1] = a_syndromes
        labels[local_round] = _state_label_from_signatures(current, signatures)
        weights[local_round] = int(current.sum())
        score_trace[local_round] = collapsed_complete_score(
            b_columns, a_syndromes, log_mass, log_odds,
        )
        b_weight_trace[local_round] = sum(int(value).bit_count() for value in b_columns)
        if record:
            packed[local_round] = _pack_state(current)
            b_trace[local_round] = b_columns
            a_trace[local_round] = a_syndromes
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
        "auxiliary_counters": auxiliary_counters,
        "auxiliary_assignments": auxiliary_assignments,
        "a_column_draws": int(rounds * columns),
    }


def run_auxiliary_stabilizer_replica_exchange_trajectory(model, frame, H, syndrome,
                                                          config, seed_identity, initial_state,
                                                          *, engine="reference", mass=None):
    """Run an immutable trajectory with UARE transport plus cold ASRH blocks."""
    if engine not in ("reference", "numba"):
        raise ValueError("auxiliary-stabilizer engine must be reference or numba")
    if not isinstance(config, AuxiliaryStabilizerReplicaExchangeConfig):
        raise TypeError("auxiliary-stabilizer config has the wrong type")
    if not isinstance(seed_identity, AuxiliaryStabilizerReplicaExchangeSeedIdentity):
        raise TypeError("auxiliary-stabilizer seed identity has the wrong type")
    if config.method_id != seed_identity.method_id:
        raise AuxiliaryStabilizerPtConflictError("auxiliary-stabilizer config/seed mismatch")
    validate_hgp_wiring(H, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise AuxiliaryStabilizerPtConflictError("auxiliary-stabilizer observable frame mismatch") from exc
    matrix = _as_binary_matrix(H)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    if syndrome.shape != (model.num_checks,) or state.shape != (model.num_qubits,):
        raise ValueError("auxiliary-stabilizer syndrome or state shape mismatch")
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
        raise AuxiliaryStabilizerPtConflictError("auxiliary-stabilizer mass table mismatch")
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    log_odds = math.log(config.p / (1.0 - config.p))
    lambdas = config.lambda_values
    rows, columns = matrix.shape
    replicas = config.num_replicas
    b_states = np.repeat(b_columns[None, :], replicas, axis=0)
    a_states = np.repeat(a_syndromes[None, :], replicas, axis=0)
    factor_neighbors = _factor_neighbors(matrix)
    y_columns = _y_columns(syndrome, matrix)
    section_masks, kernel_combinations = _section_and_kernel_masks(matrix)
    odds = config.p / (1.0 - config.p)
    odds_powers = np.ones(columns + 1, dtype=np.float64)
    for index in range(1, odds_powers.size):
        odds_powers[index] = odds_powers[index - 1] * odds
    signatures = _qubit_signatures(frame)
    inner = _inner_config(config)
    common = {
        "config": config, "inner": inner, "matrix": matrix, "syndrome": syndrome,
        "plan": plan, "compiled": compiled, "y_columns": y_columns,
        "log_mass": log_mass, "log_odds": log_odds, "lambdas": lambdas,
        "section_masks": section_masks, "kernel_combinations": kernel_combinations,
        "odds_powers": odds_powers, "signatures": signatures,
        "seed_identity": seed_identity,
    }
    if engine == "reference":
        burn = _phase_reference(
            state, b_states, a_states, phase="burn", rounds=config.burn_rounds,
            round_offset=0, factor_neighbors=factor_neighbors, record=False, **common,
        )
        measured = _phase_reference(
            burn["state"], b_states, a_states, phase="measurement",
            rounds=config.measurement_rounds, round_offset=config.burn_rounds,
            factor_neighbors=factor_neighbors, record=True, **common,
        )
    else:
        burn = _phase_numba(
            state, b_states, a_states, phase="burn", rounds=config.burn_rounds,
            round_offset=0, numba_plan=numba_plan, record=False, **common,
        )
        measured = _phase_numba(
            burn["state"], b_states, a_states, phase="measurement",
            rounds=config.measurement_rounds, round_offset=config.burn_rounds,
            numba_plan=numba_plan, record=True, **common,
        )
    unpacked = np.unpackbits(
        measured["packed"], axis=1, count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    replay_labels = np.asarray([_state_label(frame, row) for row in unpacked], dtype=np.uint64)
    if (residuals.any() or not np.array_equal(measured["labels"], replay_labels)
            or not np.array_equal(measured["weights"], unpacked.sum(axis=1))):
        raise AuxiliaryStabilizerPtConflictError("auxiliary-stabilizer raw replay failed")
    if not np.array_equal(
            measured["a_trace"],
            np.asarray([
                _recompute_a_syndromes(y_columns, row, factor_neighbors)
                for row in measured["b_trace"]
            ], dtype=np.uint32)):
        raise AuxiliaryStabilizerPtConflictError("auxiliary-stabilizer B/A trace drifted")
    return {
        "raw_version": AUX_STABILIZER_PT_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "plan_json": canonical_json(plan.as_dict()),
        "plan_sha256": plan.sha256,
        "initial_state_packed": _pack_state(initial),
        "burn_state_packed": _pack_state(burn["state"]),
        "final_state_packed": _pack_state(measured["state"]),
        "measurement_states_packed": measured["packed"],
        "measurement_b_columns": measured["b_trace"],
        "measurement_a_syndromes": measured["a_trace"],
        "burn_labels": burn["labels"],
        "measurement_labels": measured["labels"],
        "measurement_weights": measured["weights"],
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_block": np.repeat(np.arange(8, dtype=np.int8), config.measurement_rounds // 8),
        "burn_complete_scores": burn["score_trace"],
        "measurement_complete_scores": measured["score_trace"],
        "burn_b_weights": burn["b_weight_trace"],
        "measurement_b_weights": measured["b_weight_trace"],
        "burn_row_counters": burn["row_counters"],
        "measurement_row_counters": measured["row_counters"],
        "burn_hot_refresh_changed_bits": burn["hot_changed_bits"],
        "measurement_hot_refresh_changed_bits": measured["hot_changed_bits"],
        "burn_auxiliary_counters": burn["auxiliary_counters"],
        "measurement_auxiliary_counters": measured["auxiliary_counters"],
        "measurement_auxiliary_assignments": measured["auxiliary_assignments"],
        "burn_cold_a_column_draws": np.asarray(burn["a_column_draws"], dtype=np.int64),
        "measurement_cold_a_column_draws": np.asarray(measured["a_column_draws"], dtype=np.int64),
        "burn_swap_attempts": burn["swap_attempts"],
        "burn_swap_accepts": burn["swap_accepts"],
        "measurement_swap_attempts": measured["swap_attempts"],
        "measurement_swap_accepts": measured["swap_accepts"],
        "lambda_values": lambdas,
        "lambda_sha256": config.lambda_sha256,
        "mass_sha256": _float64_sha256(mass),
        "initial_label": _state_label(frame, initial),
        "burn_label": _state_label(frame, burn["state"]),
        "final_label": _state_label(frame, measured["state"]),
        "engine": engine,
    }
