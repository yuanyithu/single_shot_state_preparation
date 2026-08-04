"""Defect-tempered conditional sampler prototype for the q=0 hard coset.

The finite-rung target is

    pi_j(e) proportional to exp[-K_p |e| - Kq_j D(e)],
    D(e) = |H_Z e xor y|.

At any finite rung, conditioning a fixed-clock observation on ``D=0`` gives
the exact q=0 hard-coset posterior.  The hot rung has ``Kq=0`` and is refreshed
from its exact iid Bernoulli(p) target.  This is intentionally distinct from
the collapsed-HGP power ladder: it samples full physical states, temporarily
allows defects, and uses only fixed-clock D=0 observations as estimators.

This module is a reference-only V0 kernel.  It is diagnostic infrastructure,
not a formal exp102 sampler or a source of q_top results.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .io import canonical_json, sha256_json
from .q0_global import (
    _column_check_csr,
    pack_state,
    qubit_signatures,
    state_label,
    unpack_states,
    validate_observable_frame,
)
from .q0_pt import coupling
from .seeds import derive_seed

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised on environments without Numba
    njit = None


DEFECT_TEMPERED_VERSION = "exp102.q0_defect_tempered.v0"
DEFECT_TEMPERED_RAW_VERSION = "exp102.q0_defect_tempered.raw.v0"
DEFECT_TEMPERED_KERNEL = "syndrome_penalty_replica_exchange_iid_hot.v1"


class DefectTemperedConflictError(ValueError):
    """A target, identity, or replay invariant was violated."""


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


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _float64_sha256(values):
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=">f8").tobytes(order="C")
    ).hexdigest()


def _flip_probability(log_ratio):
    """Stable exact-heatbath probability for a binary flip."""
    if log_ratio >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_ratio))
    value = math.exp(log_ratio)
    return value / (1.0 + value)


def defect_tempered_swap_log_acceptance(kq_lower, kq_upper,
                                        defects_lower, defects_upper):
    """Log Metropolis ratio for swapping adjacent syndrome-penalty rungs."""
    return ((float(kq_lower) - float(kq_upper))
            * (int(defects_lower) - int(defects_upper)))


@dataclass(frozen=True)
class DefectTemperedConfig:
    """A fixed ladder with a finite cold penalty and exact iid hot endpoint."""

    method_id: str
    p: float
    kq_values: tuple[float, ...]
    burn_rounds: int
    measurement_rounds: int
    sweeps_per_round: int = 1

    def __post_init__(self):
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be a nonempty string")
        if not 0.0 < float(self.p) < 0.5:
            raise ValueError("p must lie in (0,0.5)")
        values = tuple(float(value) for value in self.kq_values)
        if len(values) < 2 or not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("kq_values must contain at least two finite nonnegative values")
        if values[-1] != 0.0 or values[0] <= 0.0:
            raise ValueError("ladder must have a positive finite cold rung and exact Kq=0 hot rung")
        if any(left <= right for left, right in zip(values, values[1:])):
            raise ValueError("kq_values must be strictly descending from cold to hot")
        object.__setattr__(self, "kq_values", values)
        _positive_integer(self.burn_rounds, "burn_rounds")
        measurement = _positive_integer(self.measurement_rounds, "measurement_rounds")
        if measurement % 8:
            raise ValueError("measurement_rounds must divide into eight time blocks")
        _positive_integer(self.sweeps_per_round, "sweeps_per_round")

    @property
    def num_replicas(self):
        return len(self.kq_values)

    @property
    def cold_kq(self):
        return float(self.kq_values[0])

    @property
    def ladder_sha256(self):
        return _float64_sha256(self.kq_values)

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": float(self.p),
            "kq_values": [float(value) for value in self.kq_values],
            "burn_rounds": int(self.burn_rounds),
            "measurement_rounds": int(self.measurement_rounds),
            "sweeps_per_round": int(self.sweeps_per_round),
            "num_replicas": self.num_replicas,
            "cold_kq": self.cold_kq,
            "hot_endpoint": "exact_iid_bernoulli_p",
        }


@dataclass(frozen=True)
class DefectTemperedSeedIdentity:
    """Stage-separated, replayable identity for a diagnostic trajectory."""

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
        _strict_commit(self.source_commit)
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            _strict_sha256(getattr(self, name), name)
        if self.init_family not in ("P", "U", "L"):
            raise ValueError("unknown defect-tempered initialization family")
        if (isinstance(self.trajectory_index, (bool, np.bool_))
                or int(self.trajectory_index) != self.trajectory_index
                or int(self.trajectory_index) < 0):
            raise ValueError("trajectory_index must be a nonnegative integer")
        if not isinstance(self.trajectory_namespace, str) or not self.trajectory_namespace:
            raise ValueError("trajectory_namespace must be a nonempty string")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            DEFECT_TEMPERED_VERSION,
            self.source_commit,
            self.config_sha256,
            self.registry_sha256,
            self.cell_fingerprint,
            self.method_id,
            self.resource_tier,
            self.init_family,
            int(self.trajectory_index),
            self.trajectory_namespace,
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
            "trajectory_index": int(self.trajectory_index),
            "trajectory_namespace": self.trajectory_namespace,
        }


@dataclass
class _ReplicaState:
    state: np.ndarray
    label: np.uint64
    weight: int
    residual: np.ndarray
    defects: int


def _residual(model, state, syndrome):
    return (
        model.H_check.astype(np.int64) @ np.asarray(state, dtype=np.uint8).astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome


def _make_replica(model, frame, state, syndrome):
    state = np.ascontiguousarray(state, dtype=np.uint8).copy()
    residual = _residual(model, state, syndrome)
    return _ReplicaState(
        state=state,
        label=np.uint64(state_label(frame, state)),
        weight=int(state.sum()),
        residual=np.ascontiguousarray(residual, dtype=np.uint8),
        defects=int(residual.sum()),
    )


def _single_bit_heatbath_sweep(replica, rng, kp, kq, check_indices,
                               check_offsets, signatures):
    attempts = 0
    changes = 0
    for raw_qubit in rng.permutation(replica.state.size):
        qubit = int(raw_qubit)
        start = int(check_offsets[qubit])
        stop = int(check_offsets[qubit + 1])
        ones = int(replica.residual[check_indices[start:stop]].sum())
        degree = stop - start
        delta_defects = degree - 2 * ones
        delta_weight = 1 - 2 * int(replica.state[qubit])
        log_ratio = -kp * delta_weight - kq * delta_defects
        attempts += 1
        if rng.random() < _flip_probability(log_ratio):
            replica.state[qubit] ^= np.uint8(1)
            replica.label ^= signatures[qubit]
            replica.weight += delta_weight
            replica.residual[check_indices[start:stop]] ^= np.uint8(1)
            replica.defects += delta_defects
            changes += 1
    return attempts, changes


def _iid_hot_refresh(replica, rng, model, frame, syndrome, p):
    """Replace the hot state by an exact iid draw from its Kq=0 target."""
    previous = replica.state.copy()
    for qubit in range(replica.state.size):
        replica.state[qubit] = np.uint8(rng.random() < p)
    replica.label = np.uint64(state_label(frame, replica.state))
    replica.weight = int(replica.state.sum())
    replica.residual = _residual(model, replica.state, syndrome)
    replica.defects = int(replica.residual.sum())
    return int(np.count_nonzero(previous ^ replica.state))


def _validate_replica(model, frame, syndrome, replica):
    residual = _residual(model, replica.state, syndrome)
    if not np.array_equal(residual, replica.residual):
        raise DefectTemperedConflictError("cached defect residual drifted")
    if int(residual.sum()) != int(replica.defects):
        raise DefectTemperedConflictError("cached defect count drifted")
    if int(replica.state.sum()) != int(replica.weight):
        raise DefectTemperedConflictError("cached data weight drifted")
    if int(state_label(frame, replica.state)) != int(replica.label):
        raise DefectTemperedConflictError("cached signature drifted")


def _run_phase_reference(model, frame, syndrome, config, replicas, origins,
                         seed_identity, stage, rounds, *, record):
    """Advance a fixed number of replica-exchange rounds and optionally retain cold clocks."""
    from exp101_certified_src.prng import PortablePrng

    kp = coupling(config.p)
    signatures = qubit_signatures(frame)
    check_indices, check_offsets = _column_check_csr(model)
    rung_rngs = [
        PortablePrng(seed_identity.seed(stage, "rung", rung))
        for rung in range(config.num_replicas)
    ]
    swap_rng = PortablePrng(seed_identity.seed(stage, "swap"))
    counters = np.zeros((config.num_replicas, 2), dtype=np.int64)
    hot_refresh_changes = 0
    swap_attempts = np.zeros(config.num_replicas - 1, dtype=np.int64)
    swap_accepts = np.zeros(config.num_replicas - 1, dtype=np.int64)
    hot_visits = np.zeros(config.num_replicas, dtype=np.int64)
    cold_visits = np.zeros(config.num_replicas, dtype=np.int64)
    states = None
    labels = None
    weights = None
    defects = None
    if record:
        states = np.empty(
            (rounds, (model.num_qubits + 7) // 8), dtype=np.uint8,
        )
        labels = np.empty(rounds, dtype=np.uint64)
        weights = np.empty(rounds, dtype=np.int32)
        defects = np.empty(rounds, dtype=np.int32)
    d0_leaves = 0
    d0_returns = 0
    d0_label_changes = 0
    previous_d0 = replicas[0].defects == 0
    previous_d0_label = np.uint64(replicas[0].label)
    parity = 0
    for round_index in range(rounds):
        for rung, kq in enumerate(config.kq_values):
            if kq == 0.0:
                hot_refresh_changes += _iid_hot_refresh(
                    replicas[rung], rung_rngs[rung], model, frame, syndrome,
                    config.p,
                )
                continue
            for _ in range(config.sweeps_per_round):
                attempts, changes = _single_bit_heatbath_sweep(
                    replicas[rung], rung_rngs[rung], kp, kq, check_indices,
                    check_offsets, signatures,
                )
                counters[rung, 0] += attempts
                counters[rung, 1] += changes
        for lower in range(parity, config.num_replicas - 1, 2):
            upper = lower + 1
            swap_attempts[lower] += 1
            log_ratio = defect_tempered_swap_log_acceptance(
                config.kq_values[lower], config.kq_values[upper],
                replicas[lower].defects, replicas[upper].defects,
            )
            if log_ratio >= 0.0 or swap_rng.random() < math.exp(log_ratio):
                swap_accepts[lower] += 1
                replicas[lower], replicas[upper] = replicas[upper], replicas[lower]
                origins[lower], origins[upper] = origins[upper], origins[lower]
        parity ^= 1
        hot_visits[origins[-1]] += 1
        cold_visits[origins[0]] += 1
        current_d0 = replicas[0].defects == 0
        if previous_d0 and not current_d0:
            d0_leaves += 1
        elif not previous_d0 and current_d0:
            d0_returns += 1
        elif previous_d0 and current_d0 and replicas[0].label != previous_d0_label:
            d0_label_changes += 1
        if current_d0:
            previous_d0_label = np.uint64(replicas[0].label)
        previous_d0 = current_d0
        if record:
            states[round_index] = pack_state(replicas[0].state)
            labels[round_index] = replicas[0].label
            weights[round_index] = replicas[0].weight
            defects[round_index] = replicas[0].defects
    for replica in replicas:
        _validate_replica(model, frame, syndrome, replica)
    return {
        "states_packed": states,
        "labels": labels,
        "weights": weights,
        "defects": defects,
        "bit_counters_by_rung": counters,
        "hot_refresh_bit_changes": int(hot_refresh_changes),
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
        "hot_visits_by_origin": hot_visits,
        "cold_visits_by_origin": cold_visits,
        "cold_d0_leaves": int(d0_leaves),
        "cold_d0_returns": int(d0_returns),
        "cold_d0_label_changes": int(d0_label_changes),
    }


if njit is not None:
    @njit(cache=True, inline="always")
    def _dt_next_uint64(state):
        x = state[0]
        y = state[1]
        state[0] = y
        x = x ^ (x << np.uint64(23))
        x = x ^ (x >> np.uint64(17))
        x = x ^ y ^ (y >> np.uint64(26))
        state[1] = x
        return x + y


    @njit(cache=True, inline="always")
    def _dt_random(state):
        return float(_dt_next_uint64(state) >> np.uint64(11)) * (
            1.0 / 9007199254740992.0
        )


    @njit(cache=True, inline="always")
    def _dt_fill_permutation(state, order):
        for index in range(order.size):
            order[index] = index
        for index in range(order.size - 1, 0, -1):
            selected = int(_dt_next_uint64(state) % np.uint64(index + 1))
            temporary = order[index]
            order[index] = order[selected]
            order[selected] = temporary


    @njit(cache=True, inline="always")
    def _dt_flip_probability(log_ratio):
        if log_ratio >= 0.0:
            return 1.0 / (1.0 + math.exp(-log_ratio))
        value = math.exp(log_ratio)
        return value / (1.0 + value)


    @njit(cache=True, inline="always")
    def _dt_pack_row(state, output):
        for byte in range(output.size):
            value = np.uint8(0)
            start = byte * 8
            stop = min(start + 8, state.size)
            for bit in range(start, stop):
                value |= state[bit] << np.uint8(bit - start)
            output[byte] = value


    @njit(cache=True)
    def _run_phase_numba_core(states, labels, weights, residuals, defects,
                              origins, rung_rng_states, swap_rng_state,
                              kq_values, p, kp, syndrome, check_indices,
                              check_offsets, signatures, rounds,
                              sweeps_per_round, record):
        rungs = states.shape[0]
        num_qubits = states.shape[1]
        num_checks = syndrome.size
        bytes_per_state = (num_qubits + 7) // 8
        packed_count = rounds if record else 0
        packed = np.empty((packed_count, bytes_per_state), dtype=np.uint8)
        output_labels = np.empty(packed_count, dtype=np.uint64)
        output_weights = np.empty(packed_count, dtype=np.int32)
        output_defects = np.empty(packed_count, dtype=np.int32)
        counters = np.zeros((rungs, 2), dtype=np.int64)
        hot_refresh_changes = np.int64(0)
        swap_attempts = np.zeros(rungs - 1, dtype=np.int64)
        swap_accepts = np.zeros(rungs - 1, dtype=np.int64)
        hot_visits = np.zeros(rungs, dtype=np.int64)
        cold_visits = np.zeros(rungs, dtype=np.int64)
        order = np.empty(num_qubits, dtype=np.int32)
        d0_leaves = np.int64(0)
        d0_returns = np.int64(0)
        d0_label_changes = np.int64(0)
        previous_d0 = defects[0] == 0
        previous_d0_label = labels[0]
        parity = 0
        for round_index in range(rounds):
            for rung in range(rungs):
                if kq_values[rung] == 0.0:
                    label = np.uint64(0)
                    weight = 0
                    for qubit in range(num_qubits):
                        previous_bit = states[rung, qubit]
                        bit = np.uint8(_dt_random(rung_rng_states[rung]) < p)
                        states[rung, qubit] = bit
                        hot_refresh_changes += int(previous_bit != bit)
                        if bit:
                            weight += 1
                            label ^= signatures[qubit]
                    for check in range(num_checks):
                        residuals[rung, check] = syndrome[check]
                    for qubit in range(num_qubits):
                        if states[rung, qubit]:
                            for position in range(
                                    check_offsets[qubit], check_offsets[qubit + 1]):
                                residuals[rung, check_indices[position]] ^= np.uint8(1)
                    defect = 0
                    for check in range(num_checks):
                        defect += int(residuals[rung, check])
                    labels[rung] = label
                    weights[rung] = weight
                    defects[rung] = defect
                    continue
                for unused in range(sweeps_per_round):
                    _dt_fill_permutation(rung_rng_states[rung], order)
                    for slot in range(num_qubits):
                        qubit = order[slot]
                        start = check_offsets[qubit]
                        stop = check_offsets[qubit + 1]
                        ones = 0
                        for position in range(start, stop):
                            ones += int(residuals[rung, check_indices[position]])
                        delta_defects = (stop - start) - 2 * ones
                        delta_weight = 1 - 2 * int(states[rung, qubit])
                        log_ratio = -kp * delta_weight - kq_values[rung] * delta_defects
                        counters[rung, 0] += 1
                        if _dt_random(rung_rng_states[rung]) < _dt_flip_probability(log_ratio):
                            states[rung, qubit] ^= np.uint8(1)
                            labels[rung] ^= signatures[qubit]
                            weights[rung] += delta_weight
                            defects[rung] += delta_defects
                            for position in range(start, stop):
                                residuals[rung, check_indices[position]] ^= np.uint8(1)
                            counters[rung, 1] += 1
            for lower in range(parity, rungs - 1, 2):
                upper = lower + 1
                log_ratio = ((kq_values[lower] - kq_values[upper])
                             * (defects[lower] - defects[upper]))
                swap_attempts[lower] += 1
                if log_ratio >= 0.0 or _dt_random(swap_rng_state) < math.exp(log_ratio):
                    swap_accepts[lower] += 1
                    for qubit in range(num_qubits):
                        temporary = states[lower, qubit]
                        states[lower, qubit] = states[upper, qubit]
                        states[upper, qubit] = temporary
                    for check in range(num_checks):
                        temporary = residuals[lower, check]
                        residuals[lower, check] = residuals[upper, check]
                        residuals[upper, check] = temporary
                    temporary_label = labels[lower]
                    labels[lower] = labels[upper]
                    labels[upper] = temporary_label
                    temporary_weight = weights[lower]
                    weights[lower] = weights[upper]
                    weights[upper] = temporary_weight
                    temporary_defect = defects[lower]
                    defects[lower] = defects[upper]
                    defects[upper] = temporary_defect
                    temporary_origin = origins[lower]
                    origins[lower] = origins[upper]
                    origins[upper] = temporary_origin
            parity = 1 - parity
            hot_visits[origins[-1]] += 1
            cold_visits[origins[0]] += 1
            current_d0 = defects[0] == 0
            if previous_d0 and not current_d0:
                d0_leaves += 1
            elif not previous_d0 and current_d0:
                d0_returns += 1
            elif previous_d0 and current_d0 and labels[0] != previous_d0_label:
                d0_label_changes += 1
            if current_d0:
                previous_d0_label = labels[0]
            previous_d0 = current_d0
            if record:
                _dt_pack_row(states[0], packed[round_index])
                output_labels[round_index] = labels[0]
                output_weights[round_index] = weights[0]
                output_defects[round_index] = defects[0]
        return (states, labels, weights, residuals, defects, origins, packed,
                output_labels, output_weights, output_defects, counters,
                hot_refresh_changes, swap_attempts, swap_accepts, hot_visits,
                cold_visits, d0_leaves, d0_returns, d0_label_changes)
else:  # pragma: no cover
    _run_phase_numba_core = None


def _run_phase_numba(model, frame, syndrome, config, replicas, origins,
                     seed_identity, stage, rounds, *, record):
    """Numba twin of the reference phase; all discrete draws must agree."""
    if _run_phase_numba_core is None:
        raise RuntimeError("Numba is required for the accelerated defect-tempered kernel")
    from exp101_certified_src.prng import PortablePrng

    states = np.ascontiguousarray([replica.state for replica in replicas], dtype=np.uint8)
    labels = np.ascontiguousarray([replica.label for replica in replicas], dtype=np.uint64)
    weights = np.ascontiguousarray([replica.weight for replica in replicas], dtype=np.int32)
    residuals = np.ascontiguousarray([replica.residual for replica in replicas], dtype=np.uint8)
    defects = np.ascontiguousarray([replica.defects for replica in replicas], dtype=np.int32)
    rung_rng_states = np.asarray([
        PortablePrng(seed_identity.seed(stage, "rung", rung)).state_array()
        for rung in range(config.num_replicas)
    ], dtype=np.uint64)
    swap_rng_state = PortablePrng(seed_identity.seed(stage, "swap")).state_array()
    check_indices, check_offsets = _column_check_csr(model)
    result = _run_phase_numba_core(
        states, labels, weights, residuals, defects,
        np.ascontiguousarray(origins, dtype=np.int32), rung_rng_states,
        swap_rng_state, np.asarray(config.kq_values, dtype=np.float64), float(config.p),
        float(coupling(config.p)), syndrome, check_indices, check_offsets,
        qubit_signatures(frame), int(rounds), int(config.sweeps_per_round), bool(record),
    )
    for rung in range(config.num_replicas):
        replicas[rung] = _ReplicaState(
            state=np.ascontiguousarray(result[0][rung], dtype=np.uint8),
            label=np.uint64(result[1][rung]),
            weight=int(result[2][rung]),
            residual=np.ascontiguousarray(result[3][rung], dtype=np.uint8),
            defects=int(result[4][rung]),
        )
    origins[:] = result[5]
    return {
        "states_packed": result[6],
        "labels": result[7],
        "weights": result[8],
        "defects": result[9],
        "bit_counters_by_rung": result[10],
        "hot_refresh_bit_changes": int(result[11]),
        "swap_attempts": result[12],
        "swap_accepts": result[13],
        "hot_visits_by_origin": result[14],
        "cold_visits_by_origin": result[15],
        "cold_d0_leaves": int(result[16]),
        "cold_d0_returns": int(result[17]),
        "cold_d0_label_changes": int(result[18]),
    }


def run_defect_tempered_trajectory(model, frame, syndrome, config,
                                   seed_identity, initial_state, *, engine="reference"):
    """Run one reference-only fixed-clock conditional trajectory.

    The stored cold-rung sequence includes nonzero-defect clocks.  Consumers
    must select only ``measurement_defects == 0`` before treating labels as
    hard-coset logical labels.
    """
    if not isinstance(config, DefectTemperedConfig):
        raise TypeError("config must be DefectTemperedConfig")
    if not isinstance(seed_identity, DefectTemperedSeedIdentity):
        raise TypeError("seed_identity must be DefectTemperedSeedIdentity")
    if config.method_id != seed_identity.method_id:
        raise DefectTemperedConflictError("config and seed method_id differ")
    if engine not in ("reference", "numba"):
        raise ValueError("engine must be reference or numba")
    try:
        validate_observable_frame(model, frame)
    except Exception as exc:
        raise DefectTemperedConflictError("observable frame does not match model") from exc
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    initial_state = np.ascontiguousarray(initial_state, dtype=np.uint8)
    if syndrome.shape != (model.num_checks,) or initial_state.shape != (model.num_qubits,):
        raise ValueError("state or syndrome dimensions do not match model")
    initial_residual = _residual(model, initial_state, syndrome)
    if initial_residual.any():
        raise DefectTemperedConflictError("diagnostic starts must lie in the hard coset")
    replicas = [
        _make_replica(model, frame, initial_state, syndrome)
        for _ in range(config.num_replicas)
    ]
    origins = np.arange(config.num_replicas, dtype=np.int32)
    initial = initial_state.copy()
    runner = _run_phase_reference if engine == "reference" else _run_phase_numba
    burn = runner(
        model, frame, syndrome, config, replicas, origins, seed_identity,
        "burn", int(config.burn_rounds), record=True,
    )
    burn_endpoint = replicas[0].state.copy()
    measurement = runner(
        model, frame, syndrome, config, replicas, origins, seed_identity,
        "measurement", int(config.measurement_rounds), record=True,
    )
    final = replicas[0]
    _validate_replica(model, frame, syndrome, final)
    measurement_states = unpack_states(
        measurement["states_packed"], model.num_qubits,
    )
    residuals = (
        model.H_check.astype(np.int64) @ measurement_states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    residual_weights = residuals.sum(axis=1).astype(np.int32)
    if not np.array_equal(residual_weights, measurement["defects"]):
        raise DefectTemperedConflictError("measurement defect replay drifted")
    replay_labels = np.asarray(
        [state_label(frame, state) for state in measurement_states], dtype=np.uint64,
    )
    if not np.array_equal(replay_labels, measurement["labels"]):
        raise DefectTemperedConflictError("measurement label replay drifted")
    if not np.array_equal(
            measurement_states.sum(axis=1).astype(np.int32), measurement["weights"]):
        raise DefectTemperedConflictError("measurement weight replay drifted")
    burn_d0 = np.asarray(burn["defects"] == 0, dtype=np.uint8)
    measurement_d0 = np.asarray(measurement["defects"] == 0, dtype=np.uint8)
    return {
        "raw_version": DEFECT_TEMPERED_RAW_VERSION,
        "kernel": DEFECT_TEMPERED_KERNEL,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "initial_state_packed": pack_state(initial),
        "burn_state_packed": pack_state(burn_endpoint),
        "final_state_packed": pack_state(final.state),
        "burn_states_packed": burn["states_packed"],
        "burn_labels": burn["labels"],
        "burn_weights": burn["weights"],
        "burn_defects": burn["defects"],
        "burn_d0_mask": burn_d0,
        "measurement_states_packed": measurement["states_packed"],
        "measurement_labels": measurement["labels"],
        "measurement_weights": measurement["weights"],
        "measurement_defects": measurement["defects"],
        "measurement_d0_mask": measurement_d0,
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), int(config.measurement_rounds) // 8,
        ),
        "initial_label": np.uint64(state_label(frame, initial)),
        "burn_label": np.uint64(state_label(frame, burn_endpoint)),
        "final_label": np.uint64(final.label),
        "ladder_kq": np.asarray(config.kq_values, dtype=np.float64),
        "ladder_sha256": config.ladder_sha256,
        "burn_bit_counters_by_rung": burn["bit_counters_by_rung"],
        "measurement_bit_counters_by_rung": measurement["bit_counters_by_rung"],
        "burn_hot_refresh_bit_changes": np.int64(burn["hot_refresh_bit_changes"]),
        "measurement_hot_refresh_bit_changes": np.int64(measurement["hot_refresh_bit_changes"]),
        "burn_swap_attempts": burn["swap_attempts"],
        "burn_swap_accepts": burn["swap_accepts"],
        "measurement_swap_attempts": measurement["swap_attempts"],
        "measurement_swap_accepts": measurement["swap_accepts"],
        "burn_hot_visits_by_origin": burn["hot_visits_by_origin"],
        "burn_cold_visits_by_origin": burn["cold_visits_by_origin"],
        "measurement_hot_visits_by_origin": measurement["hot_visits_by_origin"],
        "measurement_cold_visits_by_origin": measurement["cold_visits_by_origin"],
        "burn_cold_d0_leaves": np.int64(burn["cold_d0_leaves"]),
        "burn_cold_d0_returns": np.int64(burn["cold_d0_returns"]),
        "burn_cold_d0_label_changes": np.int64(burn["cold_d0_label_changes"]),
        "measurement_cold_d0_leaves": np.int64(measurement["cold_d0_leaves"]),
        "measurement_cold_d0_returns": np.int64(measurement["cold_d0_returns"]),
        "measurement_cold_d0_label_changes": np.int64(measurement["cold_d0_label_changes"]),
        "measurement_residual_weights": residual_weights,
        "engine": engine,
    }
