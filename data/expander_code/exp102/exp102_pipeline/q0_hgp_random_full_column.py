"""Random-scan exact full-B-column Gibbs for the collapsed q=0 HGP target.

One clock selects one of the ``r`` B columns with a state-independent portable
draw and heatbaths all ``r`` bits exactly.  This differs deliberately from the
historical full-column runner, whose public sweep updated every column and was
therefore rejected by an overly expensive T1 projection.  Random scan is a
mixture of exact coordinate heatbaths and has the same collapsed posterior as
its invariant distribution without requiring 24 conditionals per clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .exp101_bridge import load_exp101
from .q0_global import GlobalConflictError, validate_observable_frame
from .q0_hgp_collapsed import (
    _initial_collapsed_masks,
    _qubit_signatures,
    _reference_sample_full_state,
    _section_and_kernel_masks,
    _state_label,
    build_classical_coset_mass,
    validate_hgp_wiring,
)
from .q0_hgp_full_column_gibbs import (
    FullColumnGibbsConflictError,
    build_full_column_candidate_cache,
    build_full_column_streaming_cache,
    build_full_column_streaming_workspace,
    build_full_column_workspace,
    full_column_gibbs_update,
    full_column_streaming_gibbs_update,
)


RANDOM_FULL_COLUMN_VERSION = "exp102.q0_hgp_random_full_column.v0"
RANDOM_FULL_COLUMN_RAW_VERSION = "exp102.q0_hgp_random_full_column.raw.v0"
RANDOM_FULL_COLUMN_METHOD_ID = "RFCG-C24"
RANDOM_FULL_COLUMN_STREAMING_VERSION = "exp102.q0_hgp_random_full_column_streaming.v1"
RANDOM_FULL_COLUMN_STREAMING_RAW_VERSION = (
    "exp102.q0_hgp_random_full_column_streaming.raw.v1"
)
RANDOM_FULL_COLUMN_STREAMING_METHOD_ID = "RFCG-C24-S1"
RANDOM_FULL_COLUMN_COUNTERS = (
    "column_updates", "column_changes", "column_changed_bits", "a_conditional_draws",
    "logical_label_changes",
)


class RandomFullColumnError(ValueError):
    """Raised when a random-scan full-column trajectory loses an invariant."""


def _require(condition, message):
    if not condition:
        raise RandomFullColumnError(message)


@dataclass(frozen=True)
class RandomFullColumnConfig:
    p: float
    burn_updates: int
    measurement_updates: int
    method_id: str = RANDOM_FULL_COLUMN_METHOD_ID
    schedule: str = "portable_random_scan_one_full_B_column_per_clock"

    def __post_init__(self):
        p = float(self.p)
        _require(math.isfinite(p) and 0.0 < p < 0.5,
                 "random full-column p must lie in (0,.5)")
        object.__setattr__(self, "p", p)
        for name in ("burn_updates", "measurement_updates"):
            value = getattr(self, name)
            _require(not isinstance(value, bool) and int(value) > 0,
                     f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        _require(self.measurement_updates % 8 == 0,
                 "measurement updates must divide into eight blocks")
        _require(self.method_id == RANDOM_FULL_COLUMN_METHOD_ID,
                 "random full-column method changed")
        _require(self.schedule == "portable_random_scan_one_full_B_column_per_clock",
                 "random full-column schedule changed")

    def as_dict(self):
        return {
            "burn_updates": self.burn_updates,
            "measurement_updates": self.measurement_updates,
            "method_id": self.method_id,
            "p": self.p,
            "schedule": self.schedule,
        }


@dataclass(frozen=True)
class RandomFullColumnStreamingConfig:
    """Fresh exact-clock config for the memory-streaming successor."""

    p: float
    burn_updates: int
    measurement_updates: int
    method_id: str = RANDOM_FULL_COLUMN_STREAMING_METHOD_ID
    schedule: str = "portable_random_scan_one_full_B_column_per_clock"
    conditional_engine: str = "numba_streaming_cdf"

    def __post_init__(self):
        p = float(self.p)
        _require(math.isfinite(p) and 0.0 < p < 0.5,
                 "streaming random full-column p must lie in (0,.5)")
        object.__setattr__(self, "p", p)
        for name in ("burn_updates", "measurement_updates"):
            value = getattr(self, name)
            _require(not isinstance(value, bool) and int(value) > 0,
                     f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        _require(self.measurement_updates % 8 == 0,
                 "streaming measurement updates must divide into eight blocks")
        _require(self.method_id == RANDOM_FULL_COLUMN_STREAMING_METHOD_ID,
                 "streaming random full-column method changed")
        _require(self.schedule == "portable_random_scan_one_full_B_column_per_clock",
                 "streaming random full-column schedule changed")
        _require(self.conditional_engine == "numba_streaming_cdf",
                 "streaming full-column conditional engine changed")

    def as_dict(self):
        return {
            "burn_updates": self.burn_updates,
            "conditional_engine": self.conditional_engine,
            "measurement_updates": self.measurement_updates,
            "method_id": self.method_id,
            "p": self.p,
            "schedule": self.schedule,
        }


def _new_rng(seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    return PortablePrng(int(seed))


def _b_weight(columns):
    return sum(int(value).bit_count() for value in np.asarray(columns, dtype=np.uint32))


def _run_stage(model, frame, matrix, syndrome_matrix, b_columns, a_syndromes,
               state, config, update_rng, observation_rng, updates, *, record,
               log_mass, cache, workspace, gibbs_update=full_column_gibbs_update,
               gibbs_engine=None):
    rows, columns = matrix.shape
    counters = np.zeros(len(RANDOM_FULL_COLUMN_COUNTERS), dtype=np.int64)
    transcript = {
        "selected_columns": np.empty(updates, dtype=np.int16),
        "old_columns": np.empty(updates, dtype=np.uint32),
        "new_columns": np.empty(updates, dtype=np.uint32),
    }
    if record:
        transcript.update({
            "b_columns": np.empty((updates, rows), dtype=np.uint32),
            "b_likelihood": np.empty(updates, dtype=np.float64),
            "b_weights": np.empty(updates, dtype=np.int16),
            "blocks": np.empty(updates, dtype=np.int8),
            "labels": np.empty(updates, dtype=np.uint64),
            "states_packed": np.empty(
                (updates, (model.num_qubits + 7) // 8), dtype=np.uint8,
            ),
            "weights": np.empty(updates, dtype=np.int32),
        })
    previous_label = np.uint64(_state_label(frame, state))
    section_masks, kernel_combinations = _section_and_kernel_masks(matrix)
    odds = config.p / (1.0 - config.p)
    odds_powers = np.asarray(
        [odds ** weight for weight in range(columns + 1)], dtype=np.float64,
    )
    qubit_signatures = _qubit_signatures(frame)
    for update in range(updates):
        selected = int(update_rng.randbelow(rows))
        old = np.uint32(b_columns[selected])
        if gibbs_engine is None:
            changed, changed_bits = gibbs_update(
                b_columns, a_syndromes, matrix, syndrome_matrix, selected,
                log_mass, cache, workspace, update_rng,
            )
        else:
            changed, changed_bits = gibbs_update(
                b_columns, a_syndromes, matrix, syndrome_matrix, selected,
                log_mass, cache, workspace, update_rng, engine=gibbs_engine,
            )
        counters[0] += 1
        if changed:
            counters[1] += 1
            counters[2] += int(changed_bits)
        transcript["selected_columns"][update] = selected
        transcript["old_columns"][update] = old
        transcript["new_columns"][update] = b_columns[selected]
        if record:
            state, label, weight = _reference_sample_full_state(
                state, b_columns, a_syndromes, observation_rng, columns, rows,
                section_masks, kernel_combinations, odds_powers, qubit_signatures,
            )
            counters[3] += 1
            if label != previous_label:
                counters[4] += 1
            previous_label = label
            residual = (
                model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
            ).astype(np.uint8)
            _require(np.array_equal(residual, syndrome_matrix.reshape(-1)),
                     "random full-column state left the hard coset")
            _require(_state_label(frame, state) == label and int(state.sum()) == int(weight),
                     "random full-column state statistics drifted")
            transcript["b_columns"][update] = b_columns
            transcript["b_likelihood"][update] = float(log_mass[a_syndromes].sum())
            transcript["b_weights"][update] = _b_weight(b_columns)
            transcript["blocks"][update] = min(7, 8 * update // updates)
            transcript["labels"][update] = label
            transcript["states_packed"][update] = np.packbits(state, bitorder="little")
            transcript["weights"][update] = weight
    return state, counters, transcript


def run_random_full_column_trajectory(model, frame, H, syndrome, config,
                                      initial_state, burn_update_seed,
                                      measurement_update_seed,
                                      observation_seed, *, mass=None, cache=None,
                                      workspace=None):
    """Run one fixed-clock exact random-scan full-column trajectory."""
    _require(isinstance(config, RandomFullColumnConfig),
             "random full-column config has the wrong type")
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    validate_hgp_wiring(matrix, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise RandomFullColumnError("random full-column frame changed") from exc
    y = np.asarray(syndrome, dtype=np.uint8)
    initial = np.asarray(initial_state, dtype=np.uint8)
    _require(y.shape == (model.num_checks,)
             and initial.shape == (model.num_qubits,),
             "random full-column input dimensions changed")
    residual = (
        model.H_check.astype(np.int64) @ initial.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(np.array_equal(residual, y),
             "random full-column initial state leaves the hard coset")
    b_columns, a_syndromes, _ = _initial_collapsed_masks(initial, y, matrix)
    if mass is None:
        mass = build_classical_coset_mass(matrix, config.p, engine="reference")
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    _require(mass.shape == (1 << matrix.shape[0],)
             and np.all(np.isfinite(mass)) and np.all(mass > 0.0),
             "random full-column mass table is invalid")
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    cache = (
        build_full_column_candidate_cache(matrix.shape[0], config.p)
        if cache is None else cache
    )
    workspace = build_full_column_workspace(cache) if workspace is None else workspace
    _require(cache.rows == matrix.shape[0] and cache.p == config.p,
             "random full-column cache changed")
    syndrome_matrix = y.reshape(matrix.shape)
    initial_b = b_columns.copy()
    state, burn_counters, burn = _run_stage(
        model, frame, matrix, syndrome_matrix, b_columns, a_syndromes,
        initial.copy(), config, _new_rng(burn_update_seed),
        _new_rng(observation_seed ^ 0x243F6A8885A308D3), config.burn_updates,
        record=False, log_mass=log_mass, cache=cache, workspace=workspace,
    )
    burn_b = b_columns.copy()
    state, measurement_counters, measurement = _run_stage(
        model, frame, matrix, syndrome_matrix, b_columns, a_syndromes,
        state, config, _new_rng(measurement_update_seed),
        _new_rng(observation_seed), config.measurement_updates,
        record=True, log_mass=log_mass, cache=cache, workspace=workspace,
    )
    identity = hashlib.sha256(
        np.asarray([burn_update_seed, measurement_update_seed, observation_seed],
                   dtype=">u8").tobytes()
        + np.asarray(config.p, dtype=">f8").tobytes()
        + np.asarray([config.burn_updates, config.measurement_updates],
                     dtype=">u8").tobytes()
    ).hexdigest()
    return {
        "burn__counters": burn_counters,
        "burn__final_b_columns": burn_b,
        **{f"burn__{key}": value for key, value in burn.items()},
        "final_b_columns": b_columns.copy(),
        "final_state_packed": np.packbits(state, bitorder="little"),
        "initial_b_columns": initial_b,
        "initial_state_packed": np.packbits(initial, bitorder="little"),
        "measurement__counters": measurement_counters,
        **{f"measurement__{key}": value for key, value in measurement.items()},
        "seed_identity_sha256": np.array(identity),
    }


def replay_random_full_column_trajectory(model, frame, H, syndrome, config,
                                         initial_state, burn_update_seed,
                                         measurement_update_seed,
                                         observation_seed, raw, *, mass=None,
                                         cache=None, workspace=None):
    """Rerun a trajectory and require every stored discrete/float value exactly."""
    expected = run_random_full_column_trajectory(
        model, frame, H, syndrome, config, initial_state, burn_update_seed,
        measurement_update_seed, observation_seed, mass=mass, cache=cache,
        workspace=workspace,
    )
    _require(set(expected) == set(raw), "random full-column raw schema changed")
    for name, value in expected.items():
        _require(np.array_equal(np.asarray(value), np.asarray(raw[name]), equal_nan=False),
                 f"random full-column replay mismatch: {name}")
    return True


def run_random_full_column_streaming_trajectory(
        model, frame, H, syndrome, config, initial_state, burn_update_seed,
        measurement_update_seed, observation_seed, *, mass=None, cache=None,
        workspace=None):
    """Run the fresh memory-streaming implementation of the same exact kernel."""
    _require(isinstance(config, RandomFullColumnStreamingConfig),
             "streaming random full-column config has the wrong type")
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    validate_hgp_wiring(matrix, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise RandomFullColumnError("streaming random full-column frame changed") from exc
    y = np.asarray(syndrome, dtype=np.uint8)
    initial = np.asarray(initial_state, dtype=np.uint8)
    _require(y.shape == (model.num_checks,)
             and initial.shape == (model.num_qubits,),
             "streaming random full-column input dimensions changed")
    residual = (
        model.H_check.astype(np.int64) @ initial.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(np.array_equal(residual, y),
             "streaming random full-column initial state leaves the hard coset")
    b_columns, a_syndromes, _ = _initial_collapsed_masks(initial, y, matrix)
    if mass is None:
        mass = build_classical_coset_mass(matrix, config.p, engine="numba")
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    _require(mass.shape == (1 << matrix.shape[0],)
             and np.all(np.isfinite(mass)) and np.all(mass > 0.0),
             "streaming random full-column mass table is invalid")
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    cache = (
        build_full_column_streaming_cache(matrix.shape[0], config.p)
        if cache is None else cache
    )
    workspace = (
        build_full_column_streaming_workspace(cache)
        if workspace is None else workspace
    )
    _require(cache.rows == matrix.shape[0] and cache.p == config.p,
             "streaming random full-column cache changed")
    syndrome_matrix = y.reshape(matrix.shape)
    initial_b = b_columns.copy()
    state, burn_counters, burn = _run_stage(
        model, frame, matrix, syndrome_matrix, b_columns, a_syndromes,
        initial.copy(), config, _new_rng(burn_update_seed),
        _new_rng(observation_seed ^ 0x243F6A8885A308D3), config.burn_updates,
        record=False, log_mass=log_mass, cache=cache, workspace=workspace,
        gibbs_update=full_column_streaming_gibbs_update, gibbs_engine="numba",
    )
    burn_b = b_columns.copy()
    state, measurement_counters, measurement = _run_stage(
        model, frame, matrix, syndrome_matrix, b_columns, a_syndromes,
        state, config, _new_rng(measurement_update_seed),
        _new_rng(observation_seed), config.measurement_updates,
        record=True, log_mass=log_mass, cache=cache, workspace=workspace,
        gibbs_update=full_column_streaming_gibbs_update, gibbs_engine="numba",
    )
    identity = hashlib.sha256(
        RANDOM_FULL_COLUMN_STREAMING_VERSION.encode("ascii") + b"\0"
        + np.asarray(
            [burn_update_seed, measurement_update_seed, observation_seed],
            dtype=">u8",
        ).tobytes()
        + np.asarray(config.p, dtype=">f8").tobytes()
        + np.asarray(
            [config.burn_updates, config.measurement_updates], dtype=">u8",
        ).tobytes()
    ).hexdigest()
    return {
        "burn__counters": burn_counters,
        "burn__final_b_columns": burn_b,
        **{f"burn__{key}": value for key, value in burn.items()},
        "conditional_engine": np.array(config.conditional_engine),
        "final_b_columns": b_columns.copy(),
        "final_state_packed": np.packbits(state, bitorder="little"),
        "initial_b_columns": initial_b,
        "initial_state_packed": np.packbits(initial, bitorder="little"),
        "measurement__counters": measurement_counters,
        **{f"measurement__{key}": value for key, value in measurement.items()},
        "seed_identity_sha256": np.array(identity),
        "version": np.array(RANDOM_FULL_COLUMN_STREAMING_VERSION),
    }


def replay_random_full_column_streaming_trajectory(
        model, frame, H, syndrome, config, initial_state, burn_update_seed,
        measurement_update_seed, observation_seed, raw, *, mass=None, cache=None,
        workspace=None):
    """Rerun the streaming implementation and require bit-exact raw."""
    expected = run_random_full_column_streaming_trajectory(
        model, frame, H, syndrome, config, initial_state, burn_update_seed,
        measurement_update_seed, observation_seed, mass=mass, cache=cache,
        workspace=workspace,
    )
    _require(set(expected) == set(raw),
             "streaming random full-column raw schema changed")
    for name, value in expected.items():
        _require(np.array_equal(np.asarray(value), np.asarray(raw[name]), equal_nan=False),
                 f"streaming random full-column replay mismatch: {name}")
    return True
