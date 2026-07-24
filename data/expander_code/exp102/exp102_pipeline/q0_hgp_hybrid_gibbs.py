"""Two-scale exact collapsed-B Gibbs using one column and one row per clock."""

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
    build_full_column_direct_block_cache,
    build_full_column_direct_block_workspace,
    full_column_direct_block_gibbs_update,
)
from .q0_hgp_full_row_gibbs import (
    build_full_row_elimination_plan,
    full_row_elimination_gibbs_update,
)


HYBRID_GIBBS_VERSION = "exp102.q0_hgp_hybrid_row_column.v0"
HYBRID_GIBBS_RAW_VERSION = "exp102.q0_hgp_hybrid_row_column.raw.v0"
HYBRID_GIBBS_METHOD_ID = "HRC1-C24-DPB12-R24-VE12"
HYBRID_GIBBS_SCHEDULE = "uniform_column_then_uniform_row_per_macroclock"
HYBRID_GIBBS_ENGINE = "numba_direct_positive_block12_plus_numpy_ve12"
HYBRID_COUNTERS = (
    "macroclocks",
    "column_updates",
    "column_changes",
    "column_changed_bits",
    "row_updates",
    "row_changes",
    "row_changed_bits",
    "a_conditional_draws",
    "logical_label_changes",
)


class HybridGibbsError(ValueError):
    """Raised when the hybrid collapsed sampler loses an invariant."""


def _require(condition, message):
    if not condition:
        raise HybridGibbsError(message)


@dataclass(frozen=True)
class HybridGibbsConfig:
    p: float
    burn_clocks: int
    measurement_clocks: int
    method_id: str = HYBRID_GIBBS_METHOD_ID
    schedule: str = HYBRID_GIBBS_SCHEDULE
    conditional_engine: str = HYBRID_GIBBS_ENGINE

    def __post_init__(self):
        p = float(self.p)
        _require(math.isfinite(p) and 0.0 < p < 0.5,
                 "hybrid Gibbs p must lie in (0,.5)")
        object.__setattr__(self, "p", p)
        for name in ("burn_clocks", "measurement_clocks"):
            value = getattr(self, name)
            _require(not isinstance(value, bool) and int(value) > 0,
                     f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        _require(self.measurement_clocks % 8 == 0,
                 "hybrid measurement clocks must divide into eight blocks")
        _require(self.method_id == HYBRID_GIBBS_METHOD_ID,
                 "hybrid Gibbs method changed")
        _require(self.schedule == HYBRID_GIBBS_SCHEDULE,
                 "hybrid Gibbs schedule changed")
        _require(self.conditional_engine == HYBRID_GIBBS_ENGINE,
                 "hybrid Gibbs engine changed")

    def as_dict(self):
        return {
            "burn_clocks": self.burn_clocks,
            "conditional_engine": self.conditional_engine,
            "measurement_clocks": self.measurement_clocks,
            "method_id": self.method_id,
            "p": self.p,
            "schedule": self.schedule,
        }


def _new_rng(seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    return PortablePrng(int(seed))


def _b_weight(columns):
    return sum(
        int(value).bit_count()
        for value in np.asarray(columns, dtype=np.uint32)
    )


def hybrid_gibbs_clock(
        b_columns, a_syndromes, H, syndrome_matrix, mass, log_mass, log_odds,
        column_cache, column_workspace, row_plan, rng, *,
        column_engine="numba"):
    """Apply one state-independent exact column-then-row macroclock."""
    rows = H.shape[0]
    selected_column = int(rng.randbelow(rows))
    old_column = np.uint32(b_columns[selected_column])
    column_changed, column_changed_bits = full_column_direct_block_gibbs_update(
        b_columns, a_syndromes, H, syndrome_matrix, selected_column, mass,
        column_cache, column_workspace, rng, engine=column_engine,
    )
    new_column = np.uint32(b_columns[selected_column])
    selected_row = int(rng.randbelow(rows))
    row_changed, row_changed_bits, old_row, new_row = (
        full_row_elimination_gibbs_update(
            b_columns, a_syndromes, H, syndrome_matrix, selected_row,
            log_mass, log_odds, rng, plan=row_plan,
        )
    )
    return {
        "column_changed": bool(column_changed),
        "column_changed_bits": int(column_changed_bits),
        "new_column": int(new_column),
        "new_row": int(new_row),
        "old_column": int(old_column),
        "old_row": int(old_row),
        "row_changed": bool(row_changed),
        "row_changed_bits": int(row_changed_bits),
        "selected_column": selected_column,
        "selected_row": selected_row,
    }


def _run_stage(model, frame, matrix, syndrome_matrix, b_columns, a_syndromes,
               state, config, update_rng, observation_rng, clocks, *, record,
               mass, log_mass, log_odds, column_cache, column_workspace,
               row_plan, column_engine):
    rows, columns = matrix.shape
    counters = np.zeros(len(HYBRID_COUNTERS), dtype=np.int64)
    transcript = {
        "selected_columns": np.empty(clocks, dtype=np.int16),
        "old_columns": np.empty(clocks, dtype=np.uint32),
        "new_columns": np.empty(clocks, dtype=np.uint32),
        "selected_rows": np.empty(clocks, dtype=np.int16),
        "old_rows": np.empty(clocks, dtype=np.uint32),
        "new_rows": np.empty(clocks, dtype=np.uint32),
    }
    if record:
        transcript.update({
            "b_columns": np.empty((clocks, rows), dtype=np.uint32),
            "b_likelihood": np.empty(clocks, dtype=np.float64),
            "b_weights": np.empty(clocks, dtype=np.int16),
            "blocks": np.empty(clocks, dtype=np.int8),
            "labels": np.empty(clocks, dtype=np.uint64),
            "states_packed": np.empty(
                (clocks, (model.num_qubits + 7) // 8), dtype=np.uint8,
            ),
            "weights": np.empty(clocks, dtype=np.int32),
        })
    previous_label = np.uint64(_state_label(frame, state))
    section_masks, kernel_combinations = _section_and_kernel_masks(matrix)
    odds = config.p / (1.0 - config.p)
    odds_powers = np.asarray(
        [odds ** weight for weight in range(columns + 1)], dtype=np.float64,
    )
    qubit_signatures = _qubit_signatures(frame)
    for clock in range(clocks):
        step = hybrid_gibbs_clock(
            b_columns, a_syndromes, matrix, syndrome_matrix, mass, log_mass,
            log_odds, column_cache, column_workspace, row_plan, update_rng,
            column_engine=column_engine,
        )
        counters[0] += 1
        counters[1] += 1
        counters[2] += int(step["column_changed"])
        counters[3] += step["column_changed_bits"]
        counters[4] += 1
        counters[5] += int(step["row_changed"])
        counters[6] += step["row_changed_bits"]
        transcript["selected_columns"][clock] = step["selected_column"]
        transcript["old_columns"][clock] = step["old_column"]
        transcript["new_columns"][clock] = step["new_column"]
        transcript["selected_rows"][clock] = step["selected_row"]
        transcript["old_rows"][clock] = step["old_row"]
        transcript["new_rows"][clock] = step["new_row"]
        if record:
            state, label, weight = _reference_sample_full_state(
                state, b_columns, a_syndromes, observation_rng, columns, rows,
                section_masks, kernel_combinations, odds_powers,
                qubit_signatures,
            )
            counters[7] += 1
            if label != previous_label:
                counters[8] += 1
            previous_label = label
            residual = (
                model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
            ).astype(np.uint8)
            _require(np.array_equal(residual, syndrome_matrix.reshape(-1)),
                     "hybrid Gibbs observation left the hard coset")
            _require(
                _state_label(frame, state) == label
                and int(state.sum()) == int(weight),
                "hybrid Gibbs observation statistics changed",
            )
            transcript["b_columns"][clock] = b_columns
            transcript["b_likelihood"][clock] = float(
                log_mass[a_syndromes].sum()
            )
            transcript["b_weights"][clock] = _b_weight(b_columns)
            transcript["blocks"][clock] = min(7, 8 * clock // clocks)
            transcript["labels"][clock] = label
            transcript["states_packed"][clock] = np.packbits(
                state, bitorder="little",
            )
            transcript["weights"][clock] = weight
    return state, counters, transcript


def run_hybrid_gibbs_trajectory(
        model, frame, H, syndrome, config, initial_state, burn_update_seed,
        measurement_update_seed, observation_seed, *, mass=None,
        column_cache=None, column_workspace=None, row_plan=None,
        column_engine="numba"):
    """Run one fixed-clock exact hybrid collapsed-B trajectory."""
    _require(isinstance(config, HybridGibbsConfig),
             "hybrid Gibbs config has the wrong type")
    matrix = np.ascontiguousarray(H, dtype=np.uint8)
    validate_hgp_wiring(matrix, model)
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise HybridGibbsError("hybrid Gibbs frame changed") from exc
    y = np.asarray(syndrome, dtype=np.uint8)
    initial = np.asarray(initial_state, dtype=np.uint8)
    _require(
        y.shape == (model.num_checks,)
        and initial.shape == (model.num_qubits,),
        "hybrid Gibbs input dimensions changed",
    )
    residual = (
        model.H_check.astype(np.int64) @ initial.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(np.array_equal(residual, y),
             "hybrid Gibbs initial state leaves the hard coset")
    b_columns, a_syndromes, _ = _initial_collapsed_masks(initial, y, matrix)
    if mass is None:
        mass = build_classical_coset_mass(matrix, config.p, engine="numba")
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    _require(
        mass.shape == (1 << matrix.shape[0],)
        and np.all(np.isfinite(mass)) and np.all(mass > 0.0),
        "hybrid Gibbs mass table is invalid",
    )
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    log_odds = math.log(config.p / (1.0 - config.p))
    column_cache = (
        build_full_column_direct_block_cache(matrix.shape[0], config.p, mass)
        if column_cache is None else column_cache
    )
    column_workspace = (
        build_full_column_direct_block_workspace(column_cache)
        if column_workspace is None else column_workspace
    )
    row_plan = (
        build_full_row_elimination_plan(matrix)
        if row_plan is None else row_plan
    )
    _require(
        column_cache.rows == matrix.shape[0]
        and column_cache.p == config.p
        and column_cache.mass is mass,
        "hybrid Gibbs column cache changed",
    )
    row_plan.validate(matrix)
    syndrome_matrix = y.reshape(matrix.shape)
    initial_b = b_columns.copy()
    state, burn_counters, burn = _run_stage(
        model, frame, matrix, syndrome_matrix, b_columns, a_syndromes,
        initial.copy(), config, _new_rng(burn_update_seed),
        _new_rng(observation_seed ^ 0x243F6A8885A308D3),
        config.burn_clocks, record=False, mass=mass, log_mass=log_mass,
        log_odds=log_odds, column_cache=column_cache,
        column_workspace=column_workspace, row_plan=row_plan,
        column_engine=column_engine,
    )
    burn_b = b_columns.copy()
    burn_a = a_syndromes.copy()
    state, measurement_counters, measurement = _run_stage(
        model, frame, matrix, syndrome_matrix, b_columns, a_syndromes, state,
        config, _new_rng(measurement_update_seed), _new_rng(observation_seed),
        config.measurement_clocks, record=True, mass=mass,
        log_mass=log_mass, log_odds=log_odds, column_cache=column_cache,
        column_workspace=column_workspace, row_plan=row_plan,
        column_engine=column_engine,
    )
    identity = hashlib.sha256(
        HYBRID_GIBBS_VERSION.encode("ascii") + b"\0"
        + np.asarray(
            [burn_update_seed, measurement_update_seed, observation_seed],
            dtype=">u8",
        ).tobytes()
        + np.asarray(config.p, dtype=">f8").tobytes()
        + np.asarray(
            [config.burn_clocks, config.measurement_clocks], dtype=">u8",
        ).tobytes()
    ).hexdigest()
    return {
        "burn__counters": burn_counters,
        "burn__final_a_syndromes": burn_a,
        "burn__final_b_columns": burn_b,
        **{f"burn__{key}": value for key, value in burn.items()},
        "conditional_engine": np.array(config.conditional_engine),
        "final_a_syndromes": a_syndromes.copy(),
        "final_b_columns": b_columns.copy(),
        "final_state_packed": np.packbits(state, bitorder="little"),
        "initial_b_columns": initial_b,
        "initial_state_packed": np.packbits(initial, bitorder="little"),
        "measurement__counters": measurement_counters,
        **{f"measurement__{key}": value
           for key, value in measurement.items()},
        "seed_identity_sha256": np.array(identity),
        "version": np.array(HYBRID_GIBBS_VERSION),
    }


def replay_hybrid_gibbs_trajectory(
        model, frame, H, syndrome, config, initial_state, burn_update_seed,
        measurement_update_seed, observation_seed, raw, *, mass=None,
        column_cache=None, column_workspace=None, row_plan=None,
        column_engine="numba"):
    """Rerun a hybrid trajectory and require exact raw equality."""
    expected = run_hybrid_gibbs_trajectory(
        model, frame, H, syndrome, config, initial_state, burn_update_seed,
        measurement_update_seed, observation_seed, mass=mass,
        column_cache=column_cache, column_workspace=column_workspace,
        row_plan=row_plan, column_engine=column_engine,
    )
    _require(set(expected) == set(raw), "hybrid Gibbs raw schema changed")
    for name, value in expected.items():
        _require(
            np.array_equal(
                np.asarray(value), np.asarray(raw[name]), equal_nan=False,
            ),
            f"hybrid Gibbs replay mismatch: {name}",
        )
    return True
