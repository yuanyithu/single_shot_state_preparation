"""Frozen Q32 ladder construction and identity for exp102 PT v2."""

import math

import numpy as np

from .io import sha256_json
from .q0_pt import (
    Q32_ONE,
    Q0PtConfig,
    coupling,
    ladder_x_q32_sha256,
    validate_ladder_x_q32,
)


LADDER_RECORD_FIELDS = {
    "ladder_id",
    "p_hot",
    "num_temperatures",
    "ladder_x_q32",
    "ladder_x_sha256",
    "ladder_generation",
}
PT_RUNTIME_FIELDS = {
    "gamma",
    "burn_rounds",
    "measurement_rounds",
    "sweeps_per_round",
    "logical_move_repeat",
    "swap_sweeps_per_round",
}
PT_CANDIDATE_FIELDS = LADDER_RECORD_FIELDS | PT_RUNTIME_FIELDS


def uniform_ladder_q32(num_temperatures):
    num_temperatures = _positive_integer(num_temperatures, "num_temperatures")
    if num_temperatures < 2 or num_temperatures > Q32_ONE + 1:
        raise ValueError("num_temperatures cannot form a strict Q32 ladder")
    denominator = num_temperatures - 1
    values = tuple(
        (index * Q32_ONE + denominator // 2) // denominator
        for index in range(num_temperatures)
    )
    return validate_ladder_x_q32(values, num_temperatures)


def piecewise_density_ladder_q32(
    p_hot,
    num_temperatures,
    density,
    reference_p_cold=0.04,
    focus_p_min=0.20,
    focus_p_max=0.32,
):
    """Place ``density`` times as many rungs per coupling unit in a fixed window."""
    p_hot = float(p_hot)
    num_temperatures = _positive_integer(num_temperatures, "num_temperatures")
    density = float(density)
    if (not math.isfinite(density) or density <= 0.0
            or not 0.0 < reference_p_cold < focus_p_min < focus_p_max < p_hot < 0.5):
        raise ValueError("invalid piecewise-density ladder parameters")
    cold_k = coupling(reference_p_cold)
    hot_k = coupling(p_hot)

    def fraction(probability):
        return (coupling(probability) - cold_k) / (hot_k - cold_k)

    left = fraction(focus_p_min)
    right = fraction(focus_p_max)
    weighted_total = left + density * (right - left) + (1.0 - right)
    fractions = []
    for index in range(num_temperatures):
        target = weighted_total * index / (num_temperatures - 1)
        if target <= left:
            value = target
        elif target <= left + density * (right - left):
            value = left + (target - left) / density
        else:
            value = right + target - left - density * (right - left)
        fractions.append(value)
    return _quantize_strict_q32(fractions)


def make_uniform_ladder(ladder_id, p_hot, num_temperatures):
    generation = {"algorithm": "uniform_q32.v1"}
    return _make_ladder_record(
        ladder_id, p_hot, num_temperatures,
        uniform_ladder_q32(num_temperatures), generation,
    )


def make_piecewise_density_ladder(
    ladder_id,
    p_hot,
    num_temperatures,
    density,
    reference_p_cold=0.04,
    focus_p_min=0.20,
    focus_p_max=0.32,
):
    generation = {
        "algorithm": "piecewise_density_q32.v1",
        "density": float(density),
        "reference_p_cold": float(reference_p_cold),
        "focus_p_min": float(focus_p_min),
        "focus_p_max": float(focus_p_max),
    }
    values = piecewise_density_ladder_q32(
        p_hot, num_temperatures, density, reference_p_cold,
        focus_p_min, focus_p_max,
    )
    return _make_ladder_record(ladder_id, p_hot, num_temperatures, values, generation)


def validate_ladder_record(record):
    if not isinstance(record, dict) or set(record) != LADDER_RECORD_FIELDS:
        raise ValueError("ladder record fields are incomplete")
    ladder_id = record["ladder_id"]
    if (not isinstance(ladder_id, str) or not ladder_id
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
                   for character in ladder_id)):
        raise ValueError("ladder_id is invalid")
    p_hot = float(record["p_hot"])
    if not 0.0 < p_hot < 0.5:
        raise ValueError("ladder p_hot must lie in (0,0.5)")
    num_temperatures = _positive_integer(record["num_temperatures"], "num_temperatures")
    values = validate_ladder_x_q32(record["ladder_x_q32"], num_temperatures)
    if record["ladder_x_sha256"] != ladder_x_q32_sha256(values):
        raise ValueError("ladder_x_q32 SHA256 mismatch")
    generation = record["ladder_generation"]
    if not isinstance(generation, dict) or "algorithm" not in generation:
        raise ValueError("ladder generation metadata is incomplete")
    algorithm = generation["algorithm"]
    if algorithm == "uniform_q32.v1":
        if set(generation) != {"algorithm"}:
            raise ValueError("uniform ladder generation fields are invalid")
        regenerated = uniform_ladder_q32(num_temperatures)
    elif algorithm == "piecewise_density_q32.v1":
        expected_fields = {
            "algorithm", "density", "reference_p_cold",
            "focus_p_min", "focus_p_max",
        }
        if set(generation) != expected_fields:
            raise ValueError("piecewise ladder generation fields are invalid")
        regenerated = piecewise_density_ladder_q32(
            p_hot, num_temperatures, generation["density"],
            generation["reference_p_cold"], generation["focus_p_min"],
            generation["focus_p_max"],
        )
    else:
        raise ValueError("unknown ladder generation algorithm")
    if values != regenerated:
        raise ValueError("frozen ladder differs from its generation parameters")
    return {
        "ladder_id": ladder_id,
        "p_hot": p_hot,
        "num_temperatures": num_temperatures,
        "ladder_x_q32": list(values),
        "ladder_x_sha256": record["ladder_x_sha256"],
        "ladder_generation": dict(generation),
    }


def ladder_fingerprint(record):
    return sha256_json(validate_ladder_record({key: record[key] for key in LADDER_RECORD_FIELDS}))


def make_pt_candidate(ladder, burn_rounds, measurement_rounds,
                      swap_sweeps_per_round, gamma=1.0,
                      sweeps_per_round=1, logical_move_repeat=1):
    candidate = {
        **validate_ladder_record(ladder),
        "gamma": float(gamma),
        "burn_rounds": int(burn_rounds),
        "measurement_rounds": int(measurement_rounds),
        "sweeps_per_round": int(sweeps_per_round),
        "logical_move_repeat": int(logical_move_repeat),
        "swap_sweeps_per_round": int(swap_sweeps_per_round),
    }
    return validate_pt_candidate(candidate)


def validate_pt_candidate(candidate):
    if not isinstance(candidate, dict) or set(candidate) != PT_CANDIDATE_FIELDS:
        raise ValueError("PT candidate fields are incomplete")
    ladder = validate_ladder_record({key: candidate[key] for key in LADDER_RECORD_FIELDS})
    config = Q0PtConfig(
        p_hot=ladder["p_hot"],
        num_temperatures=ladder["num_temperatures"],
        gamma=float(candidate["gamma"]),
        burn_rounds=_nonnegative_integer(candidate["burn_rounds"], "burn_rounds"),
        measurement_rounds=_positive_integer(
            candidate["measurement_rounds"], "measurement_rounds",
        ),
        sweeps_per_round=_positive_integer(candidate["sweeps_per_round"], "sweeps_per_round"),
        logical_move_repeat=_positive_integer(
            candidate["logical_move_repeat"], "logical_move_repeat",
        ),
        ladder_x_q32=tuple(ladder["ladder_x_q32"]),
        swap_sweeps_per_round=_positive_integer(
            candidate["swap_sweeps_per_round"], "swap_sweeps_per_round",
        ),
    )
    if config.gamma != 1.0:
        raise ValueError("explicit Q32 ladders require gamma=1.0")
    return {
        **ladder,
        "gamma": config.gamma,
        "burn_rounds": config.burn_rounds,
        "measurement_rounds": config.measurement_rounds,
        "sweeps_per_round": config.sweeps_per_round,
        "logical_move_repeat": config.logical_move_repeat,
        "swap_sweeps_per_round": config.swap_sweeps_per_round,
    }


def q0_config_from_candidate(candidate):
    normalized = validate_pt_candidate(candidate)
    return Q0PtConfig(
        p_hot=normalized["p_hot"],
        num_temperatures=normalized["num_temperatures"],
        gamma=normalized["gamma"],
        burn_rounds=normalized["burn_rounds"],
        measurement_rounds=normalized["measurement_rounds"],
        sweeps_per_round=normalized["sweeps_per_round"],
        logical_move_repeat=normalized["logical_move_repeat"],
        ladder_x_q32=tuple(normalized["ladder_x_q32"]),
        swap_sweeps_per_round=normalized["swap_sweeps_per_round"],
    )


def _make_ladder_record(ladder_id, p_hot, num_temperatures, values, generation):
    raw = {
        "ladder_id": ladder_id,
        "p_hot": float(p_hot),
        "num_temperatures": int(num_temperatures),
        "ladder_x_q32": list(values),
        "ladder_x_sha256": ladder_x_q32_sha256(values),
        "ladder_generation": generation,
    }
    return validate_ladder_record(raw)


def _quantize_strict_q32(fractions):
    fractions = np.asarray(fractions, dtype=np.float64).copy()
    if (fractions.ndim != 1 or fractions.size < 2 or not np.all(np.isfinite(fractions))
            or not np.isclose(fractions[0], 0.0, rtol=0.0, atol=1e-15)
            or not np.isclose(fractions[-1], 1.0, rtol=0.0, atol=1e-15)
            or np.any(np.diff(fractions) <= 0.0)):
        raise ValueError("coupling fractions must be strictly increasing from zero to one")
    fractions[0], fractions[-1] = 0.0, 1.0
    values = np.rint(fractions * float(Q32_ONE)).astype(np.int64)
    values[0], values[-1] = 0, Q32_ONE
    for index in range(1, values.size):
        values[index] = max(values[index], values[index - 1] + 1)
    values[-1] = Q32_ONE
    for index in range(values.size - 2, -1, -1):
        values[index] = min(values[index], values[index + 1] - 1)
    return validate_ladder_x_q32(tuple(int(value) for value in values), values.size)


def _positive_integer(value, name):
    result = _nonnegative_integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
