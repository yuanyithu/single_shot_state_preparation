"""Fail-closed aggregation of the exp104 ensemble scan.

Fail-closed means the same thing it meant in exp103: a cell with any invalid or
missing task is not repaired, not dropped and not silently down-weighted. A
missing task makes its whole (m, p) column INCOMPLETE and the run's overall
status INCOMPLETE, and every published statistic that depends on it becomes NaN.
"""

import numpy as np

from . import AGGREGATE_SCHEMA, EXPERIMENT_ID
from .config import (
    CODES_PER_M,
    M_VALUES,
    TASKS_PER_M,
    block_code_indices,
    ensure_config,
)
from .crossing import (
    classify_crossing,
    cluster_bootstrap,
    crossing_location,
    wilson_interval,
)
from .io import arrays_sha256, canonical_json
from .raw import load_raw


DISTANCE_STRATA = (2, 4, 6, 8, 10, 12)

ARRAY_FIELDS = (
    "p_values", "m_values", "code_m", "code_index", "classical_distance",
    "code_status", "failure_counts", "trial_counts", "code_rates",
    "wilson_low", "wilson_high", "bp_convergence_rate", "mean_bp_iterations",
    "syndrome_mismatch_rate", "mean_logical_weight",
    "m_status", "primary_mean", "primary_failures", "primary_trials",
    "pooled_binomial_se", "cluster_se", "between_code_std",
    "primary_band_low", "primary_band_high",
    "delta38", "delta38_band_low", "delta38_band_high",
    "adjacent_delta", "adjacent_band_low", "adjacent_band_high",
    "distance_values", "strata_failures", "strata_trials", "strata_rate",
    "strata_code_counts",
)
SCALAR_FIELDS = (
    "schema_version", "experiment_id", "config_sha256", "registry_sha256",
    "source_commit", "source_tree_sha256", "decoder_binary_sha256",
    "overall_status", "terminal_status", "crossing_bracket_low",
    "crossing_bracket_high", "certified_negative_p_json",
    "certified_positive_p_json", "bootstrap_half_width", "p_cross",
    "p_cross_low", "p_cross_high", "p_cross_defined_fraction",
    "p_cross_reason", "codes_per_m", "trials_per_code_p",
    "replay_status", "replay_scope", "replay_report_sha256",
    "raw_manifest_sha256", "replay_report_json", "unexpected_raw_errors_json",
    "payload_sha256",
)


def _blank_arrays(config):
    n_codes = len(M_VALUES) * CODES_PER_M
    n_p = len(config["p_tokens"])
    n_m = len(M_VALUES)
    n_d = len(DISTANCE_STRATA)
    return {
        "p_values": np.asarray([float(token) for token in config["p_tokens"]]),
        "m_values": np.asarray(M_VALUES, dtype=np.int16),
        "distance_values": np.asarray(DISTANCE_STRATA, dtype=np.int16),
        "code_m": np.asarray(
            [m for m in M_VALUES for _ in range(CODES_PER_M)], dtype=np.int16,
        ),
        "code_index": np.asarray(
            [index for _ in M_VALUES for index in range(CODES_PER_M)], dtype=np.int32,
        ),
        "classical_distance": np.full(n_codes, -1, dtype=np.int16),
        # Wide enough for the longest status token; a narrow dtype would
        # truncate SAMPLING_INSUFFICIENT silently.
        "code_status": np.full((n_codes, n_p), "MISSING", dtype="<U24"),
        "failure_counts": np.zeros((n_codes, n_p), dtype=np.int64),
        "trial_counts": np.zeros((n_codes, n_p), dtype=np.int64),
        "code_rates": np.full((n_codes, n_p), np.nan),
        "wilson_low": np.full((n_codes, n_p), np.nan),
        "wilson_high": np.full((n_codes, n_p), np.nan),
        "bp_convergence_rate": np.full((n_codes, n_p), np.nan),
        "mean_bp_iterations": np.full((n_codes, n_p), np.nan),
        "syndrome_mismatch_rate": np.full((n_codes, n_p), np.nan),
        "mean_logical_weight": np.full((n_codes, n_p), np.nan),
        "m_status": np.full((n_m, n_p), "MISSING", dtype="<U24"),
        "primary_mean": np.full((n_m, n_p), np.nan),
        "primary_failures": np.zeros((n_m, n_p), dtype=np.int64),
        "primary_trials": np.zeros((n_m, n_p), dtype=np.int64),
        "pooled_binomial_se": np.full((n_m, n_p), np.nan),
        "cluster_se": np.full((n_m, n_p), np.nan),
        "between_code_std": np.full((n_m, n_p), np.nan),
        "primary_band_low": np.full((n_m, n_p), np.nan),
        "primary_band_high": np.full((n_m, n_p), np.nan),
        "delta38": np.full(n_p, np.nan),
        "delta38_band_low": np.full(n_p, np.nan),
        "delta38_band_high": np.full(n_p, np.nan),
        "adjacent_delta": np.full((n_m - 1, n_p), np.nan),
        "adjacent_band_low": np.full((n_m - 1, n_p), np.nan),
        "adjacent_band_high": np.full((n_m - 1, n_p), np.nan),
        "strata_failures": np.zeros((n_m, n_d, n_p), dtype=np.int64),
        "strata_trials": np.zeros((n_m, n_d, n_p), dtype=np.int64),
        "strata_rate": np.full((n_m, n_d, n_p), np.nan),
        "strata_code_counts": np.zeros((n_m, n_d), dtype=np.int32),
    }


def _global_code_slot(m, index):
    return M_VALUES.index(int(m)) * CODES_PER_M + int(index)


def _ingest_task(arrays, raw, config, errors):
    m = int(raw["m"])
    block_index = int(raw["block_index"])
    tokens = list(config["p_tokens"])
    trials = int(config["trials_per_code_p"])
    if raw["schema_version"] != "exp104.raw.v1" or raw["experiment_id"] != EXPERIMENT_ID:
        errors.append({"m": m, "block_index": block_index, "reason": "schema_mismatch"})
        return
    if raw["p_tokens"] != ",".join(tokens) or int(raw["trials_per_code_p"]) != trials:
        errors.append({"m": m, "block_index": block_index, "reason": "grid_mismatch"})
        return
    indices = block_code_indices(m, block_index)
    if raw["status"] != "VALID" or int(raw["completed_codes"]) != len(indices):
        for index in indices:
            arrays["code_status"][_global_code_slot(m, index), :] = "INVALID"
        return
    for slot, index in enumerate(indices):
        code_slot = _global_code_slot(m, index)
        arrays["classical_distance"][code_slot] = int(raw["classical_distance"][slot])
        for p_index in range(len(tokens)):
            failures = raw["failure_flags"][slot, p_index]
            arrays["code_status"][code_slot, p_index] = "REPORTABLE"
            arrays["failure_counts"][code_slot, p_index] = int(failures.sum())
            arrays["trial_counts"][code_slot, p_index] = trials
            rate = float(failures.sum()) / trials
            arrays["code_rates"][code_slot, p_index] = rate
            low, high = wilson_interval(int(failures.sum()), trials)
            arrays["wilson_low"][code_slot, p_index] = low
            arrays["wilson_high"][code_slot, p_index] = high
            arrays["bp_convergence_rate"][code_slot, p_index] = float(
                raw["bp_converged"][slot, p_index].mean()
            )
            arrays["mean_bp_iterations"][code_slot, p_index] = float(
                raw["bp_iterations"][slot, p_index].mean()
            )
            arrays["syndrome_mismatch_rate"][code_slot, p_index] = float(
                1.0 - raw["syndrome_match"][slot, p_index].mean()
            )
            arrays["mean_logical_weight"][code_slot, p_index] = float(
                raw["logical_labels"][slot, p_index].sum(axis=1).mean()
            )


def aggregate_scan(raw_root, config, replay_report=None):
    from pathlib import Path

    config = ensure_config(config)
    arrays = _blank_arrays(config)
    errors = []
    seen = set()
    expected = {(m, block) for m in M_VALUES for block in range(TASKS_PER_M[m])}
    loaded = []
    for path in sorted(Path(raw_root).rglob("*.npz")):
        raw = load_raw(path)
        key = (int(raw["m"]), int(raw["block_index"]))
        if key not in expected:
            raise ValueError(f"raw tree contains an unplanned task: {key!r}")
        if key in seen:
            raise ValueError(f"duplicate raw task: {key!r}")
        seen.add(key)
        loaded.append(raw)
    for raw in loaded:
        _ingest_task(arrays, raw, config, errors)

    tokens = list(config["p_tokens"])
    trials = int(config["trials_per_code_p"])
    n_p = len(tokens)

    complete = True
    failures_by_m = []
    for m_index, m in enumerate(M_VALUES):
        code_slice = slice(m_index * CODES_PER_M, (m_index + 1) * CODES_PER_M)
        status = arrays["code_status"][code_slice]
        counts = arrays["failure_counts"][code_slice]
        rates = arrays["code_rates"][code_slice]
        distances = arrays["classical_distance"][code_slice]
        for p_index in range(n_p):
            column = status[:, p_index]
            if np.any(column == "INVALID"):
                arrays["m_status"][m_index, p_index] = "SAMPLING_INSUFFICIENT"
                complete = False
            elif np.any(column != "REPORTABLE"):
                arrays["m_status"][m_index, p_index] = "INCOMPLETE"
                complete = False
            else:
                arrays["m_status"][m_index, p_index] = "REPORTABLE"
                total_failures = int(counts[:, p_index].sum())
                total_trials = CODES_PER_M * trials
                mean = total_failures / total_trials
                arrays["primary_failures"][m_index, p_index] = total_failures
                arrays["primary_trials"][m_index, p_index] = total_trials
                arrays["primary_mean"][m_index, p_index] = mean
                arrays["pooled_binomial_se"][m_index, p_index] = np.sqrt(
                    mean * (1.0 - mean) / total_trials
                )
                std = float(np.std(rates[:, p_index], ddof=1))
                arrays["between_code_std"][m_index, p_index] = std
                arrays["cluster_se"][m_index, p_index] = std / np.sqrt(CODES_PER_M)
        for d_index, distance in enumerate(DISTANCE_STRATA):
            member = distances == distance
            arrays["strata_code_counts"][m_index, d_index] = int(member.sum())
            if not member.any():
                continue
            reportable = member[:, None] & (status == "REPORTABLE")
            stratum_failures = np.where(reportable, counts, 0).sum(axis=0)
            stratum_trials = reportable.sum(axis=0) * trials
            arrays["strata_failures"][m_index, d_index] = stratum_failures
            arrays["strata_trials"][m_index, d_index] = stratum_trials
            with np.errstate(invalid="ignore", divide="ignore"):
                arrays["strata_rate"][m_index, d_index] = np.where(
                    stratum_trials > 0, stratum_failures / stratum_trials, np.nan,
                )
        failures_by_m.append(counts)

    arrays["overall_status"] = "COMPLETE" if complete and not errors else "INCOMPLETE"
    if arrays["overall_status"] == "COMPLETE":
        bootstrap = cluster_bootstrap(failures_by_m, trials, config, "final_m3_m8")
        arrays["primary_band_low"] = bootstrap["point_low"]
        arrays["primary_band_high"] = bootstrap["point_high"]
        arrays["delta38"] = bootstrap["endpoint"]
        arrays["delta38_band_low"] = bootstrap["endpoint_low"]
        arrays["delta38_band_high"] = bootstrap["endpoint_high"]
        arrays["adjacent_delta"] = bootstrap["adjacent"]
        arrays["adjacent_band_low"] = bootstrap["adjacent_low"]
        arrays["adjacent_band_high"] = bootstrap["adjacent_high"]
        decision = classify_crossing(
            arrays["p_values"], bootstrap["endpoint"],
            bootstrap["endpoint_low"], bootstrap["endpoint_high"],
        )
        location = crossing_location(
            arrays["p_values"], bootstrap["endpoint"],
            bootstrap["endpoint_replicates"], decision,
            confidence=float(config["bootstrap"]["confidence"]),
        )
        half_width = bootstrap["half_width"]
    else:
        decision = {
            "status": "INCOMPLETE",
            "bracket": (float("nan"), float("nan")),
            "certified_negative_p": [],
            "certified_positive_p": [],
        }
        location = {
            "p_cross": float("nan"), "p_cross_low": float("nan"),
            "p_cross_high": float("nan"), "defined_fraction": float("nan"),
            "reason": "aggregate_incomplete",
        }
        half_width = float("nan")

    replay_report = replay_report or {}
    arrays.update({
        "schema_version": AGGREGATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "terminal_status": decision["status"],
        "crossing_bracket_low": decision["bracket"][0],
        "crossing_bracket_high": decision["bracket"][1],
        "certified_negative_p_json": canonical_json(decision["certified_negative_p"]),
        "certified_positive_p_json": canonical_json(decision["certified_positive_p"]),
        "bootstrap_half_width": half_width,
        "p_cross": location["p_cross"],
        "p_cross_low": location["p_cross_low"],
        "p_cross_high": location["p_cross_high"],
        "p_cross_defined_fraction": location["defined_fraction"],
        "p_cross_reason": location["reason"],
        "codes_per_m": CODES_PER_M,
        "trials_per_code_p": trials,
        "replay_status": str(replay_report.get("status", "MISSING")),
        "replay_scope": str(replay_report.get("scope", "")),
        "replay_report_sha256": str(replay_report.get("report_sha256", "")),
        "raw_manifest_sha256": str(replay_report.get("raw_manifest_sha256", "")),
        "replay_report_json": canonical_json(replay_report),
        "unexpected_raw_errors_json": canonical_json(errors),
    })
    arrays["payload_sha256"] = arrays_sha256(arrays, ARRAY_FIELDS)
    if set(arrays) != set(ARRAY_FIELDS) | set(SCALAR_FIELDS):
        raise AssertionError("internal aggregate schema field mismatch")
    return arrays
