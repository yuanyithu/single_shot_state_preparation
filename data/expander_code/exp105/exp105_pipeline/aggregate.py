"""Fail-closed aggregation of the exp105 ensemble scan.

Fail-closed means the same thing it meant in exp103: a cell with any invalid or
missing task is not repaired, not dropped and not silently down-weighted. A
missing task makes its whole (m, p) column INCOMPLETE and the run's overall
status INCOMPLETE, and every published statistic that depends on it becomes NaN.
"""

import numpy as np

from . import AGGREGATE_SCHEMA, EXPERIMENT_ID
from .config import block_code_indices, ensure_config, tasks_per_m
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
    "readout_mismatch_rate", "mean_logical_weight",
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
    "p_cross_reason", "codes_per_m_json", "trials_per_code_p_json",
    "q_token", "qtop_lower_bound_json",
    "replay_status", "replay_scope", "replay_report_sha256",
    "raw_manifest_sha256", "replay_report_json", "unexpected_raw_errors_json",
    "payload_sha256",
)


def panel_layout(config):
    """Return (m_values, codes_per_m, trials_per_m, code slot offsets).

    Panels are unequal across m in exp105, so every array that is indexed by a
    global code slot needs this table rather than a single stride.
    """
    config = ensure_config(config)
    m_values = [int(m) for m in config["m_values"]]
    counts = {int(m): int(config["codes_per_m"][str(int(m))]) for m in m_values}
    trials = {int(m): int(config["trials_per_code_p"][str(int(m))]) for m in m_values}
    offsets = {}
    running = 0
    for m in m_values:
        offsets[m] = running
        running += counts[m]
    return m_values, counts, trials, offsets, running


def _blank_arrays(config):
    m_values, counts, _, _, n_codes = panel_layout(config)
    n_p = len(config["p_tokens"])
    n_m = len(m_values)
    n_d = len(DISTANCE_STRATA)
    return {
        "p_values": np.asarray([float(token) for token in config["p_tokens"]]),
        "m_values": np.asarray(m_values, dtype=np.int16),
        "distance_values": np.asarray(DISTANCE_STRATA, dtype=np.int16),
        "code_m": np.asarray(
            [m for m in m_values for _ in range(counts[m])], dtype=np.int16,
        ),
        "code_index": np.asarray(
            [index for m in m_values for index in range(counts[m])], dtype=np.int32,
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
        "readout_mismatch_rate": np.full((n_codes, n_p), np.nan),
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


def qtop_lower_bound(m_values, p_values, primary_mean):
    """Certified lower bound on the disorder-averaged q_top of the exp101 posterior.

    Per disorder the exact posterior satisfies map_success <= sqrt(purity), and
    no decoder can beat MAP success at its own observation, so with
    S = 1 - P_fail and M = 2^k, Jensen gives

        E[q_top] = (M E[purity] - 1) / (M - 1) >= (M S^2 - 1) / (M - 1).

    This is a bound, never an estimate. It is informative only where S is large,
    which is exactly the ordered side where the sampling route is blocked. Where
    the right-hand side is negative the bound is vacuous and is reported as 0.
    """
    bound = {}
    for m_index, m in enumerate(m_values):
        M = float(2 ** (int(m) ** 2))
        row = []
        for p_index, p in enumerate(np.asarray(p_values, dtype=float)):
            mean = float(primary_mean[m_index, p_index])
            if not np.isfinite(mean):
                row.append(None)
                continue
            success = 1.0 - mean
            value = (M * success * success - 1.0) / (M - 1.0)
            row.append(max(0.0, float(value)))
        bound[str(int(m))] = row
    return bound


def _ingest_task(arrays, raw, config, errors):
    m = int(raw["m"])
    block_index = int(raw["block_index"])
    tokens = list(config["p_tokens"])
    _, _, trials_by_m, offsets, _ = panel_layout(config)
    trials = trials_by_m[m]
    if raw["schema_version"] != "exp105.raw.v1" or raw["experiment_id"] != EXPERIMENT_ID:
        errors.append({"m": m, "block_index": block_index, "reason": "schema_mismatch"})
        return
    if raw["p_tokens"] != ",".join(tokens) or int(raw["trials_per_code_p"]) != trials:
        errors.append({"m": m, "block_index": block_index, "reason": "grid_mismatch"})
        return
    if str(raw["q_token"]) != str(config["q_token"]):
        errors.append({"m": m, "block_index": block_index, "reason": "q_mismatch"})
        return
    indices = block_code_indices(config, m, block_index)
    if raw["status"] != "VALID" or int(raw["completed_codes"]) != len(indices):
        for index in indices:
            arrays["code_status"][offsets[m] + int(index), :] = "INVALID"
        return
    for slot, index in enumerate(indices):
        code_slot = offsets[m] + int(index)
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
            arrays["readout_mismatch_rate"][code_slot, p_index] = float(
                1.0 - raw["readout_match"][slot, p_index].mean()
            )
            arrays["mean_logical_weight"][code_slot, p_index] = float(
                raw["logical_labels"][slot, p_index].sum(axis=1).mean()
            )


def aggregate_scan(raw_root, config, replay_report=None):
    from pathlib import Path

    config = ensure_config(config)
    m_values, counts, trials_by_m, offsets, _ = panel_layout(config)
    per_m_tasks = tasks_per_m(counts, {
        int(m): int(config["codes_per_task"][str(int(m))]) for m in m_values
    })
    arrays = _blank_arrays(config)
    errors = []
    seen = set()
    expected = {(m, block) for m in m_values for block in range(per_m_tasks[m])}
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
    n_p = len(tokens)

    complete = True
    failures_by_m = []
    trials_list = []
    for m_index, m in enumerate(m_values):
        trials = trials_by_m[m]
        trials_list.append(trials)
        code_slice = slice(offsets[m], offsets[m] + counts[m])
        status = arrays["code_status"][code_slice]
        code_failures = arrays["failure_counts"][code_slice]
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
                total_failures = int(code_failures[:, p_index].sum())
                total_trials = counts[m] * trials
                mean = total_failures / total_trials
                arrays["primary_failures"][m_index, p_index] = total_failures
                arrays["primary_trials"][m_index, p_index] = total_trials
                arrays["primary_mean"][m_index, p_index] = mean
                arrays["pooled_binomial_se"][m_index, p_index] = np.sqrt(
                    mean * (1.0 - mean) / total_trials
                )
                std = float(np.std(rates[:, p_index], ddof=1))
                arrays["between_code_std"][m_index, p_index] = std
                arrays["cluster_se"][m_index, p_index] = std / np.sqrt(counts[m])
        for d_index, distance in enumerate(DISTANCE_STRATA):
            member = distances == distance
            arrays["strata_code_counts"][m_index, d_index] = int(member.sum())
            if not member.any():
                continue
            reportable = member[:, None] & (status == "REPORTABLE")
            stratum_failures = np.where(reportable, code_failures, 0).sum(axis=0)
            stratum_trials = reportable.sum(axis=0) * trials
            arrays["strata_failures"][m_index, d_index] = stratum_failures
            arrays["strata_trials"][m_index, d_index] = stratum_trials
            with np.errstate(invalid="ignore", divide="ignore"):
                arrays["strata_rate"][m_index, d_index] = np.where(
                    stratum_trials > 0, stratum_failures / stratum_trials, np.nan,
                )
        failures_by_m.append(code_failures)

    arrays["overall_status"] = "COMPLETE" if complete and not errors else "INCOMPLETE"
    if arrays["overall_status"] == "COMPLETE":
        bootstrap = cluster_bootstrap(failures_by_m, trials_list, config, "final_m3_m8")
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
        "codes_per_m_json": canonical_json({str(m): counts[m] for m in m_values}),
        "trials_per_code_p_json": canonical_json(
            {str(m): trials_by_m[m] for m in m_values}
        ),
        "q_token": str(config["q_token"]),
        "qtop_lower_bound_json": canonical_json(
            qtop_lower_bound(m_values, arrays["p_values"], arrays["primary_mean"])
        ),
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
