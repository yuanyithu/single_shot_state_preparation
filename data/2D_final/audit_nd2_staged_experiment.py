#!/usr/bin/env python3
"""External evidence gates and integrity checks for the nd-2 staged run."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
    temporary.replace(path)


def _write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).view(np.uint8)
    ).hexdigest()


def _finite_or_raise(name, value):
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _scalar(data, key):
    return np.asarray(data[key]).item()


def _validate_source_commit(config):
    source_commit = str(config["source_commit"])
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError(f"invalid source_commit: {source_commit!r}")
    return source_commit


def _manifest_summary(path, source_commit, expected_chunks):
    manifest = _load_json(path)
    summary = manifest["summary"]
    result = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "source_commit": str(manifest["git_commit_sha"]),
        "total_chunks": int(summary["total_chunks"]),
        "completed_chunks": int(summary["completed_chunks"]),
        "failed_chunks": int(summary["failed_chunks"]),
        "pending_chunks": int(summary["pending_chunks"]),
    }
    result["passed"] = bool(
        result["source_commit"] == source_commit
        and result["total_chunks"] == int(expected_chunks)
        and result["completed_chunks"] == result["total_chunks"]
        and result["failed_chunks"] == 0
        and result["pending_chunks"] == 0
        and all(chunk["status"] == "completed" for chunk in manifest["chunks"])
    )
    return result


def select_q0_schedule(
        a_mean_spreads,
        b_mean_spreads,
        arm_mean_q_top_differences,
        max_spread,
        max_arm_difference):
    a_mean_spreads = np.asarray(a_mean_spreads, dtype=np.float64)
    b_mean_spreads = np.asarray(b_mean_spreads, dtype=np.float64)
    arm_mean_q_top_differences = np.asarray(
        arm_mean_q_top_differences, dtype=np.float64
    )
    a_spread_pass = bool(np.all(a_mean_spreads <= max_spread))
    arm_difference_pass = bool(
        np.all(np.abs(arm_mean_q_top_differences) <= max_arm_difference)
    )
    b_spread_pass = bool(np.all(b_mean_spreads <= max_spread))
    if a_spread_pass and arm_difference_pass:
        selected = "A"
    elif b_spread_pass:
        selected = "B"
    else:
        selected = "STOP"
    return {
        "selected_schedule": selected,
        "passed": selected in ("A", "B"),
        "checks": {
            "a_spread_pass": a_spread_pass,
            "a_b_mean_q_top_difference_pass": arm_difference_pass,
            "b_spread_pass": b_spread_pass,
        },
    }


def _audit_q0_arm(path, arm, config):
    pilot = config["q0_pilot"]
    source_commit = config["source_commit"]
    with np.load(path, allow_pickle=False) as data:
        expected_p = np.asarray(
            pilot["data_error_probabilities"], dtype=np.float64
        )
        checks = {
            "source_commit": str(_scalar(data, "git_commit_sha"))
            == source_commit,
            "lattice_size": np.array_equal(
                np.asarray(data["lattice_size_list"]),
                np.asarray([pilot["lattice_size"]]),
            ),
            "p_grid": np.array_equal(
                np.asarray(data["data_error_probability_list"]), expected_p
            ),
            "q": float(_scalar(data, "syndrome_error_probability")) == 0.0,
            "num_disorder": int(_scalar(data, "num_disorder_samples"))
            == int(pilot["num_disorder_samples"]),
            "chunk_size": int(_scalar(data, "chunk_size"))
            == int(pilot["chunk_size"]),
            "seed_base": int(_scalar(data, "seed_base"))
            == int(pilot["seed_base"]),
            "num_start_chains": int(_scalar(data, "q0_num_start_chains"))
            == int(pilot["num_start_chains"]),
            "common_random": bool(
                _scalar(data, "common_random_disorder_across_p")
            ),
            "num_burn_in_sweeps": int(_scalar(data, "num_burn_in_sweeps"))
            == int(pilot["arms"][arm]["num_burn_in_sweeps"]),
            "num_sweeps_between_measurements": int(
                _scalar(data, "num_sweeps_between_measurements")
            )
            == int(
                pilot["arms"][arm]["num_sweeps_between_measurements"]
            ),
            "num_measurements_per_disorder": int(
                _scalar(data, "num_measurements_per_disorder")
            )
            == int(
                pilot["arms"][arm]["num_measurements_per_disorder"]
            ),
        }
        q_top_curve = _finite_or_raise(
            "q_top_curve_matrix", data["q_top_curve_matrix"]
        )
        mean_spread = _finite_or_raise(
            "q0_mean_q_top_spread_curve_matrix",
            data["q0_mean_q_top_spread_curve_matrix"],
        )
        per_disorder = _finite_or_raise(
            "disorder_q_top_values_tensor",
            data["disorder_q_top_values_tensor"],
        )
        per_start = _finite_or_raise(
            "q0_q_top_values_per_disorder_per_start_tensor",
            data["q0_q_top_values_per_disorder_per_start_tensor"],
        )
        expected_shapes = {
            "q_top_curve": (1, 2),
            "mean_spread": (1, 2),
            "per_disorder": (1, 2, 128),
            "per_start": (1, 2, 128, 4),
        }
        observed_shapes = {
            "q_top_curve": q_top_curve.shape,
            "mean_spread": mean_spread.shape,
            "per_disorder": per_disorder.shape,
            "per_start": per_start.shape,
        }
        checks["shapes"] = observed_shapes == expected_shapes
        checks["aggregate_rebuild"] = bool(
            np.array_equal(np.mean(per_disorder, axis=2), q_top_curve)
        )
        checks["spread_rebuild"] = bool(
            np.array_equal(
                np.mean(np.ptp(per_start, axis=3), axis=2), mean_spread
            )
        )
        result = {
            "arm": arm,
            "path": str(path),
            "sha256": _sha256_file(path),
            "checks": checks,
            "passed_integrity": bool(all(checks.values())),
            "mean_q_top": q_top_curve[0].tolist(),
            "mean_q_top_start_spread": mean_spread[0].tolist(),
        }
    return result


def _normalized_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.full(counts.shape, np.nan, dtype=np.float64)
    return counts / total


def evaluate_qpositive_gate(arrays, thresholds):
    never_flipped = np.asarray(
        arrays["num_chains_that_never_flipped_sector_per_disorder"],
        dtype=np.int64,
    )
    histograms = np.asarray(
        arrays[
            "chain_cold_sector_histogram_counts_per_disorder_per_start_replica"
        ],
        dtype=np.int64,
    )
    r_hat = np.asarray(arrays["max_r_hat_per_disorder"], dtype=np.float64)
    ess = np.asarray(
        arrays["min_effective_sample_size_per_disorder"], dtype=np.float64
    )
    q_top_spread = np.asarray(
        arrays["q_top_spread_per_disorder"], dtype=np.float64
    )
    winding = np.asarray(
        arrays[
            "chain_winding_acceptance_rate_per_disorder_per_start_replica"
        ],
        dtype=np.float64,
    )
    support_count = np.count_nonzero(histograms > 0, axis=-1)
    every_chain_changed = bool(
        np.all(never_flipped == 0) and np.all(support_count >= 2)
    )
    checks = {
        "every_chain_changed_logical_sector": every_chain_changed,
        "all_r_hat_below_limit": bool(
            np.all(np.isfinite(r_hat))
            and np.all(r_hat < thresholds["q_positive_max_r_hat"])
        ),
        "all_ess_above_limit": bool(
            np.all(np.isfinite(ess))
            and np.all(
                ess > thresholds["q_positive_min_effective_sample_size"]
            )
        ),
        "mean_q_top_spread_below_limit": bool(
            np.all(np.isfinite(q_top_spread))
            and float(np.mean(q_top_spread))
            < thresholds["q_positive_max_mean_q_top_spread"]
        ),
        "mean_winding_acceptance_above_limit": bool(
            np.all(np.isfinite(winding))
            and float(np.mean(winding))
            > thresholds["q_positive_min_mean_winding_acceptance"]
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": {
            "max_r_hat": float(np.max(r_hat)),
            "min_effective_sample_size": float(np.min(ess)),
            "mean_q_top_spread": float(np.mean(q_top_spread)),
            "mean_winding_acceptance": float(np.mean(winding)),
            "max_num_never_flipped_chains_per_disorder": int(
                np.max(never_flipped)
            ),
            "min_occupied_sector_count_per_chain": int(
                np.min(support_count)
            ),
        },
    }


def _audit_qpositive_arm(path, arm, config):
    sentinel = config["q_positive_sentinel"]
    legacy = sentinel["legacy_reference"]
    thresholds = config["gates"]
    with np.load(path, allow_pickle=False) as data:
        invariant_checks = {
            "source_commit": str(_scalar(data, "source_commit"))
            == config["source_commit"],
            "arm": str(_scalar(data, "arm")) == arm,
            "lattice_size": int(_scalar(data, "lattice_size"))
            == int(sentinel["lattice_size"]),
            "p": float(_scalar(data, "data_error_probability"))
            == float(sentinel["data_error_probability"]),
            "q": float(_scalar(data, "syndrome_error_probability"))
            == float(sentinel["syndrome_error_probability"]),
            "num_disorder": int(_scalar(data, "num_disorder_samples"))
            == int(sentinel["num_disorder_samples"]),
            "num_start_chains": int(_scalar(data, "num_start_chains"))
            == int(sentinel["num_start_chains"]),
            "num_replicas": int(_scalar(data, "num_replicas_per_start"))
            == int(sentinel["num_replicas_per_start"]),
            "cluster_disabled": not bool(
                _scalar(data, "cluster_update_enabled_config")
            ),
            "pt_disabled": not bool(
                _scalar(data, "parallel_tempering_enabled_config")
            )
            and not bool(_scalar(data, "pt_enabled")),
            "zero_syndrome_sweep_enabled": int(
                _scalar(data, "num_zero_syndrome_sweeps_per_cycle")
            )
            == int(sentinel["num_zero_syndrome_sweeps_per_cycle"]),
            "legacy_disorder_seed": int(
                _scalar(data, "legacy_disorder_seed")
            )
            == int(legacy["disorder_seed"]),
            "legacy_uniform_hashes": str(
                _scalar(data, "syndrome_uniforms_first16_sha256")
            )
            == legacy["syndrome_uniforms_first16_sha256"]
            and str(_scalar(data, "data_uniforms_first16_sha256"))
            == legacy["data_uniforms_first16_sha256"],
        }
        arrays = {
            key: _finite_or_raise(key, data[key])
            for key in (
                "disorder_q_top_values",
                "num_chains_that_never_flipped_sector_per_disorder",
                "chain_cold_sector_histogram_counts_per_disorder_per_start_replica",
                "max_r_hat_per_disorder",
                "min_effective_sample_size_per_disorder",
                "q_top_spread_per_disorder",
                "chain_winding_acceptance_rate_per_disorder_per_start_replica",
                "chain_cold_sector_histogram_first_half_counts_per_disorder_per_start_replica",
                "chain_cold_sector_histogram_second_half_counts_per_disorder_per_start_replica",
                "chain_cold_sector_histogram_first_second_tv_per_disorder_per_start_replica",
                "q_top_block_drift_per_disorder",
                "q_top_block_values_per_disorder",
            )
        }
        expected_chain_shape = (
            int(sentinel["num_disorder_samples"]),
            int(sentinel["num_start_chains"]),
            int(sentinel["num_replicas_per_start"]),
        )
        invariant_checks["chain_shape"] = (
            arrays[
                "chain_winding_acceptance_rate_per_disorder_per_start_replica"
            ].shape
            == expected_chain_shape
        )
        gate = evaluate_qpositive_gate(arrays, thresholds)
        legacy_q_top = np.asarray(legacy["q_top_first16"], dtype=np.float64)
        current_q_top = np.asarray(
            arrays["disorder_q_top_values"], dtype=np.float64
        )
        paired_delta = current_q_top - legacy_q_top
        first_counts = np.sum(
            arrays[
                "chain_cold_sector_histogram_first_half_counts_per_disorder_per_start_replica"
            ],
            axis=(0, 1, 2),
        )
        second_counts = np.sum(
            arrays[
                "chain_cold_sector_histogram_second_half_counts_per_disorder_per_start_replica"
            ],
            axis=(0, 1, 2),
        )
        result = {
            "arm": arm,
            "path": str(path),
            "sha256": _sha256_file(path),
            "invariant_checks": invariant_checks,
            "passed_integrity": bool(all(invariant_checks.values())),
            "transport_gate": gate,
            "occupation": {
                "first_half": _normalized_counts(first_counts).tolist(),
                "second_half": _normalized_counts(second_counts).tolist(),
                "mean_per_chain_first_second_tv": float(np.mean(
                    arrays[
                        "chain_cold_sector_histogram_first_second_tv_per_disorder_per_start_replica"
                    ]
                )),
            },
            "block_drift": {
                "per_disorder": arrays[
                    "q_top_block_drift_per_disorder"
                ].tolist(),
                "mean": float(np.mean(
                    arrays["q_top_block_drift_per_disorder"]
                )),
                "mean_absolute": float(np.mean(np.abs(
                    arrays["q_top_block_drift_per_disorder"]
                ))),
            },
            "paired_difference_from_legacy": {
                "per_disorder": paired_delta.tolist(),
                "mean": float(np.mean(paired_delta)),
                "mean_absolute": float(np.mean(np.abs(paired_delta))),
                "max_absolute": float(np.max(np.abs(paired_delta))),
            },
            "q_top_per_disorder": current_q_top.tolist(),
        }
        result["passed"] = bool(
            result["passed_integrity"] and gate["passed"]
        )
        return result


def audit_pilot(args):
    config = _load_json(args.config)
    source_commit = _validate_source_commit(config)
    q0_a = _audit_q0_arm(args.q0_a, "A", config)
    q0_b = _audit_q0_arm(args.q0_b, "B", config)
    q0_gate = select_q0_schedule(
        q0_a["mean_q_top_start_spread"],
        q0_b["mean_q_top_start_spread"],
        np.asarray(q0_b["mean_q_top"]) - np.asarray(q0_a["mean_q_top"]),
        config["gates"]["q0_max_mean_start_spread"],
        config["gates"]["q0_max_abs_arm_mean_q_top_difference"],
    )
    if not (q0_a["passed_integrity"] and q0_b["passed_integrity"]):
        q0_gate["selected_schedule"] = "STOP"
        q0_gate["passed"] = False
        q0_gate["checks"]["input_integrity"] = False
    else:
        q0_gate["checks"]["input_integrity"] = True
    q0_gate["arm_mean_q_top_difference_b_minus_a"] = (
        np.asarray(q0_b["mean_q_top"]) - np.asarray(q0_a["mean_q_top"])
    ).tolist()

    qpositive = None
    if args.qpositive_a and args.qpositive_b:
        qp_a = _audit_qpositive_arm(args.qpositive_a, "A", config)
        qp_b = _audit_qpositive_arm(args.qpositive_b, "B", config)
        qpositive = {
            "arms": {"A": qp_a, "B": qp_b},
            "a_b_q_top_difference_b_minus_a": (
                np.asarray(qp_b["q_top_per_disorder"])
                - np.asarray(qp_a["q_top_per_disorder"])
            ).tolist(),
            "passed": bool(qp_a["passed"] and qp_b["passed"]),
            "decision": (
                "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_NEXT_STAGE"
                if qp_a["passed"] and qp_b["passed"]
                else "STOP_Q_POSITIVE"
            ),
        }
    else:
        qpositive = {
            "passed": False,
            "decision": "STOP_Q_POSITIVE",
            "reason": "q-positive sentinel output missing",
        }

    expected_pilot_chunks = (
        len(config["q0_pilot"]["data_error_probabilities"])
        * config["q0_pilot"]["num_disorder_samples"]
        // config["q0_pilot"]["chunk_size"]
    )
    manifests = {}
    if args.q0_a_manifest:
        manifests["q0_A"] = _manifest_summary(
            args.q0_a_manifest, source_commit, expected_pilot_chunks
        )
    if args.q0_b_manifest:
        manifests["q0_B"] = _manifest_summary(
            args.q0_b_manifest, source_commit, expected_pilot_chunks
        )
    if manifests and not all(value["passed"] for value in manifests.values()):
        q0_gate["selected_schedule"] = "STOP"
        q0_gate["passed"] = False
        q0_gate["checks"]["manifests_complete"] = False
    else:
        q0_gate["checks"]["manifests_complete"] = True

    report = {
        "schema_version": 1,
        "created_at": _timestamp(),
        "source_commit": source_commit,
        "config_path": str(args.config),
        "config_sha256": _sha256_file(args.config),
        "q0": {
            "arms": {"A": q0_a, "B": q0_b},
            "gate": q0_gate,
        },
        "q_positive": qpositive,
        "manifests": manifests,
    }
    _write_json_atomic(args.output_json, report)
    rows = []
    for arm, arm_result in (("A", q0_a), ("B", q0_b)):
        for point_index, p_value in enumerate(
                config["q0_pilot"]["data_error_probabilities"]):
            rows.append({
                "track": "q0",
                "arm": arm,
                "point": point_index,
                "p": p_value,
                "q": 0.0,
                "mean_q_top": arm_result["mean_q_top"][point_index],
                "mean_q_top_spread": arm_result[
                    "mean_q_top_start_spread"
                ][point_index],
                "passed": q0_gate["passed"],
                "decision": q0_gate["selected_schedule"],
            })
    if "arms" in qpositive:
        for arm, arm_result in qpositive["arms"].items():
            rows.append({
                "track": "q_positive",
                "arm": arm,
                "point": 0,
                "p": config["q_positive_sentinel"][
                    "data_error_probability"
                ],
                "q": config["q_positive_sentinel"][
                    "syndrome_error_probability"
                ],
                "mean_q_top": float(np.mean(
                    arm_result["q_top_per_disorder"]
                )),
                "mean_q_top_spread": arm_result["transport_gate"][
                    "metrics"
                ]["mean_q_top_spread"],
                "passed": arm_result["passed"],
                "decision": qpositive["decision"],
            })
    _write_csv_atomic(
        args.output_csv,
        (
            "track",
            "arm",
            "point",
            "p",
            "q",
            "mean_q_top",
            "mean_q_top_spread",
            "passed",
            "decision",
        ),
        rows,
    )
    print(q0_gate["selected_schedule"])


def _audit_formal_one(npz_path, manifest_path, lattice_size, config):
    formal = config["q0_formal"]
    lattice_config = formal["lattice_sizes"][str(lattice_size)]
    num_points = len(formal["data_error_probabilities"])
    expected_chunks = (
        num_points * formal["num_disorder_samples"] // formal["chunk_size"]
    )
    manifest = _manifest_summary(
        manifest_path, config["source_commit"], expected_chunks
    )
    with np.load(npz_path, allow_pickle=False) as data:
        per_disorder = _finite_or_raise(
            "disorder_q_top_values_tensor",
            data["disorder_q_top_values_tensor"],
        )
        per_start = _finite_or_raise(
            "q0_q_top_values_per_disorder_per_start_tensor",
            data["q0_q_top_values_per_disorder_per_start_tensor"],
        )
        curve = _finite_or_raise(
            "q_top_curve_matrix", data["q_top_curve_matrix"]
        )
        spread = _finite_or_raise(
            "q0_mean_q_top_spread_curve_matrix",
            data["q0_mean_q_top_spread_curve_matrix"],
        )
        checks = {
            "manifest": manifest["passed"],
            "source_commit": str(_scalar(data, "git_commit_sha"))
            == config["source_commit"],
            "lattice_size": np.array_equal(
                data["lattice_size_list"], np.asarray([lattice_size])
            ),
            "p_grid": np.array_equal(
                data["data_error_probability_list"],
                np.asarray(formal["data_error_probabilities"]),
            ),
            "num_disorder": int(_scalar(data, "num_disorder_samples"))
            == int(formal["num_disorder_samples"]),
            "chunk_size": int(_scalar(data, "chunk_size"))
            == int(formal["chunk_size"]),
            "seed_base": int(_scalar(data, "seed_base"))
            == int(lattice_config["seed_base"]),
            "common_random": bool(
                _scalar(data, "common_random_disorder_across_p")
            ),
            "per_disorder_shape": per_disorder.shape
            == (1, num_points, formal["num_disorder_samples"]),
            "per_start_shape": per_start.shape
            == (1, num_points, formal["num_disorder_samples"], 4),
            "curve_rebuild": np.array_equal(
                np.mean(per_disorder, axis=2), curve
            ),
            "spread_rebuild": np.array_equal(
                np.mean(np.ptp(per_start, axis=3), axis=2), spread
            ),
        }
        standard_error = np.std(per_disorder[0], axis=1, ddof=1) / math.sqrt(
            formal["num_disorder_samples"]
        )
        return {
            "lattice_size": lattice_size,
            "npz_path": str(npz_path),
            "npz_sha256": _sha256_file(npz_path),
            "manifest": manifest,
            "checks": checks,
            "passed": bool(all(checks.values())),
            "q_top_curve": curve[0].tolist(),
            "q_top_standard_error": standard_error.tolist(),
            "mean_start_spread": spread[0].tolist(),
        }


def audit_formal(args):
    config = _load_json(args.config)
    _validate_source_commit(config)
    l9 = _audit_formal_one(args.l9_npz, args.l9_manifest, 9, config)
    l11 = _audit_formal_one(args.l11_npz, args.l11_manifest, 11, config)
    report = {
        "schema_version": 1,
        "created_at": _timestamp(),
        "source_commit": config["source_commit"],
        "selected_l11_schedule": args.selected_l11_schedule,
        "passed": bool(l9["passed"] and l11["passed"]),
        "lattice_sizes": {"9": l9, "11": l11},
    }
    _write_json_atomic(args.output_json, report)
    rows = []
    for result in (l9, l11):
        for point_index, p_value in enumerate(
                config["q0_formal"]["data_error_probabilities"]):
            rows.append({
                "lattice_size": result["lattice_size"],
                "point_index": point_index,
                "p": p_value,
                "mean_q_top": result["q_top_curve"][point_index],
                "standard_error": result[
                    "q_top_standard_error"
                ][point_index],
                "mean_start_spread": result[
                    "mean_start_spread"
                ][point_index],
                "passed": result["passed"],
            })
    _write_csv_atomic(
        args.output_csv,
        (
            "lattice_size",
            "point_index",
            "p",
            "mean_q_top",
            "standard_error",
            "mean_start_spread",
            "passed",
        ),
        rows,
    )
    if not report["passed"]:
        raise SystemExit(2)


def _synthetic_qpositive_arrays(mode):
    num_disorder = 2
    num_start = 4
    num_replica = 2
    num_sector = 8
    hist = np.zeros(
        (num_disorder, num_start, num_replica, num_sector), dtype=np.int64
    )
    if mode == "frozen_same":
        hist[..., 0] = 600
        never = np.full(num_disorder, num_start * num_replica, dtype=np.int64)
    elif mode == "frozen_different":
        for start in range(num_start):
            hist[:, start, :, start] = 600
        never = np.full(num_disorder, num_start * num_replica, dtype=np.int64)
    elif mode == "flipping":
        hist[..., 0] = 300
        hist[..., 1] = 300
        never = np.zeros(num_disorder, dtype=np.int64)
    else:
        raise ValueError(mode)
    return {
        "num_chains_that_never_flipped_sector_per_disorder": never,
        "chain_cold_sector_histogram_counts_per_disorder_per_start_replica": hist,
        "max_r_hat_per_disorder": np.full(num_disorder, 1.01),
        "min_effective_sample_size_per_disorder": np.full(num_disorder, 300.0),
        "q_top_spread_per_disorder": np.full(num_disorder, 0.02),
        "chain_winding_acceptance_rate_per_disorder_per_start_replica": np.full(
            (num_disorder, num_start, num_replica), 0.001
        ),
    }


def run_self_test(args):
    config = _load_json(args.config)
    _validate_source_commit(config)
    sentinel_dir = str(Path(__file__).resolve().parent)
    if sentinel_dir not in sys.path:
        sys.path.insert(0, sentinel_dir)
    repo_src = str(Path(args.repo_root).resolve() / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    from build_toric_code_examples import build_toric_code_by_family
    from nd2_qpositive_sentinel import verify_legacy_reconstruction

    parity_check_matrix, _ = build_toric_code_by_family(
        code_family=config["code_family"],
        lattice_size=config["q_positive_sentinel"]["lattice_size"],
    )
    legacy = config["q_positive_sentinel"]["legacy_reference"]
    assert parity_check_matrix.shape == (
        legacy["num_checks"], legacy["num_qubits"]
    )
    syndrome_uniforms, data_uniforms, _ = verify_legacy_reconstruction(config)
    assert syndrome_uniforms.shape == (16, 121)
    assert data_uniforms.shape == (16, 242)

    thresholds = config["gates"]
    assert not evaluate_qpositive_gate(
        _synthetic_qpositive_arrays("frozen_same"), thresholds
    )["passed"]
    assert not evaluate_qpositive_gate(
        _synthetic_qpositive_arrays("frozen_different"), thresholds
    )["passed"]
    assert evaluate_qpositive_gate(
        _synthetic_qpositive_arrays("flipping"), thresholds
    )["passed"]

    assert select_q0_schedule(
        [0.05, 0.05], [0.04, 0.04], [0.005, -0.005], 0.1, 0.01
    )["selected_schedule"] == "A"
    assert select_q0_schedule(
        [0.11, 0.05], [0.09, 0.09], [0.02, 0.02], 0.1, 0.01
    )["selected_schedule"] == "B"
    assert select_q0_schedule(
        [0.11, 0.05], [0.11, 0.09], [0.02, 0.02], 0.1, 0.01
    )["selected_schedule"] == "STOP"

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "sha-test"
        path.write_bytes(b"abc")
        assert _sha256_file(path) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
    assert len(legacy["q_top_first16"]) == 16
    assert re.fullmatch(r"[0-9a-f]{64}", legacy["source_npz_sha256"])
    print("self_test_passed=1")


def _build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test = subparsers.add_parser("self-test")
    self_test.add_argument("--config", required=True)
    self_test.add_argument("--repo-root", required=True)
    self_test.set_defaults(function=run_self_test)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--q0-a", required=True)
    pilot.add_argument("--q0-b", required=True)
    pilot.add_argument("--q0-a-manifest")
    pilot.add_argument("--q0-b-manifest")
    pilot.add_argument("--qpositive-a")
    pilot.add_argument("--qpositive-b")
    pilot.add_argument("--output-json", required=True)
    pilot.add_argument("--output-csv", required=True)
    pilot.set_defaults(function=audit_pilot)

    formal = subparsers.add_parser("formal")
    formal.add_argument("--config", required=True)
    formal.add_argument("--l9-npz", required=True)
    formal.add_argument("--l9-manifest", required=True)
    formal.add_argument("--l11-npz", required=True)
    formal.add_argument("--l11-manifest", required=True)
    formal.add_argument("--selected-l11-schedule", choices=("A", "B"), required=True)
    formal.add_argument("--output-json", required=True)
    formal.add_argument("--output-csv", required=True)
    formal.set_defaults(function=audit_formal)
    return parser


def main():
    args = _build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
