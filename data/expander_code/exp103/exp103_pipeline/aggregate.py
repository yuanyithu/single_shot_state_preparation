import hashlib
import json
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.registry import load_registry

from . import AGGREGATE_SCHEMA, EXPERIMENT_ID, RAW_SCHEMA
from .config import CODE_IDS, M_VALUES, P_TOKENS, ensure_config
from .crossing import classify_final_crossing, simultaneous_bootstrap, wilson_interval
from .io import arrays_sha256, atomic_npz, canonical_json, sha256_file, sha256_json
from .raw import load_raw
from .replay import expected_replay_keys, raw_manifest, validate_replay_report
from .seeds import derive_seed


ARRAY_FIELDS = (
    "p_values", "m_values", "code_ids", "code_m", "classical_distance",
    "code_status", "failure_counts", "trial_counts", "code_rates",
    "wilson_low", "wilson_high", "bp_convergence_rate", "mean_bp_iterations",
    "syndrome_mismatch_rate", "mean_logical_weight", "m_status", "primary_mean",
    "primary_median", "fixed_panel_mc_se", "between_code_sem", "between_code_std",
    "primary_band_low", "primary_band_high", "delta38", "delta38_band_low",
    "delta38_band_high", "adjacent_delta", "adjacent_band_low",
    "adjacent_band_high", "stage1_delta35", "stage1_band_low", "stage1_band_high",
    "stage1_adjacent_delta", "stage1_adjacent_band_low", "stage1_adjacent_band_high",
    "stage1_primary_band_low", "stage1_primary_band_high",
)
SCALAR_FIELDS = (
    "schema_version", "experiment_id", "config_sha256", "registry_sha256",
    "source_commit", "source_tree_sha256", "decoder_binary_sha256", "overall_status",
    "terminal_status", "crossing_bracket_low", "crossing_bracket_high",
    "compatible_triple_json", "stage1_status", "stage1_bracket_low",
    "stage1_bracket_high", "bootstrap_half_width", "stage1_bootstrap_half_width",
    "stage1_compatible_triple_json", "payload_sha256", "unexpected_raw_errors_json",
    "replay_status", "replay_scope", "replay_report_sha256",
    "raw_manifest_sha256", "replay_report_json",
)


def _registry(config):
    root = Path(__file__).resolve().parents[4]
    registry = load_registry(root / config["registry_path"])
    if registry["registry_sha256"] != config["registry_sha256"]:
        raise ValueError("registry identity mismatch")
    rows = {row["code_id"]: row for row in registry["codes"]}
    if set(rows) != set(CODE_IDS):
        raise ValueError("registry does not retain the full 48-code panel")
    return rows


def _validate_raw(raw, config, row, code_id, p_token, shard_index):
    if raw.get("conda_prefix_matches_python") is not True:
        return "identity_mismatch:conda_prefix_matches_python"
    expected_scalars = {
        "schema_version": RAW_SCHEMA,
        "status": "VALID",
        "invalid_reason": "",
        "exception_type": "",
        "exception_message": "",
        "experiment_id": EXPERIMENT_ID,
        "code_id": code_id,
        "m": row["m"],
        "p_token": p_token,
        "shard_index": shard_index,
        "planned_trials": config["trials_per_shard"],
        "completed_trials": config["trials_per_shard"],
        "seed": derive_seed(config, "measurement", code_id, p_token, shard_index),
        "seed_namespace": config["namespaces"]["measurement"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "python_version": config["environment"]["python"],
        "numpy_version": config["environment"]["numpy"],
        "scipy_version": config["environment"]["scipy"],
        "ldpc_version": config["environment"]["ldpc"],
        "device_name": config["environment"]["device_name"],
        "hostname": config["environment"]["hostname"],
        "conda_environment": config["environment"]["conda_environment"],
        "conda_prefix_matches_python": True,
        "n": row["n"],
        "k": row["k"],
        "classical_distance": row["classical_distance"],
    }
    for field, expected in expected_scalars.items():
        if raw[field] != expected:
            return f"identity_mismatch:{field}"
    if type(raw["conda_prefix_matches_python"]) is not bool:
        return "identity_mismatch:conda_prefix_matches_python"
    if float(raw["p"]) != float(p_token):
        return "identity_mismatch:p"
    trials, k = config["trials_per_shard"], row["k"]
    expected_shapes = {
        "failure_flags": ((trials,), np.dtype(np.bool_)),
        "logical_labels": ((trials, k), np.dtype(np.uint8)),
        "syndrome_match": ((trials,), np.dtype(np.bool_)),
        "bp_converged": ((trials,), np.dtype(np.bool_)),
        "bp_iterations": ((trials,), np.dtype(np.int32)),
    }
    for field, (shape, dtype) in expected_shapes.items():
        if raw[field].shape != shape or raw[field].dtype != dtype:
            return f"array_identity_mismatch:{field}"
    if (
        np.any(raw["logical_labels"] > 1)
        or np.any(raw["bp_iterations"] < 0)
        or np.any(raw["bp_iterations"] > row["n"])
    ):
        return "invalid_trial_values"
    recomputed_failures = np.logical_or(
        np.logical_not(raw["syndrome_match"]), raw["logical_labels"].any(axis=1),
    )
    if not np.array_equal(recomputed_failures, raw["failure_flags"]):
        return "failure_flags_do_not_match_trial_fields"
    if hashlib.sha256(raw["logical_labels"].tobytes()).hexdigest() != raw["label_stream_sha256"]:
        return "label_stream_hash_mismatch"
    for field in ("error_stream_sha256", "correction_stream_sha256", "label_stream_sha256"):
        value = str(raw[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            return f"invalid_stream_hash:{field}"
    return None


def _blank_arrays(rows):
    shape = (48, 13)
    return {
        "p_values": np.asarray([float(token) for token in P_TOKENS], dtype=np.float64),
        "m_values": np.asarray(M_VALUES, dtype=np.int16),
        "code_ids": np.asarray(CODE_IDS, dtype="U8"),
        "code_m": np.asarray([rows[code]["m"] for code in CODE_IDS], dtype=np.int16),
        "classical_distance": np.asarray([rows[code]["classical_distance"] for code in CODE_IDS], dtype=np.int16),
        "code_status": np.full(shape, "INCOMPLETE", dtype="U12"),
        "failure_counts": np.zeros(shape, dtype=np.int64),
        "trial_counts": np.zeros(shape, dtype=np.int64),
        "code_rates": np.full(shape, np.nan),
        "wilson_low": np.full(shape, np.nan),
        "wilson_high": np.full(shape, np.nan),
        "bp_convergence_rate": np.full(shape, np.nan),
        "mean_bp_iterations": np.full(shape, np.nan),
        "syndrome_mismatch_rate": np.full(shape, np.nan),
        "mean_logical_weight": np.full(shape, np.nan),
        "m_status": np.full((6, 13), "INCOMPLETE", dtype="U12"),
        "primary_mean": np.full((6, 13), np.nan),
        "primary_median": np.full((6, 13), np.nan),
        "fixed_panel_mc_se": np.full((6, 13), np.nan),
        "between_code_sem": np.full((6, 13), np.nan),
        "between_code_std": np.full((6, 13), np.nan),
        "primary_band_low": np.full((6, 13), np.nan),
        "primary_band_high": np.full((6, 13), np.nan),
        "delta38": np.full(13, np.nan),
        "delta38_band_low": np.full(13, np.nan),
        "delta38_band_high": np.full(13, np.nan),
        "adjacent_delta": np.full((5, 13), np.nan),
        "adjacent_band_low": np.full((5, 13), np.nan),
        "adjacent_band_high": np.full((5, 13), np.nan),
        "stage1_delta35": np.full(13, np.nan),
        "stage1_band_low": np.full(13, np.nan),
        "stage1_band_high": np.full(13, np.nan),
        "stage1_adjacent_delta": np.full((2, 13), np.nan),
        "stage1_adjacent_band_low": np.full((2, 13), np.nan),
        "stage1_adjacent_band_high": np.full((2, 13), np.nan),
        "stage1_primary_band_low": np.full((3, 13), np.nan),
        "stage1_primary_band_high": np.full((3, 13), np.nan),
    }


def aggregate_decoder_scan(raw_root, config):
    config = ensure_config(config)
    rows = _registry(config)
    arrays = _blank_arrays(rows)
    grouped = {}
    unexpected = []
    for path in sorted(Path(raw_root).rglob("*.npz")):
        try:
            raw = load_raw(path)
            key = (str(raw["code_id"]), str(raw["p_token"]), int(raw["shard_index"]))
            grouped.setdefault(key, []).append((path, raw, sha256_file(path)))
        except Exception as error:
            unexpected.append(f"{path}:{type(error).__name__}:{error}")
    expected_keys = {
        (code_id, p_token, shard_index)
        for code_id in CODE_IDS
        for p_token in P_TOKENS
        for shard_index in range(config["shards_per_code_p"])
    }
    for key in sorted(set(grouped) - expected_keys):
        unexpected.append(f"unexpected_raw_identity:{key!r}")
    observed_keys = set(grouped)
    replay_status = "NOT_REQUIRED_INCOMPLETE"
    replay_scope = "none"
    replay_report_sha256 = ""
    raw_manifest_sha256 = ""
    replay_report_json = "{}"
    required_replay_scope = None
    if observed_keys == expected_replay_keys(config, "stage1"):
        required_replay_scope = "stage1"
    elif observed_keys == expected_replay_keys(config, "stage2"):
        required_replay_scope = "stage2"
    elif observed_keys == expected_replay_keys(config, "final"):
        required_replay_scope = "final_combined"
    if required_replay_scope is not None:
        try:
            if required_replay_scope == "final_combined":
                reports = {}
                for scope in ("stage1", "stage2"):
                    stage_root = Path(raw_root) / scope
                    report_path = stage_root / f"REPLAY_{scope.upper()}.json"
                    report = json.loads(report_path.read_text(encoding="ascii"))
                    validate_replay_report(report, stage_root, config, scope)
                    reports[scope] = report
                report = {
                    "schema_version": "exp103.replay_bundle.v1",
                    "stage1": reports["stage1"],
                    "stage2": reports["stage2"],
                }
                _, raw_manifest_sha256, _ = raw_manifest(raw_root)
                report["raw_manifest_sha256"] = raw_manifest_sha256
            else:
                stage_root = Path(raw_root) / required_replay_scope
                if not stage_root.is_dir():
                    stage_root = Path(raw_root)
                report_path = stage_root / f"REPLAY_{required_replay_scope.upper()}.json"
                report = json.loads(report_path.read_text(encoding="ascii"))
                validate_replay_report(report, stage_root, config, required_replay_scope)
                raw_manifest_sha256 = report["raw_manifest_sha256"]
            replay_status = "PASS"
            replay_scope = required_replay_scope
            replay_report_sha256 = sha256_json(report)
            replay_report_json = canonical_json(report)
        except Exception as error:
            replay_status = "INVALID"
            replay_scope = required_replay_scope
            unexpected.append(f"replay_gate:{type(error).__name__}:{error}")
    for code_index, code_id in enumerate(CODE_IDS):
        row = rows[code_id]
        for p_index, p_token in enumerate(P_TOKENS):
            shard_payloads = []
            cell_invalid = False
            for shard_index in range(config["shards_per_code_p"]):
                copies = grouped.get((code_id, p_token, shard_index), [])
                if not copies:
                    continue
                if len({item[2] for item in copies}) != 1:
                    cell_invalid = True
                    continue
                raw = copies[0][1]
                reason = _validate_raw(raw, config, row, code_id, p_token, shard_index)
                if reason is not None:
                    cell_invalid = True
                    continue
                shard_payloads.append(raw)
            if cell_invalid:
                arrays["code_status"][code_index, p_index] = "INVALID"
                continue
            if len(shard_payloads) != config["shards_per_code_p"]:
                continue
            failures = sum(int(raw["failure_flags"].sum()) for raw in shard_payloads)
            trials = sum(int(raw["completed_trials"]) for raw in shard_payloads)
            arrays["code_status"][code_index, p_index] = "REPORTABLE"
            arrays["failure_counts"][code_index, p_index] = failures
            arrays["trial_counts"][code_index, p_index] = trials
            arrays["code_rates"][code_index, p_index] = failures / trials
            low, high = wilson_interval(failures, trials)
            arrays["wilson_low"][code_index, p_index] = low
            arrays["wilson_high"][code_index, p_index] = high
            arrays["bp_convergence_rate"][code_index, p_index] = sum(
                int(raw["bp_converged"].sum()) for raw in shard_payloads
            ) / trials
            arrays["mean_bp_iterations"][code_index, p_index] = sum(
                int(raw["bp_iterations"].sum()) for raw in shard_payloads
            ) / trials
            arrays["syndrome_mismatch_rate"][code_index, p_index] = sum(
                int((~raw["syndrome_match"]).sum()) for raw in shard_payloads
            ) / trials
            arrays["mean_logical_weight"][code_index, p_index] = sum(
                int(raw["logical_labels"].sum()) for raw in shard_payloads
            ) / trials
    for m_index in range(6):
        code_slice = slice(m_index * 8, (m_index + 1) * 8)
        for p_index in range(13):
            statuses = arrays["code_status"][code_slice, p_index]
            if np.any(statuses == "INVALID"):
                arrays["m_status"][m_index, p_index] = "INVALID"
                continue
            if not np.all(statuses == "REPORTABLE"):
                continue
            arrays["m_status"][m_index, p_index] = "REPORTABLE"
            rates = arrays["code_rates"][code_slice, p_index]
            trials = arrays["trial_counts"][code_slice, p_index]
            arrays["primary_mean"][m_index, p_index] = rates.mean()
            arrays["primary_median"][m_index, p_index] = np.median(rates)
            arrays["fixed_panel_mc_se"][m_index, p_index] = np.sqrt(
                np.sum(rates * (1.0 - rates) / trials)
            ) / 8.0
            arrays["between_code_std"][m_index, p_index] = np.std(rates, ddof=1)
            arrays["between_code_sem"][m_index, p_index] = arrays["between_code_std"][m_index, p_index] / np.sqrt(8.0)
    if unexpected:
        overall = "INVALID"
    elif np.any(arrays["code_status"] == "INVALID"):
        overall = "INVALID"
    elif np.any(arrays["code_status"] == "INCOMPLETE"):
        overall = "INCOMPLETE"
    else:
        overall = "COMPLETE"
    scalars = {
        "schema_version": AGGREGATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "overall_status": overall,
        "terminal_status": "EXP103_INVALID" if overall == "INVALID" else "EXP103_INCOMPLETE",
        "crossing_bracket_low": np.nan,
        "crossing_bracket_high": np.nan,
        "compatible_triple_json": "null",
        "stage1_status": "INCOMPLETE",
        "stage1_bracket_low": np.nan,
        "stage1_bracket_high": np.nan,
        "bootstrap_half_width": np.nan,
        "stage1_bootstrap_half_width": np.nan,
        "stage1_compatible_triple_json": "null",
        "replay_status": replay_status,
        "replay_scope": replay_scope,
        "replay_report_sha256": replay_report_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "replay_report_json": replay_report_json,
        "unexpected_raw_errors_json": canonical_json(unexpected),
    }
    failures_3d = arrays["failure_counts"].reshape(6, 8, 13)
    trials_3d = arrays["trial_counts"].reshape(6, 8, 13)
    stage1_complete = (
        np.all(arrays["m_status"][:3] == "REPORTABLE")
        and replay_status == "PASS"
        and replay_scope in {"stage1", "final_combined"}
    )
    if stage1_complete:
        boot = simultaneous_bootstrap(failures_3d, trials_3d, (0, 1, 2), config, "stage1_m3_m5")
        arrays["stage1_delta35"] = boot["endpoint"]
        arrays["stage1_band_low"] = boot["endpoint_low"]
        arrays["stage1_band_high"] = boot["endpoint_high"]
        arrays["stage1_adjacent_delta"] = boot["adjacent"]
        arrays["stage1_adjacent_band_low"] = boot["adjacent_low"]
        arrays["stage1_adjacent_band_high"] = boot["adjacent_high"]
        arrays["stage1_primary_band_low"] = boot["point_low"]
        arrays["stage1_primary_band_high"] = boot["point_high"]
        scalars["stage1_bootstrap_half_width"] = boot["half_width"]
        padded_adjacent = np.full((5, 13), np.nan)
        padded_low = np.full((5, 13), np.nan)
        padded_high = np.full((5, 13), np.nan)
        padded_adjacent[:2] = boot["adjacent"]
        padded_low[:2] = boot["adjacent_low"]
        padded_high[:2] = boot["adjacent_high"]
        stage1_result = classify_final_crossing(
            arrays["p_values"], boot["endpoint"], boot["endpoint_low"], boot["endpoint_high"],
            padded_adjacent, padded_low, padded_high,
        )
        scalars["stage1_status"] = "STAGE1_RESTRICTED_" + stage1_result["status"].removeprefix("EXP103_")
        scalars["stage1_compatible_triple_json"] = canonical_json(stage1_result["compatible_triple"])
        if stage1_result["bracket"] is not None:
            scalars["stage1_bracket_low"], scalars["stage1_bracket_high"] = stage1_result["bracket"]
    if overall == "COMPLETE":
        boot = simultaneous_bootstrap(failures_3d, trials_3d, tuple(range(6)), config, "final_m3_m8")
        arrays["primary_band_low"] = boot["point_low"]
        arrays["primary_band_high"] = boot["point_high"]
        arrays["delta38"] = boot["endpoint"]
        arrays["delta38_band_low"] = boot["endpoint_low"]
        arrays["delta38_band_high"] = boot["endpoint_high"]
        arrays["adjacent_delta"] = boot["adjacent"]
        arrays["adjacent_band_low"] = boot["adjacent_low"]
        arrays["adjacent_band_high"] = boot["adjacent_high"]
        scalars["bootstrap_half_width"] = boot["half_width"]
        result = classify_final_crossing(
            arrays["p_values"], arrays["delta38"], arrays["delta38_band_low"],
            arrays["delta38_band_high"], arrays["adjacent_delta"],
            arrays["adjacent_band_low"], arrays["adjacent_band_high"],
        )
        scalars["terminal_status"] = result["status"]
        scalars["compatible_triple_json"] = canonical_json(result["compatible_triple"])
        if result["bracket"] is not None:
            scalars["crossing_bracket_low"], scalars["crossing_bracket_high"] = result["bracket"]
    aggregate = {**arrays, **scalars}
    aggregate["payload_sha256"] = arrays_sha256(aggregate, ARRAY_FIELDS)
    if set(aggregate) != set(ARRAY_FIELDS) | set(SCALAR_FIELDS):
        raise AssertionError("aggregate schema field mismatch")
    return aggregate


def save_aggregate(path, aggregate, refuse_overwrite=True):
    path = Path(path)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"aggregate already exists: {path}")
    if set(aggregate) != set(ARRAY_FIELDS) | set(SCALAR_FIELDS):
        raise ValueError("aggregate fields do not match exp103.aggregate.v1")
    atomic_npz(path, {key: np.asarray(value) for key, value in aggregate.items()})
    return sha256_file(path)
