import json
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from data.expander_code.exp102.exp102_pipeline.registry import load_registry

from . import AGGREGATE_SCHEMA, EXPERIMENT_ID
from .aggregate import ARRAY_FIELDS, SCALAR_FIELDS
from .config import CODE_IDS, M_VALUES, P_TOKENS, load_config
from .crossing import classify_final_crossing, simultaneous_bootstrap, wilson_interval
from .io import arrays_sha256, canonical_json, sha256_json
from .replay import validate_replay_report_payload


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        expected = set(ARRAY_FIELDS) | set(SCALAR_FIELDS)
        if set(data.files) != expected:
            raise ValueError("publication aggregate fields do not match exp103.aggregate.v1")
        result = {}
        for key in data.files:
            value = data[key]
            result[key] = value.copy() if key in ARRAY_FIELDS else value.item()
    return result


def _allclose(left, right):
    return np.allclose(left, right, rtol=0.0, atol=1e-15, equal_nan=True)


def _validate(aggregate):
    if aggregate["schema_version"] != AGGREGATE_SCHEMA:
        raise ValueError("refusing non-exp103 aggregate schema")
    if aggregate["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("exp103 aggregate experiment identity mismatch")
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "decoder_mc.remote.v3.json"
    )
    config = load_config(config_path)
    for field, expected in (
        ("config_sha256", config["config_sha256"]),
        ("registry_sha256", config["registry_sha256"]),
        ("source_commit", config["source_commit"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
        ("decoder_binary_sha256", config["decoder_binary"]["sha256"]),
    ):
        if aggregate[field] != expected:
            raise ValueError(f"aggregate identity mismatch for {field}")
    repo_root = Path(__file__).resolve().parents[4]
    registry = load_registry(repo_root / config["registry_path"])
    if registry["registry_sha256"] != config["registry_sha256"]:
        raise ValueError("actual registry SHA differs from the frozen config")
    registry_rows = registry["codes"]
    if aggregate["payload_sha256"] != arrays_sha256(aggregate, ARRAY_FIELDS):
        raise ValueError("aggregate payload hash mismatch")
    replay_tuple = (aggregate["replay_status"], aggregate["replay_scope"])
    if aggregate["replay_status"] == "PASS":
        if aggregate["replay_scope"] not in {"stage1", "stage2", "final_combined"}:
            raise ValueError("PASS replay gate has an invalid scope")
        replay_report = json.loads(aggregate["replay_report_json"])
        if aggregate["replay_scope"] == "final_combined":
            if set(replay_report) != {"schema_version", "stage1", "stage2", "raw_manifest_sha256"} or replay_report["schema_version"] != "exp103.replay_bundle.v1":
                raise ValueError("final replay bundle schema mismatch")
            validate_replay_report_payload(replay_report["stage1"], config, "stage1")
            validate_replay_report_payload(replay_report["stage2"], config, "stage2")
            combined_entries = []
            for scope in ("stage1", "stage2"):
                for item in replay_report[scope]["results"]:
                    combined_entries.append({
                        "code_id": item["code_id"], "p_token": item["p_token"],
                        "shard_index": int(item["shard_index"]), "raw_sha256": item["raw_sha256"],
                    })
            combined_entries.sort(key=lambda item: (
                item["code_id"], item["p_token"], item["shard_index"],
            ))
            combined_manifest_sha = hashlib.sha256(
                canonical_json(combined_entries).encode("ascii")
            ).hexdigest()
            if replay_report["raw_manifest_sha256"] != combined_manifest_sha:
                raise ValueError("final replay bundle combined manifest mismatch")
        else:
            validate_replay_report_payload(replay_report, config, aggregate["replay_scope"])
        if aggregate["replay_report_sha256"] != sha256_json(replay_report):
            raise ValueError("aggregate replay report SHA mismatch")
        if aggregate["raw_manifest_sha256"] != replay_report["raw_manifest_sha256"]:
            raise ValueError("aggregate raw manifest SHA mismatch")
        if len(str(aggregate["raw_manifest_sha256"])) != 64:
            raise ValueError("aggregate raw manifest SHA is invalid")
    else:
        if replay_tuple not in {
            ("NOT_REQUIRED_INCOMPLETE", "none"),
            ("INVALID", "stage1"),
            ("INVALID", "stage2"),
            ("INVALID", "final_combined"),
        }:
            raise ValueError("non-PASS replay gate has an invalid status/scope")
        if any(aggregate[field] for field in (
            "replay_report_sha256", "raw_manifest_sha256",
        )) or aggregate["replay_report_json"] != "{}":
            raise ValueError("non-PASS replay gate contains a replay attestation")
        if aggregate["replay_status"] == "INVALID" and not any(
            str(item).startswith("replay_gate:")
            for item in json.loads(aggregate["unexpected_raw_errors_json"])
        ):
            raise ValueError("INVALID replay gate lacks its fail-closed error record")
    if not np.array_equal(aggregate["p_values"], [float(token) for token in P_TOKENS]):
        raise ValueError("aggregate p axis differs from the preregistered grid")
    if not np.array_equal(aggregate["m_values"], M_VALUES):
        raise ValueError("aggregate m axis differs from the frozen panel")
    if aggregate["code_ids"].tolist() != CODE_IDS:
        raise ValueError("aggregate code axis drops, adds, or reorders frozen codes")
    expected_code_m = np.repeat(M_VALUES, 8)
    if not np.array_equal(aggregate["code_m"], expected_code_m):
        raise ValueError("aggregate code-to-m mapping is invalid")
    if aggregate["classical_distance"].tolist() != [row["classical_distance"] for row in registry_rows]:
        raise ValueError("aggregate classical-distance metadata differs from the frozen registry")
    expected_shapes_dtypes = {
        "p_values": ((13,), np.dtype(np.float64)),
        "m_values": ((6,), np.dtype(np.int16)),
        "code_ids": ((48,), np.dtype("U8")),
        "code_m": ((48,), np.dtype(np.int16)),
        "classical_distance": ((48,), np.dtype(np.int16)),
        "code_status": ((48, 13), np.dtype("U12")),
        "failure_counts": ((48, 13), np.dtype(np.int64)),
        "trial_counts": ((48, 13), np.dtype(np.int64)),
        "m_status": ((6, 13), np.dtype("U12")),
        "stage1_adjacent_delta": ((2, 13), np.dtype(np.float64)),
        "stage1_adjacent_band_low": ((2, 13), np.dtype(np.float64)),
        "stage1_adjacent_band_high": ((2, 13), np.dtype(np.float64)),
        "stage1_primary_band_low": ((3, 13), np.dtype(np.float64)),
        "stage1_primary_band_high": ((3, 13), np.dtype(np.float64)),
    }
    for field in (
        "code_rates", "wilson_low", "wilson_high", "bp_convergence_rate",
        "mean_bp_iterations", "syndrome_mismatch_rate", "mean_logical_weight",
    ):
        expected_shapes_dtypes[field] = ((48, 13), np.dtype(np.float64))
    for field in (
        "primary_mean", "primary_median", "fixed_panel_mc_se", "between_code_sem",
        "between_code_std", "primary_band_low", "primary_band_high",
    ):
        expected_shapes_dtypes[field] = ((6, 13), np.dtype(np.float64))
    for field in (
        "delta38", "delta38_band_low", "delta38_band_high", "stage1_delta35",
        "stage1_band_low", "stage1_band_high",
    ):
        expected_shapes_dtypes[field] = ((13,), np.dtype(np.float64))
    for field in ("adjacent_delta", "adjacent_band_low", "adjacent_band_high"):
        expected_shapes_dtypes[field] = ((5, 13), np.dtype(np.float64))
    for field, (shape, dtype) in expected_shapes_dtypes.items():
        if aggregate[field].shape != shape or aggregate[field].dtype != dtype:
            raise ValueError(f"aggregate axis or dtype mismatch for {field}")
    statuses = aggregate["code_status"]
    if statuses.shape != (48, 13) or not set(np.unique(statuses)).issubset({"REPORTABLE", "INCOMPLETE", "INVALID"}):
        raise ValueError("invalid code-p status matrix")
    if aggregate["failure_counts"].shape != (48, 13) or aggregate["trial_counts"].shape != (48, 13):
        raise ValueError("invalid code-p count axes")
    for code_index in range(48):
        for p_index in range(13):
            status = statuses[code_index, p_index]
            failures = int(aggregate["failure_counts"][code_index, p_index])
            trials = int(aggregate["trial_counts"][code_index, p_index])
            rate = aggregate["code_rates"][code_index, p_index]
            if status == "REPORTABLE":
                if trials != 10000 or not 0 <= failures <= trials:
                    raise ValueError("reportable code-p does not contain exactly 10000 trials")
                if rate != failures / trials:
                    raise ValueError("code-p rate does not match raw-derived counts")
                low, high = wilson_interval(failures, trials)
                if not _allclose([aggregate["wilson_low"][code_index, p_index], aggregate["wilson_high"][code_index, p_index]], [low, high]):
                    raise ValueError("code-p Wilson interval mismatch")
                diagnostics = (
                    aggregate["bp_convergence_rate"][code_index, p_index],
                    aggregate["syndrome_mismatch_rate"][code_index, p_index],
                )
                if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in diagnostics):
                    raise ValueError("code-p diagnostic rate is invalid")
                if not 0.0 <= aggregate["mean_bp_iterations"][code_index, p_index] <= registry_rows[code_index]["n"]:
                    raise ValueError("code-p BP iteration diagnostic is invalid")
                if not 0.0 <= aggregate["mean_logical_weight"][code_index, p_index] <= registry_rows[code_index]["k"]:
                    raise ValueError("code-p logical-weight diagnostic is invalid")
            else:
                if failures != 0 or trials != 0:
                    raise ValueError("nonreportable code-p contains counts")
                for field in (
                    "code_rates", "wilson_low", "wilson_high", "bp_convergence_rate",
                    "mean_bp_iterations", "syndrome_mismatch_rate", "mean_logical_weight",
                ):
                    if not np.isnan(aggregate[field][code_index, p_index]):
                        raise ValueError("nonreportable code-p leaked a statistic")
    for m_index in range(6):
        code_slice = slice(8 * m_index, 8 * (m_index + 1))
        for p_index in range(13):
            code_status = statuses[code_slice, p_index]
            expected_status = "INVALID" if np.any(code_status == "INVALID") else (
                "REPORTABLE" if np.all(code_status == "REPORTABLE") else "INCOMPLETE"
            )
            if aggregate["m_status"][m_index, p_index] != expected_status:
                raise ValueError("m-p status does not fail closed over all eight codes")
            values = aggregate["code_rates"][code_slice, p_index]
            if expected_status == "REPORTABLE":
                if aggregate["primary_mean"][m_index, p_index] != values.mean():
                    raise ValueError("primary mean is not the equal-weight eight-code mean")
                if aggregate["primary_median"][m_index, p_index] != np.median(values):
                    raise ValueError("secondary median mismatch")
                trials = aggregate["trial_counts"][code_slice, p_index]
                expected_mc_se = np.sqrt(np.sum(values * (1.0 - values) / trials)) / 8.0
                expected_std = np.std(values, ddof=1)
                if aggregate["fixed_panel_mc_se"][m_index, p_index] != expected_mc_se:
                    raise ValueError("fixed-panel Monte Carlo SE mismatch")
                if aggregate["between_code_std"][m_index, p_index] != expected_std:
                    raise ValueError("between-code standard deviation mismatch")
                if aggregate["between_code_sem"][m_index, p_index] != expected_std / np.sqrt(8.0):
                    raise ValueError("between-code SEM mismatch")
            elif not np.isnan(aggregate["primary_mean"][m_index, p_index]):
                raise ValueError("valid-only m-p aggregation is forbidden")
            if expected_status != "REPORTABLE":
                for field in (
                    "primary_mean", "primary_median", "fixed_panel_mc_se",
                    "between_code_sem", "between_code_std",
                ):
                    if not np.isnan(aggregate[field][m_index, p_index]):
                        raise ValueError("nonreportable m-p leaked a statistic")
    stage1_complete = (
        np.all(aggregate["m_status"][:3] == "REPORTABLE")
        and aggregate["replay_status"] == "PASS"
        and aggregate["replay_scope"] in {"stage1", "final_combined"}
    )
    if stage1_complete:
        failures_3d = aggregate["failure_counts"].reshape(6, 8, 13)
        trials_3d = aggregate["trial_counts"].reshape(6, 8, 13)
        stage1_bootstrap = simultaneous_bootstrap(
            failures_3d, trials_3d, (0, 1, 2), config, "stage1_m3_m5",
        )
        for field, expected in (
            ("stage1_primary_band_low", stage1_bootstrap["point_low"]),
            ("stage1_primary_band_high", stage1_bootstrap["point_high"]),
            ("stage1_delta35", stage1_bootstrap["endpoint"]),
            ("stage1_band_low", stage1_bootstrap["endpoint_low"]),
            ("stage1_band_high", stage1_bootstrap["endpoint_high"]),
            ("stage1_adjacent_delta", stage1_bootstrap["adjacent"]),
            ("stage1_adjacent_band_low", stage1_bootstrap["adjacent_low"]),
            ("stage1_adjacent_band_high", stage1_bootstrap["adjacent_high"]),
        ):
            if not _allclose(aggregate[field], expected):
                raise ValueError(f"Stage 1 deterministic bootstrap mismatch for {field}")
        if aggregate["stage1_bootstrap_half_width"] != stage1_bootstrap["half_width"]:
            raise ValueError("Stage 1 bootstrap half-width mismatch")
        expected_delta35 = aggregate["primary_mean"][2] - aggregate["primary_mean"][0]
        expected_adjacent = np.diff(aggregate["primary_mean"][:3], axis=0)
        if not _allclose(aggregate["stage1_delta35"], expected_delta35):
            raise ValueError("Stage 1 Delta35 does not match primary means")
        if not _allclose(aggregate["stage1_adjacent_delta"], expected_adjacent):
            raise ValueError("Stage 1 adjacent contrasts do not match primary means")
        padded_adjacent = np.full((5, 13), np.nan)
        padded_low = np.full((5, 13), np.nan)
        padded_high = np.full((5, 13), np.nan)
        padded_adjacent[:2] = aggregate["stage1_adjacent_delta"]
        padded_low[:2] = aggregate["stage1_adjacent_band_low"]
        padded_high[:2] = aggregate["stage1_adjacent_band_high"]
        stage1_decision = classify_final_crossing(
            aggregate["p_values"], aggregate["stage1_delta35"],
            aggregate["stage1_band_low"], aggregate["stage1_band_high"],
            padded_adjacent, padded_low, padded_high,
        )
        expected_stage1_status = "STAGE1_RESTRICTED_" + stage1_decision["status"].removeprefix("EXP103_")
        if aggregate["stage1_status"] != expected_stage1_status:
            raise ValueError("Stage 1 restricted terminal decision mismatch")
        if json.loads(aggregate["stage1_compatible_triple_json"]) != stage1_decision["compatible_triple"]:
            raise ValueError("Stage 1 compatible-triple decision mismatch")
        saved_stage1_bracket = None if np.isnan(aggregate["stage1_bracket_low"]) else (
            aggregate["stage1_bracket_low"], aggregate["stage1_bracket_high"],
        )
        if np.isnan(aggregate["stage1_bracket_low"]) != np.isnan(aggregate["stage1_bracket_high"]):
            raise ValueError("Stage 1 bracket endpoints must both be finite or both NaN")
        if saved_stage1_bracket != stage1_decision["bracket"]:
            raise ValueError("Stage 1 crossing bracket mismatch")
    else:
        for field in (
            "stage1_delta35", "stage1_band_low", "stage1_band_high",
            "stage1_adjacent_delta", "stage1_adjacent_band_low", "stage1_adjacent_band_high",
            "stage1_primary_band_low", "stage1_primary_band_high",
        ):
            if np.any(np.isfinite(aggregate[field])):
                raise ValueError("incomplete Stage 1 contains restricted crossing input")
        if aggregate["stage1_status"] != "INCOMPLETE":
            raise ValueError("incomplete Stage 1 status mismatch")
        if not (np.isnan(aggregate["stage1_bracket_low"]) and np.isnan(aggregate["stage1_bracket_high"])):
            raise ValueError("incomplete Stage 1 contains a bracket")
        if aggregate["stage1_compatible_triple_json"] != "null":
            raise ValueError("incomplete Stage 1 contains a compatible triple")
        if np.isfinite(aggregate["stage1_bootstrap_half_width"]):
            raise ValueError("incomplete Stage 1 contains a bootstrap width")
    full_panel_reportable = np.all(statuses == "REPORTABLE")
    final_replay_pass = replay_tuple == ("PASS", "final_combined")
    if np.any(statuses == "INVALID") or json.loads(aggregate["unexpected_raw_errors_json"]):
        expected_overall, expected_terminal = "INVALID", "EXP103_INVALID"
    elif full_panel_reportable and not final_replay_pass:
        expected_overall, expected_terminal = "INVALID", "EXP103_INVALID"
    elif np.any(statuses == "INCOMPLETE"):
        expected_overall, expected_terminal = "INCOMPLETE", "EXP103_INCOMPLETE"
    else:
        final_bootstrap = simultaneous_bootstrap(
            aggregate["failure_counts"].reshape(6, 8, 13),
            aggregate["trial_counts"].reshape(6, 8, 13), tuple(range(6)),
            config, "final_m3_m8",
        )
        for field, expected in (
            ("primary_band_low", final_bootstrap["point_low"]),
            ("primary_band_high", final_bootstrap["point_high"]),
            ("delta38", final_bootstrap["endpoint"]),
            ("delta38_band_low", final_bootstrap["endpoint_low"]),
            ("delta38_band_high", final_bootstrap["endpoint_high"]),
            ("adjacent_delta", final_bootstrap["adjacent"]),
            ("adjacent_band_low", final_bootstrap["adjacent_low"]),
            ("adjacent_band_high", final_bootstrap["adjacent_high"]),
        ):
            if not _allclose(aggregate[field], expected):
                raise ValueError(f"final deterministic bootstrap mismatch for {field}")
        if aggregate["bootstrap_half_width"] != final_bootstrap["half_width"]:
            raise ValueError("final bootstrap half-width mismatch")
        expected_overall, expected_terminal = "COMPLETE", None
    if aggregate["overall_status"] != expected_overall:
        raise ValueError("aggregate overall status mismatch")
    if expected_terminal is not None:
        if aggregate["terminal_status"] != expected_terminal:
            raise ValueError("incomplete/invalid terminal status mismatch")
        for field in (
            "primary_band_low", "primary_band_high", "delta38", "delta38_band_low",
            "delta38_band_high", "adjacent_delta", "adjacent_band_low", "adjacent_band_high",
        ):
            if np.any(np.isfinite(aggregate[field])):
                raise ValueError("noncomplete aggregate contains final crossing input")
        if np.isfinite(aggregate["bootstrap_half_width"]):
            raise ValueError("noncomplete aggregate contains a final bootstrap width")
        if not (np.isnan(aggregate["crossing_bracket_low"]) and np.isnan(aggregate["crossing_bracket_high"])):
            raise ValueError("noncomplete aggregate contains a final bracket")
        if aggregate["compatible_triple_json"] != "null":
            raise ValueError("noncomplete aggregate contains a final compatible triple")
    else:
        if not final_replay_pass:
            raise ValueError("complete aggregate lacks a full replay PASS")
        expected_delta = aggregate["primary_mean"][5] - aggregate["primary_mean"][0]
        expected_adjacent = np.diff(aggregate["primary_mean"], axis=0)
        if not _allclose(aggregate["delta38"], expected_delta) or not _allclose(aggregate["adjacent_delta"], expected_adjacent):
            raise ValueError("saved contrasts do not match primary means")
        decision = classify_final_crossing(
            aggregate["p_values"], aggregate["delta38"], aggregate["delta38_band_low"],
            aggregate["delta38_band_high"], aggregate["adjacent_delta"],
            aggregate["adjacent_band_low"], aggregate["adjacent_band_high"],
        )
        if aggregate["terminal_status"] != decision["status"]:
            raise ValueError("saved terminal crossing decision is not reproducible")
        if json.loads(aggregate["compatible_triple_json"]) != decision["compatible_triple"]:
            raise ValueError("saved compatible-triple decision mismatch")
        bracket_values = (aggregate["crossing_bracket_low"], aggregate["crossing_bracket_high"])
        if np.isnan(bracket_values[0]) != np.isnan(bracket_values[1]):
            raise ValueError("crossing bracket endpoints must both be finite or both NaN")
        saved_bracket = None if np.isnan(bracket_values[0]) else bracket_values
        if saved_bracket != decision["bracket"]:
            raise ValueError("saved crossing bracket mismatch")


def load_exp103_crossing(result_path, point_mask=None):
    """Load read-only publication data after full schema and arithmetic validation."""
    aggregate = _load_npz(Path(result_path))
    _validate(aggregate)
    full_mask = np.ones(13, dtype=np.bool_)
    if point_mask is None or (isinstance(point_mask, str) and point_mask == "full"):
        mask = full_mask
    else:
        mask = np.asarray(point_mask)
        if mask.dtype != np.bool_ or mask.shape != (13,) or not np.array_equal(mask, full_mask):
            raise ValueError("point mask is not the preregistered full-grid mask")
    result = {}
    for key, value in aggregate.items():
        if isinstance(value, np.ndarray):
            array = value.copy()
            if array.ndim and array.shape[-1] == 13:
                array = array[..., mask]
            array.flags.writeable = False
            result[key] = array
        else:
            result[key] = value
    return MappingProxyType(result)
