import hashlib
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from ldpc import BpLsdDecoder

from .config import ensure_config
from .identity import runtime_identity
from .io import canonical_json, sha256_file
from .model import clear_model_cache, load_model, parity_product
from .raw import load_raw, save_raw
from .seeds import derive_seed


_RAW_ARRAY_FIELDS = {
    "failure_flags", "logical_labels", "syndrome_match", "bp_converged",
    "bp_iterations",
}


def _rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2)


def _make_decoder(model, p_token, spec):
    return BpLsdDecoder(
        model.H_Z_sparse, error_rate=float(p_token), bp_method=spec["bp_method"],
        max_iter=model.n, schedule=spec["schedule"],
        serial_schedule_order=list(range(model.n)), lsd_method=spec["lsd_method"],
        lsd_order=spec["lsd_order"], bits_per_step=spec["bits_per_step"],
        always_run_lsd=spec["always_run_lsd"], omp_thread_count=spec["omp_thread_count"],
    )


def _time_full_trial_path(model, decoder, p_token, seed, trials, expected=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    failures = np.zeros(trials, dtype=np.bool_)
    labels_saved = np.zeros((trials, model.k), dtype=np.uint8)
    syndrome_match = np.zeros(trials, dtype=np.bool_)
    bp_converged = np.zeros(trials, dtype=np.bool_)
    bp_iterations = np.zeros(trials, dtype=np.int32)
    digests = (hashlib.sha256(), hashlib.sha256(), hashlib.sha256())
    start = time.perf_counter()
    for trial in range(trials):
        error = (rng.random(model.n) < float(p_token)).astype(np.uint8)
        syndrome = parity_product(model.H_Z, error)
        correction = decoder.decode(syndrome)
        if (
            not isinstance(correction, np.ndarray)
            or correction.dtype != np.uint8
            or correction.shape != (model.n,)
            or np.any(correction > 1)
        ):
            raise ValueError("BpLSD returned an illegal correction during preflight")
        residual = np.bitwise_xor(error, correction)
        matched = not parity_product(model.H_Z, residual).any()
        labels = parity_product(model.logical_Z, residual)
        failures[trial] = (not matched) or bool(labels.any())
        labels_saved[trial] = labels
        syndrome_match[trial] = matched
        bp_converged[trial] = bool(decoder.converge)
        bp_iterations[trial] = int(decoder.iter)
        if expected is not None:
            # Exercise the formal comparison path without making the resource
            # gate depend on, retain, or report any benchmark outcome.
            _ = (
                bool(failures[trial]) == bool(expected["failure_flags"][trial]),
                bool(syndrome_match[trial]) == bool(expected["syndrome_match"][trial]),
                np.array_equal(labels, expected["logical_labels"][trial]),
                bool(bp_converged[trial]) == bool(expected["bp_converged"][trial]),
                int(bp_iterations[trial]) == int(expected["bp_iterations"][trial]),
            )
        digests[0].update(error.tobytes())
        digests[1].update(correction.tobytes())
        digests[2].update(labels.tobytes())
    elapsed = time.perf_counter() - start
    arrays = {
        "failure_flags": failures,
        "logical_labels": labels_saved,
        "syndrome_match": syndrome_match,
        "bp_converged": bp_converged,
        "bp_iterations": bp_iterations,
    }
    hashes = {
        name: digest.hexdigest()
        for name, digest in zip(
            ("error_stream_sha256", "correction_stream_sha256", "label_stream_sha256"),
            digests,
        )
    }
    if expected is not None:
        for field, value in hashes.items():
            _ = value == expected[field]
    # Keep the complete worker path live without returning outcomes in the report.
    if sum(array.nbytes for array in arrays.values()) <= 0:
        raise AssertionError("benchmark diagnostics were not materialized")
    return elapsed, {**arrays, **hashes}


def _synthetic_raw_payload(model, code_id, p_token, seed, config, identity):
    """Build a full-size, outcome-independent raw payload for I/O timing."""
    trials = config["trials_per_shard"]
    rng = np.random.Generator(np.random.PCG64(seed))
    logical_labels = rng.integers(
        0, 2, size=(trials, model.k), dtype=np.uint8,
    )
    syndrome_match = rng.integers(0, 2, size=trials, dtype=np.uint8).astype(np.bool_)
    failure_flags = np.logical_or(
        np.logical_not(syndrome_match), logical_labels.any(axis=1),
    )
    bp_converged = rng.integers(0, 2, size=trials, dtype=np.uint8).astype(np.bool_)
    bp_iterations = rng.integers(
        0, model.n + 1, size=trials, dtype=np.int32,
    )
    tag = f"{code_id}:{p_token}:{seed}".encode("ascii")
    return {
        "schema_version": "exp103.raw.v1",
        "status": "VALID",
        "invalid_reason": "",
        "exception_type": "",
        "exception_message": "",
        "experiment_id": "exp103.decoder_mc.v1",
        "code_id": code_id,
        "m": model.m,
        "p_token": p_token,
        "p": float(p_token),
        "shard_index": 0,
        "planned_trials": trials,
        "completed_trials": trials,
        "seed": seed,
        "seed_namespace": config["namespaces"]["benchmark"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "bplsd_binary_sha256": identity["bplsd_binary_sha256"],
        "python_version": identity["python_version"],
        "numpy_version": identity["numpy_version"],
        "scipy_version": identity["scipy_version"],
        "ldpc_version": identity["ldpc_version"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
        "n": model.n,
        "k": model.k,
        "classical_distance": model.classical_distance,
        "error_stream_sha256": hashlib.sha256(b"error:" + tag).hexdigest(),
        "correction_stream_sha256": hashlib.sha256(b"correction:" + tag).hexdigest(),
        "label_stream_sha256": hashlib.sha256(logical_labels.tobytes()).hexdigest(),
        "failure_flags": failure_flags,
        "logical_labels": logical_labels,
        "syndrome_match": syndrome_match,
        "bp_converged": bp_converged,
        "bp_iterations": bp_iterations,
    }


def _time_raw_io(payload):
    """Time formal raw save/load/hash/manifest paths with no retained artifact."""
    with tempfile.TemporaryDirectory(prefix="exp103-preflight-") as temporary:
        path = Path(temporary) / "benchmark_raw.npz"
        start = time.perf_counter()
        save_raw(path, payload)
        raw_serialization_seconds = time.perf_counter() - start

        start = time.perf_counter()
        loaded = load_raw(path)
        for field, expected in payload.items():
            observed = loaded[field]
            if field in _RAW_ARRAY_FIELDS:
                if observed.shape != expected.shape or observed.dtype != expected.dtype:
                    raise ValueError(f"preflight raw array identity mismatch: {field}")
            elif observed != expected:
                raise ValueError(f"preflight raw scalar identity mismatch: {field}")
        raw_load_seconds = time.perf_counter() - start

        start = time.perf_counter()
        raw_sha256 = sha256_file(path)
        replay_raw_sha256_seconds = time.perf_counter() - start

        start = time.perf_counter()
        manifest_raw = load_raw(path)
        entry = {
            "code_id": str(manifest_raw["code_id"]),
            "p_token": str(manifest_raw["p_token"]),
            "shard_index": int(manifest_raw["shard_index"]),
            "raw_sha256": sha256_file(path),
        }
        hashlib.sha256(canonical_json([entry]).encode("ascii")).hexdigest()
        manifest_seconds = time.perf_counter() - start
    return {
        "raw_serialization_seconds": raw_serialization_seconds,
        "raw_load_seconds": raw_load_seconds,
        "replay_raw_sha256_seconds": replay_raw_sha256_seconds,
        "manifest_seconds": manifest_seconds,
        "loaded": loaded,
        "raw_sha256": raw_sha256,
    }


def benchmark_task(code_id, p_token, config):
    """Time a frozen task without persisting, reporting, or selecting on outcomes."""
    config = ensure_config(config)
    clear_model_cache()
    start = time.perf_counter()
    model = load_model(config, code_id)
    model_seconds = time.perf_counter() - start
    spec = config["decoder"]
    start = time.perf_counter()
    measurement_identity = runtime_identity(config)
    measurement_identity_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decoder = _make_decoder(model, p_token, spec)
    decoder_setup_seconds = time.perf_counter() - start
    trials = config["preflight"]["trials_per_task"]
    benchmark_seed = derive_seed(config, "benchmark", code_id, p_token, 0)
    measurement_seconds, expected = _time_full_trial_path(
        model, decoder, p_token, benchmark_seed, trials,
    )
    synthetic = _synthetic_raw_payload(
        model, code_id, p_token,
        derive_seed(config, "benchmark", code_id, p_token, 2),
        config, measurement_identity,
    )
    io_timings = _time_raw_io(synthetic)
    # Hold a full raw shard during replay, as the formal replay worker does. Only
    # the benchmarked prefix is replaced in memory; decoder outcomes never hit disk.
    loaded = io_timings.pop("loaded")
    io_timings.pop("raw_sha256")
    for field in (
        "failure_flags", "logical_labels", "syndrome_match", "bp_converged",
        "bp_iterations",
    ):
        loaded[field][:trials] = expected[field]
    for field in (
        "error_stream_sha256", "correction_stream_sha256", "label_stream_sha256",
    ):
        loaded[field] = expected[field]
    start = time.perf_counter()
    runtime_identity(config)
    replay_identity_seconds = time.perf_counter() - start
    start = time.perf_counter()
    replay_decoder = _make_decoder(model, p_token, spec)
    replay_setup_seconds = time.perf_counter() - start
    replay_seconds, _ = _time_full_trial_path(
        model, replay_decoder, p_token, benchmark_seed, trials, expected=loaded,
    )
    return {
        "code_id": code_id,
        "m": model.m,
        "p_token": p_token,
        "trials": trials,
        "model_seconds": model_seconds,
        "measurement_identity_seconds": measurement_identity_seconds,
        "decoder_setup_seconds": decoder_setup_seconds,
        "replay_identity_seconds": replay_identity_seconds,
        "replay_setup_seconds": replay_setup_seconds,
        "measurement_seconds": measurement_seconds,
        "replay_seconds": replay_seconds,
        "measurement_seconds_per_trial": measurement_seconds / trials,
        "replay_seconds_per_trial": replay_seconds / trials,
        "peak_rss_gib": _rss_gib(),
        "seed_namespace": config["namespaces"]["benchmark"],
        **io_timings,
    }


def _stage_estimate(name, m_values, tasks, config):
    anchor_fields = (
        "measurement_seconds_per_trial", "replay_seconds_per_trial",
        "model_seconds", "measurement_identity_seconds", "decoder_setup_seconds",
        "raw_serialization_seconds", "replay_identity_seconds",
        "replay_setup_seconds", "raw_load_seconds", "replay_raw_sha256_seconds",
        "manifest_seconds",
    )
    upper_by_anchor = {}
    for field in anchor_fields:
        upper_by_anchor[field] = {}
        for m in (3, 5, 8):
            selected = [task for task in tasks if task["m"] == m]
            upper_by_anchor[field][m] = max(task[field] for task in selected)
    upper_by_m = {field: {} for field in anchor_fields}
    rss_anchor_m_values = set()
    for m in m_values:
        if m == 3:
            anchor = 3
        elif m <= 5:
            anchor = 5
        else:
            anchor = 8
        rss_anchor_m_values.add(anchor)
        for field in anchor_fields:
            upper_by_m[field][m] = upper_by_anchor[field][anchor]
    shards_per_m = 8 * len(config["p_tokens"]) * config["shards_per_code_p"]
    trials_per_m = shards_per_m * config["trials_per_shard"]
    measurement_core_hours = sum(
        upper_by_m["measurement_seconds_per_trial"][m] * trials_per_m
        + upper_by_m["model_seconds"][m] * 8
        + (
            upper_by_m["measurement_identity_seconds"][m]
            + upper_by_m["decoder_setup_seconds"][m]
            + upper_by_m["raw_serialization_seconds"][m]
        ) * shards_per_m
        for m in m_values
    ) / 3600.0
    replay_core_hours = sum(
        upper_by_m["replay_seconds_per_trial"][m] * trials_per_m
        + upper_by_m["model_seconds"][m] * 8
        + (
            upper_by_m["replay_identity_seconds"][m]
            + upper_by_m["replay_setup_seconds"][m]
            + upper_by_m["raw_load_seconds"][m]
            + upper_by_m["replay_raw_sha256_seconds"][m]
            + upper_by_m["manifest_seconds"][m]
        ) * shards_per_m
        for m in m_values
    ) / 3600.0
    generation_core_hours = measurement_core_hours
    preflight = config["preflight"]
    base_core_hours = (
        generation_core_hours + replay_core_hours + preflight["analysis_core_hours"]
        + preflight["fixed_overhead_core_hours"]
    )
    reserved_core_hours = preflight["reserve_multiplier"] * base_core_hours
    predicted_wall_hours = (
        (generation_core_hours + replay_core_hours) / preflight["num_workers"]
        + preflight["analysis_core_hours"] + preflight["fixed_overhead_core_hours"]
    )
    peak_rss_gib = max(
        task["peak_rss_gib"] for task in tasks if task["m"] in rss_anchor_m_values
    ) * preflight["num_workers"] * 3.0
    checks = {
        "reserved_core_hours_le_cap": reserved_core_hours <= preflight["stage_core_hour_cap"],
        "predicted_wall_hours_le_cap": predicted_wall_hours <= preflight["stage_wall_hour_cap"],
        "peak_rss_gib_le_cap": peak_rss_gib <= preflight["peak_rss_gib_cap"],
    }
    return {
        "stage": name,
        "m_values": list(m_values),
        "status": "PASS" if all(checks.values()) else "BLOCKED_LOCAL_RESOURCE_PREFLIGHT",
        "measurement_generation_core_hours": generation_core_hours,
        "full_replay_core_hours": replay_core_hours,
        "analysis_core_hours": preflight["analysis_core_hours"],
        "fixed_overhead_core_hours": preflight["fixed_overhead_core_hours"],
        "reserved_core_hours": reserved_core_hours,
        "predicted_wall_hours": predicted_wall_hours,
        "projected_peak_rss_gib": peak_rss_gib,
        "rss_anchor_m_values": sorted(rss_anchor_m_values),
        "checks": checks,
        "measurement_seconds_per_trial_upper_by_m": {
            str(key): value
            for key, value in upper_by_m["measurement_seconds_per_trial"].items()
        },
        "replay_seconds_per_trial_upper_by_m": {
            str(key): value
            for key, value in upper_by_m["replay_seconds_per_trial"].items()
        },
        "generation_per_shard_seconds_upper_by_m": {
            str(m): (
                upper_by_m["measurement_identity_seconds"][m]
                + upper_by_m["decoder_setup_seconds"][m]
                + upper_by_m["raw_serialization_seconds"][m]
            )
            for m in m_values
        },
        "replay_per_shard_seconds_upper_by_m": {
            str(m): (
                upper_by_m["replay_identity_seconds"][m]
                + upper_by_m["replay_setup_seconds"][m]
                + upper_by_m["raw_load_seconds"][m]
                + upper_by_m["replay_raw_sha256_seconds"][m]
                + upper_by_m["manifest_seconds"][m]
            )
            for m in m_values
        },
    }


def run_resource_preflight(config):
    config = ensure_config(config)
    identity = runtime_identity(config, verify_source=True)
    tasks = []
    for code_id in config["preflight"]["code_ids"]:
        for p_token in config["preflight"]["p_tokens"]:
            tasks.append(benchmark_task(code_id, p_token, config))
    stages = {
        name: _stage_estimate(name, values, tasks, config)
        for name, values in config["stage_m_values"].items()
    }
    return {
        "schema_version": "exp103.resource_preflight.v1",
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "bplsd_binary_sha256": identity["bplsd_binary_sha256"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
        "outcome_blind": True,
        "logical_outcomes_saved": False,
        "tasks": tasks,
        "stages": stages,
        "status": "PASS_ALL_STAGES" if all(stage["status"] == "PASS" for stage in stages.values()) else "BLOCKED_LOCAL_RESOURCE_PREFLIGHT",
    }


def validate_resource_preflight_report(report, config):
    config = ensure_config(config)
    required = {
        "schema_version", "config_sha256", "registry_sha256", "source_commit",
        "source_tree_sha256", "bplsd_binary_sha256", "device_name", "hostname",
        "conda_environment", "conda_prefix_matches_python", "outcome_blind",
        "logical_outcomes_saved", "tasks", "stages", "status",
    }
    if set(report) != required or report["schema_version"] != "exp103.resource_preflight.v1":
        raise ValueError("resource preflight schema mismatch")
    if (
        report["conda_prefix_matches_python"] is not True
        or report["outcome_blind"] is not True
        or report["logical_outcomes_saved"] is not False
    ):
        raise ValueError("resource preflight boolean attestation mismatch")
    for field, expected in (
        ("config_sha256", config["config_sha256"]),
        ("registry_sha256", config["registry_sha256"]),
        ("source_commit", config["source_commit"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
        ("bplsd_binary_sha256", config["bplsd_binary"]["sha256"]),
        ("device_name", config["environment"]["device_name"]),
        ("hostname", config["environment"]["hostname"]),
        ("conda_environment", config["environment"]["conda_environment"]),
        (
            "conda_prefix_matches_python",
            config["environment"]["conda_prefix_matches_python"],
        ),
        ("outcome_blind", True),
        ("logical_outcomes_saved", False),
    ):
        if report[field] != expected:
            raise ValueError(f"resource preflight identity mismatch for {field}")
    tasks = report["tasks"]
    expected_task_ids = {
        (code_id, p_token)
        for code_id in config["preflight"]["code_ids"]
        for p_token in config["preflight"]["p_tokens"]
    }
    if len(tasks) != 9 or {(task["code_id"], task["p_token"]) for task in tasks} != expected_task_ids:
        raise ValueError("resource preflight task panel mismatch")
    for task in tasks:
        if set(task) != {
            "code_id", "m", "p_token", "trials", "model_seconds",
            "measurement_identity_seconds", "decoder_setup_seconds",
            "raw_serialization_seconds", "replay_identity_seconds",
            "replay_setup_seconds", "raw_load_seconds",
            "replay_raw_sha256_seconds", "manifest_seconds",
            "measurement_seconds", "replay_seconds",
            "measurement_seconds_per_trial", "replay_seconds_per_trial",
            "peak_rss_gib", "seed_namespace",
        }:
            raise ValueError("resource preflight task fields mismatch")
        if task["trials"] != config["preflight"]["trials_per_task"]:
            raise ValueError("resource preflight trial count mismatch")
        if task["seed_namespace"] != config["namespaces"]["benchmark"]:
            raise ValueError("measurement namespace leaked into benchmark")
        if task["measurement_seconds_per_trial"] != task["measurement_seconds"] / task["trials"]:
            raise ValueError("resource preflight timing arithmetic mismatch")
        if task["replay_seconds_per_trial"] != task["replay_seconds"] / task["trials"]:
            raise ValueError("resource preflight replay timing arithmetic mismatch")
        if min(
            task["model_seconds"], task["measurement_identity_seconds"],
            task["decoder_setup_seconds"], task["raw_serialization_seconds"],
            task["replay_identity_seconds"], task["replay_setup_seconds"],
            task["raw_load_seconds"], task["replay_raw_sha256_seconds"],
            task["manifest_seconds"], task["measurement_seconds"],
            task["replay_seconds"], task["peak_rss_gib"],
        ) < 0:
            raise ValueError("resource preflight contains a negative measurement")
    expected_stages = {
        name: _stage_estimate(name, values, tasks, config)
        for name, values in config["stage_m_values"].items()
    }
    if report["stages"] != expected_stages:
        raise ValueError("resource preflight stage arithmetic mismatch")
    expected_status = "PASS_ALL_STAGES" if all(
        stage["status"] == "PASS" for stage in expected_stages.values()
    ) else "BLOCKED_LOCAL_RESOURCE_PREFLIGHT"
    if report["status"] != expected_status:
        raise ValueError("resource preflight terminal status mismatch")
    return report
