"""Resource gate: measure the real cost, project it, compare against frozen caps.

Every projection is an upper bound. Per-trial cost is taken as the maximum over
the benchmarked grid points, and per-m cost is taken from the next anchor at or
above that m. exp103 showed how conservative that is (75.20 predicted wall hours
against 33.55 measured), so the caps must leave room for the bound rather than
for the expectation.
"""

import hashlib
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from ldpc import BpOsdDecoder

from .config import ensure_config, tasks_per_m
from .identity import runtime_identity
from .io import sha256_file
from .model import clear_model_cache, load_model, logical_label, parity_product
from .raw import load_raw, save_raw
from .replay import committed_replay_blocks
from .seeds import derive_seed
from .worker import RAW_FIELDS


# Every m is benchmarked directly. exp103 extrapolated m=6 and m=7 from the m=8
# anchor, which inflated their projected cost by more than a factor two; a gate
# with that much slop in it does not measure anything.


def benchmark_p_tokens(config):
    """The frozen positional selection of grid points to benchmark."""
    tokens = list(config["p_tokens"])
    if config["preflight"]["p_token_selection"] != "first_middle_last":
        raise ValueError("unexpected preflight p token selection")
    chosen = {tokens[0], tokens[len(tokens) // 2], tokens[-1]}
    return [token for token in tokens if token in chosen]


def _rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2)


def _make_decoder(model, p_token, q, spec):
    width = model.n + model.n_checks
    return BpOsdDecoder(
        model.H_augmented_sparse,
        error_channel=[float(p_token)] * model.n + [float(q)] * model.n_checks,
        bp_method=spec["bp_method"],
        max_iter=width, schedule=spec["schedule"],
        serial_schedule_order=list(range(width)), osd_method=spec["osd_method"],
        osd_order=spec["osd_order"], omp_thread_count=spec["omp_thread_count"],
    )


def _time_trials(model, decoder, p_token, q, seed, trials, expected=None):
    """Exercise the full worker path without reporting or selecting on outcomes."""
    rng = np.random.Generator(np.random.PCG64(seed))
    failures = np.zeros(trials, dtype=np.bool_)
    labels_saved = np.zeros((trials, model.k), dtype=np.uint8)
    readout_match = np.zeros(trials, dtype=np.bool_)
    bp_converged = np.zeros(trials, dtype=np.bool_)
    bp_iterations = np.zeros(trials, dtype=np.int32)
    start = time.perf_counter()
    width = model.n + model.n_checks
    for trial in range(trials):
        error = (rng.random(model.n) < float(p_token)).astype(np.uint8)
        readout = (rng.random(model.n_checks) < float(q)).astype(np.uint8)
        effective = np.bitwise_xor(parity_product(model.H_Z, error), readout)
        correction = decoder.decode(effective)
        if (
            not isinstance(correction, np.ndarray)
            or correction.dtype != np.uint8
            or correction.shape != (width,)
            or np.any(correction > 1)
        ):
            raise ValueError("decoder returned an illegal correction during preflight")
        residual = np.bitwise_xor(error, correction[:model.n])
        matched = bool(np.array_equal(correction[model.n:], readout))
        labels = logical_label(model, residual)
        failures[trial] = bool(labels.any())
        labels_saved[trial] = labels
        readout_match[trial] = matched
        bp_converged[trial] = bool(decoder.converge)
        bp_iterations[trial] = int(decoder.iter)
        if expected is not None:
            # Keep the comparison path live without letting any outcome reach
            # the report or the gate.
            _ = (
                bool(failures[trial]) == bool(expected["failure_flags"][trial]),
                np.array_equal(labels, expected["logical_labels"][trial]),
                int(bp_iterations[trial]) == int(expected["bp_iterations"][trial]),
            )
    elapsed = time.perf_counter() - start
    return elapsed, {
        "failure_flags": failures,
        "logical_labels": labels_saved,
        "readout_match": readout_match,
        "bp_converged": bp_converged,
        "bp_iterations": bp_iterations,
    }


def _synthetic_raw(m, config, identity):
    """Full-size, outcome-independent payload used only to time raw I/O."""
    codes = int(config["codes_per_task"][str(int(m))])
    tokens = list(config["p_tokens"])
    trials = int(config["trials_per_code_p"][str(int(m))])
    k = m ** 2
    rng = np.random.Generator(np.random.PCG64(12345 + m))
    shape = (codes, len(tokens), trials)
    return {
        "schema_version": "exp105.raw.v1",
        "status": "VALID",
        "invalid_reason": "",
        "exception_type": "",
        "exception_message": "",
        "experiment_id": config["experiment_id"],
        "m": int(m),
        "block_index": 0,
        "planned_codes": codes,
        "completed_codes": codes,
        "p_tokens": ",".join(tokens),
        "q_token": config["q_token"],
        "trials_per_code_p": trials,
        "seed_namespace": config["namespaces"]["benchmark"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "decoder_binary_sha256": identity["decoder_binary_sha256"],
        "python_version": identity["python_version"],
        "numpy_version": identity["numpy_version"],
        "scipy_version": identity["scipy_version"],
        "ldpc_version": identity["ldpc_version"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
        "n": 25 * m ** 2,
        "n_checks": 12 * m ** 2,
        "k": k,
        "code_index": np.arange(codes, dtype=np.int32),
        "graph_seed": rng.integers(0, 1 << 62, size=codes, dtype=np.int64),
        "classical_distance": np.full(codes, 6, dtype=np.int16),
        "classical_H_sha256": np.full(codes, "0" * 64, dtype="<U64"),
        "logical_frame_sha256": np.full(codes, "0" * 64, dtype="<U64"),
        "observable_frame_fingerprint": np.full(codes, "0" * 64, dtype="<U64"),
        "trial_seed": rng.integers(
            0, 1 << 62, size=(codes, len(tokens)), dtype=np.int64,
        ),
        "failure_flags": rng.integers(0, 2, size=shape, dtype=np.uint8).astype(np.bool_),
        "logical_labels": rng.integers(0, 2, size=shape + (k,), dtype=np.uint8),
        "readout_match": np.ones(shape, dtype=np.bool_),
        "bp_converged": rng.integers(0, 2, size=shape, dtype=np.uint8).astype(np.bool_),
        "bp_iterations": rng.integers(
            0, 25 * m ** 2 + 1, size=shape, dtype=np.int32,
        ),
        "error_stream_sha256": hashlib.sha256(b"error").hexdigest(),
        "readout_stream_sha256": hashlib.sha256(b"readout").hexdigest(),
        "correction_stream_sha256": hashlib.sha256(b"correction").hexdigest(),
        "label_stream_sha256": hashlib.sha256(b"label").hexdigest(),
    }


def _time_raw_io(payload):
    with tempfile.TemporaryDirectory(prefix="exp105-preflight-") as temporary:
        path = Path(temporary) / "benchmark_raw.npz"
        start = time.perf_counter()
        save_raw(path, payload)
        serialization = time.perf_counter() - start
        start = time.perf_counter()
        loaded = load_raw(path)
        load_seconds = time.perf_counter() - start
        if set(loaded) != RAW_FIELDS:
            raise ValueError("preflight raw round trip lost fields")
        start = time.perf_counter()
        sha256_file(path)
        hash_seconds = time.perf_counter() - start
        size = path.stat().st_size
    return {
        "raw_serialization_seconds": serialization,
        "raw_load_seconds": load_seconds,
        "raw_sha256_seconds": hash_seconds,
        "raw_bytes": int(size),
    }


def benchmark_task(m, code_index, p_token, config, registry_rows, trials=None):
    """Time one frozen code-p path without persisting or reporting outcomes."""
    config = ensure_config(config)
    trials = int(
        config["trials_per_code_p"][str(int(m))] if trials is None else trials
    )
    q = float(config["q_token"])
    from .config import code_id as make_code_id

    identifier = make_code_id(m, code_index)
    clear_model_cache()
    start = time.perf_counter()
    model = load_model(registry_rows[identifier])
    model_seconds = time.perf_counter() - start

    start = time.perf_counter()
    identity = runtime_identity(config)
    identity_seconds = time.perf_counter() - start

    start = time.perf_counter()
    decoder = _make_decoder(model, p_token, q, config["decoder"])
    decoder_setup_seconds = time.perf_counter() - start

    seed = derive_seed(config, "benchmark", identifier, p_token, 0)
    measurement_seconds, expected = _time_trials(
        model, decoder, p_token, q, seed, trials,
    )

    start = time.perf_counter()
    replay_decoder = _make_decoder(model, p_token, q, config["decoder"])
    replay_setup_seconds = time.perf_counter() - start
    replay_seconds, _ = _time_trials(
        model, replay_decoder, p_token, q, seed, trials, expected=expected,
    )
    io_timings = _time_raw_io(_synthetic_raw(m, config, identity))
    clear_model_cache()
    return {
        "m": int(m),
        "code_index": int(code_index),
        "p_token": p_token,
        "trials": trials,
        "model_seconds": model_seconds,
        "identity_seconds": identity_seconds,
        "decoder_setup_seconds": decoder_setup_seconds,
        "replay_setup_seconds": replay_setup_seconds,
        "measurement_seconds_per_trial": measurement_seconds / trials,
        "replay_seconds_per_trial": replay_seconds / trials,
        "peak_rss_gib": _rss_gib(),
        "seed_namespace": config["namespaces"]["benchmark"],
        **io_timings,
    }


def _upper_by_m(tasks, field, m_values):
    by_anchor = {}
    for anchor in sorted(set(m_values)):
        selected = [task for task in tasks if task["m"] == anchor]
        if not selected:
            raise ValueError(f"resource preflight is missing anchor m={anchor}")
        by_anchor[anchor] = max(task[field] for task in selected)
    return by_anchor


def estimate_resources(tasks, config, num_workers, caps):
    """Project the whole single-stage run from the benchmarked anchors."""
    config = ensure_config(config)
    tokens = config["p_tokens"]
    m_values = [int(m) for m in config["m_values"]]
    codes_per_m = {int(k): int(v) for k, v in config["codes_per_m"].items()}
    codes_per_task = {int(k): int(v) for k, v in config["codes_per_task"].items()}
    trials_by_m = {int(k): int(v) for k, v in config["trials_per_code_p"].items()}
    per_m_tasks = tasks_per_m(codes_per_m, codes_per_task)
    fields = (
        "measurement_seconds_per_trial", "replay_seconds_per_trial",
        "model_seconds", "identity_seconds", "decoder_setup_seconds",
        "replay_setup_seconds", "raw_serialization_seconds", "raw_load_seconds",
        "raw_sha256_seconds",
    )
    upper = {field: _upper_by_m(tasks, field, m_values) for field in fields}
    replay_blocks = committed_replay_blocks(config)

    def per_code_seconds(m, kind):
        trial_field = (
            "measurement_seconds_per_trial" if kind == "measurement"
            else "replay_seconds_per_trial"
        )
        setup_field = (
            "decoder_setup_seconds" if kind == "measurement" else "replay_setup_seconds"
        )
        return (
            upper["model_seconds"][m]
            + len(tokens) * upper[setup_field][m]
            + len(tokens) * trials_by_m[m] * upper[trial_field][m]
        )

    generation = 0.0
    replay = 0.0
    per_m = {}
    for m in m_values:
        code_cost = codes_per_m[m] * per_code_seconds(m, "measurement")
        task_cost = per_m_tasks[m] * (
            upper["identity_seconds"][m] + upper["raw_serialization_seconds"][m]
        )
        replay_codes = len(replay_blocks[m]) * codes_per_task[m]
        replay_cost = replay_codes * per_code_seconds(m, "replay")
        replay_task_cost = len(replay_blocks[m]) * (
            upper["identity_seconds"][m]
            + upper["raw_load_seconds"][m]
            + upper["raw_sha256_seconds"][m]
        )
        generation += code_cost + task_cost
        replay += replay_cost + replay_task_cost
        per_m[str(m)] = {
            "codes": codes_per_m[m],
            "tasks": per_m_tasks[m],
            "replay_tasks": len(replay_blocks[m]),
            "generation_core_hours": (code_cost + task_cost) / 3600.0,
            "replay_core_hours": (replay_cost + replay_task_cost) / 3600.0,
            "measurement_seconds_per_trial_upper": upper[
                "measurement_seconds_per_trial"
            ][m],
            "model_seconds_upper": upper["model_seconds"][m],
        }

    generation_core_hours = generation / 3600.0
    replay_core_hours = replay / 3600.0
    analysis = float(config["preflight"]["analysis_core_hours"])
    overhead = float(config["preflight"]["fixed_overhead_core_hours"])
    base = generation_core_hours + replay_core_hours + analysis + overhead
    reserved = float(caps["reserve_multiplier"]) * base
    wall = (generation_core_hours + replay_core_hours) / float(num_workers) + analysis + overhead
    peak_rss = max(task["peak_rss_gib"] for task in tasks) * float(num_workers) * 3.0
    checks = {
        "reserved_core_hours_le_cap": reserved <= float(caps["stage_core_hour_cap"]),
        "predicted_wall_hours_le_cap": wall <= float(caps["stage_wall_hour_cap"]),
        "peak_rss_gib_le_cap": peak_rss <= float(caps["peak_rss_gib_cap"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "BLOCKED_RESOURCE_PREFLIGHT",
        "m_values": list(m_values),
        "num_workers": int(num_workers),
        "total_codes": sum(codes_per_m[m] for m in m_values),
        "total_tasks": sum(per_m_tasks[m] for m in m_values),
        "total_replay_tasks": sum(len(value) for value in replay_blocks.values()),
        "total_trials": sum(
            codes_per_m[m] * len(tokens) * trials_by_m[m] for m in m_values
        ),
        "measurement_generation_core_hours": generation_core_hours,
        "committed_replay_core_hours": replay_core_hours,
        "analysis_core_hours": analysis,
        "fixed_overhead_core_hours": overhead,
        "reserved_core_hours": reserved,
        "predicted_wall_hours": wall,
        "projected_peak_rss_gib": peak_rss,
        "per_m": per_m,
        "checks": checks,
    }


def run_local_preflight(config, registry_rows):
    config = ensure_config(config)
    spec = config["preflight"]
    tasks = [
        benchmark_task(m, code_index, token, config, registry_rows)
        for m in config["m_values"]
        for code_index in spec["code_indices"]
        for token in benchmark_p_tokens(config)
    ]
    estimate = estimate_resources(
        tasks, config, spec["num_workers"],
        {
            "reserve_multiplier": spec["reserve_multiplier"],
            "stage_core_hour_cap": spec["stage_core_hour_cap"],
            "stage_wall_hour_cap": spec["stage_wall_hour_cap"],
            "peak_rss_gib_cap": spec["peak_rss_gib_cap"],
        },
    )
    return {
        "schema_version": "exp105.local_resource_preflight.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "outcome_blind": True,
        "estimate": estimate,
        "tasks": tasks,
    }
