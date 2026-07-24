"""Outcome-blind equivalence and runtime preflight for the streaming CDF."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib import import_module
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    _column_log_weights_unchecked,
    _column_streaming_cdf_unchecked,
    build_full_column_candidate_cache,
    build_full_column_streaming_cache,
    build_full_column_streaming_workspace,
    build_full_column_workspace,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column import (
    RandomFullColumnStreamingConfig,
    replay_random_full_column_streaming_trajectory,
    run_random_full_column_streaming_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


VERSION = "exp102.q0_random_full_column_streaming.preflight.v1"
REPORT_VERSION = "exp102.q0_random_full_column_streaming.preflight.report.v1"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = EXP102_ROOT / "config/q0_random_full_column_streaming.preflight.v1.json"
REVIEW_PATH = EXP102_ROOT / "RANDOM_FULL_COLUMN_STREAMING_REVIEW.md"
CONTROL_ROOT = EXP102_ROOT / "validation/052_q0_random_full_column_t1_m8_20260724"
WORKFLOW_052 = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.workflow"
)
STATE_NAMES = ("P", "M0", "S0", "U0")


class StreamingPreflightError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise StreamingPreflightError(message)


def load_canonical(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    value = json.loads(serialized)
    require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def verify_clean_source(expected_commit):
    head = subprocess.run(
        ("git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=all"),
        check=True, capture_output=True, text=True,
    ).stdout
    require(head == expected_commit, "source commit does not match HEAD")
    require(not status, "streaming preflight requires a clean source worktree")


def validate_config(config):
    require(config["version"] == config["contract_version"] == VERSION,
            "streaming preflight version changed")
    require(config["equivalence"] == {
        "columns": [0, 11, 17],
        "states": ["P", "M0", "S0", "U0"],
        "uniform_initialization_seed": 12817309175145032741,
    }, "equivalence panel changed")
    require(config["resource"] == {
        "fixed_workers": 4,
        "legacy_timing_repeats": 3,
        "local_min_speedup": 4.2,
        "runtime_probe_burn_updates": 8,
        "runtime_probe_measurement_updates": 128,
        "safety_factor": 1.2,
        "t1_burn_updates": 2048,
        "t1_measurement_updates": 8192,
        "trajectory_wall_cap_seconds": 7200.0,
    }, "resource contract changed")
    require(config["source_control"] == {
        "control_content_sha256": (
            "b99fb047e787fd999cde113bd3c64a1e9ef0e41e805d79a3d6d5f7995b6b8df6"
        ),
        "control_file_sha256": (
            "a43865186be0865ba8f1eac35ec22354ebe92ea6528091ce32e6f6dcaa118a41"
        ),
        "validation": "052_q0_random_full_column_t1_m8_20260724",
    }, "source control identity changed")


def load_context(config, config_sha):
    old_config, old_config_sha = WORKFLOW_052._load_config()
    context = WORKFLOW_052._load_control(
        CONTROL_ROOT / "control", old_config, old_config_sha,
    )
    require(
        context["metadata"]["control_content_sha256"]
        == config["source_control"]["control_content_sha256"],
        "control content changed",
    )
    require(
        sha256_file(CONTROL_ROOT / "control/control.npz")
        == config["source_control"]["control_file_sha256"],
        "control file changed",
    )
    fixed = context["fixed_states"]
    states = {
        "P": fixed[0].copy(),
        "M0": fixed[1].copy(),
        "S0": fixed[3].copy(),
        "U0": uniform_hard_coset_state(
            context["model"], context["syndrome"],
            config["equivalence"]["uniform_initialization_seed"],
        ),
    }
    require(tuple(states) == STATE_NAMES, "state panel changed")
    for state in states.values():
        residual = (
            context["model"].H_check.astype(np.int64)
            @ state.astype(np.int64) % 2
        ).astype(np.uint8)
        require(np.array_equal(residual, context["syndrome"]),
                "preflight state left the hard coset")
    context.update({"config_sha": config_sha, "states": states})
    return context


def cdf_digest(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_legacy_cdf(context, state, column, log_mass, cache, workspace):
    H = context["H"]
    b_columns, a_syndromes, _ = _initial_collapsed_masks(
        state, context["syndrome"], H,
    )
    _column_log_weights_unchecked(
        H, b_columns, a_syndromes, column, log_mass, cache, workspace,
    )
    maximum = float(np.max(workspace.log_weights))
    np.subtract(workspace.log_weights, maximum, out=workspace.log_weights)
    np.exp(workspace.log_weights, out=workspace.log_weights)
    np.cumsum(workspace.log_weights, out=workspace.log_weights)
    return workspace.log_weights


def build_streaming_cdf(context, state, column, log_mass, cache, workspace):
    H = context["H"]
    b_columns, a_syndromes, _ = _initial_collapsed_masks(
        state, context["syndrome"], H,
    )
    _, total = _column_streaming_cdf_unchecked(
        H, b_columns, a_syndromes, column, log_mass, cache, workspace,
        engine="numba",
    )
    require(total == float(workspace.cdf[-1]), "streaming CDF total drifted")
    return workspace.cdf


def equivalence_and_speed(context, config, mass):
    H = context["H"]
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    legacy_cache = build_full_column_candidate_cache(H.shape[0], 0.04)
    legacy_workspace = build_full_column_workspace(legacy_cache)
    streaming_cache = build_full_column_streaming_cache(H.shape[0], 0.04)
    streaming_workspace = build_full_column_streaming_workspace(streaming_cache)
    probes = []
    all_equal = True
    for state_name in config["equivalence"]["states"]:
        for column in config["equivalence"]["columns"]:
            legacy = build_legacy_cdf(
                context, context["states"][state_name], column, log_mass,
                legacy_cache, legacy_workspace,
            )
            streaming = build_streaming_cdf(
                context, context["states"][state_name], column, log_mass,
                streaming_cache, streaming_workspace,
            )
            equal = np.array_equal(legacy, streaming)
            all_equal = all_equal and equal
            probes.append({
                "cdf_sha256": cdf_digest(streaming),
                "column": int(column),
                "equal": bool(equal),
                "state": state_name,
            })

    timing_state = context["states"]["P"]
    timing_column = 0
    repeats = config["resource"]["legacy_timing_repeats"]
    legacy_seconds = []
    streaming_seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        legacy = build_legacy_cdf(
            context, timing_state, timing_column, log_mass,
            legacy_cache, legacy_workspace,
        )
        legacy_seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        streaming = build_streaming_cdf(
            context, timing_state, timing_column, log_mass,
            streaming_cache, streaming_workspace,
        )
        streaming_seconds.append(time.perf_counter() - start)
        require(np.array_equal(legacy, streaming), "timed CDFs diverged")
    legacy_median = statistics.median(legacy_seconds)
    streaming_median = statistics.median(streaming_seconds)
    speedup = legacy_median / streaming_median
    return {
        "all_cdfs_byte_equal": bool(all_equal),
        "legacy_seconds": legacy_seconds,
        "legacy_seconds_median": legacy_median,
        "probes": probes,
        "speedup": speedup,
        "streaming_seconds": streaming_seconds,
        "streaming_seconds_median": streaming_median,
    }


_RUNTIME_CONTEXT = None


def runtime_worker(index):
    context = _RUNTIME_CONTEXT
    config = context["preflight_config"]
    resource = config["resource"]
    state_name = STATE_NAMES[int(index)]
    sampler = RandomFullColumnStreamingConfig(
        p=0.04,
        burn_updates=resource["runtime_probe_burn_updates"],
        measurement_updates=resource["runtime_probe_measurement_updates"],
    )
    prefix = (
        config["seed_namespace"], context["config_sha"], "runtime",
        state_name, int(index),
    )
    cache = build_full_column_streaming_cache(context["H"].shape[0], 0.04)
    workspace = build_full_column_streaming_workspace(cache)
    arguments = (
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, context["states"][state_name],
        derive_seed(*prefix, "burn"), derive_seed(*prefix, "measurement"),
        derive_seed(*prefix, "observation"),
    )
    start = time.perf_counter()
    raw = run_random_full_column_streaming_trajectory(
        *arguments, mass=context["mass"], cache=cache, workspace=workspace,
    )
    sampling_seconds = time.perf_counter() - start
    start = time.perf_counter()
    replay_random_full_column_streaming_trajectory(
        *arguments, raw, mass=context["mass"], cache=cache, workspace=workspace,
    )
    replay_seconds = time.perf_counter() - start
    updates = sampler.burn_updates + sampler.measurement_updates
    replay_inclusive_seconds_per_update = (
        sampling_seconds + replay_seconds
    ) / updates
    projected = (
        replay_inclusive_seconds_per_update
        * (resource["t1_burn_updates"] + resource["t1_measurement_updates"])
        * resource["safety_factor"]
    )
    transcript = hashlib.sha256()
    for name in (
        "burn__selected_columns", "burn__old_columns", "burn__new_columns",
        "measurement__selected_columns", "measurement__old_columns",
        "measurement__new_columns", "measurement__b_columns",
        "measurement__labels", "measurement__states_packed",
        "measurement__weights", "final_b_columns", "final_state_packed",
    ):
        value = np.ascontiguousarray(raw[name])
        transcript.update(name.encode("ascii") + b"\0")
        transcript.update(value.dtype.str.encode("ascii") + b"\0")
        transcript.update(np.asarray(value.shape, dtype=">u8").tobytes())
        transcript.update(value.tobytes())
    return {
        "index": int(index),
        "projected_replay_inclusive_t1_seconds": projected,
        "replay_inclusive_seconds_per_update": replay_inclusive_seconds_per_update,
        "replay_seconds": replay_seconds,
        "sampling_seconds": sampling_seconds,
        "state": state_name,
        "transcript_sha256": transcript.hexdigest(),
        "updates": updates,
    }


def source_identity(source_commit, config_sha):
    files = {
        "config": CONFIG_PATH,
        "full_column": (
            EXP102_ROOT / "exp102_pipeline/q0_hgp_full_column_gibbs.py"
        ),
        "random_full_column": (
            EXP102_ROOT / "exp102_pipeline/q0_hgp_random_full_column.py"
        ),
        "review": REVIEW_PATH,
        "runner": Path(__file__).resolve(),
        "test_full_column": EXP102_ROOT / "tests/test_q0_hgp_full_column_gibbs.py",
        "test_random_full_column": (
            EXP102_ROOT / "tests/test_q0_hgp_random_full_column.py"
        ),
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    require(hashes["config"] == config_sha, "config source hash changed")
    core = {"files": hashes, "source_commit": source_commit}
    return {**core, "source_identity_sha256": sha256_json(core)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", default="macmini")
    args = parser.parse_args()
    require(not args.output.exists(), "immutable streaming preflight output exists")
    verify_clean_source(args.source_commit)
    config = load_canonical(CONFIG_PATH)
    validate_config(config)
    config_sha = sha256_file(CONFIG_PATH)
    context = load_context(config, config_sha)
    mass = build_classical_coset_mass(context["H"], 0.04, engine="numba")
    equivalence = equivalence_and_speed(context, config, mass)

    context.update({
        "config_sha": config_sha,
        "mass": mass,
        "preflight_config": config,
    })
    global _RUNTIME_CONTEXT
    _RUNTIME_CONTEXT = context
    with ProcessPoolExecutor(
        max_workers=config["resource"]["fixed_workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        runtime = list(executor.map(runtime_worker, range(len(STATE_NAMES))))
    _RUNTIME_CONTEXT = None

    resource = config["resource"]
    exact_pass = equivalence["all_cdfs_byte_equal"]
    speed_pass = equivalence["speedup"] >= resource["local_min_speedup"]
    runtime_pass = all(
        row["projected_replay_inclusive_t1_seconds"]
        <= resource["trajectory_wall_cap_seconds"]
        for row in runtime
    )
    if not exact_pass:
        status = "CONFLICT"
    elif not speed_pass or not runtime_pass:
        status = "RUNTIME_EXHAUSTED"
    else:
        status = "STREAMING_PREFLIGHT_LOCAL_PASS"
    core = {
        "checks": {
            "m8_cdf_byte_equivalence": exact_pass,
            "minimum_speedup": speed_pass,
            "runtime_projection": runtime_pass,
        },
        "config_sha256": config_sha,
        "equivalence": equivalence,
        "node": args.node,
        "report_version": REPORT_VERSION,
        "runtime": runtime,
        "source_identity": source_identity(args.source_commit, config_sha),
        "status": status,
        "worst_projected_replay_inclusive_t1_seconds": max(
            row["projected_replay_inclusive_t1_seconds"] for row in runtime
        ),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
