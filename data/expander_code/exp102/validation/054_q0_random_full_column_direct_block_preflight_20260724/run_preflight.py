"""Correctness and runtime preflight for direct-positive block sampling."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib import import_module
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - this preflight requires Numba.
    njit = None


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
    FULL_COLUMN_DIRECT_BLOCK_BITS,
    _column_log_weights_unchecked,
    build_full_column_candidate_cache,
    build_full_column_direct_block_cache,
    build_full_column_direct_block_workspace,
    build_full_column_streaming_cache,
    build_full_column_streaming_workspace,
    build_full_column_workspace,
    full_column_direct_block_gibbs_update,
    full_column_direct_block_subtotals,
    full_column_streaming_gibbs_update,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column import (
    RandomFullColumnDirectBlockConfig,
    replay_random_full_column_direct_block_trajectory,
    run_random_full_column_direct_block_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


VERSION = "exp102.q0_random_full_column_direct_block.preflight.v1"
REPORT_VERSION = "exp102.q0_random_full_column_direct_block.preflight.report.v1"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = EXP102_ROOT / "config/q0_random_full_column_direct_block.preflight.v1.json"
REVIEW_PATH = EXP102_ROOT / "RANDOM_FULL_COLUMN_DIRECT_BLOCK_REVIEW.md"
CONTROL_ROOT = EXP102_ROOT / "validation/052_q0_random_full_column_t1_m8_20260724"
WORKFLOW_052 = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.workflow"
)
STATE_NAMES = ("P", "M0", "S0", "U0")


class DirectBlockPreflightError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise DirectBlockPreflightError(message)


def load_canonical(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    value = json.loads(serialized)
    require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def verify_clean_source(expected_commit):
    if (PROJECT_ROOT / ".git").exists():
        head = subprocess.run(
            ("git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(PROJECT_ROOT), "status", "--porcelain",
             "--untracked-files=all"),
            check=True, capture_output=True, text=True,
        ).stdout
        require(head == expected_commit, "source commit does not match HEAD")
        require(not status, "direct-block preflight requires a clean source worktree")
    else:
        require(os.environ.get("EXP102_SOURCE_COMMIT") == expected_commit,
                "verified archive source identity is missing")


def validate_config(config):
    require(config["version"] == config["contract_version"] == VERSION,
            "direct-block preflight version changed")
    require(config["correctness"] == {
        "block_bits": 12,
        "columns": [0, 11, 17],
        "max_relative_weight_error": 5e-12,
        "max_scaled_weight_absolute_error": 5e-13,
        "max_total_variation": 2e-12,
        "minimum_normal_margin_factor": 1048576,
        "states": ["P", "M0", "S0", "U0"],
        "uniform_initialization_seed": 12817309175145032741,
    }, "direct-block correctness contract changed")
    require(config["resource"] == {
        "fixed_workers": 4,
        "local_min_streaming_speedup": 1.25,
        "runtime_probe_burn_updates": 8,
        "runtime_probe_measurement_updates": 128,
        "safety_factor": 1.2,
        "t1_burn_updates": 2048,
        "t1_measurement_updates": 8192,
        "trajectory_wall_cap_seconds": 7200.0,
    }, "direct-block resource contract changed")
    require(config["source_control"] == {
        "control_content_sha256": (
            "b99fb047e787fd999cde113bd3c64a1e9ef0e41e805d79a3d6d5f7995b6b8df6"
        ),
        "control_file_sha256": (
            "a43865186be0865ba8f1eac35ec22354ebe92ea6528091ce32e6f6dcaa118a41"
        ),
        "validation": "052_q0_random_full_column_t1_m8_20260724",
    }, "direct-block control identity changed")
    require(
        config["runtime_seed_key"]
        == "679dde13a0e6ea3058d56435964013c63df520eb5da39f04ed2feab06da6eecc",
        "direct-block runtime seed schedule changed",
    )
    require(FULL_COLUMN_DIRECT_BLOCK_BITS == config["correctness"]["block_bits"],
            "direct-block source partition changed")


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
            config["correctness"]["uniform_initialization_seed"],
        ),
    }
    require(tuple(states) == STATE_NAMES, "direct-block state panel changed")
    for state in states.values():
        residual = (
            context["model"].H_check.astype(np.int64)
            @ state.astype(np.int64) % 2
        ).astype(np.uint8)
        require(np.array_equal(residual, context["syndrome"]),
                "direct-block preflight state left the hard coset")
    context.update({"config_sha": config_sha, "states": states})
    return context


if njit is not None:
    @njit(cache=True)
    def _fill_direct_weights_numba(mass, bases, odds_powers, popcount16, output):
        for candidate in range(output.size):
            popcount = (
                int(popcount16[candidate & 0xFFFF])
                + int(popcount16[candidate >> 16])
            )
            weight = odds_powers[popcount]
            for factor in range(bases.size):
                weight *= mass[candidate ^ int(bases[factor])]
            output[candidate] = weight
else:  # pragma: no cover
    _fill_direct_weights_numba = None


def _bases(context, state, column):
    H = context["H"]
    b_columns, a_syndromes, _ = _initial_collapsed_masks(
        state, context["syndrome"], H,
    )
    old = np.uint32(b_columns[int(column)])
    neighbors = np.flatnonzero(H[int(column)]).astype(np.int32)
    bases = np.ascontiguousarray(a_syndromes[neighbors] ^ old, dtype=np.uint32)
    return b_columns, a_syndromes, bases


def _sha_float_array(value):
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float64).tobytes()
    ).hexdigest()


def full_weight_checks(context, config, mass):
    require(_fill_direct_weights_numba is not None, "Numba is unavailable")
    H = context["H"]
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    legacy_cache = build_full_column_candidate_cache(H.shape[0], 0.04)
    legacy_workspace = build_full_column_workspace(legacy_cache)
    direct_cache = build_full_column_direct_block_cache(H.shape[0], 0.04, mass)
    direct_workspace = build_full_column_direct_block_workspace(direct_cache)
    difference = np.empty_like(legacy_workspace.log_weights)
    probes = []
    gates = config["correctness"]
    all_pass = True
    for state_name in gates["states"]:
        state = context["states"][state_name]
        for column in gates["columns"]:
            b_columns, a_syndromes, bases = _bases(context, state, column)
            subtotals, block_total, log_lower = full_column_direct_block_subtotals(
                H, context["syndrome"].reshape(H.shape), b_columns,
                a_syndromes, column, mass, cache=direct_cache,
                workspace=direct_workspace, engine="numba",
            )
            _column_log_weights_unchecked(
                H, b_columns, a_syndromes, column, log_mass,
                legacy_cache, legacy_workspace,
            )
            _fill_direct_weights_numba(
                mass, bases, direct_cache.odds_powers,
                direct_cache.popcount16, legacy_workspace.scratch,
            )
            direct_min = float(np.min(legacy_workspace.scratch))
            direct_max = float(np.max(legacy_workspace.scratch))
            normal = (
                np.all(np.isfinite(legacy_workspace.scratch))
                and direct_min >= float(np.finfo(np.float64).tiny)
                and np.all(np.isfinite(subtotals))
                and np.all(subtotals >= float(np.finfo(np.float64).tiny))
            )
            maximum = float(np.max(legacy_workspace.log_weights))
            np.subtract(
                legacy_workspace.log_weights, maximum,
                out=legacy_workspace.log_weights,
            )
            np.exp(legacy_workspace.log_weights, out=legacy_workspace.log_weights)
            np.divide(
                legacy_workspace.scratch, direct_max,
                out=legacy_workspace.scratch,
            )
            np.subtract(
                legacy_workspace.scratch, legacy_workspace.log_weights,
                out=difference,
            )
            np.abs(difference, out=difference)
            max_absolute = float(np.max(difference))
            np.divide(
                difference, legacy_workspace.log_weights, out=difference,
            )
            max_relative = float(np.max(difference))
            direct_sum = float(np.sum(legacy_workspace.scratch, dtype=np.float64))
            legacy_sum = float(np.sum(legacy_workspace.log_weights, dtype=np.float64))
            np.divide(
                legacy_workspace.scratch, direct_sum,
                out=legacy_workspace.scratch,
            )
            np.divide(
                legacy_workspace.log_weights, legacy_sum,
                out=legacy_workspace.log_weights,
            )
            np.subtract(
                legacy_workspace.scratch, legacy_workspace.log_weights,
                out=difference,
            )
            np.abs(difference, out=difference)
            total_variation = 0.5 * float(np.sum(difference, dtype=np.float64))
            margin_log = (
                math.log(float(np.finfo(np.float64).tiny))
                + math.log(float(gates["minimum_normal_margin_factor"]))
            )
            passed = bool(
                normal
                and math.isfinite(block_total) and block_total > 0.0
                and log_lower > margin_log
                and max_absolute <= gates["max_scaled_weight_absolute_error"]
                and max_relative <= gates["max_relative_weight_error"]
                and total_variation <= gates["max_total_variation"]
            )
            all_pass = all_pass and passed
            probes.append({
                "block_subtotals_sha256": _sha_float_array(subtotals),
                "block_total": block_total,
                "column": int(column),
                "direct_max_weight": direct_max,
                "direct_min_weight": direct_min,
                "log_candidate_weight_lower_bound": log_lower,
                "max_relative_weight_error": max_relative,
                "max_scaled_weight_absolute_error": max_absolute,
                "normal_positive_weights": bool(normal),
                "passed": passed,
                "state": state_name,
                "total_variation": total_variation,
            })
    return {"all_pass": bool(all_pass), "probes": probes}


class _FixedRng:
    def __init__(self, value):
        self.value = float(value)

    def random(self):
        return self.value


def conditional_timing(context, config, mass):
    H = context["H"]
    state = context["states"]["P"]
    column = 11
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    stream_cache = build_full_column_streaming_cache(H.shape[0], 0.04)
    stream_workspace = build_full_column_streaming_workspace(stream_cache)
    direct_cache = build_full_column_direct_block_cache(H.shape[0], 0.04, mass)
    direct_workspace = build_full_column_direct_block_workspace(direct_cache)

    def timed(engine):
        b_columns, a_syndromes, _ = _initial_collapsed_masks(
            state, context["syndrome"], H,
        )
        start = time.perf_counter()
        if engine == "streaming":
            full_column_streaming_gibbs_update(
                b_columns, a_syndromes, H, context["syndrome"].reshape(H.shape),
                column, log_mass, stream_cache, stream_workspace,
                _FixedRng(0.381), engine="numba",
            )
        else:
            full_column_direct_block_gibbs_update(
                b_columns, a_syndromes, H, context["syndrome"].reshape(H.shape),
                column, mass, direct_cache, direct_workspace,
                _FixedRng(0.381), engine="numba",
            )
        return time.perf_counter() - start

    # Compile both paths before any measurement.
    timed("streaming")
    timed("direct")
    stream_seconds = []
    direct_seconds = []
    for _ in range(3):
        stream_seconds.append(timed("streaming"))
        direct_seconds.append(timed("direct"))
    stream_median = statistics.median(stream_seconds)
    direct_median = statistics.median(direct_seconds)
    return {
        "direct_seconds": direct_seconds,
        "direct_seconds_median": direct_median,
        "speedup_over_streaming": stream_median / direct_median,
        "streaming_seconds": stream_seconds,
        "streaming_seconds_median": stream_median,
    }


def run_focused_tests():
    command = (
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "data/expander_code/exp102/tests/test_q0_hgp_full_column_gibbs.py",
        "data/expander_code/exp102/tests/test_q0_hgp_random_full_column.py",
    )
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    )
    transcript = completed.stdout + completed.stderr
    require("failed" not in transcript.lower(), "focused tests reported a failure")
    return {
        "command": list(command[1:]),
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
    }


_RUNTIME_CONTEXT = None


def runtime_worker(index):
    context = _RUNTIME_CONTEXT
    config = context["preflight_config"]
    resource = config["resource"]
    state_name = STATE_NAMES[int(index)]
    sampler = RandomFullColumnDirectBlockConfig(
        p=0.04,
        burn_updates=resource["runtime_probe_burn_updates"],
        measurement_updates=resource["runtime_probe_measurement_updates"],
    )
    prefix = (
        config["seed_namespace"], config["runtime_seed_key"], "runtime",
        state_name, int(index),
    )
    cache = build_full_column_direct_block_cache(
        context["H"].shape[0], 0.04, context["mass"],
    )
    workspace = build_full_column_direct_block_workspace(cache)
    arguments = (
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, context["states"][state_name],
        derive_seed(*prefix, "burn"), derive_seed(*prefix, "measurement"),
        derive_seed(*prefix, "observation"),
    )
    start = time.perf_counter()
    raw = run_random_full_column_direct_block_trajectory(
        *arguments, mass=context["mass"], cache=cache, workspace=workspace,
    )
    sampling_seconds = time.perf_counter() - start
    start = time.perf_counter()
    replay_random_full_column_direct_block_trajectory(
        *arguments, raw, mass=context["mass"], cache=cache, workspace=workspace,
    )
    replay_seconds = time.perf_counter() - start
    updates = sampler.burn_updates + sampler.measurement_updates
    rate = (sampling_seconds + replay_seconds) / updates
    projected = (
        rate
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
        "replay_inclusive_seconds_per_update": rate,
        "replay_seconds": replay_seconds,
        "sampling_seconds": sampling_seconds,
        "state": state_name,
        "transcript_sha256": transcript.hexdigest(),
        "updates": updates,
    }


def source_identity(source_commit, config_sha):
    files = {
        "collapsed": EXP102_ROOT / "exp102_pipeline/q0_hgp_collapsed.py",
        "config": CONFIG_PATH,
        "full_column": EXP102_ROOT / "exp102_pipeline/q0_hgp_full_column_gibbs.py",
        "global": EXP102_ROOT / "exp102_pipeline/q0_global.py",
        "random_full_column": EXP102_ROOT / "exp102_pipeline/q0_hgp_random_full_column.py",
        "review": REVIEW_PATH,
        "runner": Path(__file__).resolve(),
        "test_full_column": EXP102_ROOT / "tests/test_q0_hgp_full_column_gibbs.py",
        "test_random_full_column": EXP102_ROOT / "tests/test_q0_hgp_random_full_column.py",
        "workflow_052": CONTROL_ROOT / "workflow.py",
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
    require(not args.output.exists(), "immutable direct-block preflight output exists")
    verify_clean_source(args.source_commit)
    config = load_canonical(CONFIG_PATH)
    validate_config(config)
    config_sha = sha256_file(CONFIG_PATH)
    tests = run_focused_tests()
    context = load_context(config, config_sha)
    mass = build_classical_coset_mass(context["H"], 0.04, engine="numba")
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    correctness = full_weight_checks(context, config, mass)
    timing = conditional_timing(context, config, mass)

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
    exact_pass = correctness["all_pass"]
    speed_pass = (
        timing["speedup_over_streaming"]
        >= resource["local_min_streaming_speedup"]
    )
    runtime_pass = all(
        row["projected_replay_inclusive_t1_seconds"]
        <= resource["trajectory_wall_cap_seconds"]
        for row in runtime
    )
    if not exact_pass:
        status = "CONFLICT"
    elif not runtime_pass or (args.node == "macmini" and not speed_pass):
        status = "RUNTIME_EXHAUSTED"
    else:
        status = (
            "DIRECT_BLOCK_PREFLIGHT_LOCAL_PASS"
            if args.node == "macmini" else "DIRECT_BLOCK_PREFLIGHT_NODE_PASS"
        )
    core = {
        "checks": {
            "full_m8_weight_identity": exact_pass,
            "local_minimum_streaming_speedup": speed_pass,
            "runtime_projection": runtime_pass,
        },
        "config_sha256": config_sha,
        "correctness": correctness,
        "focused_tests": tests,
        "node": args.node,
        "report_version": REPORT_VERSION,
        "runtime": runtime,
        "source_identity": source_identity(args.source_commit, config_sha),
        "status": status,
        "timing": timing,
        "worst_projected_replay_inclusive_t1_seconds": max(
            row["projected_replay_inclusive_t1_seconds"] for row in runtime
        ),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
