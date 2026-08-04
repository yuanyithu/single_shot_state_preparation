"""Outcome-blind runtime preflight for the reduced-coordinate Houdayer pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer import coordinates_to_state
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    HOUDAYER_PAIR_KERNEL,
    build_reduced_houdayer_pair_kernel,
    run_houdayer_pair_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PREFLIGHT_VERSION = "exp102.q0_houdayer_pair.runtime_preflight.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"


class HoudayerRuntimePreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise HoudayerRuntimePreflightError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise HoudayerRuntimePreflightError("Houdayer runtime config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "Houdayer runtime config is not canonical")
    expected = {
        "cell", "config_version", "contract_version", "kernel", "registry_sha256",
        "resource", "scope", "timing_initial_pair", "version",
    }
    _require(set(config) == expected and config["version"] == PREFLIGHT_VERSION
             and config["contract_version"] == PREFLIGHT_VERSION
             and config["config_version"]
             == "exp102.q0_houdayer_pair.runtime_preflight.config.v0",
             "Houdayer runtime config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "Houdayer runtime cell changed")
    _require(config["kernel"] == {
        "coordinate_basis": "h_x_plus_canonical_reduced_logical_complement.v0",
        "local_kernel": "independent_random_scan_coordinate_heatbath.v0",
        "local_updates_per_clock": 832,
        "pair_kernel": "complete_component_houdayer_swap.v0",
    }, "Houdayer runtime kernel changed")
    _require(config["resource"] == {
        "burn_clocks": 128,
        "measurement_clocks": 1024,
        "per_trajectory_cap_seconds": 7200,
        "projection_safety_factor": 2.0,
        "timing_clocks": 16,
    }, "Houdayer runtime resource tier changed")
    _require(config["timing_initial_pair"]
             == "section_zero_coordinates_vs_all_one_coordinates",
             "Houdayer runtime timing pair changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "Houdayer runtime registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "outcome_blind_houdayer_pair_runtime_preflight_only",
        "remote_authorization": False,
    }, "Houdayer runtime scope changed")
    return config, sha256_file(path)


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "config": sha256_file(config_path),
        "houdayer": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_houdayer.py"),
        "pair_kernel": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_houdayer_pair.py"),
        "probe": sha256_file(Path(__file__)),
        "registry": sha256_file(REGISTRY_PATH),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def run_preflight(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "Houdayer runtime registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    _uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, config["cell"])
    kernel = build_reduced_houdayer_pair_kernel(H, model, frame, syndrome, config["cell"]["p"])
    _require(kernel.coordinate_count == config["kernel"]["local_updates_per_clock"],
             "Houdayer runtime coordinate count changed")
    all_one_coordinates = np.ones(kernel.coordinate_count, dtype=np.uint8)
    right = coordinates_to_state(kernel.base_state, kernel.generators, all_one_coordinates)
    start = time.perf_counter()
    trace = run_houdayer_pair_trajectory(
        kernel, kernel.base_state, right, 0x4A564355, 0,
        config["resource"]["timing_clocks"], config["kernel"]["local_updates_per_clock"],
    )
    elapsed = time.perf_counter() - start
    per_clock = elapsed / config["resource"]["timing_clocks"]
    projected = (
        per_clock
        * (config["resource"]["burn_clocks"] + config["resource"]["measurement_clocks"])
        * config["resource"]["projection_safety_factor"]
    )
    runtime_pass = projected <= config["resource"]["per_trajectory_cap_seconds"]
    return {
        "status": "RUNTIME_PASS" if runtime_pass else "RUNTIME_EXHAUSTED",
        "cell": config["cell"],
        "registry_sha256": registry["registry_sha256"],
        "kernel": HOUDAYER_PAIR_KERNEL,
        "dimensions": {
            "physical_qubits": kernel.num_qubits,
            "coordinate_count": kernel.coordinate_count,
            "logical_coordinate_count": kernel.logical_count,
            "stabilizer_coordinate_count": kernel.stabilizer_count,
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "per_clock_seconds": per_clock,
            "timing_clocks": config["resource"]["timing_clocks"],
            "projected_seconds_with_safety": projected,
            "cap_seconds": config["resource"]["per_trajectory_cap_seconds"],
            "safety_factor": config["resource"]["projection_safety_factor"],
        },
        "outcome_blind_counters": trace["counters"],
        "does_not_establish": [
            "A posterior, q_top, logical-sector probability, or convergence claim.",
            "A favorable energy, weight, label, character, or sampler outcome.",
            "Any remote, formal, held-out, or production authorization.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime_preflight.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace Houdayer runtime report: {args.output}")
    config, config_sha256 = _load_config(args.config)
    core = {
        "preflight_version": PREFLIGHT_VERSION,
        "config_sha256": config_sha256,
        "scope": config["scope"],
        "source_binding": _source_binding(args.config),
        "preflight": run_preflight(config),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
