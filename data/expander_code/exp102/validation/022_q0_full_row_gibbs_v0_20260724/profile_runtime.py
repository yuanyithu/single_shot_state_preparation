"""Runtime-only profile for the new exact full-row collapsed-HGP Gibbs kernel.

This is not sampler raw and does not calculate or gate q_top.  It measures a
fixed number of local m08 hard-cell sweeps after a separate JIT warm-up, so a
later local diagnostic can choose its clock from runtime rather than outcome.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    FULL_ROW_GIBBS_VERSION,
    FullRowGibbsConfig,
    FullRowGibbsSeedIdentity,
    run_full_row_gibbs_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROFILE_VERSION = "exp102.q0_hgp_full_row_gibbs.runtime_profile.v0"
CELL = {
    "code_id": "m08_c06",
    "disorder_index": 0,
    "disorder_source": "attempt022",
    "p": 0.04,
}
ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT.parents[1] / "registry" / "registry.json"


def _source_commit():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _hard_context():
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, CELL["code_id"])
    model, frame = build_model(H)
    uniform_seed = derive_seed(
        "pilot_ladder_m8_attempt22", registry["registry_sha256"],
        code["code_id"], CELL["disorder_index"], "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < CELL["p"]
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    if not syndrome.any():
        raise RuntimeError("runtime profile hard syndrome unexpectedly vanished")
    return registry, code, H, model, frame, uniform_seed, epsilon, syndrome


def _identity(registry, config, namespace):
    return FullRowGibbsSeedIdentity(
        source_commit=_source_commit(),
        config_sha256=sha256_json(config.as_dict()),
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(CELL),
        method_id=config.method_id,
        resource_tier="RUNTIME_PROFILE",
        init_family="P",
        trajectory_index=0,
        trajectory_namespace=namespace,
    )


def profile(measurement_sweeps):
    measurement_sweeps = int(measurement_sweeps)
    if measurement_sweeps <= 0 or measurement_sweeps % 8:
        raise ValueError("measurement_sweeps must be a positive multiple of eight")
    registry, code, H, model, frame, uniform_seed, epsilon, syndrome = _hard_context()
    warmup = FullRowGibbsConfig(CELL["p"], 1, 8)
    run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, warmup,
        _identity(registry, warmup, "q0_full_row_runtime_warmup_v0"), epsilon,
        engine="numba",
    )
    timed = FullRowGibbsConfig(CELL["p"], 1, measurement_sweeps)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    result = run_full_row_gibbs_trajectory(
        model, frame, H, syndrome, timed,
        _identity(registry, timed, "q0_full_row_runtime_timed_v0"), epsilon,
        engine="numba",
    )
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    sweeps = timed.burn_sweeps + timed.measurement_sweeps
    return {
        "profile_version": PROFILE_VERSION,
        "sampler_version": FULL_ROW_GIBBS_VERSION,
        "source_commit": _source_commit(),
        "source_files": {
            "q0_hgp_full_row_gibbs.py": sha256_file(
                ROOT.parents[1] / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
            ),
            "profile_runtime.py": sha256_file(__file__),
        },
        "cell": CELL,
        "registry_sha256": registry["registry_sha256"],
        "code_npz_sha256": code["code_npz_sha256"],
        "uniform_seed": int(uniform_seed),
        "syndrome_weight": int(syndrome.sum()),
        "initial_weight": int(epsilon.sum()),
        "plan_sha256": result["plan_sha256"],
        "plan_max_table_entries": 8192,
        "config": timed.as_dict(),
        "sweeps": sweeps,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "wall_seconds_per_sweep": wall_seconds / sweeps,
        "cpu_seconds_per_sweep": cpu_seconds / sweeps,
        "measurement_row_changes": int(result["measurement_counters"][1]),
        "measurement_row_changed_bits": int(result["measurement_counters"][2]),
        "measurement_label_count": int(np.unique(result["measurement_labels"]).size),
        "raw_produced": False,
        "used_for_method_selection": "runtime_only",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-sweeps", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime_probe.json")
    args = parser.parse_args(argv)
    result = profile(args.measurement_sweeps)
    result["report_sha256"] = sha256_json(result)
    atomic_json(args.output, result)
    print(result["report_sha256"])


if __name__ == "__main__":
    main()
