"""Runtime-only probe for uniform-anchored collapsed replica exchange.

This script deliberately records no posterior estimator, labels, or raw state.
It warms the Numba path and measures fixed, predeclared tiny clocks solely to
freeze a local-screen resource tier before any scientific UARE raw exists.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_hgp_uniform_anchor_pt import (
    UNIFORM_ANCHOR_PT_KERNEL,
    UniformAnchorReplicaExchangeConfig,
    UniformAnchorReplicaExchangeSeedIdentity,
    run_uniform_anchor_replica_exchange_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
CELL = {
    "code_id": "m08_c06",
    "disorder_index": 0,
    "disorder_source": "attempt022",
    "p": 0.04,
}


def _source_binding():
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    source_paths = {
        "q0_hgp_collapsed.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        "q0_hgp_full_row_gibbs.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
        "q0_hgp_uniform_anchor_pt.py": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py",
    }
    files = {name: sha256_file(path) for name, path in source_paths.items()}
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _epsilon_and_syndrome(registry, code, model):
    seed = derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22", registry["registry_sha256"],
        code["code_id"], int(CELL["disorder_index"]), "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(seed)).random(model.num_qubits)
        < CELL["p"]
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    if not syndrome.any():
        raise RuntimeError("fixed hard-sentinel syndrome unexpectedly vanished")
    return epsilon, syndrome, int(seed)


def _run(config, binding, registry, H, model, frame, epsilon, syndrome, index):
    config_sha = sha256_json(config.as_dict())
    identity = UniformAnchorReplicaExchangeSeedIdentity(
        source_commit=binding["source_commit"], config_sha256=config_sha,
        registry_sha256=registry["registry_sha256"], cell_fingerprint=sha256_json(CELL),
        method_id=config.method_id, resource_tier="runtime_probe", init_family="P",
        trajectory_index=index,
        trajectory_namespace="exp102.q0_hgp_uniform_anchor_pt.v0.runtime_probe",
    )
    started = time.perf_counter()
    raw = run_uniform_anchor_replica_exchange_trajectory(
        model, frame, H, syndrome, config, identity, epsilon, engine="numba",
    )
    elapsed = time.perf_counter() - started
    if raw["measurement_residual_weights"].any():
        raise RuntimeError("runtime probe left the hard coset")
    return {
        "method_id": config.method_id,
        "num_replicas": config.num_replicas,
        "positive_row_updates_per_round": config.positive_row_updates_per_round,
        "rounds": config.burn_rounds + config.measurement_rounds,
        "wall_seconds": elapsed,
        "seconds_per_round": elapsed / (config.burn_rounds + config.measurement_rounds),
        "sampler_config_sha256": config_sha,
        "lambda_sha256": config.lambda_sha256,
        "hard_coset_verified": True,
        "raw_version": raw["raw_version"],
    }


def main(output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing runtime evidence: {output_path}")
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, CELL["code_id"])
    model, frame = build_model(H)
    epsilon, syndrome, uniform_seed = _epsilon_and_syndrome(registry, code, model)
    binding = _source_binding()
    # This first run is deliberately excluded from timing because it pays JIT cost.
    warmup = UniformAnchorReplicaExchangeConfig(
        p=CELL["p"], burn_rounds=1, measurement_rounds=8, num_replicas=4,
    )
    _run(warmup, binding, registry, H, model, frame, epsilon, syndrome, 0)
    candidates = [
        UniformAnchorReplicaExchangeConfig(
            p=CELL["p"], burn_rounds=2, measurement_rounds=8, num_replicas=replicas,
        )
        for replicas in (8, 16, 32, 64)
    ]
    timings = [
        _run(config, binding, registry, H, model, frame, epsilon, syndrome, index + 1)
        for index, config in enumerate(candidates)
    ]
    core = {
        "probe_version": "exp102.q0_hgp_uniform_anchor_pt.runtime_probe.v0",
        "purpose": "runtime_only_no_raw_no_estimator_no_method_selection_by_physics",
        "cell": CELL,
        "kernel": UNIFORM_ANCHOR_PT_KERNEL,
        "registry_sha256": registry["registry_sha256"],
        "source_binding": binding,
        "uniform_seed": uniform_seed,
        "warmup_excluded_from_timing": warmup.as_dict(),
        "timings": timings,
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(output_path, report)
    print(sha256_json(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runtime_probe.json")
    arguments = parser.parse_args()
    main(arguments.output)
