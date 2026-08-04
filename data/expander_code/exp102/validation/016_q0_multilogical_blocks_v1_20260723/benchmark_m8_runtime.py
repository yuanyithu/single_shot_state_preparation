"""Measure the fixed-cost MLB8-J16 hot path on the hard m8 sentinel.

This is a runtime feasibility diagnostic only.  It records a short warmed
Numba trace and extrapolates linearly to the pre-registered discovery tiers;
it neither estimates a posterior observable nor selects a physics result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

# Running this file by path puts its validation directory on sys.path, not the
# project root that owns the namespace-package imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from data.expander_code.exp102.exp102_pipeline.q0_global import GlobalSeedIdentity
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified_v0 import (
    _context,
    load_v0_config,
)
from data.expander_code.exp102.exp102_pipeline.q0_multilogical_blocks import (
    MultiLogicalBlockConfig,
    build_multilogical_empty_catalog,
    build_multilogical_blocks,
    run_multilogical_block_trajectory,
)


ROOT = Path("data/expander_code/exp102")
REGISTRY = ROOT / "registry/registry.json"
V0_CONFIG = ROOT / "config/q0_logical_stratified.v0.v2.json"


def _seed(namespace: str) -> GlobalSeedIdentity:
    return GlobalSeedIdentity(
        source_commit="0" * 40,
        config_sha256="1" * 64,
        registry_sha256="2" * 64,
        cell_fingerprint="3" * 64,
        method_id="MLB8-J16",
        resource_tier="runtime_diagnostic",
        init_family="P",
        trajectory_index=0,
        trajectory_namespace=namespace,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-sweeps", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("m8_runtime.json"),
    )
    args = parser.parse_args()
    if args.measurement_sweeps <= 0 or args.measurement_sweeps % 8:
        raise SystemExit("measurement sweeps must be a positive multiple of eight")
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")

    v0_config = load_v0_config(V0_CONFIG)
    _, code, _, model, frame, _, epsilon, syndrome = _context(REGISTRY, v0_config)
    if code["code_id"] != "m08_c06" or float(v0_config["cell"]["p"]) != 0.04:
        raise RuntimeError("runtime diagnostic lost its frozen hard sentinel")

    started = time.perf_counter()
    catalog = build_multilogical_empty_catalog(model, frame)
    empty_catalog_seconds = time.perf_counter() - started
    started = time.perf_counter()
    blocks = build_multilogical_blocks(model, frame)
    block_seconds = time.perf_counter() - started

    # Compile and execute the exact same kernel once before measuring its slope.
    warm = MultiLogicalBlockConfig(p=0.04, burn_sweeps=1, measurement_sweeps=8)
    run_multilogical_block_trajectory(
        model, frame, syndrome, warm, _seed("mlb8_runtime_warm"), epsilon,
        engine="numba", catalog=catalog, blocks=blocks,
    )

    config = MultiLogicalBlockConfig(
        p=0.04, burn_sweeps=64, measurement_sweeps=args.measurement_sweeps,
    )
    trace_seconds_per_repeat = []
    raw = None
    for repeat in range(args.repeats):
        started = time.perf_counter()
        raw = run_multilogical_block_trajectory(
            model, frame, syndrome, config,
            _seed(f"mlb8_runtime_measure_{repeat}"), epsilon,
            engine="numba", catalog=catalog, blocks=blocks,
        )
        trace_seconds_per_repeat.append(time.perf_counter() - started)
    assert raw is not None
    trace_sweeps = config.burn_sweeps + config.measurement_sweeps
    trace_seconds = statistics.median(trace_seconds_per_repeat)
    seconds_per_sweep = trace_seconds / trace_sweeps
    payload = {
        "contract": "exp102.q0_multilogical_blocks.runtime_diagnostic.v1",
        "purpose": "runtime_only_not_sampling_evidence",
        "code_id": code["code_id"],
        "p": config.p,
        "empty_catalog_seconds": empty_catalog_seconds,
        "block_seconds": block_seconds,
        "warm_sweeps": warm.burn_sweeps + warm.measurement_sweeps,
        "trace_sweeps": trace_sweeps,
        "trace_seconds_median": trace_seconds,
        "trace_seconds_per_repeat": trace_seconds_per_repeat,
        "seconds_per_sweep": seconds_per_sweep,
        "projected_t1_seconds": seconds_per_sweep * (2048 + 8192),
        "projected_t3_seconds": seconds_per_sweep * (8192 + 32768),
        "joint_attempts": int(raw["measurement_counters"][8]),
        "catalog_attempts": int(raw["measurement_counters"][2]),
        "catalog_changes": int(raw["measurement_counters"][3]),
        "measurement_count": int(raw["measurement_labels"].size),
        "hard_coset_residual_zero": bool(not raw["measurement_residual_weights"].any()),
        "joint_sha256": str(raw["joint_sha256"]),
        "catalog_sha256": str(raw["catalog_sha256"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
