"""Frozen runtime benchmark and full-schedule budget projection for PA."""

import argparse
import json
import platform
from pathlib import Path
import socket
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_json,
    verify_source_identity,
)
from data.expander_code.exp102.exp102_pipeline.pa_discovery import (
    PA_NODE_CAPACITY,
    _uniform_seed,
    load_pa_discovery_config,
)
from data.expander_code.exp102.exp102_pipeline.q0_pa import (
    PaSeedIdentity,
    Q0PaConfig,
    canonical_population_digest,
    run_q0_pa_population,
    theta_schedule_q32,
)
from data.expander_code.exp102.exp102_pipeline.q0_pt import ladder_x_q32_sha256
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _config(p, particles, steps, kernel):
    schedule = theta_schedule_q32(p, steps)
    return Q0PaConfig(
        p, particles, steps, 1, kernel, schedule,
        ladder_x_q32_sha256(schedule),
    )


def _seed(source_commit, tag, population):
    return PaSeedIdentity(
        source_commit=source_commit,
        config_sha256=sha256_json({"benchmark": tag}),
        cell_fingerprint=sha256_json({"cell": tag}),
        population_index=population,
        trajectory_namespace="q0_pa_runtime_benchmark_v1",
    )


def _run_pair(model, frame, syndrome, p, source_commit, tag, particles, steps, kernel):
    config = _config(p, particles, steps, kernel)
    start = time.perf_counter()
    core_start = time.process_time()
    digests = []
    for population in range(2):
        result = run_q0_pa_population(
            model, frame, syndrome, config,
            _seed(source_commit, tag, population), engine="numba",
        )
        digests.append(canonical_population_digest(result))
    return {
        "wall_seconds": time.perf_counter() - start,
        "core_seconds": time.process_time() - core_start,
        "particle_sweeps": 2 * particles * steps,
        "digests": digests,
    }


def run_benchmark(registry_path, config_path, source_commit, *, verified_source=False):
    source_identity = (
        verify_source_identity(Path.cwd(), source_commit)
        if verified_source else None
    )
    registry = load_registry(registry_path)
    protocol = load_pa_discovery_config(config_path, registry)
    rows = []
    startup_seconds = 0.0
    slopes = {}
    for code_id in ("m06_c00", "m08_c06"):
        _, code, H = load_frozen_code(registry_path, code_id)
        model, frame = build_model(H)
        cell = next(value for value in protocol["hard_screen"]["cells"]
                    if value["code_id"] == code_id)
        uniform_seed = _uniform_seed(registry, code, cell)
        uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        epsilon = (uniforms < cell["p"]).astype(np.uint8)
        syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
        for kernel in ("coordinate", "block4"):
            warm_config = _config(cell["p"], 8, 2, kernel)
            warm_start = time.perf_counter()
            run_q0_pa_population(
                model, frame, syndrome, warm_config,
                _seed(source_commit, f"warm-{code_id}-{kernel}", 0), engine="numba",
            )
            startup_seconds = max(startup_seconds, time.perf_counter() - warm_start)
            small = _run_pair(
                model, frame, syndrome, cell["p"], source_commit,
                f"small-{code_id}-{kernel}", 128, 96, kernel,
            )
            large = _run_pair(
                model, frame, syndrome, cell["p"], source_commit,
                f"large-{code_id}-{kernel}", 256, 192, kernel,
            )
            work_delta = large["particle_sweeps"] - small["particle_sweeps"]
            slope = max(
                0.0, (large["core_seconds"] - small["core_seconds"]) / work_delta,
            )
            if slope == 0.0:
                slope = large["core_seconds"] / large["particle_sweeps"]
            slopes[(int(code["m"]), kernel)] = slope
            rows.append({
                "code_id": code_id, "m": int(code["m"]), "kernel": kernel,
                "warmup_seconds": startup_seconds, "small": small, "large": large,
                "differential_seconds_per_particle_sweep": slope,
                "differential_us_per_particle_sweep": 1e6 * slope,
            })

    conservative = {
        kernel: max(slopes[(m, kernel)] for m in (6, 8))
        for kernel in ("coordinate", "block4")
    }
    methods = [*protocol["base_methods"], protocol["rescue_method"]]

    def task_seconds(method, particles=None):
        count = method["num_particles"] if particles is None else particles
        return conservative[method["logical_kernel"]] * count * method["num_anneal_steps"] * method["rejuvenation_sweeps"]

    base_seconds = sum(16 * task_seconds(method) for method in protocol["base_methods"])
    rescue_seconds = 16 * task_seconds(protocol["rescue_method"])
    confirmation_rank = sorted(
        methods, key=lambda method: task_seconds(method, 512), reverse=True,
    )[:2]
    confirmation_seconds = sum(136 * task_seconds(method, 512) for method in confirmation_rank)
    resolution_seconds = sum(48 * task_seconds(method, 256) for method in confirmation_rank)
    total_core_seconds = base_seconds + rescue_seconds + confirmation_seconds + resolution_seconds
    # The frozen contingency excludes a busy nd-1, so the budget must still
    # pass on the predeclared nd-2/nd-3 ownership rather than assuming all nodes.
    projection_capacity = PA_NODE_CAPACITY["nd-2"] + PA_NODE_CAPACITY["nd-3"]
    projected_minutes_with_safety = (
        2.0 * total_core_seconds / projection_capacity / 60.0
    )
    max_population_seconds = max(
        [task_seconds(method) for method in protocol["base_methods"]]
        + [task_seconds(protocol["rescue_method"])]
        + [task_seconds(method, 512) for method in methods]
        + [task_seconds(method, 256) for method in methods]
    )
    gate = protocol["runtime_gate"]
    checks = {
        "m8_slowest_kernel_us": max(
            1e6 * slopes[(8, kernel)] for kernel in ("coordinate", "block4")
        ) <= gate["max_m8_particle_sweep_us"],
        "startup_seconds": startup_seconds <= gate["max_startup_seconds"],
        "max_population_minutes": max_population_seconds / 60.0 <= gate["max_population_minutes"],
        "full_schedule_minutes_with_safety_factor_2": (
            projected_minutes_with_safety
            <= gate["max_schedule_minutes_with_safety_factor_2"]
        ),
    }
    return {
        "benchmark_version": "exp102.q0_pa.runtime.v1",
        "source_commit": source_commit,
        "source_identity": source_identity,
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": protocol["discovery_config_sha256"],
        "rows": rows,
        "conservative_seconds_per_particle_sweep": conservative,
        "startup_seconds": startup_seconds,
        "max_population_minutes": max_population_seconds / 60.0,
        "projected_core_seconds": total_core_seconds,
        "projection_nodes": ["nd-2", "nd-3"],
        "projection_capacity": projection_capacity,
        "projected_minutes_with_safety_factor_2": projected_minutes_with_safety,
        "projected_confirmation_methods": [
            method["method_id"] for method in confirmation_rank
        ],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "RUNTIME_BUDGET_EXCEEDED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("config")
    parser.add_argument("source_commit")
    parser.add_argument("output")
    parser.add_argument("--require-verified-source", action="store_true")
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.registry, args.config, args.source_commit,
        verified_source=args.require_verified_source,
    )
    atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "checks": result["checks"],
        "projected_minutes": result["projected_minutes_with_safety_factor_2"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
