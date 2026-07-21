"""Sampler-only Linux timing and worst-node diagnostic-screen projection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import socket
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import verify_source_identity
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    DefectTraceConfig,
    HardCosetConfig,
    build_joint_blocks,
    build_logical_proposal_catalog,
    run_defect_trace_trajectory,
    run_hardcoset_trajectory,
    tune_defect_bias,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model

from .common import (
    CONTRACT_VERSION,
    EXECUTION_NODES,
    EXPECTED_PREFLIGHT_NODES,
    RUNTIME_CONSENSUS_VERSION,
    RUNTIME_NODE_VERSION,
    SAFETY_FACTOR,
    SCREEN_WINDOW_SECONDS,
    TRAJECTORY_LIMIT_SECONDS,
    all_methods,
    atomic_json,
    config_sha256,
    defect_methods,
    hard_methods,
    load_config,
    load_registry,
    node_capacity,
    pipeline_attr,
    resource_tiers,
    sha256_file,
    sha256_json,
    uniform_seed_for_cell,
)


def _contains_physics_outcome(value):
    forbidden = {
        "q_top", "posterior_purity", "posterior_mass_on_planted_class",
        "map_success_probability", "d2_norm", "label", "labels", "weight",
        "weights", "characters", "measurement_weights", "physical_result",
        "sampler_pass", "sampler_fail", "sampling_pass", "sampling_fail",
    }
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or _contains_physics_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_physics_outcome(item) for item in value)
    return False


def _identity(source_commit, config, registry, cell, method, family,
              trajectory, namespace):
    identity_type = pipeline_attr("ScreenSeedIdentity")
    return identity_type(
        seed_root=config["seed_root"],
        source_commit=source_commit,
        config_sha256=config_sha256(config),
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(cell),
        method_id=method,
        resource_tier="RUNTIME",
        init_family=family,
        trajectory_index=trajectory,
        trajectory_namespace=namespace,
    )


def _timed_hard(model, frame, syndrome, epsilon, p, method, source_commit,
                config, registry, cell, catalog, joint):
    warm = HardCosetConfig(method, p, 2, 8)
    warm_identity = _identity(
        source_commit, config, registry, cell, method, "P", 0,
        "q0_global_screen_runtime_warm_v1",
    )
    started = time.perf_counter()
    run_hardcoset_trajectory(
        model, frame, syndrome, warm, warm_identity, epsilon,
        engine="numba", catalog=catalog, joint=joint,
    )
    warmup = time.perf_counter() - started

    timed = HardCosetConfig(method, p, 32, 128)
    timed_identity = _identity(
        source_commit, config, registry, cell, method, "P", 1,
        "q0_global_screen_runtime_timed_v1",
    )
    wall_start, core_start = time.perf_counter(), time.process_time()
    run_hardcoset_trajectory(
        model, frame, syndrome, timed, timed_identity, epsilon,
        engine="numba", catalog=catalog, joint=joint,
    )
    wall = time.perf_counter() - wall_start
    core = time.process_time() - core_start
    return {
        "warmup_seconds": warmup,
        "timed_sweeps": 160,
        "wall_seconds": wall,
        "core_seconds": core,
        "core_seconds_per_sweep": core / 160.0,
    }


def _timed_defect(model, frame, syndrome, epsilon, p, method,
                  source_commit, config, registry, cell):
    warm = DefectTraceConfig(method, p, 2, 8)
    warm_identity = _identity(
        source_commit, config, registry, cell, method, "P", 0,
        "q0_global_screen_runtime_warm_v1",
    )
    bias = np.zeros(warm.dmax + 1)
    started = time.perf_counter()
    run_defect_trace_trajectory(
        model, frame, syndrome, warm, warm_identity, epsilon, bias,
        "a" * 64, engine="numba",
    )
    warmup = time.perf_counter() - started

    timed = DefectTraceConfig(method, p, 32, 128)
    timed_identity = _identity(
        source_commit, config, registry, cell, method, "P", 1,
        "q0_global_screen_runtime_timed_v1",
    )
    wall_start, core_start = time.perf_counter(), time.process_time()
    run_defect_trace_trajectory(
        model, frame, syndrome, timed, timed_identity, epsilon, bias,
        "a" * 64, engine="numba",
    )
    wall = time.perf_counter() - wall_start
    core = time.process_time() - core_start

    tuning_identities = [
        _identity(
            source_commit, config, registry, cell, method, "TUNE", index,
            "q0_global_screen_runtime_bias_v1",
        )
        for index in range(8)
    ]
    tuning_wall_start = time.perf_counter()
    tune_defect_bias(
        model, syndrome, timed, tuning_identities, engine="numba",
    )
    tuning_wall = time.perf_counter() - tuning_wall_start
    return {
        "warmup_seconds": warmup,
        "timed_sweeps": 160,
        "wall_seconds": wall,
        "core_seconds": core,
        "core_seconds_per_sweep": core / 160.0,
        "bias_tuning_wall_seconds": tuning_wall,
    }


def _hard_cells(config):
    cells = config["panels"]["HARD2"]["cells"]
    if [cell["code_id"] for cell in cells] != ["m06_c00", "m08_c06"]:
        raise ValueError("diagnostic runtime requires the frozen HARD2 order")
    return cells


def _reconstruct_node_projection(rows):
    methods = all_methods()
    expected_coordinates = [
        (code_id, m, method)
        for code_id, m in (("m06_c00", 6), ("m08_c06", 8))
        for method in methods
    ]
    coordinates = [
        (row.get("code_id"), row.get("m"), row.get("method_id"))
        for row in rows
    ]
    if coordinates != expected_coordinates:
        raise ValueError("diagnostic runtime rows changed order or coverage")
    for row in rows:
        expected_fields = {
            "code_id", "m", "method_id", "catalog_seconds",
            "joint_build_seconds", "warmup_seconds", "timed_sweeps",
            "wall_seconds", "core_seconds", "core_seconds_per_sweep",
        }
        if row["method_id"] in defect_methods():
            expected_fields.add("bias_tuning_wall_seconds")
        if set(row) != expected_fields or row["timed_sweeps"] != 160:
            raise ValueError("diagnostic runtime row schema is noncanonical")
        numeric = expected_fields - {"code_id", "m", "method_id"}
        if any(not math.isfinite(float(row[key])) or float(row[key]) < 0.0
               for key in numeric):
            raise ValueError("diagnostic runtime row contains invalid timing")
        if not math.isclose(
                float(row["core_seconds_per_sweep"]),
                float(row["core_seconds"]) / 160.0,
                rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("diagnostic runtime per-sweep timing is inconsistent")
    m8 = {row["method_id"]: row for row in rows if row["m"] == 8}
    capacities = node_capacity()
    capacity = sum(capacities[node] for node in EXECUTION_NODES)
    projections = []
    eligible_by_tier = {}
    for tier, resources in resource_tiers().items():
        sweeps = int(resources["burn_sweeps"]) + int(
            resources["measurement_sweeps"]
        )
        trajectory_seconds = {}
        eligible = []
        for method in methods:
            row = m8[method]
            estimate = (
                float(row["catalog_seconds"])
                + float(row["joint_build_seconds"])
                + float(row["core_seconds_per_sweep"]) * sweeps
            )
            trajectory_seconds[method] = estimate
            if estimate <= TRAJECTORY_LIMIT_SECONDS:
                eligible.append(method)
        bias_seconds = {
            method: float(m8[method]["bias_tuning_wall_seconds"])
            for method in defect_methods()
        }
        measurement_core = sum(
            5 * 2 * 16 * trajectory_seconds[method] for method in methods
        )
        bias_core = sum(5 * bias_seconds[method] for method in defect_methods())
        # Bias is tuned once, then replay-validated once before measurement.
        core = measurement_core + 2.0 * bias_core
        wall = SAFETY_FACTOR * core / capacity
        passed = eligible == list(methods) and wall <= SCREEN_WINDOW_SECONDS
        projections.append({
            "resource_tier": tier,
            "trajectory_seconds_m8": trajectory_seconds,
            "bias_tuning_seconds_m8": bias_seconds,
            "eligible_methods": eligible,
            "projected_core_seconds": core,
            "projected_screen_wall_seconds": wall,
            "safety_factor": SAFETY_FACTOR,
            "execution_nodes": list(EXECUTION_NODES),
            "execution_capacity": capacity,
            "pass": bool(passed),
        })
        eligible_by_tier[tier] = eligible
    passing = [value for value in projections if value["pass"]]
    selected = passing[-1]["resource_tier"] if passing else None
    checks = {
        "all_methods_timed": set(m8) == set(methods),
        "all_numeric_finite": True,
        "at_least_T1_fits": selected is not None,
        "selected_tier_retains_all_methods": bool(
            selected and eligible_by_tier[selected] == list(methods)
        ),
    }
    return projections, selected, (
        [] if selected is None else eligible_by_tier[selected]
    ), checks


def run_benchmark(registry_path, config_path, source_commit, *,
                  verified_source=False, node=None):
    source_identity = (
        verify_source_identity(Path.cwd(), source_commit)
        if verified_source else None
    )
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    rows = []
    for cell in _hard_cells(config):
        _, code, H = load_frozen_code(registry_path, cell["code_id"])
        model, frame = build_model(H)
        uniform_seed = uniform_seed_for_cell(registry, code, cell)
        uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(
            model.num_qubits
        )
        epsilon = (uniforms < cell["p"]).astype(np.uint8)
        syndrome = (
            model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
        ).astype(np.uint8)

        catalog_start = time.perf_counter()
        catalog = build_logical_proposal_catalog(model, frame)
        catalog_seconds = time.perf_counter() - catalog_start
        for method in hard_methods():
            block_size = int(method[-2:]) if "-J" in method else 0
            joint_start = time.perf_counter()
            joint = (
                build_joint_blocks(model, frame, catalog, block_size)
                if block_size else None
            )
            joint_seconds = time.perf_counter() - joint_start
            timing = _timed_hard(
                model, frame, syndrome, epsilon, cell["p"], method,
                source_commit, config, registry, cell, catalog, joint,
            )
            rows.append({
                "code_id": cell["code_id"], "m": int(code["m"]),
                "method_id": method, "catalog_seconds": catalog_seconds,
                "joint_build_seconds": joint_seconds, **timing,
            })
        for method in defect_methods():
            timing = _timed_defect(
                model, frame, syndrome, epsilon, cell["p"], method,
                source_commit, config, registry, cell,
            )
            rows.append({
                "code_id": cell["code_id"], "m": int(code["m"]),
                "method_id": method, "catalog_seconds": 0.0,
                "joint_build_seconds": 0.0, **timing,
            })

    m8 = {row["method_id"]: row for row in rows if row["m"] == 8}
    methods = all_methods()
    capacities = node_capacity()
    execution_capacity = sum(capacities[node] for node in EXECUTION_NODES)
    projections = []
    eligible_by_tier = {}
    for tier, resources in resource_tiers(config).items():
        sweeps = int(resources["burn_sweeps"]) + int(
            resources["measurement_sweeps"]
        )
        trajectory_seconds = {}
        eligible = []
        for method in methods:
            row = m8[method]
            setup = row["catalog_seconds"] + row["joint_build_seconds"]
            estimate = setup + row["core_seconds_per_sweep"] * sweeps
            trajectory_seconds[method] = estimate
            if math.isfinite(estimate) and estimate <= TRAJECTORY_LIMIT_SECONDS:
                eligible.append(method)
        bias_seconds = {
            method: float(m8[method]["bias_tuning_wall_seconds"])
            for method in defect_methods()
        }
        measurement_core = sum(
            5 * 2 * 16 * trajectory_seconds[method] for method in methods
        )
        bias_core = sum(5 * bias_seconds[method] for method in defect_methods())
        projected_wall = SAFETY_FACTOR * (
            measurement_core + 2.0 * bias_core
        ) / execution_capacity
        passed = (
            eligible == list(methods)
            and math.isfinite(projected_wall)
            and projected_wall <= SCREEN_WINDOW_SECONDS
        )
        projections.append({
            "resource_tier": tier,
            "trajectory_seconds_m8": trajectory_seconds,
            "bias_tuning_seconds_m8": bias_seconds,
            "eligible_methods": eligible,
            "projected_core_seconds": measurement_core + 2.0 * bias_core,
            "projected_screen_wall_seconds": projected_wall,
            "safety_factor": SAFETY_FACTOR,
            "execution_nodes": list(EXECUTION_NODES),
            "execution_capacity": execution_capacity,
            "pass": bool(passed),
        })
        eligible_by_tier[tier] = eligible
    passing = [value for value in projections if value["pass"]]
    selected = passing[-1]["resource_tier"] if passing else None
    checks = {
        "all_methods_timed": set(m8) == set(methods),
        "all_numeric_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "warmup_seconds", "wall_seconds", "core_seconds",
                "core_seconds_per_sweep",
            )
        ),
        "at_least_T1_fits": selected is not None,
        "selected_tier_retains_all_methods": bool(
            selected and eligible_by_tier[selected] == list(methods)
        ),
    }
    status = "PASS" if all(checks.values()) else "RUNTIME_EXHAUSTED"
    return {
        "benchmark_version": RUNTIME_NODE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_commit": source_commit,
        "source_identity": source_identity,
        "registry_sha256": registry["registry_sha256"],
        "diagnostic_config_sha256": config_sha256(config),
        "node": str(node) if node is not None else socket.gethostname(),
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "completed_unix": time.time(),
        "rows": rows,
        "projections": projections,
        "selected_resource_tier": selected,
        "selected_eligible_methods": (
            [] if selected is None else eligible_by_tier[selected]
        ),
        "checks": checks,
        "status": status,
    }


def combine_runtime_reports(report_paths, output_path=None):
    if set(report_paths) != set(EXPECTED_PREFLIGHT_NODES):
        raise ValueError("diagnostic runtime consensus requires all three nodes")
    reports = {
        node: json.loads(Path(report_paths[node]).read_text(encoding="ascii"))
        for node in EXPECTED_PREFLIGHT_NODES
    }
    first = reports[EXPECTED_PREFLIGHT_NODES[0]]
    identity_fields = (
        "source_commit", "source_identity", "registry_sha256",
        "diagnostic_config_sha256",
    )
    expected_identity = {
        key: first.get(key) for key in identity_fields
    }
    check_names = {
        "all_methods_timed", "all_numeric_finite", "at_least_T1_fits",
        "selected_tier_retains_all_methods",
    }
    report_fields = {
        "benchmark_version", "contract_version", "source_commit",
        "source_identity", "registry_sha256", "diagnostic_config_sha256",
        "node", "environment", "completed_unix", "rows", "projections",
        "selected_resource_tier", "selected_eligible_methods", "checks",
        "status",
    }
    environment_fields = {"system", "machine", "hostname", "python", "numpy"}
    for node, report in reports.items():
        checks = report.get("checks")
        checks_valid = (
            isinstance(checks, dict) and set(checks) == check_names
            and all(isinstance(value, bool) for value in checks.values())
        )
        expected_status = (
            "PASS" if checks_valid and all(checks.values())
            else "RUNTIME_EXHAUSTED"
        )
        source_identity = report.get("source_identity")
        if (any(key.startswith("ti_") or "wmc" in key for key in report)
                or _contains_physics_outcome(report)):
            raise ValueError("diagnostic runtime report contains excluded work")
        rows = report.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"diagnostic runtime rows are missing: {node}"
            )
        (expected_projections, expected_selected, expected_eligible,
         expected_checks) = _reconstruct_node_projection(rows)
        if (set(report) != report_fields
                or set(report.get("environment", {})) != environment_fields
                or report.get("benchmark_version") != RUNTIME_NODE_VERSION
                or report.get("contract_version") != CONTRACT_VERSION
                or report.get("node") != node
                or report.get("environment", {}).get("system") != "Linux"
                or {key: report.get(key) for key in identity_fields}
                != expected_identity
                or not isinstance(source_identity, dict)
                or source_identity.get("mode") != "archive"
                or source_identity.get("source_commit")
                != report.get("source_commit")
                or not checks_valid
                or report.get("projections") != expected_projections
                or report.get("selected_resource_tier") != expected_selected
                or report.get("selected_eligible_methods") != expected_eligible
                or checks != expected_checks
                or report.get("status") != expected_status
                or not math.isfinite(float(
                    report.get("completed_unix", math.nan)
                ))):
            raise ValueError(
                f"diagnostic runtime report is not verified evidence: {node}"
            )

    tiers = list(resource_tiers().keys())
    methods = all_methods()
    projections = []
    common_by_tier = {}
    for tier in tiers:
        per_node = {}
        common = set(methods)
        trajectory_seconds = {}
        bias_seconds = {}
        for node, report in reports.items():
            matches = [
                value for value in report["projections"]
                if value.get("resource_tier") == tier
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"diagnostic runtime tier is malformed: {node}/{tier}"
                )
            value = matches[0]
            common &= set(value.get("eligible_methods", []))
            per_node[node] = {
                "projected_screen_wall_seconds": value.get(
                    "projected_screen_wall_seconds"
                ),
                "eligible_methods": value.get("eligible_methods"),
                "pass": value.get("pass"),
            }
        common_ordered = [method for method in methods if method in common]
        for method in methods:
            trajectory_seconds[method] = max(
                float(next(
                    value for value in reports[node]["projections"]
                    if value["resource_tier"] == tier
                )["trajectory_seconds_m8"][method])
                for node in EXPECTED_PREFLIGHT_NODES
            )
        for method in defect_methods():
            bias_seconds[method] = max(
                float(next(
                    value for value in reports[node]["projections"]
                    if value["resource_tier"] == tier
                )["bias_tuning_seconds_m8"][method])
                for node in EXPECTED_PREFLIGHT_NODES
            )
        capacity = sum(node_capacity()[node] for node in EXECUTION_NODES)
        measurement_core = sum(
            5 * 2 * 16 * trajectory_seconds[method] for method in methods
        )
        bias_core = sum(5 * bias_seconds[method] for method in defect_methods())
        projected_core = measurement_core + 2.0 * bias_core
        projected_wall = SAFETY_FACTOR * projected_core / capacity
        passed = (
            common_ordered == list(methods)
            and projected_wall <= SCREEN_WINDOW_SECONDS
            and all(
                trajectory_seconds[method] <= TRAJECTORY_LIMIT_SECONDS
                for method in methods
            )
        )
        projections.append({
            "resource_tier": tier,
            "eligible_methods": common_ordered,
            "per_node": per_node,
            "trajectory_seconds_m8": trajectory_seconds,
            "bias_tuning_seconds_m8": bias_seconds,
            "projected_core_seconds": projected_core,
            "projected_screen_wall_seconds": projected_wall,
            "safety_factor": SAFETY_FACTOR,
            "execution_nodes": list(EXECUTION_NODES),
            "execution_capacity": capacity,
            "pass": bool(passed),
        })
        common_by_tier[tier] = common_ordered
    passing = [value for value in projections if value["pass"]]
    selected = passing[-1]["resource_tier"] if passing else None
    all_node_reports_valid = all(
        report["status"] == "PASS" for report in reports.values()
    )
    consensus_pass = selected is not None and all_node_reports_valid
    result = {
        "benchmark_version": RUNTIME_CONSENSUS_VERSION,
        "contract_version": CONTRACT_VERSION,
        **expected_identity,
        "environment": {
            "system": "Linux", "nodes": list(EXPECTED_PREFLIGHT_NODES),
        },
        "node_report_sha256": {
            node: sha256_file(report_paths[node])
            for node in EXPECTED_PREFLIGHT_NODES
        },
        "completed_unix_max": max(
            float(report["completed_unix"]) for report in reports.values()
        ),
        "projections": projections,
        "selected_resource_tier": selected,
        "selected_eligible_methods": (
            [] if selected is None else common_by_tier[selected]
        ),
        "excluded_work": ["full_sector_ti", "wmc"],
        "status": "PASS" if consensus_pass else "RUNTIME_EXHAUSTED",
    }
    if output_path is not None:
        atomic_json(output_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("config")
    parser.add_argument("source_commit")
    parser.add_argument("output")
    parser.add_argument("--require-verified-source", action="store_true")
    parser.add_argument("--node", choices=EXPECTED_PREFLIGHT_NODES)
    parser.add_argument("--combine-report", action="append", default=[])
    args = parser.parse_args(argv)
    if args.combine_report:
        mappings = dict(value.split("=", 1) for value in args.combine_report)
        result = combine_runtime_reports(mappings, args.output)
    else:
        result = run_benchmark(
            args.registry, args.config, args.source_commit,
            verified_source=args.require_verified_source, node=args.node,
        )
        atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "selected_resource_tier": result["selected_resource_tier"],
        "eligible_methods": result["selected_eligible_methods"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
