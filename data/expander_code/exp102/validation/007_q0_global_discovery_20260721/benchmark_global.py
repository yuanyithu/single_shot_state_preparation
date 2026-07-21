"""Linux runtime benchmark and frozen 72-hour resource-tier projection."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import socket
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    DEFECT_METHODS,
    HARD_METHODS,
    NODE_CAPACITY,
    RESOURCE_TIERS,
    character_seed,
    load_global_discovery_config,
    uniform_seed_for_cell,
)
from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
    verify_source_identity,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    DefectTraceConfig,
    GlobalSeedIdentity,
    HardCosetConfig,
    build_joint_blocks,
    build_logical_proposal_catalog,
    run_defect_trace_trajectory,
    run_hardcoset_trajectory,
    tune_defect_bias,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _identity(source_commit, config, registry, cell, method, family, trajectory, tag):
    return GlobalSeedIdentity(
        source_commit=source_commit,
        config_sha256=config["discovery_config_sha256"],
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(cell),
        method_id=method,
        resource_tier="RUNTIME",
        init_family=family,
        trajectory_index=trajectory,
        trajectory_namespace=f"q0_global_runtime_{tag}_v1",
    )


def _timed_hard(model, frame, syndrome, epsilon, p, method, source_commit,
                config, registry, cell, catalog, joint):
    warm = HardCosetConfig(method, p, 2, 8)
    identity = _identity(
        source_commit, config, registry, cell, method, "P", 0, "warm",
    )
    start = time.perf_counter()
    run_hardcoset_trajectory(
        model, frame, syndrome, warm, identity, epsilon,
        engine="numba", catalog=catalog, joint=joint,
    )
    warmup = time.perf_counter() - start
    timed = HardCosetConfig(method, p, 32, 128)
    identity = _identity(
        source_commit, config, registry, cell, method, "P", 1, "timed",
    )
    wall_start, core_start = time.perf_counter(), time.process_time()
    run_hardcoset_trajectory(
        model, frame, syndrome, timed, identity, epsilon,
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


def _timed_defect(model, frame, syndrome, epsilon, p, method, source_commit,
                  config, registry, cell):
    warm = DefectTraceConfig(method, p, 2, 8)
    identity = _identity(
        source_commit, config, registry, cell, method, "P", 0, "warm",
    )
    bias = np.zeros(warm.dmax + 1)
    start = time.perf_counter()
    run_defect_trace_trajectory(
        model, frame, syndrome, warm, identity, epsilon, bias, "a" * 64,
        engine="numba",
    )
    warmup = time.perf_counter() - start
    timed = DefectTraceConfig(method, p, 32, 128)
    identity = _identity(
        source_commit, config, registry, cell, method, "P", 1, "timed",
    )
    wall_start, core_start = time.perf_counter(), time.process_time()
    run_defect_trace_trajectory(
        model, frame, syndrome, timed, identity, epsilon, bias, "a" * 64,
        engine="numba",
    )
    wall = time.perf_counter() - wall_start
    core = time.process_time() - core_start
    tuning_identities = [
        _identity(
            source_commit, config, registry, cell, method, "TUNE", index,
            "bias_timing",
        )
        for index in range(8)
    ]
    tuning_start, tuning_core_start = time.perf_counter(), time.process_time()
    tune_defect_bias(
        model, syndrome, timed, tuning_identities, engine="numba",
    )
    tuning_wall = time.perf_counter() - tuning_start
    tuning_core = time.process_time() - tuning_core_start
    return {
        "warmup_seconds": warmup,
        "timed_sweeps": 160,
        "wall_seconds": wall,
        "core_seconds": core,
        "core_seconds_per_sweep": core / 160.0,
        "bias_tuning_wall_seconds": tuning_wall,
        "bias_tuning_core_seconds": tuning_core,
    }


def _timed_ti_anchor(registry_path, registry, config, source_commit):
    load_exp101()
    from exp101_certified_src.model import DisorderRealization, wire_ensemble
    from exp101_certified_src.sector_ti import SectorTiConfig, run_sector_ti

    _, code, H = load_frozen_code(registry_path, "m03_c00")
    model, frame = build_model(H)
    cell = next(
        value for value in config["panels"]["SMALL6"]["cells"]
        if value["code_id"] == "m03_c00" and value["p"] == 0.10
    )
    uniform_seed = uniform_seed_for_cell(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(
        model.num_qubits
    )
    epsilon = (uniforms < cell["p"]).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    disorder = DisorderRealization(
        epsilon_data_true=epsilon,
        measurement_error=np.zeros(model.num_checks, dtype=np.uint8),
        effective_syndrome=syndrome,
        p=cell["p"], q=0.0,
    )
    wiring = wire_ensemble(model, disorder, "true_posterior", frame)
    probe = SectorTiConfig(
        num_kp_grid_points=3, num_burn_in_sweeps=1,
        num_measurements=8, block_count=8, num_bootstrap=2,
    )
    seed = int.from_bytes(
        hashlib.sha256(
            f"{source_commit}:q0_global_ti_runtime".encode("ascii")
        ).digest()[:8], "big",
    ) & ((1 << 63) - 1)
    wall_start, core_start = time.perf_counter(), time.process_time()
    run_sector_ti(model, frame, wiring, probe, seed=seed)
    probe_wall = time.perf_counter() - wall_start
    probe_core = time.process_time() - core_start
    frozen = config["ti_anchor"]
    probe_work = 3 * (1 + 8)
    frozen_work = int(frozen["num_kp_grid_points"]) * (
        int(frozen["num_burn_in_sweeps"]) + int(frozen["num_measurements"])
    )
    ratio = frozen_work / probe_work
    projected = probe_wall * ratio
    return {
        "probe_wall_seconds": probe_wall,
        "probe_core_seconds": probe_core,
        "probe_work_units": probe_work,
        "frozen_work_units": frozen_work,
        "linear_scale_ratio": ratio,
        "projected_seconds_per_anchor": projected,
        "factor_two_stage_seconds_two_node_contingency": 4.0 * projected,
        "pass": bool(
            projected <= 10.0 * 3600.0 and 4.0 * projected <= 22.0 * 3600.0
        ),
    }


def run_benchmark(registry_path, config_path, source_commit, *, verified_source=False,
                  node=None):
    source_identity = (
        verify_source_identity(Path.cwd(), source_commit)
        if verified_source else None
    )
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    rows = []
    catalog_seconds = {}
    joint_seconds = {}
    for code_id in ("m06_c00", "m08_c06"):
        _, code, H = load_frozen_code(registry_path, code_id)
        model, frame = build_model(H)
        cell = next(
            value for value in config["panels"]["HARD2"]["cells"]
            if value["code_id"] == code_id
        )
        uniform_seed = uniform_seed_for_cell(registry, code, cell)
        uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        epsilon = (uniforms < cell["p"]).astype(np.uint8)
        syndrome = (
            model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
        ).astype(np.uint8)
        start = time.perf_counter()
        catalog = build_logical_proposal_catalog(model, frame)
        catalog_seconds[int(code["m"])] = time.perf_counter() - start
        for method in HARD_METHODS:
            block_size = int(method[-2:]) if "-J" in method else 0
            start = time.perf_counter()
            joint = build_joint_blocks(model, frame, catalog, block_size) if block_size else None
            joint_seconds[(int(code["m"]), method)] = time.perf_counter() - start
            timing = _timed_hard(
                model, frame, syndrome, epsilon, cell["p"], method,
                source_commit, config, registry, cell, catalog, joint,
            )
            rows.append({
                "code_id": code_id, "m": int(code["m"]), "method_id": method,
                "catalog_seconds": catalog_seconds[int(code["m"])],
                "joint_build_seconds": joint_seconds[(int(code["m"]), method)],
                **timing,
            })
        for method in DEFECT_METHODS:
            timing = _timed_defect(
                model, frame, syndrome, epsilon, cell["p"], method,
                source_commit, config, registry, cell,
            )
            rows.append({
                "code_id": code_id, "m": int(code["m"]), "method_id": method,
                "catalog_seconds": 0.0, "joint_build_seconds": 0.0,
                **timing,
            })

    ti_anchor_projection = _timed_ti_anchor(
        registry_path, registry, config, source_commit,
    )
    m8 = {row["method_id"]: row for row in rows if row["m"] == 8}
    projections = []
    eligible_by_tier = {}
    for tier, resources in RESOURCE_TIERS.items():
        sweeps = int(resources["burn_sweeps"] + resources["measurement_sweeps"])
        trajectory_seconds = {}
        eligible = []
        for method in (*HARD_METHODS, *DEFECT_METHODS):
            row = m8[method]
            setup = row["catalog_seconds"] + row["joint_build_seconds"]
            estimate = setup + row["core_seconds_per_sweep"] * sweeps
            trajectory_seconds[method] = estimate
            if estimate <= 2.0 * 3600.0:
                eligible.append(method)
        hard_eligible = [method for method in eligible if method in HARD_METHODS]
        defect_eligible = [method for method in eligible if method in DEFECT_METHODS]
        if not hard_eligible or not defect_eligible:
            projected_core = float("inf")
        else:
            screen = sum(160 * trajectory_seconds[method] for method in eligible)
            worst_hard = max(trajectory_seconds[method] for method in hard_eligible)
            worst_defect = max(trajectory_seconds[method] for method in defect_eligible)
            # Later schedule: 192 HARD2 T/2T-equivalent tasks, 992 confirmation
            # 2T tasks, and 192 RES6 T tasks for each selected mechanism.
            later_multiplier = 192 + 2 * 992 + 192
            later = later_multiplier * (worst_hard + worst_defect)
            bias_tasks = 5 * len(defect_eligible) + 2 * 2 + 31 + 6
            worst_bias = max(m8[method]["bias_tuning_core_seconds"] for method in defect_eligible)
            projected_core = screen + later + bias_tasks * worst_bias
        projected_hours = (
            2.0 * projected_core
            / (NODE_CAPACITY["nd-2"] + NODE_CAPACITY["nd-3"])
            / 3600.0
        )
        pass_tier = np.isfinite(projected_hours) and projected_hours <= 58.0
        projections.append({
            "resource_tier": tier,
            "trajectory_seconds_m8": trajectory_seconds,
            "eligible_methods": eligible,
            "projected_core_seconds": projected_core,
            "projected_hours_with_safety_factor_2": projected_hours,
            "pass": bool(pass_tier),
        })
        eligible_by_tier[tier] = eligible
    passing = [value for value in projections if value["pass"]]
    selected = passing[-1]["resource_tier"] if passing else None
    checks = {
        "at_least_T1_fits": selected is not None,
        "hard_method_available": bool(selected and any(
            method in HARD_METHODS for method in eligible_by_tier[selected]
        )),
        "defect_method_available": bool(selected and any(
            method in DEFECT_METHODS for method in eligible_by_tier[selected]
        )),
        "all_numeric_finite": all(
            np.isfinite(float(row[key]))
            for row in rows
            for key in ("warmup_seconds", "wall_seconds", "core_seconds",
                        "core_seconds_per_sweep")
        ),
        "ti_anchor_fits_confirmation_window": ti_anchor_projection["pass"],
    }
    return {
        "benchmark_version": "exp102.q0_global.runtime.v1",
        "source_commit": source_commit,
        "source_identity": source_identity,
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "hostname": socket.gethostname(), "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "node": str(node) if node is not None else socket.gethostname(),
        "completed_unix": time.time(),
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "rows": rows,
        "ti_anchor_projection": ti_anchor_projection,
        "projections": projections,
        "selected_resource_tier": selected,
        "selected_eligible_methods": [] if selected is None else eligible_by_tier[selected],
        "projection_nodes": ["nd-2", "nd-3"],
        "projection_capacity": NODE_CAPACITY["nd-2"] + NODE_CAPACITY["nd-3"],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "RUNTIME_EXHAUSTED",
    }


def combine_runtime_reports(report_paths, output_path=None):
    expected_nodes = ("nd-1", "nd-2", "nd-3")
    if set(report_paths) != set(expected_nodes):
        raise ValueError("runtime consensus requires nd-1, nd-2, and nd-3")
    reports = {
        node: json.loads(Path(report_paths[node]).read_text(encoding="ascii"))
        for node in expected_nodes
    }
    first = reports[expected_nodes[0]]
    axes = (
        first.get("source_commit"), first.get("registry_sha256"),
        first.get("discovery_config_sha256"),
    )
    source_identity = first.get("source_identity")
    for node, report in reports.items():
        if (report.get("benchmark_version") != "exp102.q0_global.runtime.v1"
                or report.get("node") != node
                or report.get("status") != "PASS"
                or report.get("environment", {}).get("system") != "Linux"
                or (report.get("source_commit"), report.get("registry_sha256"),
                    report.get("discovery_config_sha256")) != axes
                or report.get("source_identity") != source_identity
                or not isinstance(source_identity, dict)
                or source_identity.get("mode") != "archive"
                or source_identity.get("source_commit") != axes[0]
                or not np.isfinite(float(report.get("completed_unix", np.nan)))):
            raise ValueError(f"runtime report is not verified consensus evidence: {node}")
        if not report.get("ti_anchor_projection", {}).get("pass"):
            raise ValueError(f"TI anchor runtime projection failed: {node}")
    projections = []
    common_by_tier = {}
    for tier in RESOURCE_TIERS:
        per_node = {}
        common = set((*HARD_METHODS, *DEFECT_METHODS))
        passing = True
        for node, report in reports.items():
            matches = [
                value for value in report["projections"]
                if value["resource_tier"] == tier
            ]
            if len(matches) != 1:
                raise ValueError(f"runtime report has a malformed tier: {node}/{tier}")
            projection = matches[0]
            per_node[node] = {
                "projected_hours_with_safety_factor_2": projection[
                    "projected_hours_with_safety_factor_2"
                ],
                "eligible_methods": projection["eligible_methods"],
                "pass": projection["pass"],
            }
            common &= set(projection["eligible_methods"])
            passing &= bool(projection["pass"])
        common = [
            method for method in (*HARD_METHODS, *DEFECT_METHODS)
            if method in common
        ]
        worst_hours = max(
            float(value["projected_hours_with_safety_factor_2"])
            for value in per_node.values()
        )
        trajectory_seconds = {
            method: max(
                float(next(
                    value for value in reports[node]["projections"]
                    if value["resource_tier"] == tier
                )["trajectory_seconds_m8"][method])
                for node in expected_nodes
            )
            for method in (*HARD_METHODS, *DEFECT_METHODS)
        }
        passing &= (
            worst_hours <= 58.0
            and any(method in HARD_METHODS for method in common)
            and any(method in DEFECT_METHODS for method in common)
        )
        projections.append({
            "resource_tier": tier,
            "eligible_methods": common,
            "per_node": per_node,
            "trajectory_seconds_m8": trajectory_seconds,
            "projected_hours_with_safety_factor_2": worst_hours,
            "pass": bool(passing),
        })
        common_by_tier[tier] = common
    passing_tiers = [value for value in projections if value["pass"]]
    selected = passing_tiers[-1]["resource_tier"] if passing_tiers else None
    result = {
        "benchmark_version": "exp102.q0_global.runtime_consensus.v1",
        "source_commit": axes[0],
        "source_identity": source_identity,
        "registry_sha256": axes[1],
        "discovery_config_sha256": axes[2],
        "environment": {"system": "Linux", "nodes": list(expected_nodes)},
        "node_report_sha256": {
            node: sha256_file(report_paths[node]) for node in expected_nodes
        },
        "completed_unix_max": max(
            float(report["completed_unix"]) for report in reports.values()
        ),
        "projections": projections,
        "bias_tuning_seconds_m8": {
            method: max(
                float(next(
                    row for row in reports[node]["rows"]
                    if row["m"] == 8 and row["method_id"] == method
                )["bias_tuning_wall_seconds"])
                for node in expected_nodes
            )
            for method in DEFECT_METHODS
        },
        "ti_anchor_projection": {
            "projected_seconds_per_anchor": max(
                float(report["ti_anchor_projection"]["projected_seconds_per_anchor"])
                for report in reports.values()
            ),
            "factor_two_stage_seconds_two_node_contingency": max(
                float(report["ti_anchor_projection"][
                    "factor_two_stage_seconds_two_node_contingency"
                ]) for report in reports.values()
            ),
            "pass": True,
        },
        "selected_resource_tier": selected,
        "selected_eligible_methods": (
            [] if selected is None else common_by_tier[selected]
        ),
        "status": "PASS" if selected is not None else "RUNTIME_EXHAUSTED",
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
    parser.add_argument("--node", choices=("nd-1", "nd-2", "nd-3"))
    parser.add_argument(
        "--combine-report", action="append", default=[],
        help="NODE=PATH; provide all three instead of running a benchmark",
    )
    args = parser.parse_args(argv)
    if args.combine_report:
        mappings = dict(value.split("=", 1) for value in args.combine_report)
        result = combine_runtime_reports(mappings)
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
