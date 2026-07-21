"""Freeze ownership and launch one diagnostic stage on nd-1 and nd-3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

from .common import (
    CONTRACT_VERSION,
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_REGISTRY_RELATIVE,
    EXECUTION_NODES,
    PREFLIGHT_VERSION,
    all_methods,
    config_sha256,
    fixed_ownership,
    load_config,
    load_registry,
    node_capacity,
    read_linux_loads,
    remote_command,
    sha256_file,
    sha256_json,
    validate_control,
    validate_runtime_consensus,
    validate_schedule,
    validate_source_and_hashes,
    verified_bootstrap,
    verify_remote_stage,
    wait_for_markers,
    write_exclusive,
)


MODULE_ROOT = (
    "data.expander_code.exp102.validation."
    "011_q0_global_screen_diagnostic_20260721"
)


def _freeze_copy(input_path, destination):
    input_path = Path(input_path).resolve(strict=True)
    destination = Path(destination)
    digest = sha256_file(input_path)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise FileExistsError("diagnostic frozen control copy conflicts")
    else:
        value = json.loads(input_path.read_text(encoding="ascii"))
        if write_exclusive(destination, value) != digest:
            raise ValueError("diagnostic control is not canonical JSON")
    return destination, digest


def _validate_preflight(path, source_commit, registry_sha256,
                        diagnostic_config_sha256, schedule_file_sha256,
                        schedule_sha256, runtime_sha256):
    report = json.loads(Path(path).read_text(encoding="ascii"))
    if (report.get("report_version") != PREFLIGHT_VERSION
            or report.get("contract_version") != CONTRACT_VERSION
            or report.get("status") != "PASS"
            or report.get("source_commit") != source_commit
            or report.get("registry_sha256") != registry_sha256
            or report.get("diagnostic_config_sha256")
            != diagnostic_config_sha256
            or report.get("schedule_file_sha256") != schedule_file_sha256
            or report.get("schedule_sha256") != schedule_sha256
            or report.get("runtime_consensus_sha256") != runtime_sha256
            or report.get("selected_eligible_methods") != list(all_methods())
            or report.get("excluded_work") != ["full_sector_ti", "wmc"]
            or report.get("maximum_terminal_status")
            != "DIAGNOSTIC_SCREEN_PAIR_FOUND"):
        raise ValueError("diagnostic preflight is not a PASS authority")
    return report


def _predicted_wall(control, runtime):
    selected = runtime["selected_resource_tier"]
    projection = next(
        value for value in runtime["projections"]
        if value["resource_tier"] == selected
    )
    if control["kind"] == "diagnostic_defect_bias":
        core = sum(
            float(projection["bias_tuning_seconds_m8"][
                entry["task"]["method_id"]
            ])
            for entry in control["tasks"]
        )
    else:
        core = sum(
            float(projection["trajectory_seconds_m8"][
                entry["task"]["method_id"]
            ])
            for entry in control["tasks"]
        )
        unique_biases = {
            (entry["task"]["method_id"], entry["bias_relpath"])
            for entry in control["tasks"]
            if entry.get("bias_relpath") is not None
        }
        core += sum(
            float(projection["bias_tuning_seconds_m8"][method])
            for method, _ in unique_biases
        )
    capacity = sum(node_capacity()[node] for node in EXECUTION_NODES)
    return 2.0 * core / capacity


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-file-sha256", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--max-load-fraction", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--registry-relative", default=str(DEFAULT_REGISTRY_RELATIVE))
    parser.add_argument("--config-relative", default=str(DEFAULT_CONFIG_RELATIVE))
    args = parser.parse_args(argv)
    validate_source_and_hashes(
        args.source_commit, args.archive_sha256, args.manifest_sha256,
        args.schedule_file_sha256,
    )
    if (not math.isfinite(args.max_load_fraction)
            or args.max_load_fraction <= 0.0):
        raise ValueError("diagnostic load fraction is invalid")

    registry_path = Path(args.registry_relative)
    config_path = Path(args.config_relative)
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    schedule_input = Path(args.schedule).resolve(strict=True)
    if sha256_file(schedule_input) != args.schedule_file_sha256:
        raise ValueError("diagnostic stage schedule file SHA mismatch")
    schedule = validate_schedule(
        schedule_input, registry, config, args.source_commit,
    )
    if (schedule["archive_sha256"] != args.archive_sha256
            or schedule["source_manifest_sha256"] != args.manifest_sha256):
        raise ValueError("diagnostic stage schedule/source mismatch")
    runtime_input = Path(args.runtime_report).resolve(strict=True)
    runtime_sha = sha256_file(runtime_input)
    runtime = validate_runtime_consensus(
        runtime_input, args.source_commit, registry["registry_sha256"],
        config_sha256(config), args.archive_sha256, args.manifest_sha256,
    )
    _validate_preflight(
        args.preflight_report, args.source_commit,
        registry["registry_sha256"], config_sha256(config),
        args.schedule_file_sha256, schedule["schedule_sha256"], runtime_sha,
    )
    control_input = Path(args.control).resolve(strict=True)
    control = json.loads(control_input.read_text(encoding="ascii"))
    validate_control(control, registry, config)
    if control.get("source_commit") != args.source_commit:
        raise ValueError("diagnostic control source mismatch")
    stage_key = {
        "diagnostic_defect_bias": "bias",
        "diagnostic_measurement": "measurement",
    }.get(control.get("kind"))
    if stage_key is None:
        raise ValueError("diagnostic stage control kind is unknown")
    selected_tier = runtime["selected_resource_tier"]
    expected_methods = (
        list(runtime["selected_eligible_methods"][-3:])
        if stage_key == "bias" else list(runtime["selected_eligible_methods"])
    )
    if ([method for method, tier in control["method_tiers"]]
            != expected_methods
            or any(tier != selected_tier
                   for method, tier in control["method_tiers"])):
        raise ValueError("diagnostic control does not use the maximum passing tier")
    deadline = float(schedule["deadlines_unix"][stage_key])
    available = deadline - time.time()
    predicted = _predicted_wall(control, runtime)
    if (not math.isfinite(predicted) or predicted > available
            or available <= 0.0):
        raise TimeoutError("diagnostic stage cannot fit its frozen deadline")
    timeout = available if args.timeout_seconds is None else args.timeout_seconds
    if (not math.isfinite(timeout) or timeout <= 0.0 or timeout > available):
        raise ValueError("diagnostic stage timeout exceeds remaining wall")

    loads = read_linux_loads(EXECUTION_NODES)
    capacities = node_capacity()
    overloaded = [
        node for node in EXECUTION_NODES
        if loads[node][0] > args.max_load_fraction * capacities[node]
    ]
    print(" ".join(
        f"{node}_load1={loads[node][0]:.2f}"
        for node in EXECUTION_NODES
    ), flush=True)
    if overloaded:
        raise RuntimeError(
            "fixed diagnostic execution nodes are overloaded: "
            + ",".join(overloaded)
        )

    home = Path.home()
    deployment_root = home / ".single_shot/repos" / args.run_id
    if (sha256_file(deployment_root / "SOURCE.tar") != args.archive_sha256
            or sha256_file(deployment_root / "SOURCE_MANIFEST.json")
            != args.manifest_sha256):
        raise ValueError("diagnostic stage deployment hashes mismatch")
    run_root = home / ".single_shot/runs" / args.run_id
    raw_root = run_root / "screen_diagnostic/raw"
    schedule_path, schedule_file_sha = _freeze_copy(
        schedule_input,
        run_root / "control/SCREEN_DIAGNOSTIC_24H_SCHEDULE.json",
    )
    runtime_path, frozen_runtime_sha = _freeze_copy(
        runtime_input, run_root / "control/screen_runtime_consensus.json",
    )
    if (schedule_file_sha != args.schedule_file_sha256
            or frozen_runtime_sha != runtime_sha):
        raise ValueError("diagnostic frozen schedule/runtime copy mismatch")
    input_digest = sha256_file(control_input)
    control_path, control_sha = _freeze_copy(
        control_input,
        run_root / "control" / f"screen_{stage_key}_{input_digest[:12]}.json",
    )
    tasks = [entry["task"] for entry in control["tasks"]]
    ownership = fixed_ownership(
        tasks, args.source_commit, registry["registry_sha256"],
        config_sha256(config), schedule_file_sha, schedule["schedule_sha256"],
        control_sha, runtime_sha, control["stage"], control["kind"],
    )
    ownership_path = (
        run_root / "control" / f"screen_ownership_{control_sha[:12]}.json"
    )
    ownership_sha = write_exclusive(ownership_path, ownership)
    fingerprint = ownership["stage_fingerprint"]

    source = deployment_root / "source"
    stage_dirs = {}
    screens = {}
    run_token = sha256_json({"run_id": args.run_id})[:8]
    try:
        for node in EXECUTION_NODES:
            stage_dir = (
                run_root / "screen_diagnostic/stages" / stage_key / "markers"
                / control_sha[:12] / node
            )
            stage_dirs[node] = stage_dir
            if any((stage_dir / name).exists()
                   for name in ("RUNNING", "SUCCESS", "FAILED")):
                raise FileExistsError(
                    f"diagnostic {stage_key} marker already exists for {node}"
                )
            log = home / ".single_shot/logs" / (
                f"{args.run_id}_screen_{stage_key}_{node}.log"
            )
            command = (
                "env", f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
                "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                "PYTHONDONTWRITEBYTECODE=1", "conda", "run", "-n", "11",
                "--no-capture-output", "python", "-m",
                MODULE_ROOT + ".run_screen_stage", node,
                "--num-workers", capacities[node], "--run-id", args.run_id,
                "--source-commit", args.source_commit,
                "--control", control_path, "--control-sha256", control_sha,
                "--ownership", ownership_path,
                "--ownership-sha256", ownership_sha,
                "--stage-fingerprint", fingerprint,
                "--schedule", schedule_path,
                "--schedule-file-sha256", schedule_file_sha,
                "--runtime-report", runtime_path,
                "--runtime-report-sha256", runtime_sha,
                "--raw-root", raw_root,
                "--registry-relative", registry_path,
                "--config-relative", config_path,
            )
            shell = verified_bootstrap(
                deployment_root, args.source_commit, args.archive_sha256,
                args.manifest_sha256, stage_dir, log, fingerprint, command,
            )
            screen = (
                f"exp102_sd_{run_token}_{stage_key}_{control_sha[:8]}_{node}"
            )
            screens[node] = screen
            subprocess.run((
                "ssh", node,
                remote_command((
                    "screen", "-dmS", screen, "bash", "-lc", shell,
                )),
            ), check=True)
            print(
                f"launched node={node} workers={capacities[node]} "
                f"screen={screen}", flush=True,
            )
        wait_for_markers(stage_dirs, fingerprint, timeout)
    except BaseException:
        for node, screen in screens.items():
            subprocess.run((
                "ssh", node,
                remote_command(("screen", "-S", screen, "-X", "quit")),
            ), check=False)
        raise

    evidence = verify_remote_stage(
        run_root, raw_root, control_path, ownership_path, deployment_root,
        schedule_path, runtime_path, registry_path, config_path,
    )
    evidence_path = (
        run_root / "control" / f"screen_{stage_key}_evidence_{control_sha[:12]}.json"
    )
    evidence_sha = write_exclusive(evidence_path, evidence)
    print(json.dumps({
        "status": "SUCCESS", "stage": stage_key,
        "nodes": list(EXECUTION_NODES), "loads": loads,
        "predicted_wall_seconds_with_safety_factor_2": predicted,
        "control": str(control_path), "control_sha256": control_sha,
        "ownership": str(ownership_path), "ownership_sha256": ownership_sha,
        "evidence": str(evidence_path), "evidence_sha256": evidence_sha,
        "runtime_report_sha256": runtime_sha,
        "stage_fingerprint": fingerprint,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
