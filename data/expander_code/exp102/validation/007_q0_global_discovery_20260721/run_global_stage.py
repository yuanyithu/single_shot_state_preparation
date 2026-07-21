"""Execute one immutable global-discovery manifest on its frozen owner node."""

import argparse
import concurrent.futures
import json
from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    DEFECT_METHODS,
    GLOBAL_TASKS_VERSION,
    NODE_CAPACITY,
    fixed_global_ownership,
    load_global_discovery_config,
    run_bias_task,
    run_defect_task,
    run_hard_task,
    run_ti_anchor_task,
    validate_global_control_manifest,
    validate_global_schedule,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
    verify_source_identity,
)


def _execute(arguments):
    function, values = arguments
    return function(*values)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(NODE_CAPACITY))
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--ownership", required=True)
    parser.add_argument("--ownership-sha256", required=True)
    parser.add_argument("--stage-fingerprint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-file-sha256", required=True)
    args = parser.parse_args(argv)
    if args.num_workers != NODE_CAPACITY[args.node]:
        raise ValueError("global worker count differs from frozen node capacity")
    source = Path.cwd().resolve()
    source_identity = verify_source_identity(source, args.source_commit)
    control_path = Path(args.control).resolve(strict=True)
    ownership_path = Path(args.ownership).resolve(strict=True)
    if sha256_file(control_path) != args.control_sha256:
        raise ValueError("global control SHA256 mismatch")
    if sha256_file(ownership_path) != args.ownership_sha256:
        raise ValueError("global ownership SHA256 mismatch")
    control = json.loads(control_path.read_text(encoding="ascii"))
    ownership = json.loads(ownership_path.read_text(encoding="ascii"))
    if (control.get("manifest_version") != GLOBAL_TASKS_VERSION
            or control.get("source_commit") != args.source_commit):
        raise ValueError("global control source/version mismatch")
    tasks = [entry["task"] for entry in control["tasks"]]
    expected_ownership = fixed_global_ownership(
        tasks, ownership.get("nodes", []), args.source_commit,
        args.control_sha256, f"{control['stage']}:{control['kind']}",
    )
    if ownership != expected_ownership:
        raise ValueError("global ownership is not the canonical assignment")
    if (args.node not in ownership["nodes"]
            or ownership["stage_fingerprint"] != args.stage_fingerprint):
        raise ValueError("global node/stage fingerprint mismatch")
    entry_by_fingerprint = {
        entry["task_fingerprint"]: entry for entry in control["tasks"]
    }
    if len(entry_by_fingerprint) != len(control["tasks"]):
        raise ValueError("global control contains duplicate task fingerprints")
    for fingerprint, entry in entry_by_fingerprint.items():
        if fingerprint != sha256_json(entry["task"]):
            raise ValueError("global control task fingerprint is noncanonical")
    selected = [
        (fingerprint, entry_by_fingerprint[fingerprint])
        for fingerprint in sorted(entry_by_fingerprint)
        if ownership["task_owner"][fingerprint] == args.node
    ]

    registry_path = source / "data/expander_code/exp102/registry/registry.json"
    config_path = source / "data/expander_code/exp102/config/q0_global.discovery.v1.json"
    schedule_path = Path(args.schedule).resolve(strict=True)
    if sha256_file(schedule_path) != args.schedule_file_sha256:
        raise ValueError("global worker schedule file SHA256 mismatch")
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    validate_global_control_manifest(control, registry, config)
    schedule = validate_global_schedule(
        schedule_path, registry, config, args.source_commit,
    )
    if (schedule["archive_sha256"] != source_identity.get("archive_sha256")
            or schedule["source_manifest_sha256"]
            != source_identity.get("manifest_sha256")):
        raise ValueError("global worker schedule/source identity mismatch")
    evidence_root = (
        Path.home() / ".single_shot/runs" / args.run_id
        / "global" / control["stage"]
    )
    work = []
    output_by_fingerprint = {}
    for fingerprint, entry in selected:
        task = entry["task"]
        output = evidence_root / entry["output_relpath"]
        output_by_fingerprint[fingerprint] = output
        if control["kind"] == "ti_anchor":
            function = run_ti_anchor_task
            values = (
                str(registry_path), str(config_path), args.source_commit,
                task, str(output),
            )
        elif control["kind"] == "defect_bias":
            function = run_bias_task
            values = (
                str(registry_path), str(config_path), args.source_commit,
                task, str(output),
            )
        elif task["method_id"] in DEFECT_METHODS:
            function = run_defect_task
            values = (
                str(registry_path), str(config_path), args.source_commit,
                task, str(evidence_root / entry["bias_relpath"]), str(output),
            )
        else:
            function = run_hard_task
            values = (
                str(registry_path), str(config_path), args.source_commit,
                task, str(output),
            )
        work.append((function, values))
    counts = {"computed": 0, "reused": 0}
    # Warm one task before worker fan-out to populate the verified shared Numba
    # cache without concurrent writers.
    if work:
        counts[_execute(work[0])] += 1
        if len(work) > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.num_workers, len(work) - 1),
            ) as pool:
                for status in pool.map(_execute, work[1:]):
                    counts[status] += 1
    raw_files = []
    for fingerprint, output in sorted(output_by_fingerprint.items()):
        if not output.is_file():
            raise ValueError(f"global worker did not produce {output}")
        raw_files.append({
            "task_fingerprint": fingerprint,
            "path": output.relative_to(evidence_root).as_posix(),
            "sha256": sha256_file(output),
        })
    if verify_source_identity(source, args.source_commit) != source_identity:
        raise ValueError("global stage changed the verified source tree")
    marker_root = (
        evidence_root / "node_manifests" / args.control_sha256[:12] / args.node
    )
    raw_manifest = {
        "raw_manifest_version": control["kind"],
        "node": args.node,
        "stage": control["stage"],
        "kind": control["kind"],
        "stage_fingerprint": args.stage_fingerprint,
        "source_commit": args.source_commit,
        "control_sha256": args.control_sha256,
        "ownership_sha256": args.ownership_sha256,
        "schedule_file_sha256": args.schedule_file_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": source_identity,
        "files": raw_files,
    }
    raw_manifest_path = marker_root / "raw_manifest.json"
    atomic_json(raw_manifest_path, raw_manifest)
    atomic_json(marker_root / "stage_status.json", {
        "status": "SUCCESS", "node": args.node,
        "stage_fingerprint": args.stage_fingerprint,
        "expected": len(work), **counts,
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
    })
    print(json.dumps({
        "node": args.node, "expected": len(work), **counts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
