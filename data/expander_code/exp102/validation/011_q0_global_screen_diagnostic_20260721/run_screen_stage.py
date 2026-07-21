"""Execute one immutable diagnostic bias or measurement owner partition."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time

from data.expander_code.exp102.exp102_pipeline.io import verify_source_identity

from .common import (
    CONTRACT_VERSION,
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_REGISTRY_RELATIVE,
    EXECUTION_NODES,
    RAW_MANIFEST_VERSION,
    atomic_json,
    config_sha256,
    defect_methods,
    load_config,
    load_registry,
    node_capacity,
    resolve_source_path,
    run_bias_task,
    run_defect_task,
    run_hard_task,
    sha256_file,
    sha256_json,
    validate_control,
    validate_ownership,
    validate_raw,
    validate_runtime_consensus,
    validate_schedule,
)


def _execute(arguments):
    function, values, keywords = arguments
    return function(*values, **keywords)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=EXECUTION_NODES)
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
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--runtime-report-sha256", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--registry-relative", default=str(DEFAULT_REGISTRY_RELATIVE))
    parser.add_argument("--config-relative", default=str(DEFAULT_CONFIG_RELATIVE))
    args = parser.parse_args(argv)
    capacities = node_capacity()
    if args.num_workers != capacities[args.node]:
        raise ValueError("diagnostic worker count differs from frozen capacity")

    source = Path.cwd().resolve()
    source_identity = verify_source_identity(source, args.source_commit)
    registry_path = resolve_source_path(source, args.registry_relative)
    config_path = resolve_source_path(source, args.config_relative)
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    control_path = Path(args.control).resolve(strict=True)
    ownership_path = Path(args.ownership).resolve(strict=True)
    schedule_path = Path(args.schedule).resolve(strict=True)
    runtime_path = Path(args.runtime_report).resolve(strict=True)
    if sha256_file(control_path) != args.control_sha256:
        raise ValueError("diagnostic worker control SHA mismatch")
    if sha256_file(ownership_path) != args.ownership_sha256:
        raise ValueError("diagnostic worker ownership SHA mismatch")
    if sha256_file(schedule_path) != args.schedule_file_sha256:
        raise ValueError("diagnostic worker schedule SHA mismatch")
    if sha256_file(runtime_path) != args.runtime_report_sha256:
        raise ValueError("diagnostic worker runtime SHA mismatch")
    control = json.loads(control_path.read_text(encoding="ascii"))
    ownership = json.loads(ownership_path.read_text(encoding="ascii"))
    validate_control(control, registry, config)
    if (control.get("source_commit") != args.source_commit
            or control.get("stage") != "screen"):
        raise ValueError("diagnostic worker control source/stage mismatch")
    schedule = validate_schedule(
        schedule_path, registry, config, args.source_commit,
    )
    if (schedule["archive_sha256"] != source_identity.get("archive_sha256")
            or schedule["source_manifest_sha256"]
            != source_identity.get("manifest_sha256")):
        raise ValueError("diagnostic worker schedule/source mismatch")
    validate_runtime_consensus(
        runtime_path, args.source_commit, registry["registry_sha256"],
        config_sha256(config), schedule["archive_sha256"],
        schedule["source_manifest_sha256"],
    )
    tasks = [entry["task"] for entry in control["tasks"]]
    validate_ownership(
        ownership, tasks, args.source_commit, registry["registry_sha256"],
        config_sha256(config), args.schedule_file_sha256,
        schedule["schedule_sha256"], args.control_sha256,
        args.runtime_report_sha256, control["stage"], control["kind"],
    )
    if (ownership["stage_fingerprint"] != args.stage_fingerprint
            or args.node not in ownership["nodes"]):
        raise ValueError("diagnostic worker node/fingerprint mismatch")

    entry_by_fingerprint = {
        entry["task_fingerprint"]: entry for entry in control["tasks"]
    }
    if (len(entry_by_fingerprint) != len(control["tasks"])
            or any(fingerprint != sha256_json(entry["task"])
                   for fingerprint, entry in entry_by_fingerprint.items())):
        raise ValueError("diagnostic worker control fingerprints are invalid")
    selected = [
        (fingerprint, entry_by_fingerprint[fingerprint])
        for fingerprint in sorted(entry_by_fingerprint)
        if ownership["task_owner"][fingerprint] == args.node
    ]
    raw_root = Path(args.raw_root).resolve()
    bias_cache = {}
    if control["kind"] == "diagnostic_measurement":
        for _, entry in selected:
            task = entry["task"]
            if task["method_id"] not in defect_methods():
                continue
            bias_path = (raw_root / entry["bias_relpath"]).resolve(strict=True)
            cache_key = str(bias_path)
            if cache_key not in bias_cache:
                bias_cache[cache_key] = validate_raw(
                    bias_path, "defect_bias", task["method_id"], registry,
                    config, args.source_commit,
                )
    work = []
    output_by_fingerprint = {}
    for fingerprint, entry in selected:
        task = entry["task"]
        output = raw_root / entry["output_relpath"]
        if output.exists():
            raise FileExistsError(
                "fresh diagnostic stage refuses an existing raw file"
            )
        output_by_fingerprint[fingerprint] = output
        if control["kind"] == "diagnostic_defect_bias":
            function = run_bias_task
            values = (
                str(registry_path), str(config_path), args.source_commit,
                task, str(output),
            )
            keywords = {}
        elif task["method_id"] in defect_methods():
            function = run_defect_task
            bias_path = (raw_root / entry["bias_relpath"]).resolve(strict=True)
            values = (
                str(registry_path), str(config_path), args.source_commit,
                task, str(bias_path), str(output),
            )
            keywords = {
                "_validated_bias_record": bias_cache[str(bias_path)],
            }
        else:
            function = run_hard_task
            values = (
                str(registry_path), str(config_path), args.source_commit,
                task, str(output),
            )
            keywords = {}
        work.append((function, values, keywords))

    counts = {"computed": 0, "reused": 0}
    if work:
        status = _execute(work[0])
        counts[status] += 1
        if len(work) > 1:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(args.num_workers, len(work) - 1)) as pool:
                for status in pool.map(_execute, work[1:]):
                    counts[status] += 1
    if counts["reused"]:
        raise ValueError("fresh diagnostic stage unexpectedly reused raw")

    raw_files = []
    for fingerprint, output in sorted(output_by_fingerprint.items()):
        if not output.is_file():
            raise ValueError(f"diagnostic worker did not produce {output}")
        raw_files.append({
            "task_fingerprint": fingerprint,
            "path": output.relative_to(raw_root).as_posix(),
            "sha256": sha256_file(output),
        })
    if verify_source_identity(source, args.source_commit) != source_identity:
        raise ValueError("diagnostic stage changed verified source")

    stage_key = (
        "bias" if control["kind"] == "diagnostic_defect_bias"
        else "measurement"
    )
    manifest_root = (
        Path.home() / ".single_shot/runs" / args.run_id
        / "screen_diagnostic/stages" / stage_key / "node_manifests"
        / args.control_sha256[:12] / args.node
    )
    raw_manifest = {
        "raw_manifest_version": RAW_MANIFEST_VERSION,
        "contract_version": CONTRACT_VERSION,
        "node": args.node,
        "stage": control["stage"],
        "kind": control["kind"],
        "stage_fingerprint": args.stage_fingerprint,
        "source_commit": args.source_commit,
        "registry_sha256": registry["registry_sha256"],
        "diagnostic_config_sha256": config_sha256(config),
        "control_sha256": args.control_sha256,
        "ownership_sha256": args.ownership_sha256,
        "schedule_file_sha256": args.schedule_file_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "runtime_report_sha256": args.runtime_report_sha256,
        "source_identity": source_identity,
        "files": raw_files,
    }
    raw_manifest_path = manifest_root / "raw_manifest.json"
    atomic_json(raw_manifest_path, raw_manifest)
    completed = time.time()
    if completed > float(schedule["deadlines_unix"][stage_key]):
        raise TimeoutError("diagnostic worker completed after stage deadline")
    atomic_json(manifest_root / "stage_status.json", {
        "status": "SUCCESS", "node": args.node,
        "stage_fingerprint": args.stage_fingerprint,
        "expected": len(work), **counts,
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "completed_unix": completed,
    })
    print(json.dumps({
        "node": args.node, "expected": len(work), **counts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
