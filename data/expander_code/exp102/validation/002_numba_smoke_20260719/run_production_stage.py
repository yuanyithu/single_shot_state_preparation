"""Run one fail-closed exp102 production shard from a frozen deployment manifest."""

import argparse
import concurrent.futures
from importlib import import_module
import json
import os
from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json, sha256_file, sha256_json, verify_source_identity,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.tasks import task_records
from data.expander_code.exp102.exp102_pipeline.worker import run_task


CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
build_manifest = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.build_production_deployment"
).build_manifest


def execute(task):
    return run_task(*task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(CAPACITY))
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--pilot-report", required=True)
    parser.add_argument("--task-plan", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    args = parser.parse_args()
    if args.num_workers != CAPACITY[args.node]:
        raise ValueError("worker count differs from frozen production capacity")

    registry = load_registry(args.registry)
    config = load_config(args.config)
    frozen = json.loads(Path(args.frozen).read_text(encoding="ascii"))
    plan = json.loads(Path(args.task_plan).read_text(encoding="ascii"))
    deployment = json.loads(Path(args.deployment_manifest).read_text(encoding="ascii"))
    frozen_hash = sha256_json(frozen)
    source_identity = verify_source_identity(Path.cwd(), frozen.get("source_commit", ""))
    expected_deployment = build_manifest(
        args.registry, args.config, args.frozen, args.pilot_report,
    )
    if deployment != expected_deployment:
        raise ValueError("deployment manifest differs from recomputed held-out evidence")
    expected = {
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "frozen_config_sha256": frozen_hash,
        "source_commit": frozen["source_commit"],
    }
    if frozen.get("status") != "FROZEN_HELD_OUT_PASS" or frozen.get("engine") != "numba":
        raise ValueError("production requires a held-out certified Numba freezer")
    if plan.get("status") != "PRODUCTION" or plan.get("num_tasks") != 6144:
        raise ValueError("production task plan must contain exactly 6144 tasks")
    if (plan.get("registry_sha256") != registry["registry_sha256"]
            or plan.get("config_sha256") != config["config_sha256"]
            or plan.get("tasks") != task_records(registry, config, frozen)):
        raise ValueError("production task plan identity or coverage mismatch")
    if deployment.get("capacity") != CAPACITY or set(deployment.get("code_owner", {})) != {
        code["code_id"] for code in registry["codes"]
    }:
        raise ValueError("deployment manifest is incomplete")
    if set(deployment["code_owner"].values()) - set(CAPACITY):
        raise ValueError("deployment manifest contains an unknown node")
    if deployment.get("selected_cell_count") != 2688:
        raise ValueError("deployment manifest lacks complete held-out timing evidence")
    if deployment.get("pilot_report_sha256") != frozen.get("pilot_report_sha256"):
        raise ValueError("deployment pilot report identity mismatch")
    if deployment.get("raw_evidence_sha256") != frozen.get("raw_evidence_sha256"):
        raise ValueError("deployment raw evidence identity mismatch")
    if deployment.get("selected_attempt_by_m") != frozen.get("held_out_attempt_by_m"):
        raise ValueError("deployment held-out attempt identity mismatch")
    if not isinstance(deployment.get("selected_held_out_evidence_sha256"), str):
        raise ValueError("deployment held-out evidence digest is missing")
    for key, value in expected.items():
        if deployment.get(key) != value:
            raise ValueError(f"deployment identity mismatch: {key}")

    os.environ["EXP102_FROZEN_VERIFIED_SHA256"] = frozen_hash
    os.environ["EXP102_SOURCE_VERIFIED_COMMIT"] = frozen["source_commit"]

    output_root = Path(args.run_root) / "raw" / "production"
    tasks = []
    for record in plan["tasks"]:
        if deployment["code_owner"][record["code_id"]] != args.node:
            continue
        output = output_root / record["code_id"] / f"d{record['disorder_index']:03d}.npz"
        tasks.append((args.registry, args.config, args.frozen, record["code_id"],
                      record["disorder_index"], output, "production"))
    counts = {"computed": 0, "reused": 0}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for status in pool.map(execute, tasks):
            counts[status] += 1
    raw_manifest_path = output_root / "_manifests" / f"{args.node}.json"
    raw_files = []
    for task in tasks:
        path = Path(task[5])
        raw_files.append({
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
        })
    atomic_json(raw_manifest_path, {
        "raw_manifest_version": "exp102.production.raw.v1",
        "node": args.node,
        **expected,
        "files": sorted(raw_files, key=lambda item: item["path"]),
    })
    status = {"status": "SUCCESS", "node": args.node, "expected": len(tasks), **counts,
              **expected, "source_identity": source_identity,
              "task_plan_sha256": sha256_file(args.task_plan),
              "deployment_manifest_sha256": sha256_file(args.deployment_manifest),
              "pilot_report_sha256_file": sha256_file(args.pilot_report),
              "raw_manifest_sha256": sha256_file(raw_manifest_path)}
    atomic_json(Path(args.run_root) / "status" / f"production_{args.node}.json", status)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
