"""Run one fail-closed exp102 production shard from a frozen deployment manifest."""

import argparse
import concurrent.futures
import json
from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.worker import run_task


CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}


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
    if deployment.get("capacity") != CAPACITY or set(deployment.get("code_owner", {})) != {
        code["code_id"] for code in registry["codes"]
    }:
        raise ValueError("deployment manifest is incomplete")
    for key, value in expected.items():
        if deployment.get(key) != value:
            raise ValueError(f"deployment identity mismatch: {key}")

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
    status = {"status": "SUCCESS", "node": args.node, "expected": len(tasks), **counts,
              **expected}
    atomic_json(Path(args.run_root) / "status" / f"production_{args.node}.json", status)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
