import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.io import atomic_json
from data.expander_code.exp102.exp102_pipeline.pilot_cell import run_cell
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


RUN_ID = "exp102_pilot_20260719_70191fb"
COMMIT = "70191fb"
CANDIDATE = {"p_hot": 0.45, "num_temperatures": 8, "gamma": 1.0,
             "burn_rounds": 500, "measurement_rounds": 2000,
             "sweeps_per_round": 1, "logical_move_repeat": 1}
CAPACITY = {"nd-2": 75, "nd-3": 91}


def ownership(codes):
    load = {node: 0.0 for node in CAPACITY}
    result = {}
    for code in sorted(codes, key=lambda item: (-item["n"], item["code_id"])):
        node = min(CAPACITY, key=lambda value: (load[value], value))
        result[code["code_id"]] = node
        load[node] += code["n"] / CAPACITY[node]
    return result, load


def execute(task):
    registry, config, output, code_id, p, disorder = task
    return run_cell(registry, config, code_id, p, disorder, CANDIDATE, 0,
                    "ladder", COMMIT, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(CAPACITY)); parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()
    if args.num_workers != CAPACITY[args.node]:
        raise ValueError("worker count differs from frozen deployment manifest")
    source = Path.cwd()
    registry_path = source / "data/expander_code/exp102/registry/registry.json"
    config_path = source / "data/expander_code/exp102/config/production.v1.json"
    registry = load_registry(registry_path)
    assigned, loads = ownership(registry["codes"])
    run_root = Path.home() / ".single_shot/runs" / RUN_ID
    output_root = run_root / "ladder" / args.node
    tasks = []
    for code in registry["codes"]:
        if assigned[code["code_id"]] != args.node:
            continue
        for p in (0.04, 0.07, 0.10):
            for disorder in range(4):
                output = output_root / code["code_id"] / f"p{p:.2f}_d{disorder:02d}.npz"
                tasks.append((str(registry_path), str(config_path), str(output),
                              code["code_id"], p, disorder))
    atomic_json(run_root / "deployment_manifest.json", {
        "run_id": RUN_ID, "source_commit": COMMIT, "candidate": CANDIDATE,
        "capacity": CAPACITY, "normalized_load": loads, "code_owner": assigned,
    })
    counts = {"computed": 0, "reused": 0}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for status in pool.map(execute, tasks):
            counts[status] += 1
    atomic_json(output_root / "stage_status.json", {"status": "SUCCESS", "node": args.node,
                                                     "expected": len(tasks), **counts})
    print(json.dumps({"node": args.node, "expected": len(tasks), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
