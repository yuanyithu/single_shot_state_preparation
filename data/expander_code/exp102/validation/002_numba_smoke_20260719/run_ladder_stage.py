import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re

from data.expander_code.exp102.exp102_pipeline.config import load_config, validate_pilot_candidate
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, verify_source_identity
from data.expander_code.exp102.exp102_pipeline.pilot_cell import pilot_task_identity, run_cell
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


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
    (registry, config, output, code_id, p, disorder, candidate, attempt, stage,
     source_commit) = task
    return run_cell(registry, config, code_id, p, disorder, candidate, attempt,
                    stage, source_commit, output)


def build_manifest_tasks(registry, config, candidates, selected_m, stage,
                         attempt, source_commit):
    tasks = []
    for code in registry["codes"]:
        if code["m"] not in selected_m:
            continue
        p_values = config["p_values"] if stage == "held_out" else (0.04, 0.07, 0.10)
        num_disorders = 8 if stage == "held_out" else 4
        for p in p_values:
            for disorder in range(num_disorders):
                tasks.append(pilot_task_identity(
                    registry["registry_sha256"], config["config_sha256"],
                    code["code_id"], code["m"], p, disorder,
                    candidates[code["m"]], attempt, stage, source_commit,
                ))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(CAPACITY)); parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stage", choices=("ladder", "gamma", "rounds", "held_out"), default="ladder")
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--p-hot", type=float)
    parser.add_argument("--num-temperatures", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--burn-rounds", type=int)
    parser.add_argument("--measurement-rounds", type=int)
    parser.add_argument("--by-m-config", help="JSON mapping m to a complete PT config")
    parser.add_argument("--by-m-config-sha256")
    parser.add_argument("--m-values", default="3,4,5,6,7,8")
    args = parser.parse_args()
    if args.num_workers != CAPACITY[args.node]:
        raise ValueError("worker count differs from frozen deployment manifest")
    source = Path.cwd()
    source_identity = verify_source_identity(source, args.source_commit)
    registry_path = source / "data/expander_code/exp102/registry/registry.json"
    config_path = source / "data/expander_code/exp102/config/production.v1.json"
    selected_m = {int(value) for value in args.m_values.split(",")}
    if not selected_m or not selected_m <= set(range(3, 9)):
        raise ValueError("m-values must be a nonempty subset of 3..8")
    if args.by_m_config:
        if (re.fullmatch(r"[0-9a-f]{64}", str(args.by_m_config_sha256)) is None
                or sha256_file(args.by_m_config) != args.by_m_config_sha256):
            raise ValueError("by-m config SHA256 is missing or mismatched")
        raw_by_m = json.loads(Path(args.by_m_config).read_text(encoding="ascii"))
        candidates = {int(m): value for m, value in raw_by_m.items()}
        if set(candidates) != selected_m:
            raise ValueError("by-m config keys must exactly match m-values")
    else:
        if args.by_m_config_sha256 is not None:
            raise ValueError("by-m config SHA256 was provided without a config")
        fields = (args.p_hot, args.num_temperatures, args.gamma,
                  args.burn_rounds, args.measurement_rounds)
        if any(value is None for value in fields):
            raise ValueError("scalar PT parameters are required without --by-m-config")
        candidate = {"p_hot": args.p_hot, "num_temperatures": args.num_temperatures,
                     "gamma": args.gamma, "burn_rounds": args.burn_rounds,
                     "measurement_rounds": args.measurement_rounds,
                     "sweeps_per_round": 1, "logical_move_repeat": 1}
        candidates = {m: candidate for m in selected_m}
    registry = load_registry(registry_path)
    config_data = load_config(config_path)
    candidates = {m: validate_pilot_candidate(candidate, config_data)
                  for m, candidate in candidates.items()}
    assigned, loads = ownership(registry["codes"])
    run_root = Path.home() / ".single_shot/runs" / args.run_id
    output_root = run_root / args.stage / f"attempt_{args.attempt:03d}" / args.node
    tasks = []
    manifest_tasks = build_manifest_tasks(
        registry, config_data, candidates, selected_m, args.stage,
        args.attempt, args.source_commit,
    )
    for code in registry["codes"]:
        if code["m"] not in selected_m or assigned[code["code_id"]] != args.node:
            continue
        p_values = config_data["p_values"] if args.stage == "held_out" else (0.04, 0.07, 0.10)
        num_disorders = 8 if args.stage == "held_out" else 4
        for p in p_values:
            for disorder in range(num_disorders):
                output = output_root / code["code_id"] / f"p{p:.2f}_d{disorder:02d}.npz"
                tasks.append((str(registry_path), str(config_path), str(output),
                              code["code_id"], p, disorder, candidates[code["m"]],
                              args.attempt, args.stage, args.source_commit))
    atomic_json(run_root / f"deployment_manifest_{args.stage}_{args.attempt:03d}.json", {
        "run_id": args.run_id, "source_commit": args.source_commit,
        "by_m_config": {str(m): candidates[m] for m in sorted(candidates)},
        "stage": args.stage, "attempt": args.attempt,
        "capacity": CAPACITY, "normalized_load": loads, "code_owner": assigned,
        "source_identity": source_identity,
        "by_m_config_sha256": args.by_m_config_sha256,
        "tasks": manifest_tasks,
    })
    counts = {"computed": 0, "reused": 0}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for status in pool.map(execute, tasks):
            counts[status] += 1
    raw_manifest_path = output_root / "raw_manifest.json"
    raw_files = []
    for task in tasks:
        path = Path(task[2])
        raw_files.append({
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
        })
    atomic_json(raw_manifest_path, {
        "raw_manifest_version": "exp102.pilot.raw.v1",
        "node": args.node, "stage": args.stage, "attempt": args.attempt,
        "source_commit": args.source_commit,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config_data["config_sha256"],
        "files": sorted(raw_files, key=lambda item: item["path"]),
    })
    atomic_json(output_root / "stage_status.json", {
        "status": "SUCCESS", "node": args.node, "expected": len(tasks), **counts,
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
    })
    print(json.dumps({"node": args.node, "expected": len(tasks), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
