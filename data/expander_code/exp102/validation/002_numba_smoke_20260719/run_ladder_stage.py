import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import atomic_json
from data.expander_code.exp102.exp102_pipeline.pilot_cell import run_cell
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


RUN_ID = "exp102_pilot_20260719_70191fb"
COMMIT = "70191fb"
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
    registry, config, output, code_id, p, disorder, candidate, attempt, stage = task
    return run_cell(registry, config, code_id, p, disorder, candidate, attempt,
                    stage, COMMIT, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(CAPACITY)); parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--stage", choices=("ladder", "gamma", "rounds", "held_out"), default="ladder")
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--p-hot", type=float)
    parser.add_argument("--num-temperatures", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--burn-rounds", type=int)
    parser.add_argument("--measurement-rounds", type=int)
    parser.add_argument("--by-m-config", help="JSON mapping m to a complete PT config")
    parser.add_argument("--m-values", default="3,4,5,6,7,8")
    args = parser.parse_args()
    if args.num_workers != CAPACITY[args.node]:
        raise ValueError("worker count differs from frozen deployment manifest")
    source = Path.cwd()
    registry_path = source / "data/expander_code/exp102/registry/registry.json"
    config_path = source / "data/expander_code/exp102/config/production.v1.json"
    selected_m = {int(value) for value in args.m_values.split(",")}
    if not selected_m or not selected_m <= set(range(3, 9)):
        raise ValueError("m-values must be a nonempty subset of 3..8")
    if args.by_m_config:
        raw_by_m = json.loads(Path(args.by_m_config).read_text(encoding="ascii"))
        candidates = {int(m): value for m, value in raw_by_m.items()}
        if set(candidates) != selected_m:
            raise ValueError("by-m config keys must exactly match m-values")
    else:
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
    assigned, loads = ownership(registry["codes"])
    run_root = Path.home() / ".single_shot/runs" / RUN_ID
    output_root = run_root / args.stage / f"attempt_{args.attempt:03d}" / args.node
    tasks = []
    for code in registry["codes"]:
        if assigned[code["code_id"]] != args.node or code["m"] not in selected_m:
            continue
        p_values = config_data["p_values"] if args.stage == "held_out" else (0.04, 0.07, 0.10)
        num_disorders = 8 if args.stage == "held_out" else 4
        for p in p_values:
            for disorder in range(num_disorders):
                output = output_root / code["code_id"] / f"p{p:.2f}_d{disorder:02d}.npz"
                tasks.append((str(registry_path), str(config_path), str(output),
                              code["code_id"], p, disorder, candidates[code["m"]],
                              args.attempt, args.stage))
    atomic_json(run_root / f"deployment_manifest_{args.stage}_{args.attempt:03d}.json", {
        "run_id": RUN_ID, "source_commit": COMMIT,
        "by_m_config": {str(m): candidates[m] for m in sorted(candidates)},
        "stage": args.stage, "attempt": args.attempt,
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
