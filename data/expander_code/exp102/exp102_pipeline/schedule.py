"""Assign whole codes to nodes with longest-processing-time balancing."""

import argparse
import json
from pathlib import Path

from .io import atomic_json


NODE_WORKERS = {"nd-1": 75, "nd-2": 75, "nd-3": 91}


def lpt_assign(core_seconds_by_code):
    loads = {node: 0.0 for node in NODE_WORKERS}
    assignments = {node: [] for node in NODE_WORKERS}
    for code_id, core_seconds in sorted(core_seconds_by_code.items(), key=lambda item: (-item[1], item[0])):
        node = min(loads, key=lambda name: (loads[name] / NODE_WORKERS[name], name))
        assignments[node].append(code_id)
        loads[node] += float(core_seconds)
    return {"workers": NODE_WORKERS, "assignments": assignments, "core_seconds": loads,
            "estimated_wall_seconds": {node: loads[node] / NODE_WORKERS[node] for node in loads}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("pilot_times"); parser.add_argument("output")
    args = parser.parse_args(argv)
    times = json.loads(Path(args.pilot_times).read_text(encoding="ascii"))
    result = lpt_assign(times["core_seconds_by_code"]); atomic_json(args.output, result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
