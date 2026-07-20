"""Build deterministic three-node production ownership from held-out timings."""

import argparse
import json
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registry"); parser.add_argument("config"); parser.add_argument("frozen")
    parser.add_argument("held_out_raw"); parser.add_argument("output")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    config = load_config(args.config)
    frozen = json.loads(Path(args.frozen).read_text(encoding="ascii"))
    if frozen.get("status") != "FROZEN_HELD_OUT_PASS":
        raise ValueError("held-out freezer is required")
    raw_timings = {code["code_id"]: [] for code in registry["codes"]}
    for path in Path(args.held_out_raw).rglob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            code_id = str(data["code_id"].item())
            if code_id not in raw_timings:
                raise ValueError(f"unknown held-out code ID in {path}")
            if not bool(data["valid"].item()):
                raise ValueError(f"invalid held-out timing cell for {code_id}")
            raw_timings[code_id].append(float(data["core_seconds"].item()))
    timings = {}
    for code in registry["codes"]:
        values = raw_timings[code["code_id"]]
        if len(values) != 56:
            raise ValueError(f"held-out timing coverage for {code['code_id']} is {len(values)}/56")
        timings[code["code_id"]] = float(np.median(values))
    load = {node: 0.0 for node in CAPACITY}
    owners = {}
    for code_id, seconds in sorted(timings.items(), key=lambda item: (-item[1], item[0])):
        node = min(CAPACITY, key=lambda name: (load[name], name))
        owners[code_id] = node
        load[node] += 128 * seconds / CAPACITY[node]
    manifest = {
        "deployment_version": "exp102.production.deployment.v1",
        "capacity": CAPACITY, "code_owner": owners, "estimated_node_seconds": load,
        "held_out_median_core_seconds": timings,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "frozen_config_sha256": sha256_json(frozen),
        "source_commit": frozen["source_commit"],
    }
    atomic_json(args.output, manifest)
    print(json.dumps(load, sort_keys=True))


if __name__ == "__main__":
    main()
