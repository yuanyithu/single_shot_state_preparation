import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .exp101_bridge import load_exp101
from .io import atomic_json, sha256_json
from .registry import load_registry
from .seeds import derive_seed


def task_records(registry, config, frozen=None):
    records = []
    for code in registry["codes"]:
        for disorder in range(config["num_disorders"]):
            identity = {"code_id": code["code_id"], "disorder_index": disorder,
                        "registry_sha256": registry["registry_sha256"],
                        "config_sha256": config["config_sha256"], "namespace": "production"}
            if frozen is not None:
                identity["frozen_config_sha256"] = sha256_json(frozen)
            records.append(dict(identity, task_fingerprint=sha256_json(identity)))
    return records


def write_task_plan(registry_path, config_path, output_path, frozen_path=None):
    registry = load_registry(registry_path)
    config = load_config(config_path)
    if frozen_path is None:
        raise ValueError("draft production task plans are forbidden; provide a held-out freezer")
    frozen = json.loads(Path(frozen_path).read_text(encoding="ascii"))
    if frozen.get("status") != "FROZEN_HELD_OUT_PASS":
        raise ValueError("cannot create production task plan from an uncertified PT config")
    if frozen.get("registry_sha256") != registry["registry_sha256"] or frozen.get("config_sha256") != config["config_sha256"]:
        raise ValueError("frozen/task-plan identity mismatch")
    tasks = task_records(registry, config, frozen)
    if len(tasks) != 6144 or len({row["task_fingerprint"] for row in tasks}) != 6144:
        raise AssertionError("task plan is not exactly 6144 unique units")
    manifest = {"task_plan_version": "exp102.tasks.v1", "status": "PRODUCTION",
                "num_tasks": len(tasks),
                "registry_sha256": registry["registry_sha256"],
                "config_sha256": config["config_sha256"], "tasks": tasks}
    atomic_json(output_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--frozen")
    args = parser.parse_args(argv)
    result = write_task_plan(args.registry, args.config, args.output, args.frozen)
    print(result["num_tasks"])


if __name__ == "__main__":
    main()
