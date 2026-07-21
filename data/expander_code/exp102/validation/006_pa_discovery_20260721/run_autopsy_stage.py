"""Run the four frozen PT transport-autopsy tasks on assigned nodes."""

import argparse
import concurrent.futures
import json
from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
    verify_source_identity,
)
from data.expander_code.exp102.exp102_pipeline.transport_autopsy import (
    AUTOPSY_NODE_CAPACITY,
    AUTOPSY_RAW_VERSION,
    AUTOPSY_TASKS_VERSION,
    PARENT_RUN_ID,
    fixed_autopsy_ownership,
    run_autopsy_task,
)


def _execute(arguments):
    return run_autopsy_task(*arguments)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(AUTOPSY_NODE_CAPACITY))
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--ownership", required=True)
    parser.add_argument("--ownership-sha256", required=True)
    parser.add_argument("--stage-fingerprint", required=True)
    args = parser.parse_args(argv)
    if args.num_workers != AUTOPSY_NODE_CAPACITY[args.node]:
        raise ValueError("autopsy worker count differs from frozen node capacity")
    source = Path.cwd().resolve()
    source_identity = verify_source_identity(source, args.source_commit)
    control_path = Path(args.control).resolve(strict=True)
    ownership_path = Path(args.ownership).resolve(strict=True)
    if sha256_file(control_path) != args.control_sha256:
        raise ValueError("autopsy control SHA mismatch")
    if sha256_file(ownership_path) != args.ownership_sha256:
        raise ValueError("autopsy ownership SHA mismatch")
    control = json.loads(control_path.read_text(encoding="ascii"))
    ownership = json.loads(ownership_path.read_text(encoding="ascii"))
    if control.get("manifest_version") != AUTOPSY_TASKS_VERSION:
        raise ValueError("wrong autopsy task manifest version")
    expected_ownership = fixed_autopsy_ownership(
        control["tasks"], ownership.get("nodes", []), args.source_commit,
        args.control_sha256,
    )
    if ownership != expected_ownership:
        raise ValueError("autopsy ownership is not canonical")
    if (args.node not in ownership["nodes"]
            or ownership["stage_fingerprint"] != args.stage_fingerprint):
        raise ValueError("autopsy node/stage identity mismatch")
    task_by_fingerprint = {sha256_json(task): task for task in control["tasks"]}
    if len(task_by_fingerprint) != len(control["tasks"]):
        raise ValueError("autopsy control has duplicate tasks")
    selected = [
        (fingerprint, task_by_fingerprint[fingerprint])
        for fingerprint in sorted(task_by_fingerprint)
        if ownership["task_owner"][fingerprint] == args.node
    ]
    registry = source / "data/expander_code/exp102/registry/registry.json"
    discovery_config = source / "data/expander_code/exp102/config/discovery.v2.json"
    autopsy_config = source / "data/expander_code/exp102/config/transport_autopsy.v1.json"
    parent_root = Path.home() / ".single_shot/runs" / PARENT_RUN_ID
    if not parent_root.is_dir():
        raise ValueError("frozen autopsy parent run is absent from shared storage")
    run_root = Path.home() / ".single_shot/runs" / args.run_id
    output_root = run_root / "transport_autopsy" / args.control_sha256[:12] / args.node
    work = []
    output_by_fingerprint = {}
    for fingerprint, task in selected:
        cell = task["cell"]
        output = output_root / task["ladder_id"] / cell["code_id"] / "trace.npz"
        output_by_fingerprint[fingerprint] = output
        work.append((
            str(registry), str(discovery_config), str(autopsy_config),
            args.source_commit, task, str(parent_root), str(output),
        ))
    counts = {"computed": 0, "reused": 0}
    if work:
        # Populate the shared Numba cache before any child process opens it.
        counts[_execute(work[0])] += 1
        remaining = work[1:]
        if remaining:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.num_workers, len(remaining)),
            ) as pool:
                for status in pool.map(_execute, remaining):
                    counts[status] += 1
    raw_files = []
    for fingerprint, output in sorted(output_by_fingerprint.items()):
        if not output.is_file():
            raise ValueError(f"autopsy worker did not produce {output}")
        raw_files.append({
            "task_fingerprint": fingerprint,
            "path": output.relative_to(output_root).as_posix(),
            "sha256": sha256_file(output),
        })
    raw_manifest = {
        "raw_manifest_version": AUTOPSY_RAW_VERSION,
        "node": args.node,
        "stage": "transport_autopsy",
        "stage_fingerprint": args.stage_fingerprint,
        "source_commit": args.source_commit,
        "control_sha256": args.control_sha256,
        "ownership_sha256": args.ownership_sha256,
        "source_identity": source_identity,
        "files": raw_files,
    }
    manifest_path = output_root / "raw_manifest.json"
    atomic_json(manifest_path, raw_manifest)
    atomic_json(output_root / "stage_status.json", {
        "status": "SUCCESS", "node": args.node,
        "stage_fingerprint": args.stage_fingerprint,
        "expected": len(work), **counts,
        "raw_manifest_sha256": sha256_file(manifest_path),
    })
    print(json.dumps({"node": args.node, "expected": len(work), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
