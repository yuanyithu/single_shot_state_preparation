"""Run one immutable discovery task manifest on its preassigned node."""

import argparse
import concurrent.futures
import json
from pathlib import Path

from data.expander_code.exp102.exp102_pipeline.discovery import (
    DISCOVERY_RAW_VERSION,
    run_discovery_cell,
)
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
    verify_source_identity,
)


NODE_CAPACITY = {"nd-1": 75, "nd-2": 75, "nd-3": 91}


def _execute(arguments):
    return run_discovery_cell(*arguments)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(NODE_CAPACITY))
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--ownership", required=True)
    parser.add_argument("--ownership-sha256", required=True)
    parser.add_argument("--stage-fingerprint", required=True)
    args = parser.parse_args(argv)
    if args.num_workers != NODE_CAPACITY[args.node]:
        raise ValueError("worker count differs from the frozen node capacity")
    source = Path.cwd().resolve()
    source_identity = verify_source_identity(source, args.source_commit)
    control_path = Path(args.control).resolve(strict=True)
    ownership_path = Path(args.ownership).resolve(strict=True)
    if sha256_file(control_path) != args.control_sha256:
        raise ValueError("discovery control SHA256 mismatch")
    if sha256_file(ownership_path) != args.ownership_sha256:
        raise ValueError("discovery ownership SHA256 mismatch")
    control = json.loads(control_path.read_text(encoding="ascii"))
    ownership = json.loads(ownership_path.read_text(encoding="ascii"))
    if control.get("manifest_version") != "exp102.discovery.tasks.v2":
        raise ValueError("wrong discovery task manifest version")
    if ownership.get("ownership_version") != "exp102.discovery.ownership.v2":
        raise ValueError("wrong discovery ownership version")
    if (ownership.get("control_sha256") != args.control_sha256
            or ownership.get("source_commit") != args.source_commit
            or ownership.get("stage") != control.get("stage")
            or ownership.get("stage_fingerprint") != args.stage_fingerprint):
        raise ValueError("discovery ownership identity mismatch")
    if args.node not in ownership.get("nodes", []):
        raise ValueError("node is outside the frozen discovery ownership")
    tasks = control.get("tasks")
    task_by_fingerprint = {sha256_json(task): task for task in tasks}
    if len(task_by_fingerprint) != len(tasks):
        raise ValueError("discovery control contains duplicate tasks")
    assigned = ownership.get("task_owner")
    if set(assigned) != set(task_by_fingerprint):
        raise ValueError("discovery ownership does not cover the exact control tasks")
    selected = [
        (fingerprint, task_by_fingerprint[fingerprint])
        for fingerprint in sorted(task_by_fingerprint)
        if assigned[fingerprint] == args.node
    ]

    registry_path = source / "data/expander_code/exp102/registry/registry.json"
    config_path = source / "data/expander_code/exp102/config/discovery.v2.json"
    run_root = Path.home() / ".single_shot/runs" / args.run_id
    output_root = run_root / control["stage"] / args.control_sha256[:12] / args.node
    work = []
    output_by_fingerprint = {}
    for fingerprint, task in selected:
        candidate = task["candidate"]
        cell = task["cell"]
        candidate_dir = (
            f"{candidate['ladder_id']}_S{candidate['swap_sweeps_per_round']}_"
            f"b{candidate['burn_rounds']}_m{candidate['measurement_rounds']}"
        )
        output = output_root / candidate_dir / cell["code_id"] / (
            f"p{cell['p']:.2f}_d{cell['disorder_index']:02d}.npz"
        )
        output_by_fingerprint[fingerprint] = output
        work.append((
            str(registry_path), str(config_path), args.source_commit, task, str(output),
        ))
    counts = {"computed": 0, "reused": 0}
    if work:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.num_workers, len(work)),
        ) as pool:
            for status in pool.map(_execute, work):
                counts[status] += 1
    raw_files = []
    for fingerprint, output in sorted(output_by_fingerprint.items()):
        if not output.is_file():
            raise ValueError(f"discovery worker did not produce {output}")
        raw_files.append({
            "task_fingerprint": fingerprint,
            "path": output.relative_to(output_root).as_posix(),
            "sha256": sha256_file(output),
        })
    raw_manifest = {
        "raw_manifest_version": DISCOVERY_RAW_VERSION,
        "node": args.node,
        "stage": control["stage"],
        "stage_fingerprint": args.stage_fingerprint,
        "source_commit": args.source_commit,
        "control_sha256": args.control_sha256,
        "ownership_sha256": args.ownership_sha256,
        "source_identity": source_identity,
        "files": raw_files,
    }
    raw_manifest_path = output_root / "raw_manifest.json"
    atomic_json(raw_manifest_path, raw_manifest)
    atomic_json(output_root / "stage_status.json", {
        "status": "SUCCESS",
        "node": args.node,
        "stage_fingerprint": args.stage_fingerprint,
        "expected": len(work),
        **counts,
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
    })
    print(json.dumps({"node": args.node, "expected": len(work), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
