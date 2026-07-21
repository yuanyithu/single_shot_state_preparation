"""Execute one immutable PA task manifest on its frozen owner node."""

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
from data.expander_code.exp102.exp102_pipeline.pa_discovery import (
    PA_NODE_CAPACITY,
    PA_RAW_VERSION,
    PA_TASKS_VERSION,
    fixed_pa_ownership,
    run_pa_task,
)


def _execute(arguments):
    return run_pa_task(*arguments)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("node", choices=tuple(PA_NODE_CAPACITY))
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--ownership", required=True)
    parser.add_argument("--ownership-sha256", required=True)
    parser.add_argument("--stage-fingerprint", required=True)
    args = parser.parse_args(argv)
    if args.num_workers != PA_NODE_CAPACITY[args.node]:
        raise ValueError("PA worker count differs from frozen node capacity")
    source = Path.cwd().resolve()
    source_identity = verify_source_identity(source, args.source_commit)
    control_path = Path(args.control).resolve(strict=True)
    ownership_path = Path(args.ownership).resolve(strict=True)
    if sha256_file(control_path) != args.control_sha256:
        raise ValueError("PA control SHA256 mismatch")
    if sha256_file(ownership_path) != args.ownership_sha256:
        raise ValueError("PA ownership SHA256 mismatch")
    control = json.loads(control_path.read_text(encoding="ascii"))
    ownership = json.loads(ownership_path.read_text(encoding="ascii"))
    if control.get("manifest_version") != PA_TASKS_VERSION:
        raise ValueError("wrong PA task manifest version")
    expected_ownership = fixed_pa_ownership(
        control["tasks"], ownership.get("nodes", []), args.source_commit,
        args.control_sha256, control["stage"],
    )
    if ownership != expected_ownership:
        raise ValueError("PA ownership is not the frozen canonical assignment")
    if (args.node not in ownership["nodes"]
            or ownership["stage_fingerprint"] != args.stage_fingerprint):
        raise ValueError("PA node/stage fingerprint differs from ownership")
    task_by_fingerprint = {sha256_json(task): task for task in control["tasks"]}
    if len(task_by_fingerprint) != len(control["tasks"]):
        raise ValueError("PA control contains duplicate tasks")
    selected = [
        (fingerprint, task_by_fingerprint[fingerprint])
        for fingerprint in sorted(task_by_fingerprint)
        if ownership["task_owner"][fingerprint] == args.node
    ]

    registry_path = source / "data/expander_code/exp102/registry/registry.json"
    config_path = source / "data/expander_code/exp102/config/q0_pa.discovery.v1.json"
    run_root = Path.home() / ".single_shot/runs" / args.run_id
    output_root = run_root / control["stage"] / args.control_sha256[:12] / args.node
    work = []
    output_by_fingerprint = {}
    for fingerprint, task in selected:
        pa_config = task["pa_config"]
        cell = task["cell"]
        method_dir = (
            f"{task['method_id']}_N{pa_config['num_particles']}_"
            f"G{pa_config['num_anneal_steps']}_s{pa_config['rejuvenation_sweeps']}"
        )
        output = output_root / method_dir / cell["code_id"] / (
            f"p{cell['p']:.2f}_d{cell['disorder_index']:02d}_"
            f"pop{task['population_index']:02d}.npz"
        )
        output_by_fingerprint[fingerprint] = output
        work.append((
            str(registry_path), str(config_path), args.source_commit, task, str(output),
        ))
    counts = {"computed": 0, "reused": 0}
    # Compile the Numba mutation kernel once before child processes open the
    # shared cache; the remaining tasks can then fan out without a cache race.
    if work:
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
            raise ValueError(f"PA worker did not produce {output}")
        raw_files.append({
            "task_fingerprint": fingerprint,
            "path": output.relative_to(output_root).as_posix(),
            "sha256": sha256_file(output),
        })
    raw_manifest = {
        "raw_manifest_version": PA_RAW_VERSION,
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
