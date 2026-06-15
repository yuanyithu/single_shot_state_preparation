#!/usr/bin/env python3
"""Collect exp38 P2 remote TI shard directories into the local data tree."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def collect_one(host: str, remote_run_root: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parent = str(Path(remote_run_root).parent)
    name = Path(remote_run_root).name
    remote_cmd = f"tar -cf - -C {parent!r} {name!r}"
    ssh_proc = subprocess.Popen(
        ["ssh", "yuany", f"ssh {host} {remote_cmd!r}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar_proc = subprocess.Popen(
        ["tar", "-xf", "-", "-C", str(output_dir)],
        stdin=ssh_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ssh_proc.stdout is not None:
        ssh_proc.stdout.close()
    tar_stdout, tar_stderr = tar_proc.communicate()
    ssh_stderr = ssh_proc.stderr.read() if ssh_proc.stderr is not None else b""
    ssh_rc = ssh_proc.wait()
    if ssh_rc != 0 or tar_proc.returncode != 0:
        raise RuntimeError(
            "collect failed for "
            f"{host}:{remote_run_root}; ssh_rc={ssh_rc}, "
            f"tar_rc={tar_proc.returncode}, "
            f"ssh_stderr={ssh_stderr.decode(errors='replace')[-1000:]}, "
            f"tar_stderr={tar_stderr.decode(errors='replace')[-1000:]}"
        )
    del tar_stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for shard in payload.get("shards", []):
        collect_one(
            host=str(shard["host"]),
            remote_run_root=str(shard["run_root"]),
            output_dir=args.output_dir,
        )
        copied.append({
            "host": shard["host"],
            "lattice_size": shard["lattice_size"],
            "local_dir": str(args.output_dir / Path(shard["run_root"]).name),
        })
    (args.output_dir / "collected_manifest.json").write_text(
        json.dumps({"source_manifest": str(args.manifest), "copied": copied}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"copied": copied}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
