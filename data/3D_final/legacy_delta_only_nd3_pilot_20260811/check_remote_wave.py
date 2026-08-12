#!/usr/bin/env python3
"""Read-only, fail-closed status check for one exact nd-3 pilot wave."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import PurePosixPath
from typing import Any


REMOTE_BASE = PurePosixPath("/home/DATA1/users/yuany/.single_shot")
EXPECTED_SOURCE_SHA256 = (
    "428e8383fc5ff7a9f31529f5b604c4a2fadf7302d7779367482e39af773114eb"
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


REMOTE_CHECK = r'''
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = {"root": str(root), "exists": root.is_dir()}
if not root.is_dir():
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)

def read_json(name):
    path = root / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

try:
    result["manifest"] = read_json("run_manifest.json")
    result["status"] = read_json("status.json")
    result["success"] = read_json("SUCCESS.json")
except Exception as exc:
    result["read_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)

result["failure_files"] = sorted(path.name for path in root.glob("failure_*.json"))
status = result.get("status")
pid = status.get("pid") if isinstance(status, dict) else None
if isinstance(pid, int) and pid > 1:
    proc = Path("/proc") / str(pid)
    result["pid"] = pid
    result["pid_alive"] = proc.is_dir()
    if proc.is_dir():
        try:
            result["pid_owned_by_current_user"] = proc.stat().st_uid == os.getuid()
            result["pid_cmdline"] = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except Exception as exc:
            result["proc_read_error"] = f"{type(exc).__name__}: {exc}"

success = result.get("success")
if isinstance(success, dict) and isinstance(success.get("cells"), dict):
    hashes = {}
    for relative in sorted(success["cells"]):
        path = root / relative
        if not path.is_file():
            hashes[relative] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[relative] = digest.hexdigest()
    result["actual_cell_hashes"] = hashes
print(json.dumps(result, sort_keys=True))
'''


def parse_pairs(text: str) -> list[list[float]]:
    result: list[list[float]] = []
    for entry in text.split(","):
        p_text, q_text = entry.split(":", 1)
        result.append([float(p_text), float(q_text)])
    if not result:
        raise ValueError("at least one expected pair is required")
    return result


def fetch_remote(wave_root: PurePosixPath, timeout: int) -> dict[str, Any]:
    remote_command = " ".join(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20",
            "nd-3",
            "python3", "-", shlex.quote(str(wave_root)),
        ]
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "yuany", remote_command],
        input=REMOTE_CHECK,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().replace("\n", " ")[-300:]
        raise RuntimeError(f"ssh return code {completed.returncode}: {detail}")
    return json.loads(completed.stdout)


def expected_cells(pairs: list[list[float]]) -> set[str]:
    return {
        f"cells/p{round(p_value * 1000):04d}_q{round(q_value * 1000):04d}_L{lattice_size:02d}.npz"
        for p_value, q_value in pairs
        for lattice_size in (3, 7)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--expected-pairs", required=True)
    parser.add_argument("--expected-seed-base", required=True, type=int)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if not SAFE_NAME.fullmatch(args.run_id) or not SAFE_NAME.fullmatch(args.wave_id):
        raise SystemExit("unsafe run or wave ID")
    pairs = parse_pairs(args.expected_pairs)
    wave_root = REMOTE_BASE / "runs" / args.run_id / "nd3" / "waves" / args.wave_id
    try:
        remote = fetch_remote(wave_root, args.timeout)
    except Exception as exc:  # Network/checker errors are unknown, never success.
        print(f"UNKNOWN:remote_check:{type(exc).__name__}:{exc}")
        return 0

    if remote.get("read_error"):
        print(f"UNKNOWN:malformed_remote_json:{remote['read_error']}")
        return 0
    if not remote.get("exists"):
        print("UNKNOWN:wave_root_missing")
        return 0

    manifest = remote.get("manifest")
    status = remote.get("status")
    success = remote.get("success")
    if not isinstance(manifest, dict) or not isinstance(status, dict):
        print("UNKNOWN:manifest_or_status_missing")
        return 0

    identity_checks = {
        "model": manifest.get("model") == "legacy_delta_only",
        "source": manifest.get("source_sha256") == EXPECTED_SOURCE_SHA256,
        "pairs": manifest.get("pairs") == pairs,
        "lattice_sizes": manifest.get("lattice_sizes") == [3, 7],
        "num_disorder_samples": manifest.get("num_disorder_samples") == 48,
        "seed_base": manifest.get("seed_base") == args.expected_seed_base,
        "workers": isinstance(manifest.get("workers"), int) and 1 <= manifest["workers"] <= 70,
        "config_hash": bool(SHA256.fullmatch(str(manifest.get("config_hash", "")))),
    }
    bad_identity = sorted(name for name, passed in identity_checks.items() if not passed)
    if bad_identity:
        print(f"FAILED:identity_mismatch:{','.join(bad_identity)}")
        return 0

    failure_files = remote.get("failure_files")
    if failure_files or status.get("state") == "failed":
        print(f"FAILED:remote_failure:{failure_files or 'status_failed'}")
        return 0

    total_tasks = len(pairs) * 2 * 48
    if isinstance(success, dict):
        cells = success.get("cells")
        actual_hashes = remote.get("actual_cell_hashes")
        success_checks = {
            "success_state": success.get("state") == "success",
            "status_matches": status == success,
            "total_tasks": success.get("total_tasks") == total_tasks,
            "config_hash": success.get("config_hash") == manifest["config_hash"],
            "cell_names": isinstance(cells, dict) and set(cells) == expected_cells(pairs),
            "declared_hashes": isinstance(cells, dict)
            and all(SHA256.fullmatch(str(value)) for value in cells.values()),
            "actual_hashes": isinstance(cells, dict) and actual_hashes == cells,
        }
        bad_success = sorted(name for name, passed in success_checks.items() if not passed)
        if bad_success:
            print(f"FAILED:invalid_success_marker:{','.join(bad_success)}")
            return 0
        print("SUCCESS")
        return 0

    if status.get("state") not in {"running", "merging"}:
        print(f"UNKNOWN:unexpected_status:{status.get('state')}")
        return 0
    if status.get("total_tasks") != total_tasks:
        print("FAILED:running_total_task_mismatch")
        return 0
    if not remote.get("pid_alive"):
        print("FAILED:process_missing_without_terminal_marker")
        return 0
    if not remote.get("pid_owned_by_current_user"):
        print("FAILED:pid_owner_mismatch")
        return 0
    if str(wave_root) not in str(remote.get("pid_cmdline", "")):
        print("FAILED:pid_identity_mismatch")
        return 0
    print(
        "WAITING:"
        f"state={status['state']}:completed={status.get('completed_tasks', 'unknown')}:"
        f"pending={status.get('pending_tasks', 'unknown')}:pid={remote.get('pid')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
