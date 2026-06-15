#!/usr/bin/env python3
"""Read-only checker for exp38 P3 remote second-method subset run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_remote(host: str, command: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "yuany", f"ssh {host} bash -s"],
        input=command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def check_shard(shard: dict, timeout: int) -> dict:
    host = str(shard["host"])
    run_root = str(shard["run_root"])
    screen_name = str(shard["screen_name"])
    command = (
        "set -e; "
        f"if test -f {run_root}/_FAILED.json; then echo FAILED; cat {run_root}/_FAILED.json; "
        f"elif test -f {run_root}/_SUCCESS.json; then echo SUCCESS; cat {run_root}/_SUCCESS.json; "
        f"elif screen -ls | grep -q '[.]{screen_name}[[:space:]]'; then echo RUNNING; "
        "else echo WAITING_NO_SCREEN; fi"
    )
    try:
        proc = run_remote(host, command, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {
            "host": host,
            "state": "CHECK_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    output = (proc.stdout or "") + (proc.stderr or "")
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    if proc.returncode != 0:
        return {"host": host, "state": "CHECK_ERROR", "detail": output[-1000:]}
    if first_line == "SUCCESS":
        state = "SUCCESS"
    elif first_line == "FAILED":
        state = "FAILED"
    elif first_line == "RUNNING":
        state = "RUNNING"
    else:
        state = "WAITING"
    return {"host": host, "state": state, "detail": output[-1000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    statuses = [
        check_shard(shard, timeout=int(args.timeout_seconds))
        for shard in payload.get("shards", [])
    ]
    if args.json:
        print(json.dumps({"statuses": statuses}, indent=2, sort_keys=True))
    failed = [status for status in statuses if status["state"] == "FAILED"]
    errors = [status for status in statuses if status["state"] == "CHECK_ERROR"]
    success = [status for status in statuses if status["state"] == "SUCCESS"]
    if failed:
        print("FAILED: " + json.dumps(failed, sort_keys=True))
        return 2
    if errors:
        print("CHECK_ERROR: " + json.dumps(errors, sort_keys=True))
        return 3
    if len(success) == len(statuses) and statuses:
        print("SUCCESS")
        return 0
    print("WAITING: " + json.dumps(statuses, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
