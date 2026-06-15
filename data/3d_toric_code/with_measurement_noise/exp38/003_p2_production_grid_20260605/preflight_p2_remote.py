#!/usr/bin/env python3
"""Remote preflight for exp38 P2 production grid."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "preflight_summary.json"


def _run(host: str, command: str, timeout: int) -> dict:
    proc = subprocess.run(
        ["ssh", "yuany", f"ssh {host} bash -s"],
        input=command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _check_host(host: str, timeout: int) -> dict:
    command = (
        "set -euo pipefail; "
        "hostname; "
        "printf 'nproc='; nproc; "
        "command -v screen >/dev/null; "
        "command -v conda >/dev/null; "
        "export CONDA_NO_PLUGINS=true; "
        "conda run --no-capture-output -n 11 python -c "
        "\"import importlib.util, numpy; "
        "print('python_env_ok=1'); "
        "print('numba_available=' + str(importlib.util.find_spec('numba') is not None)); "
        "import math; "
        "print('stdlib_ok=1')\""
    )
    # Keep the imported module check separate so the remote repo is not required
    # during pure environment preflight.
    result = _run(host, command, timeout)
    output = result["stdout"] + result["stderr"]
    return {
        "host": host,
        "passed": (
            result["returncode"] == 0
            and "python_env_ok=1" in output
            and "numba_available=True" in output
        ),
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", default="nd-1,nd-2,nd-3")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    hosts = [item.strip() for item in args.hosts.split(",") if item.strip()]
    checks = [_check_host(host, timeout=int(args.timeout_seconds)) for host in hosts]
    payload = {
        "stage": "P2",
        "preflight": "remote_conda11_numba_screen",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(checks and all(item["passed"] for item in checks)),
        "checks": checks,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
