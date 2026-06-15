#!/usr/bin/env python3
"""Read-only status checker for exp38 P0 regression anchor."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
RESULT_PATH = RUN_DIR / "stageD_results.json"
PROCESS_NEEDLE = "run_stageD_sector_ti.py --output-dir"


def p0_process_running() -> bool:
    proc = subprocess.run(
        ["ps", "-ax", "-o", "command="],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"UNKNOWN: ps failed: {proc.stderr.strip()}")
        return True
    output_arg = str(RUN_DIR)
    for line in proc.stdout.splitlines():
        if PROCESS_NEEDLE in line and output_arg in line:
            return True
    return False


def main() -> int:
    if RESULT_PATH.exists():
        try:
            payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            if p0_process_running():
                print(f"WAITING: result JSON incomplete: {exc}")
                return 0
            print(f"FAILED: result JSON malformed and process stopped: {exc}")
            return 0
        passed = bool(payload.get("overall_passed"))
        print(f"SUCCESS: overall_passed={passed}")
        return 0
    if p0_process_running():
        print("WAITING: P0 runner still active")
        return 0
    print("FAILED: P0 runner stopped without stageD_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
