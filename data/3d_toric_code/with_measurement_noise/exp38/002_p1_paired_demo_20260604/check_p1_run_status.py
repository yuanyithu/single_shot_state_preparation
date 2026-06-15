#!/usr/bin/env python3
"""Read-only status checker for exp38 P1 local runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIRS = {
    "strong": SCRIPT_DIR / "strong_l35_q018_d8",
    "l4": SCRIPT_DIR / "strong_l4_q018_d1_walltime",
    "p1b": SCRIPT_DIR / "p1b_q020_021_d12",
}


def _process_running(run_dir: Path) -> bool:
    proc = subprocess.run(
        ["ps", "-ax", "-o", "command="],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"UNKNOWN: ps failed: {proc.stderr.strip()}")
        return True
    run_dir_text = str(run_dir)
    for line in proc.stdout.splitlines():
        if "exp37_sector_ti.py run" in line and run_dir_text in line:
            return True
        if "run_p1_" in line and run_dir.name in line:
            return True
    return False


def _status_one(label: str) -> tuple[str, str]:
    run_dir = RUN_DIRS[label]
    success_path = run_dir / "_SUCCESS.json"
    failed_path = run_dir / "_FAILED.json"
    result_path = run_dir / "sector_ti_results.npz"
    summary_path = run_dir / "sector_ti_summary.md"
    if success_path.exists():
        try:
            payload = json.loads(success_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return "FAILED", f"{label}: malformed success JSON: {exc}"
        if result_path.exists() and summary_path.exists():
            return "SUCCESS", f"{label}: success exit={payload.get('exit_code')}"
        return "WAITING", f"{label}: success marker present, artifacts not complete"
    if failed_path.exists():
        try:
            payload = json.loads(failed_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return "FAILED", f"{label}: malformed failed JSON: {exc}"
        return "FAILED", f"{label}: failed exit={payload.get('exit_code')}"
    if _process_running(run_dir):
        return "WAITING", f"{label}: process active"
    if result_path.exists() and summary_path.exists():
        return "SUCCESS", f"{label}: artifacts present without marker"
    return "WAITING", f"{label}: not started"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("strong", "l4", "p1b", "all"), required=True)
    args = parser.parse_args()
    labels = ("strong", "l4", "p1b") if args.target == "all" else (args.target,)
    statuses = [_status_one(label) for label in labels]
    if any(status == "FAILED" for status, _ in statuses):
        print("FAILED: " + "; ".join(message for _, message in statuses))
    elif all(status == "SUCCESS" for status, _ in statuses):
        print("SUCCESS: " + "; ".join(message for _, message in statuses))
    else:
        print("WAITING: " + "; ".join(message for _, message in statuses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
