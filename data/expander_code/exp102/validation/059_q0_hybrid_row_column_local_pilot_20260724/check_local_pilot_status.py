"""Read-only terminal checker for the validation-059 local pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--command-fragment", required=True)
    args = parser.parse_args()
    report_path = args.run_root / "pilot_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="ascii"))
            expected = report.pop("report_sha256")
            actual = hashlib.sha256(canonical(report).encode("ascii")).hexdigest()
            valid_status = report.get("status") in {
                "LOCAL_HYBRID_B_NECESSARY_GATES_PASS",
                "LOCAL_HYBRID_B_NECESSARY_GATES_FAIL",
            }
            if actual == expected and valid_status and report.get("raw_count") == 16:
                print("SUCCESS")
                return
            print("FAILED:malformed_terminal_report")
            return
        except Exception as exc:  # noqa: BLE001 - checker must fail closed.
            print(f"FAILED:terminal_report_error:{type(exc).__name__}")
            return
    process = subprocess.run(
        ["ps", "-p", str(args.pid), "-o", "command="],
        capture_output=True, text=True, check=False,
    )
    command = process.stdout.strip()
    if process.returncode == 0 and args.command_fragment in command:
        raw_count = len(list((args.run_root / "raw").glob("*.npz")))
        print(f"WAITING:{raw_count}/16")
        return
    print("FAILED:pilot_process_exited_without_terminal_report")


if __name__ == "__main__":
    main()
