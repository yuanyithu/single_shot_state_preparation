#!/usr/bin/env python3
"""Read-only status checker for the nd-2 staged 2D experiment."""

import argparse
import json
import re
import shlex
import subprocess
import sys


REMOTE_BASE = "/home/DATA1/users/yuany/.single_shot"


REMOTE_CHECK = r'''
import json
import pathlib
import subprocess
import sys

stage = pathlib.Path(sys.argv[1])
screen_name = sys.argv[2]
phase_path = stage / "control/phase"
phase = phase_path.read_text().strip() if phase_path.is_file() else None
screen_result = subprocess.run(
    ["screen", "-ls"], text=True, capture_output=True, check=False
)
screen_text = (screen_result.stdout or "") + (screen_result.stderr or "")
manifests = {}
for path in sorted(stage.glob("runs/q0_*/manifest.json")):
    try:
        data = json.loads(path.read_text())
        manifests[path.parent.name] = data.get("summary", {})
    except Exception as exc:
        manifests[path.parent.name] = {"error": str(exc)}
parts = {}
for arm in ("A", "B"):
    part_dir = stage / "runs/qpositive_sentinel/parts" / arm
    parts[arm] = len(list(part_dir.glob("disorder_*.npz"))) if part_dir.is_dir() else 0
result = {
    "stage_exists": stage.is_dir(),
    "phase": phase,
    "stage_done": (stage / "control/stage_done").is_file(),
    "pilot_audit": (stage / "pilot_audit.json").is_file(),
    "formal_audit": (stage / "formal_audit.json").is_file(),
    "screen_present": screen_name in screen_text,
    "manifests": manifests,
    "qpositive_parts": parts,
}
print(json.dumps(result, sort_keys=True))
'''


def query(run_id):
    stage = f"{REMOTE_BASE}/runs/{run_id}"
    screen_name = f"ssprep_{run_id}"
    remote_command = "python -c {} {} {}".format(
        shlex.quote(REMOTE_CHECK), shlex.quote(stage), shlex.quote(screen_name)
    )
    relay_command = "ssh -o BatchMode=yes nd-2 {}".format(
        shlex.quote(remote_command)
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "yuany", relay_command],
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        print(f"UNKNOWN ssh_rc={completed.returncode} detail={detail[-300:]}")
        return 1
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        print("UNKNOWN no_json_status")
        return 1
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        print(f"UNKNOWN malformed_json={exc}")
        return 1


def classify(status, milestone):
    if not isinstance(status, dict):
        return 1
    phase = status.get("phase")
    if not status.get("stage_exists"):
        print("FAILED:stage_missing")
        return 2
    if isinstance(phase, str) and phase.startswith("done_rc_"):
        if phase == "done_rc_0" and status.get("formal_audit"):
            print("SUCCESS stage_complete")
            return 0
        print(f"FAILED:{phase}")
        return 2
    if not status.get("screen_present"):
        print("FAILED:screen_missing_without_done_marker")
        return 2
    if milestone == "stage-complete":
        print(f"WAITING phase={phase}")
        return 1
    if milestone == "pilot-complete":
        if status.get("pilot_audit") or phase == "formal":
            print(f"SUCCESS pilot_complete phase={phase}")
            return 0
        print(f"WAITING phase={phase}")
        return 1
    if phase != "pilot":
        print(f"SUCCESS phase_changed phase={phase}")
        return 0
    completed_chunks = 0
    for summary in status.get("manifests", {}).values():
        if isinstance(summary, dict):
            completed_chunks += int(summary.get("completed_chunks", 0) or 0)
    completed_parts = sum(
        int(value) for value in status.get("qpositive_parts", {}).values()
    )
    if completed_chunks > 0 or completed_parts > 0:
        print(
            f"SUCCESS first_progress chunks={completed_chunks} "
            f"parts={completed_parts}"
        )
        return 0
    print("WAITING phase=pilot chunks=0 parts=0")
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--milestone",
        choices=("first-progress", "pilot-complete", "stage-complete"),
        default="stage-complete",
    )
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id) is None:
        raise SystemExit("invalid run id")
    status = query(args.run_id)
    return classify(status, args.milestone)


if __name__ == "__main__":
    sys.exit(main())
