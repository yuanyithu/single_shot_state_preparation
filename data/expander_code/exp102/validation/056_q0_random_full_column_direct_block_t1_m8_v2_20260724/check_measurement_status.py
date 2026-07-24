"""Read-only status check for the immutable validation-056 measurement."""

from __future__ import annotations

import subprocess
import sys


RUN_ID = "exp102_q0_direct_block_t1_m8_v2_20260724_6933e31"
NODES = ("nd-1", "nd-2", "nd-3")


def main():
    remote = "\n".join((
        f'run="$HOME/.single_shot/runs/{RUN_ID}"',
        "failed=",
        "waiting=",
        *(f'if test -f "$run/stages/measurement-{node}/FAILED"; then '
          f'failed="$failed {node}"; '
          f'elif test ! -f "$run/stages/measurement-{node}/SUCCESS"; then '
          f'waiting="$waiting {node}"; fi' for node in NODES),
        'if test -n "$failed"; then echo "FAILED:$failed"; '
        'elif test -n "$waiting"; then echo "WAITING:$waiting"; '
        'else echo SUCCESS; fi',
    ))
    try:
        completed = subprocess.run(
            ["ssh", "yuany", remote],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"CHECK_ERROR:{type(exc).__name__}")
        return 1
    stdout = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not stdout:
        print(f"CHECK_ERROR:ssh_exit_{completed.returncode}")
        return 1
    status = stdout[-1]
    if status == "SUCCESS" or status.startswith(("FAILED:", "WAITING:")):
        print(status)
        return 0
    print("CHECK_ERROR:malformed_status")
    return 1


if __name__ == "__main__":
    sys.exit(main())
