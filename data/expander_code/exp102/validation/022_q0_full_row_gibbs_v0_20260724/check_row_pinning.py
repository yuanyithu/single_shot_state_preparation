"""Read-only completion checker for the exact row-pinning diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.is_file() and args.report.stat().st_size:
        try:
            report = json.loads(args.report.read_text(encoding="ascii"))
            if (report.get("diagnostic_version")
                    != "exp102.q0_hgp_full_row_gibbs.row_pinning.v0"
                    or report.get("trajectory_raw_produced") is not False
                    or len(report.get("families", [])) != 3
                    or not isinstance(report.get("report_sha256"), str)):
                raise ValueError("report identity is incomplete")
        except Exception as exc:
            print(f"FAILED: malformed pinning report: {exc}")
            return
        print("SUCCESS")
        return
    try:
        os.kill(args.pid, 0)
    except ProcessLookupError:
        print("FAILED: pinning process exited without a complete report")
        return
    except PermissionError:
        print("UNKNOWN: cannot inspect pinning process")
        return
    print("WAITING")


if __name__ == "__main__":
    main()
