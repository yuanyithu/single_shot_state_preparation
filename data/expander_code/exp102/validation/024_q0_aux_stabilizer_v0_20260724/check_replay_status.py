"""Read-only terminal-status checker for the UASRE parallel replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import sha256_json


ROOT = Path(__file__).resolve().parent


def _load(path):
    return json.loads(path.read_text(encoding="ascii"))


def check(root):
    root = Path(root)
    report_path = root / "REPLAY.json"
    failed_path = root / "REPLAY_FAILED.json"
    if report_path.is_file() and failed_path.is_file():
        return "CHECK_ERROR: conflicting replay markers"
    if failed_path.is_file():
        return f"FAILED: {_load(failed_path).get('failure', 'unknown replay failure')}"
    if not report_path.is_file():
        return "WAITING"
    report = _load(report_path)
    core = {name: value for name, value in report.items() if name != "replay_sha256"}
    if report.get("replay_sha256") != sha256_json(core):
        return "CHECK_ERROR: replay hash"
    if report.get("all_bit_identical") is not True or report.get("task_count") != 48:
        return "CHECK_ERROR: replay result"
    return "SUCCESS"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "local_hard_viability")
    args = parser.parse_args(argv)
    print(check(args.root))


if __name__ == "__main__":
    main()
