"""Read-only completion checker for the frozen UASRE local viability run."""

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
    failed = root / "FAILED.json"
    complete = root / "RUN_COMPLETE.json"
    if failed.is_file() and complete.is_file():
        return "CHECK_ERROR: conflicting terminal markers"
    if failed.is_file():
        payload = _load(failed)
        return f"FAILED: {payload.get('failure', 'unknown failure')}"
    if not complete.is_file():
        return "WAITING"
    manifest = _load(root / "MANIFEST.json")
    payload = _load(complete)
    required = {"runner_version", "manifest_sha256", "raw_count", "raw", "run_sha256"}
    if set(payload) != required:
        return "CHECK_ERROR: completion schema"
    core = {name: value for name, value in payload.items() if name != "run_sha256"}
    if payload["run_sha256"] != sha256_json(core):
        return "CHECK_ERROR: completion hash"
    if payload["manifest_sha256"] != manifest.get("manifest_sha256"):
        return "CHECK_ERROR: completion manifest"
    if payload["raw_count"] != len(manifest.get("tasks", [])):
        return "CHECK_ERROR: completion count"
    return "SUCCESS"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "local_hard_viability")
    args = parser.parse_args(argv)
    print(check(args.root))


if __name__ == "__main__":
    main()
