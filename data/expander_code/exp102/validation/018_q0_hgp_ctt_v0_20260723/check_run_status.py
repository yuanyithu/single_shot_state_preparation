#!/usr/bin/env python3
"""Read-only status checker for the frozen local CTT V0 run."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent / "local_m8_transport_ctt_v0"
MANIFEST_SHA256 = "f77add0a8b1825b117ac49ed85b3a3a138045cb233bed43fb691cac9bd31ff85"


def load_marker(name):
    try:
        return json.loads((ROOT / name).read_text(encoding="ascii"))
    except Exception as exc:
        print(f"FAILED: cannot read {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(0)


def valid_binding(marker):
    return marker.get("manifest_sha256") == MANIFEST_SHA256


def main():
    failed = ROOT / "FAILED.json"
    success = ROOT / "SUCCESS.json"
    running = ROOT / "RUNNING.json"
    if failed.exists():
        marker = load_marker("FAILED.json")
        if success.exists() or not valid_binding(marker):
            print("FAILED: terminal marker conflict")
        else:
            print(f"FAILED: {marker.get('error', 'worker failure')}")
        return
    if success.exists():
        marker = load_marker("SUCCESS.json")
        if not valid_binding(marker) or marker.get("task_count") != 24:
            print("FAILED: malformed SUCCESS marker")
        else:
            print("SUCCESS")
        return
    if running.exists():
        marker = load_marker("RUNNING.json")
        if valid_binding(marker):
            print("WAITING")
        else:
            print("FAILED: RUNNING marker has a foreign manifest")
        return
    print("UNKNOWN")


if __name__ == "__main__":
    main()
