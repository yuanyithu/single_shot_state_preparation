"""Classify the exact validation-057 T1 pair terminal markers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUCCESS = ROOT / "T1_PAIR_SUCCESS.json"
FAILURE = ROOT / "T1_PAIR_FAILED.json"
REPORT = ROOT / "m8_t1_pair_report.json"


def main():
    if SUCCESS.exists() and FAILURE.exists():
        print("FAILED: conflicting terminal markers")
        return
    if FAILURE.exists():
        payload = json.loads(FAILURE.read_text(encoding="utf-8"))
        print(f"FAILED: {payload.get('exception_type', 'unknown')}")
        return
    if not SUCCESS.exists():
        print("WAITING")
        return
    marker = json.loads(SUCCESS.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    claimed = report.pop("report_sha256")
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if (
        claimed != actual
        or marker.get("report_sha256") != claimed
        or marker.get("source_commit") != report.get("source_commit")
        or marker.get("status") != report.get("status")
    ):
        print("FAILED: terminal report identity mismatch")
        return
    print("SUCCESS")


if __name__ == "__main__":
    main()
