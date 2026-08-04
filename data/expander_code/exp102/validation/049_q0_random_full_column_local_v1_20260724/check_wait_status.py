"""Lightweight terminal-report checker for validation 049."""

import json
from pathlib import Path


report = Path(__file__).resolve().parent / "transport_report.json"
allowed = {
    "LOCAL_RANDOM_FULL_COLUMN_RUNTIME_EXHAUSTED",
    "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_UNRESOLVED",
    "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_VIABLE",
}
if not report.exists():
    print("WAITING")
else:
    try:
        value = json.loads(report.read_text(encoding="ascii"))
        if (value.get("contract_version")
                != "exp102.q0_random_full_column.local.v1"
                or value.get("status") not in allowed
                or not isinstance(value.get("report_sha256"), str)
                or len(value["report_sha256"]) != 64):
            raise ValueError("terminal report identity is invalid")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
    else:
        print("SUCCESS")
