"""Lightweight completion checker for the validation-048 blocking wait."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "transport_report.json"
ALLOWED = {
    "LOCAL_RANDOM_FULL_COLUMN_RUNTIME_EXHAUSTED",
    "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_UNRESOLVED",
    "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_VIABLE",
}


if not REPORT.exists():
    print("WAITING")
else:
    try:
        value = json.loads(REPORT.read_text(encoding="ascii"))
        if (value.get("contract_version")
                != "exp102.q0_random_full_column.local.v0"
                or value.get("status") not in ALLOWED
                or not isinstance(value.get("report_sha256"), str)
                or len(value["report_sha256"]) != 64):
            raise ValueError("terminal report identity is invalid")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
    else:
        print("SUCCESS")
