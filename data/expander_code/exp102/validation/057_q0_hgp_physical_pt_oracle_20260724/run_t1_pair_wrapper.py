"""Write exclusive success/failure markers around the long local T1 pair."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import traceback


ROOT = Path(__file__).resolve().parent
SUCCESS = ROOT / "T1_PAIR_SUCCESS.json"
FAILURE = ROOT / "T1_PAIR_FAILED.json"


def _write_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def main():
    if SUCCESS.exists() or FAILURE.exists():
        raise RuntimeError("T1 pair terminal marker already exists")
    module = importlib.import_module(
        "data.expander_code.exp102.validation."
        "057_q0_hgp_physical_pt_oracle_20260724.run_m8_t1_pair"
    )
    try:
        module.main()
        report = json.loads(module.OUTPUT.read_text(encoding="utf-8"))
        _write_exclusive(SUCCESS, {
            "report_sha256": report["report_sha256"],
            "source_commit": report["source_commit"],
            "status": report["status"],
        })
    except BaseException as exc:
        _write_exclusive(FAILURE, {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
