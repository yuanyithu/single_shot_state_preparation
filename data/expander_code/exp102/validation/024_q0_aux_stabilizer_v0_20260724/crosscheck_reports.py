"""Reconcile the frozen-run, replay, and raw-only UASRE reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json


VERSION = "exp102.q0_hgp_aux_stabilizer_pt.crosscheck.v1"
ROOT = Path(__file__).resolve().parent


def _load(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _verify_hash(report, name):
    key = {"REPORT.json": "report_sha256", "INDEPENDENT_AUDIT.json": "audit_sha256", "REPLAY.json": "replay_sha256"}[name]
    core = {field: value for field, value in report.items() if field != key}
    _require(report.get(key) == sha256_json(core), f"{name} hash mismatch")


def crosscheck(root):
    root = Path(root)
    output = root / "CROSSCHECK.json"
    _require(not output.exists(), "crosscheck output already exists")
    report = _load(root / "REPORT.json")
    audit = _load(root / "INDEPENDENT_AUDIT.json")
    replay = _load(root / "REPLAY.json")
    _verify_hash(report, "REPORT.json")
    _verify_hash(audit, "INDEPENDENT_AUDIT.json")
    _verify_hash(replay, "REPLAY.json")
    _require(report["manifest_sha256"] == audit["manifest_sha256"] == replay["manifest_sha256"],
             "manifest mismatch across reports")
    _require(report["status"] == audit["status"], "terminal status mismatch")
    _require(report["raw_sha256"] == audit["raw_sha256"] == replay["raw_sha256"],
             "raw hash mismatch across reports")
    _require(replay["all_bit_identical"] is True and replay["task_count"] == 48,
             "replay is incomplete")
    diagnostics = {}
    runner_methods = {}
    for method, summary in report["methods"].items():
        summary = dict(summary)
        diagnostics[method] = summary.pop("nonconstant_B_mask_diagnostic")
        runner_methods[method] = summary
        _require(set(diagnostics[method]) == {"P", "U", "L"}
                 and all(isinstance(value, int) and 0 <= value <= 64
                         for value in diagnostics[method].values()),
                 "runner diagnostic has an invalid range")
    _require(runner_methods == audit["methods"], "pre-registered gate summaries disagree")
    core = {
        "crosscheck_version": VERSION, "manifest_sha256": report["manifest_sha256"],
        "status": report["status"], "raw_sha256": report["raw_sha256"],
        "report_sha256": report["report_sha256"], "audit_sha256": audit["audit_sha256"],
        "replay_sha256": replay["replay_sha256"], "replay_all_bit_identical": True,
        "pre_registered_gate_summaries_equal": True,
        "runner_only_non_gate_diagnostic": diagnostics,
        "source_sha256": sha256_file(Path(__file__)),
    }
    result = {**core, "crosscheck_sha256": sha256_json(core)}
    atomic_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "local_hard_viability")
    args = parser.parse_args(argv)
    print(crosscheck(args.root)["crosscheck_sha256"])


if __name__ == "__main__":
    main()
