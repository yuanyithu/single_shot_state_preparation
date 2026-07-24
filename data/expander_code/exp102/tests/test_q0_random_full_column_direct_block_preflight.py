"""Contract and fail-closed aggregate tests for validation 054."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

from data.expander_code.exp102.exp102_pipeline.io import (
    canonical_json,
    sha256_file,
    sha256_json,
)


RUNNER = importlib.import_module(
    "data.expander_code.exp102.validation."
    "054_q0_random_full_column_direct_block_preflight_20260724.run_preflight"
)
COMBINER = importlib.import_module(
    "data.expander_code.exp102.validation."
    "054_q0_random_full_column_direct_block_preflight_20260724.combine_preflights"
)


def _write_json(path, value):
    Path(path).write_text(canonical_json(value) + "\n", encoding="ascii")


def _source_identity(commit):
    core = {"files": {"synthetic": "0" * 64}, "source_commit": commit}
    return {**core, "source_identity_sha256": sha256_json(core)}


def _node_report(node, commit, *, transcript_suffix="", runtime_pass=True):
    config = RUNNER.load_canonical(RUNNER.CONFIG_PATH)
    reference = RUNNER.load_portable_reference(config)
    probes = [
        {
            "block_subtotals_sha256": row["block_subtotals_sha256"],
            "column": row["column"],
            "state": row["state"],
        }
        for row in reference["block_subtotal_catalog"]
    ]
    runtime = [dict(row) for row in reference["runtime_transcript_catalog"]]
    if transcript_suffix:
        runtime[0]["transcript_sha256"] = transcript_suffix * 64
    core = {
        "checks": {
            "full_m8_weight_identity": True,
            "local_minimum_streaming_speedup": True,
            "portable_reference": not transcript_suffix,
            "runtime_projection": runtime_pass,
        },
        "config_sha256": sha256_file(RUNNER.CONFIG_PATH),
        "correctness": {"all_pass": True, "probes": probes},
        "focused_tests": {},
        "node": node,
        "portable_reference_sha256": reference["reference_sha256"],
        "report_version": RUNNER.REPORT_VERSION,
        "runtime": runtime,
        "source_identity": _source_identity(commit),
        "status": (
            "DIRECT_BLOCK_PREFLIGHT_NODE_PASS"
            if runtime_pass and not transcript_suffix
            else ("CONFLICT" if transcript_suffix else "RUNTIME_EXHAUSTED")
        ),
        "timing": {},
        "worst_projected_replay_inclusive_t1_seconds": (
            1000.0 if runtime_pass else 8000.0
        ),
    }
    return {**core, "report_sha256": sha256_json(core)}


def _run_combiner(tmp_path, monkeypatch, reports):
    commit = "1" * 40
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    for node, report in reports.items():
        _write_json(preflight / f"{node}.json", report)
    output = preflight / "aggregate.json"
    monkeypatch.setenv("EXP102_SOURCE_COMMIT", commit)
    monkeypatch.setattr(sys, "argv", [
        "combine_preflights.py", "--run-root", str(tmp_path),
        "--source-commit", commit, "--output", str(output),
    ])
    COMBINER.main()
    return json.loads(output.read_text(encoding="ascii"))


def test_config_and_frozen_portable_reference_are_self_consistent():
    config = RUNNER.load_canonical(RUNNER.CONFIG_PATH)
    RUNNER.validate_config(config)
    reference = RUNNER.load_portable_reference(config)
    assert len(reference["block_subtotal_catalog"]) == 12
    assert len(reference["runtime_transcript_catalog"]) == 4
    assert config["runtime_seed_key"] == (
        "679dde13a0e6ea3058d56435964013c63df520eb5da39f04ed2feab06da6eecc"
    )


def test_combiner_passes_only_exact_consensus(tmp_path, monkeypatch):
    commit = "1" * 40
    reports = {
        node: _node_report(node, commit) for node in COMBINER.NODES
    }
    aggregate = _run_combiner(tmp_path, monkeypatch, reports)
    assert aggregate["status"] == "PASS"
    assert aggregate["exact_consensus"] is True


def test_combiner_fails_closed_on_transcript_conflict(tmp_path, monkeypatch):
    commit = "1" * 40
    reports = {
        node: _node_report(
            node, commit, transcript_suffix=("f" if node == "nd-3" else ""),
        )
        for node in COMBINER.NODES
    }
    aggregate = _run_combiner(tmp_path, monkeypatch, reports)
    assert aggregate["status"] == "CONFLICT"
    assert aggregate["exact_consensus"] is False


def test_combiner_persists_runtime_exhaustion(tmp_path, monkeypatch):
    commit = "1" * 40
    reports = {
        node: _node_report(node, commit, runtime_pass=(node != "nd-2"))
        for node in COMBINER.NODES
    }
    aggregate = _run_combiner(tmp_path, monkeypatch, reports)
    assert aggregate["status"] == "RUNTIME_EXHAUSTED"
    assert aggregate["exact_consensus"] is True


def test_combiner_rejects_tampered_source_identity(tmp_path, monkeypatch):
    commit = "1" * 40
    reports = {
        node: _node_report(node, commit) for node in COMBINER.NODES
    }
    reports["nd-1"]["source_identity"]["source_identity_sha256"] = "f" * 64
    core = {key: value for key, value in reports["nd-1"].items()
            if key != "report_sha256"}
    reports["nd-1"]["report_sha256"] = sha256_json(core)
    with pytest.raises(COMBINER.CombineError, match="source identity self-hash"):
        _run_combiner(tmp_path, monkeypatch, reports)
