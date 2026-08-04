"""Frozen one-basis counterfactual for real-low-energy Houdayer components."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_houdayer import (
    HOUDAYER_REDUCED_LOGICAL_COORDINATE_VERSION,
    build_sparse_hgp_reduced_logical_coordinate_basis,
)


PROBE_VERSION = "exp102.q0_houdayer_reduced_logicals.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
BASE_PROBE_PATH = (
    EXP102_ROOT / "validation" / "038_q0_houdayer_real_logicals_feasibility_20260724"
    / "probe_houdayer_real_logicals.py"
)


class ReducedLogicalHoudayerError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ReducedLogicalHoudayerError(message)


def _base_probe():
    spec = importlib.util.spec_from_file_location("houdayer_real_logicals_base", BASE_PROBE_PATH)
    _require(spec is not None and spec.loader is not None,
             "cannot load the frozen real-logical Houdayer schedule evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ReducedLogicalHoudayerError("reduced-logical Houdayer config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "reduced-logical Houdayer config is not canonical")
    expected_keys = {
        "catalog", "cell", "config_version", "contract_version", "coordinate_basis",
        "exhaustive_component_limit", "pair_families", "registry_sha256", "scope", "version",
    }
    _require(set(config) == expected_keys and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"]
             == "exp102.q0_houdayer_reduced_logicals.feasibility.config.v0",
             "reduced-logical Houdayer config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "reduced-logical Houdayer cell changed")
    _require(config["coordinate_basis"]
             == "h_x_plus_canonical_reduced_logical_complement.v0",
             "reduced-logical Houdayer coordinate basis changed")
    _require(config["catalog"] == {
        "candidate_orders": [1, 2, 3],
        "deduplication": "one_minimum_p_derived_state_per_nonzero_logical_signature",
        "low_energy_label_count": 16,
        "rank_complete_label_count": 64,
        "selection_order": "state_weight,move_weight,signature,packed_move",
    }, "reduced-logical Houdayer catalog definition changed")
    _require(config["pair_families"] == [
        "P_vs_each_low_energy", "all_pairs_within_low_energy", "P_vs_each_rank_complete",
    ], "reduced-logical Houdayer pair schedule changed")
    _require(config["exhaustive_component_limit"] == 12,
             "reduced-logical Houdayer exhaustive component cap changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "reduced-logical Houdayer registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "real_low_energy_reduced_logical_houdayer_component_feasibility_only",
        "remote_authorization": False,
    }, "reduced-logical Houdayer scope changed")
    return config, sha256_file(path)


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "base_probe": sha256_file(BASE_PROBE_PATH),
        "config": sha256_file(config_path),
        "houdayer": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_houdayer.py"),
        "probe": sha256_file(Path(__file__)),
        "registry": sha256_file(REGISTRY_PATH),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def run_probe(config):
    """Run the frozen schedule after changing exactly one coordinate factory."""
    base = _base_probe()
    original_factory = base.build_sparse_hgp_coordinate_basis
    base.build_sparse_hgp_coordinate_basis = build_sparse_hgp_reduced_logical_coordinate_basis
    try:
        result = base.run_probe(config)
    finally:
        base.build_sparse_hgp_coordinate_basis = original_factory
    _require(result["coordinate_basis"]["version"]
             == HOUDAYER_REDUCED_LOGICAL_COORDINATE_VERSION,
             "reduced-logical Houdayer runner used the wrong coordinate basis")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "houdayer_reduced_logicals.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace reduced-logical Houdayer report: {args.output}")
    config, config_sha256 = _load_config(args.config)
    core = {
        "probe_version": PROBE_VERSION,
        "config_sha256": config_sha256,
        "scope": config["scope"],
        "source_binding": _source_binding(args.config),
        "probe": run_probe(config),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
