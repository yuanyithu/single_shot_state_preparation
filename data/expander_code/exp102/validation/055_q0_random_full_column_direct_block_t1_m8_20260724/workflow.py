"""Fresh direct-block schedule, preflight, and measurement workflow.

The frozen validation-052 orchestration is reused only as control-flow code.
All contract identities, seeds, artifacts, and sampler entry points are rebound
here before its CLI is entered; importing this module has no mutation side
effect on the historical module.
"""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path

import numpy as np


_legacy = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.workflow"
)
_direct = import_module(
    "data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column"
)
_conditional = import_module(
    "data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs"
)

CONTRACT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.v1"
SCHEDULE_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.schedule.v1"
PREFLIGHT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.preflight.v1"
RAW_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.raw.v1"
NODE_REPORT_VERSION = (
    "exp102.q0_random_full_column_direct_block.t1_m8.node_report.v1"
)
CONTROL_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.control.v1"
REPORT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.report.v1"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = (
    EXP102_ROOT / "config/q0_random_full_column_direct_block.t1_m8.v1.json"
)
SOURCE_CONTROL_DIR = ROOT / "control"
FAMILIES = ("P", "U", "M0", "M1", "S")

_ORIGINAL_LOAD_CONTROL = _legacy._load_control
_ORIGINAL_BUILD_MASS = _legacy.build_classical_coset_mass
_MASS_BY_KEY = {}

_load_canonical_json = _legacy._load_canonical_json
_verify_self_hash = _legacy._verify_self_hash
_source_identity = _legacy._source_identity
_require = _legacy._require
canonical_json = _legacy.canonical_json
sha256_file = _legacy.sha256_file
sha256_json = _legacy.sha256_json


def _load_config():
    config = _load_canonical_json(CONFIG_PATH)
    _require(
        config["version"] == config["contract_version"] == CONTRACT_VERSION,
        "direct-block T1 contract version changed",
    )
    _require(
        config["config_version"]
        == "exp102.q0_random_full_column_direct_block.t1_m8.config.v1",
        "direct-block T1 config version changed",
    )
    _require(
        tuple(config["initialization"]["families"]) == FAMILIES
        and config["initialization"]["trajectories_per_family"] == 8,
        "initialization panel changed",
    )
    _require(config["resource"] == {
        "allowed_nodes": ["nd-1", "nd-2", "nd-3"],
        "burn_updates": 2048,
        "fixed_workers_per_node": 4,
        "measurement_updates": 8192,
        "runtime_probe_concurrency": 4,
        "runtime_probe_updates_per_worker": 8,
        "safety_factor": 2.0,
        "trajectory_wall_cap_seconds": 7200.0,
    }, "resource clock changed")
    implementation = config["implementation"]
    _require(
        implementation["method_id"]
        == _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_METHOD_ID,
        "direct-block method identity changed",
    )
    for path_field, hash_field in (
        ("full_column_relpath", "full_column_file_sha256"),
        ("random_full_column_relpath", "random_full_column_file_sha256"),
        ("portable_reference_relpath", "portable_reference_file_sha256"),
    ):
        path = EXP102_ROOT / implementation[path_field]
        _require(sha256_file(path) == implementation[hash_field],
                 f"implementation binding changed: {path_field}")
    reference = _load_canonical_json(
        EXP102_ROOT / implementation["portable_reference_relpath"]
    )
    _verify_self_hash(reference, "reference_sha256")
    _require(
        reference["reference_sha256"]
        == implementation["portable_reference_sha256"],
        "portable reference identity changed",
    )
    aggregate = _load_canonical_json(
        EXP102_ROOT
        / "validation/054_q0_random_full_column_direct_block_preflight_20260724"
        / "remote_evidence/preflight/aggregate.json"
    )
    _verify_self_hash(aggregate, "aggregate_sha256")
    _require(
        aggregate["status"] == "PASS"
        and aggregate["aggregate_sha256"]
        == implementation["validation_054_aggregate_sha256"],
        "validation-054 terminal authorization changed",
    )
    return config, sha256_file(CONFIG_PATH)


def _load_control(control_dir, config, config_sha):
    previous = _legacy.CONTROL_VERSION
    _legacy.CONTROL_VERSION = CONTROL_VERSION
    try:
        return _ORIGINAL_LOAD_CONTROL(control_dir, config, config_sha)
    finally:
        _legacy.CONTROL_VERSION = previous


def _build_mass(H, p, engine="numba"):
    # Numba-owned arrays can cause np.ascontiguousarray to create a distinct
    # view.  Normalize once so the cache and runner share the exact object.
    mass = np.ascontiguousarray(
        _ORIGINAL_BUILD_MASS(H, p, engine=engine), dtype=np.float64,
    )
    _MASS_BY_KEY[(int(H.shape[0]), float(p))] = mass
    return mass


def _build_cache(rows, p):
    key = (int(rows), float(p))
    _require(key in _MASS_BY_KEY, "direct-block mass must be built before its cache")
    return _conditional.build_full_column_direct_block_cache(
        rows, p, _MASS_BY_KEY[key],
    )


def _configure_legacy():
    bindings = {
        "CONTRACT_VERSION": CONTRACT_VERSION,
        "SCHEDULE_VERSION": SCHEDULE_VERSION,
        "PREFLIGHT_VERSION": PREFLIGHT_VERSION,
        "RAW_VERSION": RAW_VERSION,
        "NODE_REPORT_VERSION": NODE_REPORT_VERSION,
        "CONTROL_VERSION": CONTROL_VERSION,
        "ROOT": ROOT,
        "EXP102_ROOT": EXP102_ROOT,
        "CONFIG_PATH": CONFIG_PATH,
        "SOURCE_CONTROL_DIR": SOURCE_CONTROL_DIR,
        "FAMILIES": FAMILIES,
        "RANDOM_FULL_COLUMN_METHOD_ID": (
            _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_METHOD_ID
        ),
        "RANDOM_FULL_COLUMN_VERSION": (
            _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_VERSION
        ),
        "RandomFullColumnConfig": _direct.RandomFullColumnDirectBlockConfig,
        "build_classical_coset_mass": _build_mass,
        "build_full_column_candidate_cache": _build_cache,
        "build_full_column_workspace": (
            _conditional.build_full_column_direct_block_workspace
        ),
        "run_random_full_column_trajectory": (
            _direct.run_random_full_column_direct_block_trajectory
        ),
        "replay_random_full_column_trajectory": (
            _direct.replay_random_full_column_direct_block_trajectory
        ),
        "_load_config": _load_config,
        "_load_control": _load_control,
    }
    for name, value in bindings.items():
        setattr(_legacy, name, value)


def main():
    _configure_legacy()
    _legacy.main()


if __name__ == "__main__":
    main()
