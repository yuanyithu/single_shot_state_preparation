"""Read-only feasibility audit of cross-fitted IS on historical MAM draws.

The historical 50,000 draws remain diagnostic-only raw.  This script neither
generates fresh samples nor grants them new posterior-estimation authority; it
only tests whether their iid proposal-overlap data has stable enough blocks to
motivate a separately frozen future IS contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_crossfit_importance import (
    CROSSFIT_IMPORTANCE_VERSION,
    crossfit_collision_ratio,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_crossfit_is.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
HISTORICAL_RAW = (
    EXP102_ROOT / "validation" / "013_q0_hgp_global_screen_20260722" / "remote_run"
    / "exp102_q0_hgp_screen_v2_20260722_4d134ee" / "hgp_global" / "raw"
    / "importance_sampling" / "b4ccfd16bed39aac912b8aa129485dbc2f7ac696724e12f38b23a5f83da521fa.npz"
)
HISTORICAL_RAW_SHA256 = "3469f4c567d206e2b6c23fbf312be9ef17b9d4de7f6678dc33ab8a22c72b470e"
OUTPUT_PATH = ROOT / "historical_mam_crossfit.json"
BLOCK_COUNT = 10


class ProbeError(RuntimeError):
    """Raised when the immutable old diagnostic cannot be audited safely."""


def _require(condition, message):
    if not condition:
        raise ProbeError(message)


def _source_commit():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _labels_from_packed_states(frame, packed):
    states = np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=1, count=frame.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    signatures = np.zeros(frame.num_qubits, dtype=np.uint64)
    for bit in range(frame.k):
        signatures[frame.W_basis[bit].astype(bool)] |= np.uint64(1) << np.uint64(bit)
    labels = np.zeros(states.shape[0], dtype=np.uint64)
    for qubit in np.flatnonzero(states.any(axis=0)):
        labels[states[:, qubit].astype(bool)] ^= signatures[qubit]
    return labels, states.sum(axis=1, dtype=np.int32)


def run_probe():
    _require(HISTORICAL_RAW.is_file(), "historical MAM IS raw is missing")
    _require(sha256_file(HISTORICAL_RAW) == HISTORICAL_RAW_SHA256,
             "historical MAM IS raw SHA changed")
    with np.load(HISTORICAL_RAW, allow_pickle=False) as raw:
        required = {
            "identity_json", "sample_states_packed", "sample_physical_weights",
            "sample_log_importance_weight",
        }
        _require(required <= set(raw.files), "historical MAM IS raw fields changed")
        identity = json.loads(raw["identity_json"].item())
        _require(identity.get("role") == "auxiliary_proposal_overlap_diagnostic_only",
                 "historical raw authority is not diagnostic-only")
        _require(identity.get("cell") == {
            "code_id": "m08_c06", "disorder_index": 0,
            "disorder_source": "attempt022", "p": 0.04,
        }, "historical raw cell changed")
        packed = np.ascontiguousarray(raw["sample_states_packed"], dtype=np.uint8)
        stored_weights = np.asarray(raw["sample_physical_weights"], dtype=np.int32)
        log_weights = np.asarray(raw["sample_log_importance_weight"], dtype=np.float64)

    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    _require(model.num_qubits == 1600 and frame.k == 64,
             "historical m8 dimensions changed")
    _require(packed.shape == (50000, 200) and stored_weights.shape == (50000,),
             "historical MAM IS sample shape changed")
    labels, recomputed_weights = _labels_from_packed_states(frame, packed)
    _require(np.array_equal(stored_weights, recomputed_weights),
             "historical MAM IS physical weights do not replay")
    result = crossfit_collision_ratio(
        labels, log_weights, block_count=BLOCK_COUNT, logical_dimension=frame.k,
    )
    return {
        "probe_version": PROBE_VERSION,
        "crossfit_version": CROSSFIT_IMPORTANCE_VERSION,
        "source_commit": _source_commit(),
        "source_files": {
            "probe_historical_mam_is.py": sha256_file(__file__),
            "q0_crossfit_importance.py": sha256_file(
                EXP102_ROOT / "exp102_pipeline" / "q0_crossfit_importance.py",
            ),
        },
        "historical_raw": {
            "path": HISTORICAL_RAW.relative_to(EXP102_ROOT).as_posix(),
            "sha256": HISTORICAL_RAW_SHA256,
            "identity_sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "authority": identity["role"],
        },
        "registry_sha256": registry["registry_sha256"],
        "code_npz_sha256": code["code_npz_sha256"],
        "block_count": BLOCK_COUNT,
        "samples_per_block": 50000 // BLOCK_COUNT,
        "crossfit_collision_diagnostic": result.as_dict(),
        "new_samples_generated": False,
        "raw_reuse_prohibited": True,
        "interpretation": {
            "result": "HISTORICAL_IID_PROPOSAL_FEASIBILITY_ONLY",
            "does_not_establish": [
                "A new posterior estimate or q_top result.",
                "Proposal coverage of unobserved target modes.",
                "Any formal, held-out, remote, or production authority.",
            ],
            "required_before_any_successor": (
                "A fresh contract with fresh iid samples, independently frozen proposal "
                "families, cross-proposal agreement, and a predeclared finite-sample gate."
            ),
        },
    }


def main():
    if OUTPUT_PATH.exists():
        raise ProbeError(f"refusing to overwrite immutable feasibility report: {OUTPUT_PATH}")
    core = run_probe()
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(OUTPUT_PATH, report)
    print(report["report_sha256"])


if __name__ == "__main__":
    main()
