"""Read-only cross-fit audit of iid LSI proposal draws embedded in old chains.

Each LSI IMH step first calls a state-independent proposal sampler, so the
stored proposal records are iid draws even though the subsequently accepted
states form a nonmixing Markov chain.  This script audits only those proposal
records.  The historical raw remains non-reusable for scientific estimation.
"""

from __future__ import annotations

import hashlib
import json
import math
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


PROBE_VERSION = "exp102.q0_crossfit_lsi.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
RAW_DIRECTORY = (
    EXP102_ROOT / "validation" / "015_q0_logical_stratified_v0b_20260723" / "remote_run"
    / "exp102_q0_lsi_v0d_20260723_9f0c473" / "pulled_run" / "raw"
)
OUTPUT_PATH = ROOT / "historical_lsi_crossfit.json"
P_VALUE = 0.04
EXPECTED_RAW_COUNT = 48
PROPOSALS_PER_TRAJECTORY = 512 + 4096


class ProbeError(RuntimeError):
    """Raised when the frozen LSI proposal transcript is not the expected one."""


def _require(condition, message):
    if not condition:
        raise ProbeError(message)


def _source_commit():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _read_trajectory(path):
    with np.load(path, allow_pickle=False) as raw:
        required = {
            "task_json", "task_fingerprint", "burn_proposal_labels",
            "measurement_proposal_labels", "burn_proposal_weights",
            "measurement_proposal_weights", "burn_proposal_log_q",
            "measurement_proposal_log_q",
        }
        _require(required <= set(raw.files), f"LSI proposal fields changed: {path.name}")
        task = json.loads(raw["task_json"].item())
        _require(task.get("cell") == {
            "code_id": "m08_c06", "disorder_index": 0,
            "disorder_source": "attempt022", "p": P_VALUE,
        }, f"LSI cell changed: {path.name}")
        artifact = task.get("artifact", {})
        tau = float(artifact.get("alpha_temperature"))
        _require(tau in (0.5, 1.0), f"LSI alpha temperature changed: {path.name}")
        labels = np.concatenate((
            np.asarray(raw["burn_proposal_labels"], dtype=np.uint64),
            np.asarray(raw["measurement_proposal_labels"], dtype=np.uint64),
        ))
        weights = np.concatenate((
            np.asarray(raw["burn_proposal_weights"], dtype=np.int32),
            np.asarray(raw["measurement_proposal_weights"], dtype=np.int32),
        ))
        log_q = np.concatenate((
            np.asarray(raw["burn_proposal_log_q"], dtype=np.float64),
            np.asarray(raw["measurement_proposal_log_q"], dtype=np.float64),
        ))
        _require(labels.shape == weights.shape == log_q.shape == (PROPOSALS_PER_TRAJECTORY,),
                 f"LSI proposal count changed: {path.name}")
        _require(np.all(np.isfinite(log_q)), f"LSI proposal log density is non-finite: {path.name}")
        return {
            "tau": tau,
            "task_fingerprint": raw["task_fingerprint"].item(),
            "labels": labels,
            "log_weights": weights.astype(np.float64) * math.log(P_VALUE / (1.0 - P_VALUE)) - log_q,
        }


def run_probe():
    paths = sorted(RAW_DIRECTORY.glob("*.npz"))
    _require(len(paths) == EXPECTED_RAW_COUNT, "historical LSI raw count changed")
    grouped = {0.5: [], 1.0: []}
    for path in paths:
        record = _read_trajectory(path)
        grouped[record["tau"]].append(record)
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    _require(frame.k == 64 and model.num_qubits == 1600, "historical LSI dimensions changed")

    reports = []
    for tau in (0.5, 1.0):
        records = sorted(grouped[tau], key=lambda value: value["task_fingerprint"])
        _require(len(records) == 24, f"historical LSI tau={tau} task count changed")
        fingerprints = [value["task_fingerprint"] for value in records]
        _require(len(set(fingerprints)) == len(fingerprints), "historical LSI task duplicated")
        labels = np.concatenate([value["labels"] for value in records])
        log_weights = np.concatenate([value["log_weights"] for value in records])
        result = crossfit_collision_ratio(
            labels, log_weights, block_count=len(records), logical_dimension=frame.k,
        )
        reports.append({
            "alpha_temperature": tau,
            "independent_trajectory_blocks": len(records),
            "proposal_draws_per_block": PROPOSALS_PER_TRAJECTORY,
            "task_fingerprints_sha256": hashlib.sha256(
                "".join(fingerprints).encode("ascii")
            ).hexdigest(),
            "crossfit_collision_diagnostic": result.as_dict(),
        })
    return {
        "probe_version": PROBE_VERSION,
        "crossfit_version": CROSSFIT_IMPORTANCE_VERSION,
        "source_commit": _source_commit(),
        "source_files": {
            "probe_historical_lsi_is.py": sha256_file(__file__),
            "q0_crossfit_importance.py": sha256_file(
                EXP102_ROOT / "exp102_pipeline" / "q0_crossfit_importance.py",
            ),
        },
        "historical_raw": {
            "directory": RAW_DIRECTORY.relative_to(EXP102_ROOT).as_posix(),
            "file_count": len(paths),
            "file_names_sha256": hashlib.sha256(
                "".join(path.name for path in paths).encode("ascii")
            ).hexdigest(),
            "authority": "terminal_LSI_transport_raw_diagnostic_only",
        },
        "registry_sha256": registry["registry_sha256"],
        "code_npz_sha256": code["code_npz_sha256"],
        "proposal_reports": reports,
        "new_samples_generated": False,
        "raw_reuse_prohibited": True,
        "interpretation": {
            "result": "HISTORICAL_IID_LSI_PROPOSAL_FEASIBILITY_ONLY",
            "does_not_establish": [
                "A new posterior estimate or q_top result.",
                "That the LSI Markov chains mixed.",
                "Proposal coverage of target modes not observed in these draws.",
                "Any formal, held-out, remote, or production authority.",
            ],
            "required_before_any_successor": (
                "A fresh iid-IS contract with fresh proposal samples, independent "
                "MAM/LSI-style proposal families, and predeclared cross-proposal gates."
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
