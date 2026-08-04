"""Outcome-blind runtime gate for exact full-B-column Gibbs on the m8 sentinel.

This script intentionally performs no full-state redraw and never constructs a
logical label, character, posterior estimate, or an initial-state weight.  Its
only purpose is to decide whether the exact 2**24-column conditional can meet
the predeclared T1 two-hour-per-trajectory runtime requirement.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    FULL_COLUMN_GIBBS_VERSION,
    build_full_column_candidate_cache,
    build_full_column_workspace,
    full_column_gibbs_update,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROFILE_VERSION = "exp102.q0_hgp_full_column_gibbs.runtime_profile.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
OUTPUT_PATH = ROOT / "runtime_probe.json"
CELL = {
    "code_id": "m08_c06",
    "disorder_index": 0,
    "disorder_source": "attempt022",
    "p": 0.04,
}

# This is fixed before any update timing is observed.  One warm-up touches the
# dense arrays; two subsequent full-column conditionals are the sole timing sample.
WARMUP_UPDATES = 1
TIMED_UPDATES = 2
T1_BURN_SWEEPS = 2048
T1_MEASUREMENT_SWEEPS = 8192
TRAJECTORY_WALL_SECONDS_LIMIT = 2.0 * 60.0 * 60.0
SAFETY_FACTOR = 2.0
RUNTIME_SEED = 850018170129077531


class RuntimeProbeError(RuntimeError):
    """Raised when the outcome-blind profiling contract is violated."""


def _source_commit():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes while Linux reports KiB.
    return value if sys.platform == "darwin" else 1024 * value


def _context():
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, CELL["code_id"])
    model, _ = build_model(H)
    uniform_seed = derive_seed(
        "pilot_ladder_m8_attempt22", registry["registry_sha256"],
        code["code_id"], CELL["disorder_index"], "uniforms",
    )
    initial_state = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < CELL["p"]
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ initial_state.astype(np.int64) % 2
    ).astype(np.uint8)
    if not syndrome.any():
        raise RuntimeProbeError("the frozen hard sentinel unexpectedly has zero syndrome")
    return registry, code, np.ascontiguousarray(H, dtype=np.uint8), initial_state, syndrome


def _portable_rng(seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    return PortablePrng(int(seed))


def profile():
    registry, code, H, initial_state, syndrome = _context()
    rows, columns = H.shape
    started_setup = time.perf_counter()
    mass = build_classical_coset_mass(H, CELL["p"], engine="numba")
    log_mass = np.ascontiguousarray(np.log(mass), dtype=np.float64)
    del mass
    cache = build_full_column_candidate_cache(rows, CELL["p"])
    workspace = build_full_column_workspace(cache)
    b_columns, a_syndromes, _ = _initial_collapsed_masks(initial_state, syndrome, H)
    setup_seconds = time.perf_counter() - started_setup
    rng = _portable_rng(RUNTIME_SEED)
    syndrome_matrix = syndrome.reshape(rows, columns)

    # Warm the memory path without retaining any state-derived diagnostic.
    for update in range(WARMUP_UPDATES):
        full_column_gibbs_update(
            b_columns, a_syndromes, H, syndrome_matrix, update % rows,
            log_mass, cache, workspace, rng,
        )
    started_updates = time.perf_counter()
    for update in range(TIMED_UPDATES):
        full_column_gibbs_update(
            b_columns, a_syndromes, H, syndrome_matrix,
            (WARMUP_UPDATES + update) % rows, log_mass, cache, workspace, rng,
        )
    timed_update_seconds = time.perf_counter() - started_updates

    updates_per_t1_trajectory = (T1_BURN_SWEEPS + T1_MEASUREMENT_SWEEPS) * rows
    projected_wall_seconds = SAFETY_FACTOR * (
        setup_seconds + timed_update_seconds / TIMED_UPDATES * updates_per_t1_trajectory
    )
    return {
        "profile_version": PROFILE_VERSION,
        "sampler_version": FULL_COLUMN_GIBBS_VERSION,
        "source_commit": _source_commit(),
        "source_files": {
            "profile_runtime.py": sha256_file(__file__),
            "q0_hgp_full_column_gibbs.py": sha256_file(
                EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_column_gibbs.py",
            ),
        },
        "cell": CELL,
        "registry_sha256": registry["registry_sha256"],
        "code_npz_sha256": code["code_npz_sha256"],
        "matrix_shape": [int(rows), int(columns)],
        "candidate_count": int(cache.masks.size),
        "warmup_updates": WARMUP_UPDATES,
        "timed_updates": TIMED_UPDATES,
        "setup_seconds": setup_seconds,
        "timed_update_seconds": timed_update_seconds,
        "wall_seconds_per_column_update": timed_update_seconds / TIMED_UPDATES,
        "peak_rss_bytes": _peak_rss_bytes(),
        "formal_t1": {
            "burn_sweeps": T1_BURN_SWEEPS,
            "measurement_sweeps": T1_MEASUREMENT_SWEEPS,
            "columns_per_sweep": int(rows),
            "column_updates_per_trajectory": int(updates_per_t1_trajectory),
            "trajectory_wall_seconds_limit": TRAJECTORY_WALL_SECONDS_LIMIT,
            "safety_factor": SAFETY_FACTOR,
        },
        "projected_t1_trajectory_wall_seconds": projected_wall_seconds,
        "runtime_gate": (
            "PASS" if projected_wall_seconds <= TRAJECTORY_WALL_SECONDS_LIMIT
            else "RUNTIME_EXHAUSTED"
        ),
        "raw_produced": False,
        "outcome_blind": {
            "full_state_redraws": 0,
            "logical_labels_constructed": 0,
            "character_estimates_constructed": 0,
            "weights_recorded": 0,
            "method_selection_input": "timing_and_memory_only",
        },
    }


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeProbeError(f"refusing to overwrite frozen runtime report: {OUTPUT_PATH}")
    result = profile()
    result["report_sha256"] = sha256_json(result)
    atomic_json(OUTPUT_PATH, result)
    print(result["report_sha256"])


if __name__ == "__main__":
    main()
