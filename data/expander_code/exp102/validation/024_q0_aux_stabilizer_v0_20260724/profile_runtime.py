"""Runtime-only probe for the auxiliary A-row stabilizer heatbath."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_hgp_aux_stabilizer import (
    AUX_STABILIZER_KERNEL,
    auxiliary_stabilizer_row_heatbath,
    auxiliary_stabilizer_sweep,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import hgp_syndrome_matrix, split_hgp_state
from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model

load_exp101()
from exp101_certified_src.prng import PortablePrng


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
OUTPUT = ROOT / "runtime_probe.json"


def _state(code, model, registry):
    seed = derive_seed(
        "pilot_ladder_m8_attempt22", registry["registry_sha256"], code["code_id"], 0, "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(seed)).random(model.num_qubits) < 0.04
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    return epsilon, syndrome


def main():
    if OUTPUT.exists():
        raise RuntimeError("runtime probe already exists")
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, _ = build_model(H)
    state, syndrome = _state(code, model, registry)
    A, B = split_hgp_state(state, H)
    row_seconds = []
    for repeat in range(3):
        rng = PortablePrng(derive_seed("exp102.aux_stabilizer.runtime", repeat, "row"))
        started = time.perf_counter()
        next_a, next_b, _ = auxiliary_stabilizer_row_heatbath(H, A, B, repeat, 0.04, rng)
        row_seconds.append(time.perf_counter() - started)
        if not np.array_equal(hgp_syndrome_matrix(next_a, next_b, H), syndrome.reshape(H.shape)):
            raise RuntimeError("auxiliary row heatbath left the hard coset")
    rng = PortablePrng(derive_seed("exp102.aux_stabilizer.runtime", "sweep"))
    started = time.perf_counter()
    next_a, next_b, assignments = auxiliary_stabilizer_sweep(H, A, B, 0.04, rng)
    sweep_seconds = time.perf_counter() - started
    if not np.array_equal(hgp_syndrome_matrix(next_a, next_b, H), syndrome.reshape(H.shape)):
        raise RuntimeError("auxiliary sweep left the hard coset")
    core = {
        "probe_version": "exp102.q0_hgp_aux_stabilizer.runtime_probe.v0",
        "purpose": "runtime_only_no_raw_no_estimator_no_method_selection_by_physics",
        "kernel": AUX_STABILIZER_KERNEL,
        "cell": {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0, "disorder_source": "attempt022"},
        "source": {
            "auxiliary": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer.py"),
            "full_row": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py"),
            "profile": sha256_file(Path(__file__)),
        },
        "row_seconds": row_seconds,
        "row_seconds_max": max(row_seconds),
        "sweep_seconds": sweep_seconds,
        "assignments_sha256": hashlib.sha256(assignments.tobytes()).hexdigest(),
        "hard_coset_preserved": True,
    }
    atomic_json(OUTPUT, {**core, "report_sha256": sha256_json(core)})
    print(OUTPUT)


if __name__ == "__main__":
    main()
