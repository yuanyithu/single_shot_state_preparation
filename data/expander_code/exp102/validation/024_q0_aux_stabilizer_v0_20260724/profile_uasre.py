"""Runtime-only UASRE probe; it never computes a posterior statistic."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_hgp_aux_stabilizer_pt import (
    AUX_STABILIZER_PT_KERNEL,
    AuxiliaryStabilizerReplicaExchangeConfig,
    AuxiliaryStabilizerReplicaExchangeSeedIdentity,
    run_auxiliary_stabilizer_replica_exchange_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
OUTPUT = ROOT / "uasre_runtime_probe.json"


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


def _seed(config, registry, repeat):
    return AuxiliaryStabilizerReplicaExchangeSeedIdentity(
        source_commit="0" * 40,
        config_sha256=sha256_json(config.as_dict()),
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json({
            "code_id": "m08_c06", "p": 0.04, "disorder_index": 0, "disorder_source": "attempt022",
        }),
        method_id=config.method_id,
        resource_tier="runtime_probe",
        init_family="P",
        trajectory_index=repeat,
        trajectory_namespace="exp102.q0_hgp_aux_stabilizer.runtime_probe",
    )


def main():
    if OUTPUT.exists():
        raise RuntimeError("UASRE runtime probe already exists")
    load_exp101()
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    state, syndrome = _state(code, model, registry)
    timings = []
    for replicas in (32, 64):
        warm = AuxiliaryStabilizerReplicaExchangeConfig(0.04, 1, 8, replicas, 1, 1)
        run_auxiliary_stabilizer_replica_exchange_trajectory(
            model, frame, H, syndrome, warm, _seed(warm, registry, replicas), state, engine="numba",
        )
        config = AuxiliaryStabilizerReplicaExchangeConfig(0.04, 8, 16, replicas, 1, 1)
        started = time.perf_counter()
        raw = run_auxiliary_stabilizer_replica_exchange_trajectory(
            model, frame, H, syndrome, config, _seed(config, registry, replicas + 100), state, engine="numba",
        )
        seconds = time.perf_counter() - started
        if raw["measurement_residual_weights"].any():
            raise RuntimeError("UASRE runtime probe left the hard coset")
        timings.append({
            "method_id": config.method_id,
            "rounds": config.burn_rounds + config.measurement_rounds,
            "seconds": seconds,
            "seconds_per_round": seconds / (config.burn_rounds + config.measurement_rounds),
            "auxiliary_attempts": int(raw["burn_auxiliary_counters"][0] + raw["measurement_auxiliary_counters"][0]),
            "auxiliary_changed": int(raw["burn_auxiliary_counters"][1] + raw["measurement_auxiliary_counters"][1]),
            "raw_digest": hashlib.sha256(raw["measurement_states_packed"].tobytes()).hexdigest(),
        })
    core = {
        "probe_version": "exp102.q0_hgp_aux_stabilizer_pt.runtime_probe.v0",
        "purpose": "runtime_only_no_raw_no_estimator_no_method_selection_by_physics",
        "kernel": AUX_STABILIZER_PT_KERNEL,
        "cell": {"code_id": "m08_c06", "p": 0.04, "disorder_index": 0, "disorder_source": "attempt022"},
        "source": {
            "auxiliary": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer.py"),
            "sampler": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_aux_stabilizer_pt.py"),
            "uniform_anchor": sha256_file(EXP102_ROOT / "exp102_pipeline" / "q0_hgp_uniform_anchor_pt.py"),
            "profile": sha256_file(Path(__file__)),
        },
        "timings": timings,
    }
    atomic_json(OUTPUT, {**core, "report_sha256": sha256_json(core)})
    print(OUTPUT)


if __name__ == "__main__":
    main()
