"""Canonical Linux digest for the PA mutation and transcript path."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import socket

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    canonical_json,
    sha256_json,
    verify_source_identity,
)
from data.expander_code.exp102.exp102_pipeline.q0_pa import (
    PaSeedIdentity,
    Q0PaConfig,
    canonical_population_digest,
    run_q0_pa_population,
    theta_schedule_q32,
)
from data.expander_code.exp102.exp102_pipeline.q0_pt import ladder_x_q32_sha256
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def canonical_digest(registry_path, source_commit):
    _, _, H = load_frozen_code(registry_path, "m08_c06")
    model, frame = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, 3, model.num_qubits - 1]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    records = []
    for kernel in ("coordinate", "block4"):
        schedule = theta_schedule_q32(0.04, 8)
        config = Q0PaConfig(
            0.04, 32, 8, 2, kernel, schedule,
            ladder_x_q32_sha256(schedule),
        )
        seed_identity = PaSeedIdentity(
            source_commit, sha256_json({"cross_node": kernel}),
            sha256_json({"cell": "m08_c06", "p": 0.04}), 0,
            "q0_pa_cross_node_v1",
        )
        reference = run_q0_pa_population(
            model, frame, syndrome, config, seed_identity, engine="reference",
        )
        accelerated = run_q0_pa_population(
            model, frame, syndrome, config, seed_identity, engine="numba",
        )
        reference_digest = canonical_population_digest(reference)
        numba_digest = canonical_population_digest(accelerated)
        if reference_digest != numba_digest:
            raise AssertionError("PA reference/Numba digest mismatch")
        records.append({"kernel": kernel, "digest": reference_digest})
    payload = canonical_json(records).encode("ascii")
    return {"records": records, "canonical_digest": hashlib.sha256(payload).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("source_commit")
    parser.add_argument("--require-verified-source", action="store_true")
    args = parser.parse_args(argv)
    source_identity = (
        verify_source_identity(Path.cwd(), args.source_commit)
        if args.require_verified_source else None
    )
    result = canonical_digest(args.registry, args.source_commit)
    result["source_commit"] = args.source_commit
    result["source_identity"] = source_identity
    result["environment"] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
