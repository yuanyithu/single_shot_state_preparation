"""Emit one canonical reference/Numba PT-v2 digest for cross-node attestation."""

import hashlib

import numpy as np

from data.expander_code.exp102.exp102_pipeline.discovery import load_discovery_config
from data.expander_code.exp102.exp102_pipeline.io import canonical_json
from data.expander_code.exp102.exp102_pipeline.ladders import make_pt_candidate, q0_config_from_candidate
from data.expander_code.exp102.exp102_pipeline.q0_pt import run_q0_pt_instance
from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


registry = load_registry("data/expander_code/exp102/registry/registry.json")
config = load_discovery_config(
    "data/expander_code/exp102/config/discovery.v2.json", registry,
)
candidate = make_pt_candidate(config["ladders"][0], 5, 20, 4)
pt_config = q0_config_from_candidate(candidate)

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101

load_exp101()
from exp101_certified_src.graphs import cycle_parity_check_matrix

model, frame = build_model(cycle_parity_check_matrix(3))
epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
epsilon[[0, 2]] = 1
syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
results = [
    run_q0_pt_instance(
        model, frame, syndrome, 0.04, pt_config, 0x1022026, np.uint64(1), engine=engine,
    )
    for engine in ("reference", "numba")
]
array_fields = (
    "labels", "ladder_K", "ladder_p", "swap_attempts", "swap_accepts",
    "logical_attempts", "logical_accepts", "hot_arrival_labels",
    "hot_departure_labels", "hot_touches_per_replica",
    "hot_updated_visits_per_replica", "uncertified_round_trips_per_replica",
    "round_trips_per_replica", "sector_changing_round_trips_per_replica",
    "final_replica_at_rung", "final_transport_phase",
)
scalar_fields = (
    "hot_touches", "hot_updated_visits", "uncertified_round_trips",
    "round_trips", "sector_changing_round_trips", "max_hard_coset_residual",
)
for field in array_fields:
    if not np.array_equal(results[0][field], results[1][field]):
        raise AssertionError(f"reference/Numba mismatch: {field}")
for field in scalar_fields:
    if results[0][field] != results[1][field]:
        raise AssertionError(f"reference/Numba mismatch: {field}")
digest = hashlib.sha256()
digest.update(canonical_json(candidate).encode("ascii"))
for field in array_fields:
    value = np.asarray(results[1][field])
    digest.update(field.encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=">i8").tobytes())
    digest.update(value.tobytes(order="C"))
digest.update(canonical_json({field: results[1][field] for field in scalar_fields}).encode("ascii"))
print(digest.hexdigest())
