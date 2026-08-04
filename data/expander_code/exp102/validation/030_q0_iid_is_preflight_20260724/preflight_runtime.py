"""Outcome-blind local compatibility probe for fresh iid-IS planning.

This probe deliberately does not calculate importance weights, q_top, a sector
mass, or an MCMC diagnostic.  It only checks that two independently designed
full-support proposal mechanisms can be constructed or loaded, produce legal
hard-coset states, and fit a prospective draw-time budget on the m8 sentinel.

The logical-stratified artifact is historical and is loaded only to audit the
current sampling interface.  It is never copied into a successor experiment;
a successor must build and freeze its own artifact before generating samples.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np

# Permit direct invocation while keeping all output under the validation folder.
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import state_label
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import (
    load_logical_stratified_frozen_artifact,
)
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    build_map_mixture_proposal,
    build_milp_map_anchors,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_iid_is.preflight_runtime.v0"
CELL = {
    "code_id": "m08_c06",
    "p": 0.04,
    "disorder_index": 0,
    "disorder_source": "attempt022",
}
SAMPLES_PER_PROPOSAL = 64
LSI_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "015_q0_logical_stratified_v0b_20260723"
    / "remote_run/exp102_q0_lsi_v0d_20260723_9f0c473/pulled_run/artifacts"
    / "artifacts/lsi_imh_tau_05.npz"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _full_support_lower_mass_mam(proposal):
    uniform = np.flatnonzero(proposal.theta_stabilizer == 0.5)
    _require(uniform.size == 1, "MAM has no unique theta=.5 defensive component")
    index = int(uniform[0])
    _require(proposal.theta_logical[index] == 0.5,
             "MAM defensive component is not uniform over logical coordinates")
    mass = float(proposal.component_weights[index])
    _require(mass > 0.0 and math.isfinite(mass), "MAM defensive mass is invalid")
    return mass


def _full_support_lower_mass_lsi(proposal):
    uniform = np.flatnonzero(proposal.theta_stabilizer == 0.5)
    _require(uniform.size == 1, "LSI has no unique theta=.5 defensive component")
    mass = float(proposal.uniform_label_probability * proposal.component_weights[int(uniform[0])])
    _require(mass > 0.0 and math.isfinite(mass), "LSI defensive mass is invalid")
    return mass


def _validate_draw(proposal, model, frame, syndrome, draw):
    state = np.asarray(draw["state"], dtype=np.uint8)
    _require(state.shape == (model.num_qubits,), "proposal state has the wrong length")
    _require(not np.any((state != 0) & (state != 1)), "proposal state is nonbinary")
    residual = (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    _require(not residual.any(), "proposal draw escaped the hard coset")
    log_q = float(draw["log_q"])
    _require(math.isfinite(log_q), "proposal draw has a nonfinite log density")
    coordinate = np.asarray(draw["coordinate"], dtype=np.uint8)
    _require(np.array_equal(proposal.coordinates.state_from_coordinates(coordinate), state),
             "proposal coordinate does not replay its state")
    replayed = float(proposal.log_probability_coordinates(coordinate))
    _require(math.isclose(log_q, replayed, rel_tol=0.0, abs_tol=1e-12),
             "proposal log density does not replay")
    # This uses the canonical observable frame only as an algebraic check.
    state_label(frame, state)


def _benchmark(name, proposal, model, frame, syndrome, *, seed):
    from exp101_certified_src.prng import PortablePrng

    rng = PortablePrng(int(seed))
    started = time.perf_counter()
    for _ in range(SAMPLES_PER_PROPOSAL):
        _validate_draw(proposal, model, frame, syndrome, proposal.sample(rng))
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "samples": SAMPLES_PER_PROPOSAL,
        "draw_seconds_total": elapsed,
        "draw_seconds_per_sample": elapsed / SAMPLES_PER_PROPOSAL,
    }


def run_probe():
    root = Path(__file__).resolve().parents[2]
    registry_path = root / "registry/registry.json"
    registry = load_registry(registry_path)
    _, code, H = load_frozen_code(registry_path, CELL["code_id"])
    model, frame = build_model(H)
    uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, CELL)
    _require(syndrome.any(), "the frozen m8 sentinel syndrome unexpectedly vanished")

    mam_started = time.perf_counter()
    mam_catalog = build_milp_map_anchors(model.H_check, syndrome, CELL["p"], max_anchors=8)
    mam = build_map_mixture_proposal(model, mam_catalog)
    mam_construction = time.perf_counter() - mam_started

    lsi_started = time.perf_counter()
    _require(LSI_ARTIFACT.is_file(), "historical LSI artifact is missing")
    lsi_artifact = load_logical_stratified_frozen_artifact(LSI_ARTIFACT, model, frame)
    lsi = lsi_artifact.proposal
    lsi_load = time.perf_counter() - lsi_started

    mam_support = _full_support_lower_mass_mam(mam)
    lsi_support = _full_support_lower_mass_lsi(lsi)
    return {
        "probe_version": PROBE_VERSION,
        "authority": "outcome_blind_local_runtime_and_algebra_only",
        "does_not_establish": [
            "A posterior estimate, sector mass, purity, or q_top.",
            "Coverage of unobserved target modes.",
            "A fresh logical-stratified artifact or any remote/formal authority.",
        ],
        "cell": CELL,
        "registry_sha256": registry["registry_sha256"],
        "classical_matrix_sha256": hashlib.sha256(
            np.packbits(H, axis=1, bitorder="little").tobytes(),
        ).hexdigest(),
        "model": {
            "num_qubits": int(model.num_qubits),
            "num_checks": int(model.num_checks),
            "logical_dimension": int(model.k),
            "syndrome_weight": int(syndrome.sum()),
        },
        "m8_disorder_uniform_seed": int(uniform_seed),
        "proposals": [
            {
                "name": "MAM-IMH8-rebuilt",
                "construction_seconds": mam_construction,
                "proposal_sha256": mam.proposal_sha256,
                "anchor_sha256": mam_catalog.anchor_sha256,
                "anchor_count": int(mam_catalog.size),
                "defensive_uniform_coordinate_mass": mam_support,
                "artifact_status": "freshly_rebuilt_for_preflight_only",
                **_benchmark("MAM-IMH8-rebuilt", mam, model, frame, syndrome, seed=101),
            },
            {
                "name": "LSI-IMH-T05-historical-artifact-load",
                "construction_seconds": lsi_load,
                "proposal_sha256": lsi.proposal_sha256,
                "anchor_count": int(lsi.catalog.size),
                "defensive_uniform_coordinate_mass": lsi_support,
                "artifact_status": "historical_load_only_not_reusable_for_successor",
                "historical_artifact_sha256": sha256_file(LSI_ARTIFACT),
                **_benchmark("LSI-IMH-T05-historical-artifact-load", lsi, model, frame, syndrome, seed=202),
            },
        ],
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing probe: {args.output}")
    report = run_probe()
    report["script_sha256"] = sha256_file(Path(__file__))
    report["report_sha256"] = sha256_json(report)
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
