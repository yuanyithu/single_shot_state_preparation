"""Outcome-blind cross-density timing for the proposed iid-MIS mixture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.q0_iid_importance import (
    mixture_log_probability_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import (
    load_logical_stratified_frozen_artifact,
)
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    build_map_mixture_proposal,
    build_milp_map_anchors,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model

from preflight_runtime import CELL, LSI_ARTIFACT, _validate_draw


PROBE_VERSION = "exp102.q0_iid_is.cross_density_runtime.v0"
SAMPLES_PER_SOURCE = 16


def _benchmark_states(name, states, proposals):
    started = time.perf_counter()
    for state in states:
        mixture_log_probability_state(proposals, [1.0 / len(proposals)] * len(proposals), state)
    elapsed = time.perf_counter() - started
    return {
        "source": name,
        "states": len(states),
        "mixture_log_density_seconds_total": elapsed,
        "mixture_log_density_seconds_per_state": elapsed / len(states),
    }


def run_probe():
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    root = Path(__file__).resolve().parents[2]
    registry_path = root / "registry/registry.json"
    registry = load_registry(registry_path)
    _, code, H = load_frozen_code(registry_path, CELL["code_id"])
    model, frame = build_model(H)
    uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, CELL)
    mam_catalog = build_milp_map_anchors(model.H_check, syndrome, CELL["p"], max_anchors=8)
    mam = build_map_mixture_proposal(model, mam_catalog)
    lsi05 = load_logical_stratified_frozen_artifact(LSI_ARTIFACT, model, frame).proposal
    lsi10_path = LSI_ARTIFACT.with_name("lsi_imh_tau_10.npz")
    lsi10 = load_logical_stratified_frozen_artifact(lsi10_path, model, frame).proposal
    proposals = (mam, lsi05, lsi10)
    states_by_source = []
    for source_index, proposal in enumerate(proposals):
        rng = PortablePrng(1000 + source_index)
        states = []
        for _ in range(SAMPLES_PER_SOURCE):
            draw = proposal.sample(rng)
            _validate_draw(proposal, model, frame, syndrome, draw)
            states.append(np.asarray(draw["state"], dtype=np.uint8))
        states_by_source.append(states)
    timings = [
        _benchmark_states(name, states, proposals)
        for name, states in zip(("MAM-IMH8", "LSI-IMH-T05", "LSI-IMH-T10"), states_by_source)
    ]
    projected_total = 16 * 1024 * len(proposals)
    conservative_per_state = max(row["mixture_log_density_seconds_per_state"] for row in timings)
    return {
        "probe_version": PROBE_VERSION,
        "authority": "outcome_blind_local_runtime_and_algebra_only",
        "does_not_establish": [
            "A posterior estimate, sector mass, purity, or q_top.",
            "That a finite iid sample covers an unobserved target mode.",
            "Remote, formal, held-out, or production authority.",
        ],
        "cell": CELL,
        "registry_sha256": registry["registry_sha256"],
        "m8_disorder_uniform_seed": int(uniform_seed),
        "proposal_sha256": [proposal.proposal_sha256 for proposal in proposals],
        "timings": timings,
        "fixed_prospective_schedule": {
            "block_count": 16,
            "draws_per_proposal_per_block": 1024,
            "proposal_count": len(proposals),
            "total_draws": projected_total,
            "conservative_projected_cross_density_seconds": conservative_per_state * projected_total,
        },
        "historical_artifact_sha256": {
            "tau_05": sha256_file(LSI_ARTIFACT),
            "tau_10": sha256_file(lsi10_path),
        },
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
