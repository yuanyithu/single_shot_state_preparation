"""Run the frozen local BP-systematic IID-MIS feasibility diagnostic once."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_iid_provenance import (
    IID_PROVENANCE_VERSION,
    draw_provenanced_stratified_iid_mixture,
    validate_provenanced_stratified_iid_mixture,
)

from bp_iid_common import (
    build_context,
    build_proposals,
    load_config,
    raw_content_sha256,
    raw_metadata,
    seed_schedule,
)


def _equal_draws(left, right):
    return (left.block_count == right.block_count
            and left.draws_per_proposal_per_block == right.draws_per_proposal_per_block
            and left.proposal_count == right.proposal_count
            and left.coordinate_dimension == right.coordinate_dimension
            and all(np.array_equal(left_value, right_value)
                    for name, left_value in left.arrays().items()
                    for right_name, right_value in right.arrays().items()
                    if name == right_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to replace BP-systematic IID raw or receipt")
    config, config_sha256 = load_config(args.config)
    _root, registry, _code, _H, model, frame, uniform_seed, syndrome = build_context(config)
    proposal_rows = build_proposals(config, model, syndrome)
    proposals = tuple(row[1] for row in proposal_rows)
    seeds = seed_schedule(config, config_sha256, registry, proposal_rows)
    schedule = config["sample_schedule"]
    draws = draw_provenanced_stratified_iid_mixture(
        model, frame, syndrome, config["cell"]["p"], proposals,
        config["proposal_mixture_weights"], seeds,
        block_count=schedule["block_count"],
        draws_per_proposal_per_block=schedule["draws_per_proposal_per_block"],
    )
    validate_provenanced_stratified_iid_mixture(
        draws, model, frame, syndrome, config["cell"]["p"], proposals,
        config["proposal_mixture_weights"],
    )
    # Regenerate from rebuilt proposal objects before writing any immutable raw.
    replay_rows = build_proposals(config, model, syndrome)
    replay = draw_provenanced_stratified_iid_mixture(
        model, frame, syndrome, config["cell"]["p"], tuple(row[1] for row in replay_rows),
        config["proposal_mixture_weights"], seed_schedule(
            config, config_sha256, registry, replay_rows,
        ), block_count=schedule["block_count"],
        draws_per_proposal_per_block=schedule["draws_per_proposal_per_block"],
    )
    if not _equal_draws(draws, replay):
        raise RuntimeError("BP-systematic IID deterministic generation replay failed")
    metadata = raw_metadata(
        config, config_sha256, registry, model, uniform_seed, syndrome, proposal_rows, seeds,
    )
    metadata["iid_provenance_version"] = IID_PROVENANCE_VERSION
    metadata_json = canonical_json(metadata)
    arrays = draws.arrays()
    content_sha256 = raw_content_sha256(metadata_json, arrays)
    atomic_npz(
        args.output, metadata_json=np.asarray(metadata_json),
        content_sha256=np.asarray(content_sha256), **arrays,
    )
    receipt_core = {
        "contract_version": config["contract_version"],
        "config_sha256": config_sha256,
        "raw_sha256": sha256_file(args.output),
        "raw_content_sha256": content_sha256,
        "deterministic_generation_replay": "PASS",
        "raw_algebra_replay": "PASS",
        "draw_count": int(draws.labels.size),
    }
    receipt = {**receipt_core, "receipt_sha256": sha256_json(receipt_core)}
    atomic_json(args.receipt, receipt)


if __name__ == "__main__":
    main()
