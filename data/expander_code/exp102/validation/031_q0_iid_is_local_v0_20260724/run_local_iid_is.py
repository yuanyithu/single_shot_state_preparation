"""Run the frozen local m8 iid-MIS feasibility diagnostic once."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import atomic_npz, canonical_json
from data.expander_code.exp102.exp102_pipeline.q0_iid_importance import (
    IID_IMPORTANCE_VERSION,
    draw_stratified_iid_mixture,
    validate_stratified_iid_mixture,
)

from local_iid_is_common import (
    build_context,
    build_proposals,
    load_config,
    raw_content_sha256,
    raw_metadata,
    seed_schedule,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace iid-MIS raw: {args.output}")
    config, config_sha256 = load_config(args.config)
    _root, registry, _code, _H, model, frame, uniform_seed, syndrome = build_context(config)
    proposal_rows = build_proposals(config, _root, model, frame, syndrome)
    proposals = tuple(row[1] for row in proposal_rows)
    seeds = seed_schedule(config, config_sha256, registry, proposal_rows)
    draws = draw_stratified_iid_mixture(
        model, frame, syndrome, config["cell"]["p"], proposals,
        config["proposal_mixture_weights"], seeds,
        block_count=config["sample_schedule"]["block_count"],
        draws_per_proposal_per_block=config["sample_schedule"]["draws_per_proposal_per_block"],
    )
    validate_stratified_iid_mixture(
        draws, model, frame, syndrome, config["cell"]["p"], proposals,
        config["proposal_mixture_weights"],
    )
    metadata = raw_metadata(
        config, config_sha256, registry, model, uniform_seed, syndrome, proposal_rows, seeds,
    )
    metadata["iid_importance_version"] = IID_IMPORTANCE_VERSION
    metadata_json = canonical_json(metadata)
    arrays = draws.arrays()
    content_sha256 = raw_content_sha256(metadata_json, arrays)
    atomic_npz(
        args.output,
        metadata_json=np.asarray(metadata_json),
        content_sha256=np.asarray(content_sha256),
        **arrays,
    )


if __name__ == "__main__":
    main()
