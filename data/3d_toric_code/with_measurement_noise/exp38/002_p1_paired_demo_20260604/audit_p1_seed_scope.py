#!/usr/bin/env python3
"""Audit exp38 P1 disorder seed scope before any expensive local TI run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exp37_sector_ti import _build_tasks  # noqa: E402


OUTPUT_PATH = SCRIPT_PATH.parent / "seed_scope_audit.json"


def main() -> int:
    args = argparse.Namespace(
        lattice_sizes="3,4,5",
        q_values="0.18",
        num_disorder_samples=8,
        seed_base=638000,
        common_disorder_across_q=True,
        disorder_seed_scope="disorder_index",
        code_family="3d_toric",
        projection_mode="linear",
        p=0.05,
        num_kp_grid_points=129,
        num_burn_in_sweeps=512,
        num_measurements=8192,
        num_sweeps_between_measurements=2,
        block_count=128,
        num_bootstrap=800,
        winding_heatbath_sweeps=1,
        use_numba=True,
        grid_tv_warning=0.02,
        grid_q_top_warning=0.02,
        debug_checks=False,
    )
    tasks, lattice_sizes, q_values = _build_tasks(args)

    disorder_seed_by_index: dict[int, dict[int, int]] = {}
    sample_seed_collisions = 0
    sample_seeds: set[int] = set()
    for task in tasks:
        disorder_seed_by_index.setdefault(
            int(task["disorder_index"]),
            {},
        )[int(task["lattice_size"])] = int(task["disorder_seed"])
        sample_seed = int(task["sample_seed"])
        if sample_seed in sample_seeds:
            sample_seed_collisions += 1
        sample_seeds.add(sample_seed)

    per_index_pass = {}
    for disorder_index, seeds_by_l in disorder_seed_by_index.items():
        per_index_pass[disorder_index] = (
            len(seeds_by_l) == len(lattice_sizes)
            and len(set(seeds_by_l.values())) == 1
        )
    passed = bool(all(per_index_pass.values()) and sample_seed_collisions == 0)
    payload = {
        "stage": "P1",
        "audit": "cross_l_common_disorder_seed_scope",
        "passed": passed,
        "lattice_sizes": [int(value) for value in lattice_sizes],
        "q_values": [float(value) for value in q_values],
        "num_disorder_samples": int(args.num_disorder_samples),
        "seed_base": int(args.seed_base),
        "common_disorder_across_q": bool(args.common_disorder_across_q),
        "disorder_seed_scope": str(args.disorder_seed_scope),
        "per_disorder_index_shared_across_l": {
            str(key): bool(value) for key, value in per_index_pass.items()
        },
        "disorder_seed_by_index": {
            str(key): {str(l): seed for l, seed in sorted(value.items())}
            for key, value in sorted(disorder_seed_by_index.items())
        },
        "sample_seed_collisions": int(sample_seed_collisions),
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
