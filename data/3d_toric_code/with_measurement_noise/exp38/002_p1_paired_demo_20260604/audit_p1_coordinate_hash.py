#!/usr/bin/env python3
"""Audit coordinate-hash cross-L disorder coupling for exp38 P1b."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exp37_sector_ti import _build_disorder_uniforms  # noqa: E402


OUTPUT_PATH = SCRIPT_PATH.parent / "coordinate_hash_audit.json"


def _index(lattice_size: int, type_index: int, i: int, j: int, k: int) -> int:
    return type_index * lattice_size ** 3 + (i * lattice_size + j) * lattice_size + k


def main() -> int:
    seed = 638500
    data3, syndrome3 = _build_disorder_uniforms(
        code_family="3d_toric",
        lattice_size=3,
        num_qubits=3 * 3 ** 3,
        num_checks=3 * 3 ** 3,
        disorder_seed=seed,
        disorder_realization_mode="coordinate_hash",
    )
    data5, syndrome5 = _build_disorder_uniforms(
        code_family="3d_toric",
        lattice_size=5,
        num_qubits=3 * 5 ** 3,
        num_checks=3 * 5 ** 3,
        disorder_seed=seed,
        disorder_realization_mode="coordinate_hash",
    )
    data_matches = []
    syndrome_matches = []
    for type_index in range(3):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx3 = _index(3, type_index, i, j, k)
                    idx5 = _index(5, type_index, i, j, k)
                    data_matches.append(data3[idx3] == data5[idx5])
                    syndrome_matches.append(syndrome3[idx3] == syndrome5[idx5])
    payload = {
        "stage": "P1",
        "audit": "coordinate_hash_cross_l_common_disorder",
        "passed": bool(all(data_matches) and all(syndrome_matches)),
        "seed": int(seed),
        "lattice_sizes": [3, 5],
        "shared_coordinate_count": int(len(data_matches)),
        "data_shared_fraction": float(np.mean(data_matches)),
        "syndrome_shared_fraction": float(np.mean(syndrome_matches)),
        "data_syndrome_same_coordinate_collision_count": int(
            np.count_nonzero(data3 == syndrome3)
        ),
        "mode": "coordinate_hash",
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
