"""Measure one-at-a-time m8 physical-p mass construction without sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT.parents[1] / "registry/registry.json"
OUTPUT = ROOT / "mass_table_benchmark.json"
P_VALUES = (0.50, 0.25, 0.10, 0.04)


def main():
    _, _, H = load_frozen_code(REGISTRY, "m08_c06")
    rows = []
    for p in P_VALUES:
        started = time.perf_counter()
        mass = build_classical_coset_mass(H, p, engine="numba")
        elapsed = time.perf_counter() - started
        rows.append({
            "bytes": int(mass.nbytes),
            "elapsed_seconds": float(elapsed),
            "maximum": float(mass.max()),
            "minimum": float(mass.min()),
            "p": float(p),
            "sha256_be_f8": hashlib.sha256(
                np.asarray(mass, dtype=">f8").tobytes()
            ).hexdigest(),
            "sum": float(mass.sum(dtype=np.float64)),
        })
        del mass
    payload = {
        "benchmark_version": "exp102.q0_hgp_physical_pt.mass_benchmark.v0",
        "code_id": "m08_c06",
        "h_shape": [int(value) for value in H.shape],
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "rows": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["benchmark_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
