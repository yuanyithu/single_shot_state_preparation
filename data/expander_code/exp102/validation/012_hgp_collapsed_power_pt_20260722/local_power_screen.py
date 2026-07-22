"""Local, non-authorizing feasibility screen for collapsed likelihood-power PT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    GlobalSeedIdentity,
    character_d2_estimate,
    character_means,
    character_qtop_estimate,
    frozen_character_set,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    CollapsedPowerPtConfig,
    run_collapsed_power_pt_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.screen_diagnostic import (
    _cell_disorder,
    _registry_with_path,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


VERSION = "exp102.q0_hgp_collapsed.local_feasibility.v1"
CELLS = {
    "m06_c00": {
        "code_id": "m06_c00", "p": 0.04, "disorder_index": 0,
        "disorder_source": "attempt022",
    },
    "m08_c06": {
        "code_id": "m08_c06", "p": 0.04, "disorder_index": 0,
        "disorder_source": "attempt022",
    },
}


def _source_identity():
    module = Path(__file__).resolve().parents[2] / "exp102_pipeline/q0_hgp_collapsed.py"
    digest = sha256_file(module)
    return digest[:40], digest


def _task(code_id, family, trajectory, method, burn, measurement):
    if code_id not in CELLS or family not in ("P", "U"):
        raise ValueError("unknown local feasibility task")
    return {
        "version": VERSION,
        "cell": CELLS[code_id],
        "family": family,
        "trajectory": int(trajectory),
        "method": method,
        "burn_rounds": int(burn),
        "measurement_rounds": int(measurement),
    }


def run_one(args):
    registry_path = Path(args.registry).resolve()
    task = _task(args.code_id, args.family, args.trajectory, args.method,
                 args.burn, args.measurement)
    source_commit, source_sha = _source_identity()
    registry, code, H = load_frozen_code(registry_path, args.code_id)
    registry = _registry_with_path(registry, registry_path)
    model, frame = build_model(H)
    _, epsilon, syndrome = _cell_disorder(
        registry, code, model, task["cell"],
    )
    config = CollapsedPowerPtConfig(
        args.method, task["cell"]["p"], args.burn, args.measurement,
    )
    seed_identity = GlobalSeedIdentity(
        source_commit=source_commit,
        config_sha256=sha256_json({
            "version": VERSION,
            "source_sha256": source_sha,
            "method": args.method,
            "burn": args.burn,
            "measurement": args.measurement,
        }),
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(task["cell"]),
        method_id=args.method,
        resource_tier=f"L{args.burn}_{args.measurement}",
        init_family=args.family,
        trajectory_index=args.trajectory,
        trajectory_namespace="q0_hgp_collapsed_local_feasibility_v1",
    )
    initial = (
        epsilon.copy() if args.family == "P" else
        uniform_hard_coset_state(
            model, syndrome, seed_identity.seed("initialize", "hard_coset"),
        )
    )
    start_wall, start_core = time.monotonic(), time.process_time()
    result = run_collapsed_power_pt_trajectory(
        model, frame, H, syndrome, config, seed_identity, initial,
    )
    wall, core = time.monotonic() - start_wall, time.process_time() - start_core
    payload = {
        "version": np.array(VERSION),
        "task_json": np.array(canonical_json(task)),
        "task_sha256": np.array(sha256_json(task)),
        "source_sha256": np.array(source_sha),
        "registry_sha256": np.array(registry["registry_sha256"]),
        "model_fingerprint": np.array(model.fingerprint()),
        "seed_identity_json": np.array(canonical_json(seed_identity.as_dict())),
        "num_qubits": np.array(model.num_qubits, dtype=np.int32),
        "k": np.array(model.k, dtype=np.int16),
        "core_seconds": np.array(core),
        "wall_seconds": np.array(wall),
    }
    for field, value in result.items():
        if field != "engine":
            payload[field] = np.asarray(value)
    payload["engine"] = np.array(result["engine"])
    output = Path(args.output)
    atomic_npz(output, **payload)
    print(json.dumps({
        "output": str(output), "task_sha256": sha256_json(task),
        "wall_seconds": wall, "core_seconds": core,
        "mean_weight": float(result["measurement_weights"].mean()),
        "round_trips": int(result["round_trips_by_origin"].sum()),
    }, sort_keys=True))


def _load_raw(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["version"].item()) != VERSION:
            raise ValueError(f"wrong prototype version: {path}")
        task = json.loads(str(data["task_json"].item()))
        if sha256_json(task) != str(data["task_sha256"].item()):
            raise ValueError(f"task identity mismatch: {path}")
        return task, {field: data[field].copy() for field in data.files}


def analyze(args):
    root = Path(args.raw_root)
    grouped = {}
    source_hashes = set()
    for path in sorted(root.glob("*.npz")):
        task, raw = _load_raw(path)
        key = (task["cell"]["code_id"], task["method"], task["family"])
        grouped.setdefault(key, []).append((task, raw))
        source_hashes.add(str(raw["source_sha256"].item()))
    if len(source_hashes) != 1:
        raise ValueError("local feasibility raw mixes source identities")
    report = {
        "version": VERSION,
        "purpose": "local_feasibility_only",
        "formal_authorization": False,
        "production_authorization": False,
        "source_sha256": next(iter(source_hashes)),
        "cells": [],
    }
    for code_id in CELLS:
        families = {}
        character_set = None
        for family in ("P", "U"):
            rows = grouped.get((code_id, args.method, family), [])
            rows.sort(key=lambda pair: pair[0]["trajectory"])
            if len(rows) != args.trajectories:
                raise ValueError(f"{code_id}/{family} has {len(rows)} raws")
            k = int(rows[0][1]["k"].item())
            character_set = frozen_character_set(
                k,
                int.from_bytes(hashlib.sha256(
                    f"{VERSION}:{code_id}:characters".encode("ascii")
                ).digest()[:8], "big") & ((1 << 63) - 1),
            )
            traces = [raw["measurement_labels"] for _, raw in rows]
            means, counts = character_means(traces, character_set.masks)
            qtop_full = character_qtop_estimate(character_set, means)
            qtop = {
                field: float(qtop_full[field]) for field in (
                    "q_top", "q_top_trajectory_se", "q_top_character_se",
                    "q_top_total_se",
                )
            }
            trajectory_weights = np.asarray([
                raw["measurement_weights"].mean() for _, raw in rows
            ])
            swap_rates = np.asarray([
                raw["swap_accepts"] / np.maximum(raw["swap_attempts"], 1)
                for _, raw in rows
            ])
            families[family] = {
                "means": means,
                "q_top": qtop,
                "mean_weight": float(trajectory_weights.mean()),
                "mean_weight_se": float(trajectory_weights.std(ddof=1) / np.sqrt(len(rows))),
                "min_swap_rate": float(swap_rates.min()),
                "median_swap_rate": float(np.median(swap_rates)),
                "round_trips": [
                    int(raw["round_trips_by_origin"].sum()) for _, raw in rows
                ],
                "cold_origin_counts": [
                    int(np.count_nonzero(raw["cold_visits_by_origin"])) for _, raw in rows
                ],
                "core_seconds": float(sum(float(raw["core_seconds"].item()) for _, raw in rows)),
                "observations_per_trajectory": counts.tolist(),
            }
        d2_full = character_d2_estimate(
            character_set, families["P"].pop("means"), families["U"].pop("means"),
        )
        d2 = {
            field: float(d2_full[field]) for field in (
                "d2_norm", "d2_trajectory_se", "d2_character_se",
                "d2_total_se",
            )
        }
        q_delta = abs(families["P"]["q_top"]["q_top"] - families["U"]["q_top"]["q_top"])
        num_qubits = int(grouped[(code_id, args.method, "P")][0][1]["num_qubits"].item())
        weight_delta = abs(families["P"]["mean_weight"] - families["U"]["mean_weight"]) / num_qubits
        report["cells"].append({
            "code_id": code_id,
            "method": args.method,
            "families": families,
            "q_top_delta": float(q_delta),
            "d2": d2,
            "normalized_weight_delta": float(weight_delta),
            "provisional_pass": bool(
                q_delta <= 0.04
                and max(0.0, d2["d2_norm"]) + 3.0 * d2["d2_total_se"] <= 0.04
                and weight_delta <= 0.01
                and min(families[family]["min_swap_rate"] for family in ("P", "U")) >= 0.05
                and min(min(families[family]["round_trips"]) for family in ("P", "U")) >= 1
            ),
        })
    report["all_cells_provisional_pass"] = all(
        cell["provisional_pass"] for cell in report["cells"]
    )
    atomic_json(args.output, report)
    print(canonical_json(report))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--registry", required=True)
    run.add_argument("--code-id", choices=sorted(CELLS), required=True)
    run.add_argument("--family", choices=("P", "U"), required=True)
    run.add_argument("--trajectory", type=int, required=True)
    run.add_argument("--method", choices=("HP16", "HP32", "HP64"), default="HP32")
    run.add_argument("--burn", type=int, default=1024)
    run.add_argument("--measurement", type=int, default=4096)
    run.add_argument("--output", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--raw-root", required=True)
    analyze_parser.add_argument("--method", choices=("HP16", "HP32", "HP64"), default="HP32")
    analyze_parser.add_argument("--trajectories", type=int, default=8)
    analyze_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_one(args) if args.command == "run" else analyze(args)


if __name__ == "__main__":
    main()
