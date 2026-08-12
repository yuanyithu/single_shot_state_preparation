#!/usr/bin/env python3
"""Fail-closed validation and adaptive decisions for the nd-3 legacy pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MODEL = "legacy_delta_only"
HISTORICAL_Q = np.array(
    [0.012, 0.018, 0.022, 0.026, 0.030, 0.034, 0.040, 0.048, 0.058, 0.070],
    dtype=np.float64,
)
PILOT_Q = (0.012, 0.022, 0.030)
HISTORICAL_L = (3, 4, 5, 7)
NBOOT = 10000
BASELINE_BOOTSTRAP_SEED = 20260811
HISTORICAL_CROSSING_SEED = 20260620
MINIMUM_PASS_FRACTION = 0.9
EXPECTED_RAW_HASHES = {
    "nd1": "4b485473380b3142518c97639e5aa5adb2a367896e16c77daabef348a4db1173",
    "nd2": "6e3f4587afb9dac384735271540da14aafa05180340879d3736e1c6aadffc09b",
}
EXPECTED_BASELINE = {
    0.012: (-0.0015961667222288503, -0.004551326, -0.000118068),
    0.022: (-0.00801918296590766, -0.016424496, -0.001936170),
    0.030: (-0.014176815408754652, -0.024755156, -0.004607196),
}
FIXED_MANIFEST = {
    "mode": "sector_ti",
    "sector_observable": "corrected_c_eta_section",
    "code_family": "3d_toric",
    "projection_mode": "linear",
    "common_disorder_across_q": True,
    "disorder_seed_scope": "disorder_index",
    "disorder_realization_mode": "rng_stream",
    "num_kp_grid_points": 129,
    "num_burn_in_sweeps": 512,
    "max_effective_num_burn_in_sweeps": 512,
    "num_measurements": 8192,
    "num_sweeps_between_measurements": 2,
    "block_count": 128,
    "num_bootstrap": 800,
    "winding_heatbath_sweeps": 1,
    "use_numba": True,
    "grid_tv_warning": 0.02,
    "grid_q_top_warning": 0.02,
}


@dataclass(frozen=True)
class Cell:
    p: float
    q: float
    lattice_size: int
    seeds: np.ndarray
    sample_seeds: np.ndarray
    w0: np.ndarray
    flags: np.ndarray
    source_paths: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def softmax_w0(delta_f: np.ndarray) -> np.ndarray:
    delta_f = np.asarray(delta_f, dtype=np.float64)
    if delta_f.shape[-1] != 8 or not np.all(np.isfinite(delta_f)):
        raise ValueError("delta_f must be finite with eight logical sectors")
    shifted = delta_f - np.min(delta_f, axis=-1, keepdims=True)
    weights = np.exp(-shifted)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    if not np.all(np.isfinite(weights)):
        raise ValueError("non-finite softmax weights")
    return weights[..., 0]


def validate_flags(flags: np.ndarray, path: Path) -> None:
    allowed = {"PASS", "TI_GRID_TV_WARN", "TI_GRID_QTOP_WARN"}
    for value in np.asarray(flags).flat:
        components = set(str(value).split(";"))
        if not components or not components <= allowed or "MISSING" in components:
            raise ValueError(f"{path}: invalid flag {value!r}")


def validate_historical_manifest(manifest: dict[str, Any], path: Path) -> None:
    for key, expected in FIXED_MANIFEST.items():
        actual = manifest.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, abs_tol=1e-15):
                raise ValueError(f"{path}: manifest {key} mismatch")
        elif actual != expected:
            raise ValueError(f"{path}: manifest {key} mismatch: {actual!r}")
    if not math.isclose(float(manifest.get("p_value")), 0.22, abs_tol=1e-15):
        raise ValueError(f"{path}: manifest p mismatch")
    if manifest.get("lattice_sizes") != list(HISTORICAL_L):
        raise ValueError(f"{path}: manifest L grid mismatch")
    if not np.array_equal(np.asarray(manifest.get("q_values")), HISTORICAL_Q):
        raise ValueError(f"{path}: manifest q grid mismatch")
    if int(manifest.get("num_disorder_samples")) != 192:
        raise ValueError(f"{path}: historical shard is not 192 disorders")


def load_historical(files: list[Path]) -> tuple[dict[tuple[float, float, int], Cell], dict[str, Any]]:
    if len(files) != 2:
        raise ValueError("exactly two historical p=.220 shards are required")
    by_node = {"nd1": None, "nd2": None}
    for path in files:
        for node in by_node:
            if node in path.parts:
                if by_node[node] is not None:
                    raise ValueError(f"duplicate {node} raw path")
                by_node[node] = path
                break
    if any(path is None for path in by_node.values()):
        raise ValueError("historical paths must identify nd1 and nd2")

    source_records: list[dict[str, Any]] = []
    pieces: dict[tuple[float, float, int], list[tuple[np.ndarray, ...]]] = {}
    global_sample_seeds: set[int] = set()
    for node in ("nd1", "nd2"):
        path = Path(by_node[node])  # type: ignore[arg-type]
        actual_hash = sha256_file(path)
        if actual_hash != EXPECTED_RAW_HASHES[node]:
            raise ValueError(f"{path}: SHA256 mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            manifest = json.loads(str(loaded["manifest_json"].item()))
            validate_historical_manifest(manifest, path)
            lattice_sizes = tuple(int(value) for value in loaded["lattice_size_list"])
            q_values = np.asarray(loaded["q_values"], dtype=np.float64)
            if lattice_sizes != HISTORICAL_L or not np.array_equal(q_values, HISTORICAL_Q):
                raise ValueError(f"{path}: stored grids mismatch")
            if not math.isclose(float(loaded["p_value"]), 0.22, abs_tol=1e-15):
                raise ValueError(f"{path}: stored p mismatch")
            delta_f = np.asarray(loaded["delta_f_per_disorder"], dtype=np.float64)
            flags = np.asarray(loaded["flags_per_disorder"])
            disorder_seeds = np.asarray(loaded["disorder_seed_per_disorder"], dtype=np.int64)
            sample_seeds = np.asarray(loaded["sample_seed_per_disorder"], dtype=np.int64)
            expected_shape = (4, 10, 192)
            if delta_f.shape != expected_shape + (8,):
                raise ValueError(f"{path}: delta_f shape mismatch")
            for array, name in (
                (flags, "flags"),
                (disorder_seeds, "disorder seeds"),
                (sample_seeds, "sample seeds"),
            ):
                if array.shape != expected_shape:
                    raise ValueError(f"{path}: {name} shape mismatch")
            if not np.all(np.isfinite(delta_f)):
                raise ValueError(f"{path}: non-finite delta_f")
            validate_flags(flags, path)
            for li, lattice_size in enumerate(lattice_sizes):
                for qi, q_value in enumerate(q_values):
                    seeds = disorder_seeds[li, qi]
                    samples = sample_seeds[li, qi]
                    if len(np.unique(seeds)) != 192 or len(np.unique(samples)) != 192:
                        raise ValueError(f"{path}: duplicate seeds in a cell")
                    for sample_seed in samples:
                        sample_seed_int = int(sample_seed)
                        if sample_seed_int in global_sample_seeds:
                            raise ValueError(f"{path}: repeated sample seed across cells")
                        global_sample_seeds.add(sample_seed_int)
                    pieces.setdefault((0.22, float(q_value), lattice_size), []).append(
                        (seeds, samples, softmax_w0(delta_f[li, qi]), flags[li, qi])
                    )
        source_records.append(
            {"node": node, "path": str(path), "sha256": actual_hash, "bytes": path.stat().st_size}
        )

    cells: dict[tuple[float, float, int], Cell] = {}
    canonical_seed_set: np.ndarray | None = None
    for key, cell_pieces in pieces.items():
        seeds = np.concatenate([piece[0] for piece in cell_pieces])
        samples = np.concatenate([piece[1] for piece in cell_pieces])
        w0 = np.concatenate([piece[2] for piece in cell_pieces])
        flags = np.concatenate([piece[3] for piece in cell_pieces])
        order = np.argsort(seeds)
        seeds, samples, w0, flags = seeds[order], samples[order], w0[order], flags[order]
        if len(seeds) != 384 or len(np.unique(seeds)) != 384:
            raise ValueError(f"historical cell {key} is not 384 unique disorders")
        if canonical_seed_set is None:
            canonical_seed_set = seeds.copy()
        elif not np.array_equal(canonical_seed_set, seeds):
            raise ValueError("historical cells do not share the complete disorder seed set")
        cells[key] = Cell(
            p=key[0],
            q=key[1],
            lattice_size=key[2],
            seeds=seeds,
            sample_seeds=samples,
            w0=w0,
            flags=flags,
            source_paths=tuple(record["path"] for record in source_records),
        )
    if len(cells) != 40:
        raise ValueError("historical grid is incomplete")
    metadata = {"sources": source_records, "num_cells": len(cells), "num_disorders": 384}
    return cells, metadata


def cross_q(q_values: np.ndarray, differences: np.ndarray) -> float | None:
    nonzero = np.flatnonzero(differences)
    if len(nonzero) == 0:
        return None
    previous = int(nonzero[0])
    for current_value in nonzero[1:]:
        current = int(current_value)
        if differences[previous] * differences[current] < 0:
            fraction = differences[previous] / (
                differences[previous] - differences[current]
            )
            return float(
                q_values[previous] + fraction * (q_values[current] - q_values[previous])
            )
        previous = current
    return None


def crossing_ci(
    q_values: np.ndarray,
    small: np.ndarray,
    large: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float | None, list[float | None], float]:
    if small.shape != large.shape or small.shape[0] != len(q_values):
        raise ValueError("crossing arrays are not aligned")
    point = cross_q(q_values, np.mean(small, axis=1) - np.mean(large, axis=1))
    ndis = small.shape[1]
    crossings: list[float] = []
    for _ in range(NBOOT):
        indices = rng.integers(0, ndis, ndis)
        value = cross_q(
            q_values,
            np.mean(small[:, indices], axis=1) - np.mean(large[:, indices], axis=1),
        )
        if value is not None:
            crossings.append(value)
    if len(crossings) <= 10:
        return point, [None, None], len(crossings) / NBOOT
    bounds = np.quantile(np.asarray(crossings), [0.025, 0.975])
    return point, [float(bounds[0]), float(bounds[1])], len(crossings) / NBOOT


def historical_headline(cells: dict[tuple[float, float, int], Cell]) -> dict[str, Any]:
    matrices = {
        lattice_size: np.stack(
            [cells[(0.22, float(q_value), lattice_size)].w0 for q_value in HISTORICAL_Q]
        )
        for lattice_size in HISTORICAL_L
    }
    rng = np.random.default_rng(HISTORICAL_CROSSING_SEED)
    pairs = [(a, b) for index, a in enumerate(HISTORICAL_L) for b in HISTORICAL_L[index + 1 :]]
    results: dict[tuple[int, int], tuple[float | None, list[float | None], float]] = {}
    for pair in pairs:
        results[pair] = crossing_ci(
            HISTORICAL_Q, matrices[pair[0]], matrices[pair[1]], rng
        )
    point, ci, fraction = results[(3, 7)]
    if point is None or ci[0] is None or ci[1] is None:
        raise ValueError("historical L3-L7 crossing is unresolved")
    rounded = [round(point, 4), round(ci[0], 4), round(ci[1], 4)]
    if rounded != [0.0349, 0.0327, 0.0363]:
        raise ValueError(f"historical headline mismatch: {rounded}")
    return {
        "observable": "w0",
        "L_pair": "L3-L7",
        "q_crossing": point,
        "ci95": ci,
        "bootstrap_crossing_fraction": fraction,
        "nboot": NBOOT,
        "rng_seed": HISTORICAL_CROSSING_SEED,
    }


def endpoint_from_cells(
    cell3: Cell,
    cell7: Cell,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not math.isclose(cell3.p, cell7.p, abs_tol=1e-15) or not math.isclose(
        cell3.q, cell7.q, abs_tol=1e-15
    ):
        raise ValueError("endpoint cells do not have the same p/q")
    if not np.array_equal(cell3.seeds, cell7.seeds):
        raise ValueError("L3/L7 disorder seeds are not exactly aligned")
    if len(cell3.seeds) == 0:
        raise ValueError("empty endpoint")
    differences = cell3.w0 - cell7.w0
    if not np.all(np.isfinite(differences)):
        raise ValueError("non-finite w0 difference")
    indices = rng.integers(0, len(differences), size=(NBOOT, len(differences)))
    bootstrap = np.mean(differences[indices], axis=1)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    point = float(np.mean(differences))
    if point > 0 and ci[0] > 0:
        state = "POS"
    elif point < 0 and ci[1] < 0:
        state = "NEG"
    else:
        state = "UNRESOLVED"
    pass3 = float(np.mean(cell3.flags == "PASS"))
    pass7 = float(np.mean(cell7.flags == "PASS"))
    eligible = state in {"POS", "NEG"} and min(pass3, pass7) >= MINIMUM_PASS_FRACTION
    return {
        "p": cell3.p,
        "q": cell3.q,
        "num_disorders": len(differences),
        "D": point,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "state": state,
        "pass_fraction_L3": pass3,
        "pass_fraction_L7": pass7,
        "eligible": eligible,
        "bootstrap_method": "common-seed L3/L7 paired resampling; legacy rng_stream",
        "nboot": NBOOT,
    }


def baseline_endpoints(cells: dict[tuple[float, float, int], Cell]) -> dict[float, dict[str, Any]]:
    rng = np.random.default_rng(BASELINE_BOOTSTRAP_SEED)
    results: dict[float, dict[str, Any]] = {}
    for q_value in PILOT_Q:
        result = endpoint_from_cells(
            cells[(0.22, q_value, 3)], cells[(0.22, q_value, 7)], rng
        )
        expected = EXPECTED_BASELINE[q_value]
        actual = (result["D"], result["ci95_low"], result["ci95_high"])
        tolerances = (1e-14, 5e-9, 5e-9)
        if any(abs(a - e) > tolerance for a, e, tolerance in zip(actual, expected, tolerances)):
            raise ValueError(f"baseline D mismatch at q={q_value}: {actual}")
        if result["state"] != "NEG" or not result["eligible"]:
            raise ValueError(f"baseline endpoint is not strict NEG at q={q_value}")
        result["bootstrap_seed"] = BASELINE_BOOTSTRAP_SEED
        results[q_value] = result
    return results


def validate_p022(args: argparse.Namespace) -> int:
    files = [Path(args.nd1).resolve(), Path(args.nd2).resolve()]
    cells, metadata = load_historical(files)
    headline = historical_headline(cells)
    baseline = baseline_endpoints(cells)
    payload = {
        "schema_version": 1,
        "validated_at": utc_now(),
        "model": MODEL,
        "status": "PASS",
        "historical_grid": metadata,
        "frozen_headline_reproduction": headline,
        "pilot_baseline": {f"{q_value:.3f}": baseline[q_value] for q_value in PILOT_Q},
        "permissions": {
            "allowed": "legacy_delta_only adaptive pilot baseline",
            "not_allowed": [
                "true_posterior",
                "reduced_MLD",
                "asymptotic_threshold",
                "paired_coordinate_disorder_claim",
            ],
        },
    }
    atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def load_new_cells(root: Path) -> dict[tuple[float, float, int], Cell]:
    paths = sorted(root.glob("**/cells/*.npz"))
    cells: dict[tuple[float, float, int], Cell] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as loaded:
            manifest = json.loads(str(loaded["manifest_json"].item()))
            if manifest.get("model") != MODEL or manifest.get("fixed_config", {}).get(
                "projection_mode"
            ) != "linear":
                raise ValueError(f"{path}: model/config mismatch")
            p_value = float(loaded["p_value"])
            q_value = float(loaded["q_value"])
            lattice_size = int(loaded["lattice_size"])
            num_disorders = int(loaded["num_disorder_samples"])
            if p_value not in {0.225, 0.23, 0.235, 0.24}:
                raise ValueError(f"{path}: p is outside the candidate pool")
            if q_value not in PILOT_Q or lattice_size not in {3, 7}:
                raise ValueError(f"{path}: q/L is outside the pilot grid")
            if num_disorders != 48:
                raise ValueError(f"{path}: pilot cell is not 48 disorders")
            seeds = np.asarray(loaded["disorder_seed_per_disorder"], dtype=np.int64)
            sample_seeds = np.asarray(loaded["sample_seed_per_disorder"], dtype=np.int64)
            delta_f = np.asarray(loaded["delta_f_per_disorder"], dtype=np.float64)
            flags = np.asarray(loaded["flags_per_disorder"])
            if seeds.shape != (48,) or sample_seeds.shape != (48,) or delta_f.shape != (48, 8):
                raise ValueError(f"{path}: cell array shape mismatch")
            if not np.array_equal(np.asarray(loaded["disorder_index_per_disorder"]), np.arange(48)):
                raise ValueError(f"{path}: disorder indices are incomplete")
            if len(np.unique(seeds)) != 48 or len(np.unique(sample_seeds)) != 48:
                raise ValueError(f"{path}: duplicate seed")
            validate_flags(flags, path)
            key = (p_value, q_value, lattice_size)
            if key in cells:
                raise ValueError(f"duplicate new cell {key}")
            cells[key] = Cell(
                p=p_value,
                q=q_value,
                lattice_size=lattice_size,
                seeds=seeds,
                sample_seeds=sample_seeds,
                w0=softmax_w0(delta_f),
                flags=flags,
                source_paths=(str(path),),
            )
    for p_value, q_value, lattice_size in cells:
        other = (p_value, q_value, 7 if lattice_size == 3 else 3)
        if other not in cells:
            raise ValueError(f"missing L pair for p={p_value}, q={q_value}")
        if not np.array_equal(cells[(p_value, q_value, lattice_size)].seeds, cells[other].seeds):
            raise ValueError(f"L3/L7 seed mismatch for p={p_value}, q={q_value}")
    return cells


def endpoint_seed(p_value: float, q_value: float) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [BASELINE_BOOTSTRAP_SEED, int(round(p_value * 1000)), int(round(q_value * 1000))]
    )
    return np.random.default_rng(seed_sequence)


def opposite(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first["eligible"] and second["eligible"] and first["state"] != second["state"]


def same_sign(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first["eligible"] and second["eligible"] and first["state"] == second["state"]


def decide_q(q_value: float, stats: dict[float, dict[str, Any]]) -> dict[str, Any]:
    baseline = stats[0.22]
    if 0.23 not in stats:
        return {"status": "NEXT", "next_p": 0.23, "reason": "initial_wave_required"}
    p230 = stats[0.23]
    if not baseline["eligible"] or not p230["eligible"]:
        return {"status": "STOP", "reason": "unresolved_or_low_pass_decision_endpoint"}
    if opposite(baseline, p230):
        if 0.225 not in stats:
            return {"status": "NEXT", "next_p": 0.225, "reason": "refine_0220_0230_flip"}
        midpoint = stats[0.225]
        if not midpoint["eligible"]:
            return {"status": "STOP", "reason": "unresolved_or_low_pass_midpoint"}
        for low, high in ((0.22, 0.225), (0.225, 0.23)):
            if opposite(stats[low], stats[high]):
                return {
                    "status": "BRACKET",
                    "bracket_low": low,
                    "bracket_high": high,
                    "reason": "strict_adjacent_width_0.005",
                }
        return {"status": "STOP", "reason": "midpoint_did_not_form_adjacent_bracket"}
    if not same_sign(baseline, p230):
        return {"status": "STOP", "reason": "non_strict_0220_0230_relation"}
    if 0.24 not in stats:
        return {"status": "NEXT", "next_p": 0.24, "reason": "no_flip_at_0230"}
    p240 = stats[0.24]
    if not p240["eligible"]:
        return {"status": "STOP", "reason": "unresolved_or_low_pass_at_0240"}
    if not opposite(baseline, p240):
        return {"status": "STOP", "reason": "no_flip_by_0240"}
    if 0.235 not in stats:
        return {"status": "NEXT", "next_p": 0.235, "reason": "refine_0230_0240_flip"}
    midpoint = stats[0.235]
    if not midpoint["eligible"]:
        return {"status": "STOP", "reason": "unresolved_or_low_pass_midpoint"}
    for low, high in ((0.23, 0.235), (0.235, 0.24)):
        if opposite(stats[low], stats[high]):
            return {
                "status": "BRACKET",
                "bracket_low": low,
                "bracket_high": high,
                "reason": "strict_adjacent_width_0.005",
            }
    return {"status": "STOP", "reason": "midpoint_did_not_form_adjacent_bracket"}


def decide(args: argparse.Namespace) -> int:
    historical, _ = load_historical([Path(args.nd1).resolve(), Path(args.nd2).resolve()])
    baseline = baseline_endpoints(historical)
    new_cells = load_new_cells(Path(args.cells_root).resolve())
    new_cell_count = len(new_cells)
    if new_cell_count > 18:
        raise ValueError("adaptive pilot exceeded 18 cells")

    all_stats: dict[float, dict[float, dict[str, Any]]] = {q_value: {0.22: baseline[q_value]} for q_value in PILOT_Q}
    for p_value, q_value, lattice_size in sorted(new_cells):
        if lattice_size != 3:
            continue
        result = endpoint_from_cells(
            new_cells[(p_value, q_value, 3)],
            new_cells[(p_value, q_value, 7)],
            endpoint_seed(p_value, q_value),
        )
        result["bootstrap_seed_material"] = [
            BASELINE_BOOTSTRAP_SEED,
            int(round(p_value * 1000)),
            int(round(q_value * 1000)),
        ]
        all_stats[q_value][p_value] = result

    actions = {q_value: decide_q(q_value, all_stats[q_value]) for q_value in PILOT_Q}
    next_pairs = sorted(
        [[action["next_p"], q_value] for q_value, action in actions.items() if action["status"] == "NEXT"]
    )
    brackets = {
        f"{q_value:.3f}": [action["bracket_low"], action["bracket_high"]]
        for q_value, action in actions.items()
        if action["status"] == "BRACKET"
    }
    production_new_endpoints = sum(
        sum(1 for p_value in bracket if not math.isclose(p_value, 0.22, abs_tol=1e-15))
        for bracket in brackets.values()
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "model": MODEL,
        "scope": "adaptive_pilot_only",
        "new_cells_completed": new_cell_count,
        "maximum_adaptive_cells": 18,
        "endpoint_statistics": {
            f"{q_value:.3f}": {
                f"{p_value:.3f}": result
                for p_value, result in sorted(all_stats[q_value].items())
            }
            for q_value in PILOT_Q
        },
        "actions": {f"{q_value:.3f}": action for q_value, action in actions.items()},
        "next_pairs": next_pairs,
        "strict_brackets": brackets,
        "pilot_complete": not next_pairs,
        "production_not_authorized": True,
        "estimated_future_production_core_hours": 960 * production_new_endpoints,
    }
    output_json = Path(args.output_json)
    atomic_write_json(output_json, payload)

    rows: list[dict[str, Any]] = []
    for q_value in PILOT_Q:
        action = actions[q_value]
        for p_value, result in sorted(all_stats[q_value].items()):
            rows.append(
                {
                    "model": MODEL,
                    "q": f"{q_value:.3f}",
                    "p": f"{p_value:.3f}",
                    "D": f"{result['D']:.12g}",
                    "ci95_low": f"{result['ci95_low']:.12g}",
                    "ci95_high": f"{result['ci95_high']:.12g}",
                    "state": result["state"],
                    "pass_fraction_L3": f"{result['pass_fraction_L3']:.6f}",
                    "pass_fraction_L7": f"{result['pass_fraction_L7']:.6f}",
                    "eligible": str(bool(result["eligible"])).lower(),
                    "q_action": action["status"],
                    "next_p": "" if "next_p" not in action else f"{action['next_p']:.3f}",
                    "bracket_low": "" if "bracket_low" not in action else f"{action['bracket_low']:.3f}",
                    "bracket_high": "" if "bracket_high" not in action else f"{action['bracket_high']:.3f}",
                    "reason": action["reason"],
                }
            )
    fieldnames = list(rows[0])
    atomic_write_csv(Path(args.output_csv), rows, fieldnames)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-p022")
    validate_parser.add_argument("--nd1", required=True)
    validate_parser.add_argument("--nd2", required=True)
    validate_parser.add_argument("--output", required=True)
    validate_parser.set_defaults(func=validate_p022)

    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--nd1", required=True)
    decide_parser.add_argument("--nd2", required=True)
    decide_parser.add_argument("--cells-root", required=True)
    decide_parser.add_argument("--output-json", required=True)
    decide_parser.add_argument("--output-csv", required=True)
    decide_parser.set_defaults(func=decide)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
