#!/usr/bin/env python3
"""Safe task-checkpoint runner for the legacy_delta_only nd-3 pilot.

This file deliberately wraps the validated sector-TI task implementation
without changing its public CLI or the frozen exp40/41 evidence tree.  Every
(p, q, L, disorder) result is written atomically before the next scientific
gate is considered, so an interrupted wave can resume without recomputing
completed tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MODEL = "legacy_delta_only"
EXPECTED_SOURCE_SHA256 = (
    "428e8383fc5ff7a9f31529f5b604c4a2fadf7302d7779367482e39af773114eb"
)
FIXED_CONFIG: dict[str, Any] = {
    "code_family": "3d_toric",
    "projection_mode": "linear",
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
    "disorder_seed_scope": "disorder_index",
    "disorder_realization_mode": "rng_stream",
}
CHECKPOINT_KEYS = {
    "task_id",
    "task_hash",
    "config_hash",
    "lattice_size",
    "p_value",
    "q_value",
    "disorder_index",
    "seed",
    "disorder_seed",
    "sample_seed",
    "projection_mode",
    "disorder_seed_scope",
    "disorder_realization_mode",
    "delta_f",
    "weights",
    "delta_f_stderr",
    "weights_stderr",
    "q_top",
    "q_top_stderr",
    "q_top_ci95",
    "grid_tv",
    "grid_q_top_abs_diff",
    "flags",
    "wall_time_seconds",
    "num_burn_in_sweeps",
    "max_effective_num_burn_in_sweeps",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "src" / "exp37_sector_ti.py").is_file():
            return candidate
    raise RuntimeError("cannot locate project root containing src/exp37_sector_ti.py")


def probability_tag(value: float) -> str:
    scaled = int(round(float(value) * 1000.0))
    if not math.isclose(scaled / 1000.0, float(value), abs_tol=1e-12):
        raise ValueError(f"probability is not on the 0.001 grid: {value}")
    return f"{scaled:04d}"


def parse_pairs(value: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for entry in value.split(","):
        if not entry.strip():
            continue
        pieces = entry.split(":")
        if len(pieces) != 2:
            raise ValueError(f"invalid p:q pair: {entry}")
        pair = (float(pieces[0]), float(pieces[1]))
        if pair in pairs:
            raise ValueError(f"duplicate p:q pair: {entry}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("at least one p:q pair is required")
    return pairs


def task_id(task: dict[str, Any]) -> str:
    return (
        f"p{probability_tag(task['p_value'])}_q{probability_tag(task['q_value'])}_"
        f"L{int(task['lattice_size']):02d}_d{int(task['disorder_index']):05d}"
    )


def checkpoint_path(run_root: Path, task: dict[str, Any]) -> Path:
    return (
        run_root
        / "tasks"
        / f"p{probability_tag(task['p_value'])}"
        / f"q{probability_tag(task['q_value'])}"
        / f"L{int(task['lattice_size']):02d}"
        / f"d{int(task['disorder_index']):05d}.npz"
    )


def build_tasks(
    pairs: list[tuple[float, float]],
    lattice_sizes: list[int],
    num_disorder_samples: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for p_value, q_value in pairs:
        q_seed_offset = 1009 * int(round(10000 * q_value))
        for lattice_size in lattice_sizes:
            for disorder_index in range(num_disorder_samples):
                disorder_seed = int(seed_base) + disorder_index
                sample_seed = (
                    int(seed_base)
                    + 1000003 * int(lattice_size)
                    + q_seed_offset
                    + disorder_index
                )
                task = {
                    **FIXED_CONFIG,
                    "lattice_size": int(lattice_size),
                    "p_value": float(p_value),
                    "q_value": float(q_value),
                    "disorder_index": int(disorder_index),
                    "seed": int(sample_seed),
                    "disorder_seed": int(disorder_seed),
                    "sample_seed": int(sample_seed),
                }
                task["task_id"] = task_id(task)
                task["task_hash"] = sha256_bytes(canonical_json(task).encode("utf-8"))
                tasks.append(task)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("task IDs are not unique")
    # Long L=7 work first minimizes the tail while a single pool owns the CPU cap.
    return sorted(
        tasks,
        key=lambda item: (
            -int(item["lattice_size"]),
            int(item["disorder_index"]),
            float(item["p_value"]),
            float(item["q_value"]),
        ),
    )


def scalar(loaded: np.lib.npyio.NpzFile, key: str) -> Any:
    value = loaded[key]
    if value.shape != ():
        raise ValueError(f"{key} is not scalar")
    return value.item()


def validate_checkpoint(path: Path, task: dict[str, Any], config_hash: str) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        missing = CHECKPOINT_KEYS - set(loaded.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        if str(scalar(loaded, "task_id")) != task["task_id"]:
            raise ValueError(f"{path}: task_id mismatch")
        if str(scalar(loaded, "task_hash")) != task["task_hash"]:
            raise ValueError(f"{path}: task_hash mismatch")
        if str(scalar(loaded, "config_hash")) != config_hash:
            raise ValueError(f"{path}: config_hash mismatch")
        for key, expected in (
            ("lattice_size", int(task["lattice_size"])),
            ("disorder_index", int(task["disorder_index"])),
            ("disorder_seed", int(task["disorder_seed"])),
            ("sample_seed", int(task["sample_seed"])),
        ):
            if int(scalar(loaded, key)) != expected:
                raise ValueError(f"{path}: {key} mismatch")
        for key, expected in (
            ("p_value", float(task["p_value"])),
            ("q_value", float(task["q_value"])),
        ):
            if not math.isclose(float(scalar(loaded, key)), expected, abs_tol=1e-15):
                raise ValueError(f"{path}: {key} mismatch")
        for key in ("delta_f", "weights", "delta_f_stderr", "weights_stderr"):
            values = np.asarray(loaded[key], dtype=np.float64)
            if values.shape != (8,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: invalid {key}")
        if not math.isclose(float(np.sum(loaded["weights"])), 1.0, abs_tol=1e-10):
            raise ValueError(f"{path}: weights do not sum to one")
        if str(scalar(loaded, "flags")) == "MISSING":
            raise ValueError(f"{path}: MISSING flag")


def execute_task(payload: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    task, path_text, config_hash = payload
    project_root = find_project_root()
    sys.path.insert(0, str(project_root / "src"))
    from exp37_sector_ti import _run_single_ti_task  # pylint: disable=import-outside-toplevel

    output_path = Path(path_text)
    result = _run_single_ti_task(task)
    atomic_savez(
        output_path,
        task_id=np.array(task["task_id"]),
        task_hash=np.array(task["task_hash"]),
        config_hash=np.array(config_hash),
        lattice_size=np.int64(result["lattice_size"]),
        p_value=np.float64(result["p_value"]),
        q_value=np.float64(result["q_value"]),
        disorder_index=np.int64(result["disorder_index"]),
        seed=np.int64(result["seed"]),
        disorder_seed=np.int64(result["disorder_seed"]),
        sample_seed=np.int64(result["sample_seed"]),
        projection_mode=np.array(result["projection_mode"]),
        disorder_seed_scope=np.array(result["disorder_seed_scope"]),
        disorder_realization_mode=np.array(result["disorder_realization_mode"]),
        delta_f=np.asarray(result["delta_f"], dtype=np.float64),
        weights=np.asarray(result["weights"], dtype=np.float64),
        delta_f_stderr=np.asarray(result["delta_f_stderr"], dtype=np.float64),
        weights_stderr=np.asarray(result["weights_stderr"], dtype=np.float64),
        q_top=np.float64(result["q_top"]),
        q_top_stderr=np.float64(result["q_top_stderr"]),
        q_top_ci95=np.asarray(result["q_top_ci95"], dtype=np.float64),
        grid_tv=np.float64(result["grid_tv"]),
        grid_q_top_abs_diff=np.float64(result["grid_q_top_abs_diff"]),
        flags=np.array(result["flags"]),
        wall_time_seconds=np.float64(result["wall_time_seconds"]),
        num_burn_in_sweeps=np.int64(result["num_burn_in_sweeps"]),
        max_effective_num_burn_in_sweeps=np.int64(
            result["max_effective_num_burn_in_sweeps"]
        ),
    )
    validate_checkpoint(output_path, task, config_hash)
    return {
        "task_id": task["task_id"],
        "wall_time_seconds": float(result["wall_time_seconds"]),
        "flags": str(result["flags"]),
    }


def merge_cell(
    run_root: Path,
    cell_tasks: list[dict[str, Any]],
    config_hash: str,
) -> Path:
    first = cell_tasks[0]
    output_path = (
        run_root
        / "cells"
        / (
            f"p{probability_tag(first['p_value'])}_q{probability_tag(first['q_value'])}_"
            f"L{int(first['lattice_size']):02d}.npz"
        )
    )
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as loaded:
            if str(scalar(loaded, "config_hash")) != config_hash:
                raise ValueError(f"{output_path}: existing cell config mismatch")
            if int(scalar(loaded, "num_disorder_samples")) != len(cell_tasks):
                raise ValueError(f"{output_path}: existing cell size mismatch")
        return output_path

    rows: list[dict[str, Any]] = []
    for task in sorted(cell_tasks, key=lambda item: int(item["disorder_index"])):
        path = checkpoint_path(run_root, task)
        validate_checkpoint(path, task, config_hash)
        with np.load(path, allow_pickle=False) as loaded:
            rows.append({key: np.array(loaded[key]) for key in loaded.files})

    disorder_indices = np.asarray([row["disorder_index"].item() for row in rows])
    disorder_seeds = np.asarray([row["disorder_seed"].item() for row in rows])
    if not np.array_equal(disorder_indices, np.arange(len(rows))):
        raise ValueError("cell disorder indices are incomplete")
    if len(np.unique(disorder_seeds)) != len(rows):
        raise ValueError("cell disorder seeds are not unique")

    cell_manifest = {
        "model": MODEL,
        "p_value": float(first["p_value"]),
        "q_value": float(first["q_value"]),
        "lattice_size": int(first["lattice_size"]),
        "num_disorder_samples": len(rows),
        "seed_base": int(disorder_seeds.min()),
        "config_hash": config_hash,
        "fixed_config": FIXED_CONFIG,
    }
    atomic_savez(
        output_path,
        manifest_json=np.array(json.dumps(cell_manifest, sort_keys=True)),
        config_hash=np.array(config_hash),
        p_value=np.float64(first["p_value"]),
        q_value=np.float64(first["q_value"]),
        lattice_size=np.int64(first["lattice_size"]),
        num_disorder_samples=np.int64(len(rows)),
        disorder_index_per_disorder=disorder_indices.astype(np.int64),
        disorder_seed_per_disorder=disorder_seeds.astype(np.int64),
        sample_seed_per_disorder=np.asarray(
            [row["sample_seed"].item() for row in rows], dtype=np.int64
        ),
        delta_f_per_disorder=np.stack([row["delta_f"] for row in rows]),
        weights_per_disorder=np.stack([row["weights"] for row in rows]),
        delta_f_stderr_per_disorder=np.stack(
            [row["delta_f_stderr"] for row in rows]
        ),
        weights_stderr_per_disorder=np.stack(
            [row["weights_stderr"] for row in rows]
        ),
        q_top_per_disorder=np.asarray([row["q_top"].item() for row in rows]),
        q_top_stderr_per_disorder=np.asarray(
            [row["q_top_stderr"].item() for row in rows]
        ),
        q_top_ci95_per_disorder=np.stack([row["q_top_ci95"] for row in rows]),
        grid_tv_per_disorder=np.asarray([row["grid_tv"].item() for row in rows]),
        grid_q_top_abs_diff_per_disorder=np.asarray(
            [row["grid_q_top_abs_diff"].item() for row in rows]
        ),
        flags_per_disorder=np.asarray([row["flags"].item() for row in rows]),
        wall_time_seconds_per_disorder=np.asarray(
            [row["wall_time_seconds"].item() for row in rows]
        ),
    )
    return output_path


def run_wave(args: argparse.Namespace) -> int:
    project_root = find_project_root()
    source_path = project_root / "src" / "exp37_sector_ti.py"
    actual_source_sha = sha256_file(source_path)
    if actual_source_sha != args.source_sha256:
        raise RuntimeError(
            f"source SHA mismatch: expected {args.source_sha256}, got {actual_source_sha}"
        )
    pairs = parse_pairs(args.pairs)
    lattice_sizes = [int(value) for value in args.lattice_sizes.split(",")]
    if lattice_sizes != [3, 7]:
        raise ValueError("pilot lattice sizes must be exactly 3,7")
    if args.num_disorder_samples != 48:
        raise ValueError("pilot disorder count must be exactly 48")
    if args.workers < 1 or args.workers > 70:
        raise ValueError("workers must be in [1,70]")

    run_root = Path(args.run_root).resolve()
    wave_manifest = {
        "schema_version": 1,
        "model": MODEL,
        "created_at": utc_now(),
        "source_sha256": actual_source_sha,
        "pairs": [[p_value, q_value] for p_value, q_value in pairs],
        "lattice_sizes": lattice_sizes,
        "num_disorder_samples": int(args.num_disorder_samples),
        "seed_base": int(args.seed_base),
        "fixed_config": FIXED_CONFIG,
        "workers": int(args.workers),
    }
    comparable_manifest = dict(wave_manifest)
    comparable_manifest.pop("created_at")
    config_hash = sha256_bytes(canonical_json(comparable_manifest).encode("utf-8"))
    wave_manifest["config_hash"] = config_hash
    manifest_path = run_root / "run_manifest.json"

    if run_root.exists():
        if not args.resume:
            raise FileExistsError(f"run root already exists: {run_root}")
        if not manifest_path.is_file():
            raise ValueError("resume requested but run manifest is missing")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_comparable = dict(existing)
        existing_comparable.pop("created_at", None)
        existing_hash = existing_comparable.pop("config_hash", None)
        if existing_hash != config_hash or existing_comparable != comparable_manifest:
            raise ValueError("resume manifest does not match requested wave")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        atomic_write_json(manifest_path, wave_manifest)

    tasks = build_tasks(
        pairs=pairs,
        lattice_sizes=lattice_sizes,
        num_disorder_samples=args.num_disorder_samples,
        seed_base=args.seed_base,
    )
    expected_count = len(pairs) * len(lattice_sizes) * args.num_disorder_samples
    if len(tasks) != expected_count:
        raise RuntimeError("task count mismatch")

    pending: list[dict[str, Any]] = []
    for task in tasks:
        path = checkpoint_path(run_root, task)
        if path.exists():
            validate_checkpoint(path, task, config_hash)
        else:
            pending.append(task)

    atomic_write_json(
        run_root / "status.json",
        {
            "state": "running" if pending else "merging",
            "updated_at": utc_now(),
            "pid": os.getpid(),
            "process_group": os.getpgrp(),
            "total_tasks": len(tasks),
            "completed_tasks": len(tasks) - len(pending),
            "pending_tasks": len(pending),
        },
    )
    print(
        f"wave tasks={len(tasks)} completed={len(tasks)-len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    failures: list[dict[str, str]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    execute_task,
                    (task, str(checkpoint_path(run_root, task)), config_hash),
                ): task
                for task in pending
            }
            completed_now = 0
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                    completed_now += 1
                    print(
                        f"[{completed_now}/{len(pending)}] {result['task_id']} "
                        f"flags={result['flags']} wall={result['wall_time_seconds']:.1f}s",
                        flush=True,
                    )
                except Exception as exc:  # fail closed, while other checkpoints survive
                    failures.append({"task_id": task["task_id"], "error": repr(exc)})
                    print(f"FAILED {task['task_id']}: {exc!r}", file=sys.stderr, flush=True)

    if failures:
        failure_path = run_root / f"failure_{int(time.time())}.json"
        atomic_write_json(
            failure_path,
            {"state": "failed", "failed_at": utc_now(), "failures": failures},
        )
        atomic_write_json(
            run_root / "status.json",
            {
                "state": "failed",
                "updated_at": utc_now(),
                "total_tasks": len(tasks),
                "failed_tasks": len(failures),
            },
        )
        return 2

    for task in tasks:
        validate_checkpoint(checkpoint_path(run_root, task), task, config_hash)

    grouped: dict[tuple[float, float, int], list[dict[str, Any]]] = {}
    for task in tasks:
        key = (
            float(task["p_value"]),
            float(task["q_value"]),
            int(task["lattice_size"]),
        )
        grouped.setdefault(key, []).append(task)
    cell_paths = [
        merge_cell(run_root, cell_tasks, config_hash)
        for _, cell_tasks in sorted(grouped.items())
    ]
    cell_hashes = {str(path.relative_to(run_root)): sha256_file(path) for path in cell_paths}
    success = {
        "state": "success",
        "finished_at": utc_now(),
        "total_tasks": len(tasks),
        "cells": cell_hashes,
        "config_hash": config_hash,
    }
    if not (run_root / "SUCCESS.json").exists():
        atomic_write_json(run_root / "SUCCESS.json", success)
    atomic_write_json(run_root / "status.json", success)
    print(f"SUCCESS cells={len(cell_paths)}", flush=True)
    return 0


def read_cpu_snapshot() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def read_swap_counters() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        if key in {"pswpin", "pswpout"}:
            values[key] = int(value)
    return values.get("pswpin", 0), values.get("pswpout", 0)


def memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def ancestor_pids() -> set[int]:
    result: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        status = Path(f"/proc/{pid}/status")
        if not status.exists():
            break
        parent = 0
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        pid = parent
    return result


def process_snapshot() -> tuple[float, list[dict[str, Any]]]:
    output = subprocess.check_output(
        ["ps", "-eo", "user=,pcpu=,pid=,etimes=,comm="],
        text=True,
    )
    current_user = os.environ.get("USER", "yuany")
    excluded = ancestor_pids()
    other_cpu = 0.0
    own_compute: list[dict[str, Any]] = []
    compute_names = {"python", "python3", "julia", "matlab", "R", "Rscript"}
    for line in output.splitlines():
        pieces = line.split(None, 4)
        if len(pieces) != 5:
            continue
        user, cpu_text, pid_text, elapsed_text, command = pieces
        cpu = float(cpu_text)
        pid = int(pid_text)
        elapsed = int(elapsed_text)
        if pid in excluded:
            continue
        if user != current_user:
            other_cpu += cpu
        elif command in compute_names and elapsed > 60 and cpu > 1.0:
            own_compute.append(
                {"pid": pid, "cpu_percent": cpu, "elapsed_seconds": elapsed, "command": command}
            )
    return other_cpu, own_compute


def preflight(args: argparse.Namespace) -> int:
    idle_samples: list[float] = []
    swap_start = read_swap_counters()
    total_before, idle_before = read_cpu_snapshot()
    for _ in range(3):
        time.sleep(5)
        total_after, idle_after = read_cpu_snapshot()
        delta_total = total_after - total_before
        delta_idle = idle_after - idle_before
        idle_samples.append(100.0 * delta_idle / delta_total if delta_total else 0.0)
        total_before, idle_before = total_after, idle_after
    swap_end = read_swap_counters()
    other_cpu, own_compute = process_snapshot()
    disk = shutil.disk_usage("/home/DATA1")
    load1 = float(os.getloadavg()[0])
    mem_available = memory_available_bytes()
    swap_delta = [swap_end[index] - swap_start[index] for index in range(2)]
    logical_cpus = os.cpu_count() or 0
    measured_busy_cores = max(
        load1,
        logical_cpus * (100.0 - min(idle_samples)) / 100.0,
    )
    recommended_workers = max(
        0,
        min(70, int(math.floor(logical_cpus - measured_busy_cores - 16))),
    )
    checks = {
        "hostname_is_nd3": os.uname().nodename == "nd-3",
        "logical_cpus_at_least_96": logical_cpus >= 96,
        "recommended_workers_at_least_8": recommended_workers >= 8,
        "memory_available_at_least_350_gib": mem_available >= 350 * 1024**3,
        "no_swap_activity": swap_delta == [0, 0],
        "disk_free_at_least_100_gib": disk.free >= 100 * 1024**3,
        "no_existing_yuany_compute": not own_compute,
    }
    payload = {
        "checked_at": utc_now(),
        "hostname": os.uname().nodename,
        "logical_cpus": logical_cpus,
        "load1": load1,
        "cpu_idle_samples_percent": idle_samples,
        "memory_available_bytes": mem_available,
        "swap_delta_pages": swap_delta,
        "disk_free_bytes": disk.free,
        "other_user_cpu_percent": other_cpu,
        "measured_busy_cores": measured_busy_cores,
        "reserved_logical_cpus": 16,
        "maximum_workers": 70,
        "recommended_workers": recommended_workers,
        "existing_yuany_compute": own_compute,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        atomic_write_json(Path(args.output), payload)
    return 0 if payload["passed"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-wave")
    run_parser.add_argument("--run-root", required=True)
    run_parser.add_argument("--pairs", required=True, help="comma-separated p:q pairs")
    run_parser.add_argument("--lattice-sizes", default="3,7")
    run_parser.add_argument("--num-disorder-samples", type=int, default=48)
    run_parser.add_argument("--seed-base", type=int, required=True)
    run_parser.add_argument("--workers", type=int, default=70)
    run_parser.add_argument("--source-sha256", default=EXPECTED_SOURCE_SHA256)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.set_defaults(func=run_wave)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output", default=None)
    preflight_parser.set_defaults(func=preflight)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
