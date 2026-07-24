"""Immutable schedule, preflight, and measurement workflow for the m8 T1 screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    frozen_character_set,
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    build_full_column_candidate_cache,
    build_full_column_workspace,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column import (
    RANDOM_FULL_COLUMN_METHOD_ID,
    RANDOM_FULL_COLUMN_VERSION,
    RandomFullColumnConfig,
    replay_random_full_column_trajectory,
    run_random_full_column_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import (
    frozen_b_character_set,
    _disorder,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_random_full_column.t1_m8.v0"
SCHEDULE_VERSION = "exp102.q0_random_full_column.t1_m8.schedule.v0"
PREFLIGHT_VERSION = "exp102.q0_random_full_column.t1_m8.preflight.v0"
RAW_VERSION = "exp102.q0_random_full_column.t1_m8.raw.v0"
NODE_REPORT_VERSION = "exp102.q0_random_full_column.t1_m8.node_report.v0"
CONTROL_VERSION = "exp102.q0_random_full_column.t1_m8.control.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_random_full_column.t1_m8.v0.json"
SOURCE_CONTROL_DIR = ROOT / "control"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FAMILIES = ("P", "U", "M0", "M1", "S")


class T1ConflictError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise T1ConflictError(message)


def _exclusive_json(path, value):
    path = Path(path)
    _require(not path.exists(), f"immutable JSON already exists: {path}")
    atomic_json(path, value)


def _load_canonical_json(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    value = json.loads(serialized)
    _require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def _verify_self_hash(value, field):
    claimed = value[field]
    _require(SHA256_RE.fullmatch(str(claimed)) is not None, f"invalid {field}")
    core = {key: item for key, item in value.items() if key != field}
    _require(sha256_json(core) == claimed, f"self-hash mismatch: {field}")
    return claimed


def _load_config():
    config = _load_canonical_json(CONFIG_PATH)
    _require(config["version"] == config["contract_version"] == CONTRACT_VERSION,
             "contract version changed")
    _require(config["config_version"]
             == "exp102.q0_random_full_column.t1_m8.config.v0",
             "config version changed")
    _require(tuple(config["initialization"]["families"]) == FAMILIES
             and config["initialization"]["trajectories_per_family"] == 8,
             "initialization panel changed")
    _require(config["resource"] == {
        "allowed_nodes": ["nd-1", "nd-2", "nd-3"],
        "burn_updates": 2048,
        "fixed_workers_per_node": 4,
        "measurement_updates": 8192,
        "runtime_probe_concurrency": 4,
        "runtime_probe_updates_per_worker": 8,
        "safety_factor": 2.0,
        "trajectory_wall_cap_seconds": 7200.0,
    }, "resource clock changed")
    return config, sha256_file(CONFIG_PATH)


def _control_content_sha(metadata, arrays):
    metadata = dict(metadata)
    metadata.pop("control_content_sha256", None)
    digest = hashlib.sha256(CONTROL_VERSION.encode("ascii") + b"\0")
    digest.update(canonical_json(metadata).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_control(control_dir, config, config_sha):
    control_dir = Path(control_dir)
    manifest_path = control_dir / "control_manifest.json"
    control_path = control_dir / "control.npz"
    manifest = _load_canonical_json(manifest_path)
    _verify_self_hash(manifest, "manifest_sha256")
    _require(manifest["control_version"] == CONTROL_VERSION
             and manifest["config_sha256"] == config_sha
             and manifest["control_file_sha256"] == sha256_file(control_path),
             "control manifest identity mismatch")
    with np.load(control_path, allow_pickle=False) as archive:
        _require("metadata_json" in archive.files, "control metadata missing")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: archive[name].copy() for name in archive.files if name != "metadata_json"
        }
    expected_fields = {
        "H", "b_character_masks_packed", "fixed_b_blocks", "fixed_labels",
        "fixed_states_packed", "fixed_weights", "logical_basis_positions",
        "logical_character_masks", "syndrome_packed",
    }
    _require(set(arrays) == expected_fields, "control array schema changed")
    _require(metadata["control_version"] == CONTROL_VERSION
             and metadata["config_sha256"] == config_sha
             and metadata["registry_sha256"] == config["registry_sha256"]
             and metadata["fixed_names"]
             == ["P", "M0", "M1", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"],
             "control metadata changed")
    content_sha = _control_content_sha(metadata, arrays)
    _require(content_sha == metadata["control_content_sha256"]
             == manifest["control_content_sha256"],
             "control content hash mismatch")

    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "registry changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(np.array_equal(arrays["H"], H)
             and np.array_equal(
                 np.unpackbits(arrays["syndrome_packed"], count=model.num_checks,
                               bitorder="little").astype(np.uint8),
                 syndrome,
             ), "control H or syndrome changed")
    fixed = np.unpackbits(
        arrays["fixed_states_packed"], axis=1, count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8)
    _require(fixed.shape == (11, model.num_qubits) and np.array_equal(fixed[0], planted),
             "control planted initialization changed")
    residuals = (
        model.H_check.astype(np.int64) @ fixed.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    _require(np.array_equal(
        residuals, np.repeat(syndrome[None, :], fixed.shape[0], axis=0),
    ), "control contains an illegal state")
    labels = np.asarray([state_label(frame, state) for state in fixed], dtype=np.uint64)
    b_blocks = np.stack([split_hgp_state(state, H)[1] for state in fixed])
    _require(np.array_equal(labels, arrays["fixed_labels"])
             and np.array_equal(fixed.sum(axis=1), arrays["fixed_weights"])
             and np.array_equal(b_blocks, arrays["fixed_b_blocks"]),
             "control state statistics mismatch")
    logical_characters = frozen_character_set(
        model.k, metadata["logical_character_seed"],
        num_nonbasis=config["statistics"]["logical_nonbasis_character_count"],
    )
    b_characters = frozen_b_character_set(
        H.shape[0], metadata["b_character_seed"],
        dense_count=config["statistics"]["b_dense_character_count"],
    )
    _require(np.array_equal(logical_characters.masks, arrays["logical_character_masks"])
             and np.array_equal(
                 logical_characters.basis_positions, arrays["logical_basis_positions"],
             ) and logical_characters.character_sha256
             == metadata["logical_character_sha256"]
             and np.array_equal(
                 b_characters.masks_packed, arrays["b_character_masks_packed"],
             ) and b_characters.character_sha256 == metadata["b_character_sha256"],
             "control character set changed")
    return {
        "arrays": arrays,
        "config": config,
        "config_sha": config_sha,
        "control_path": control_path,
        "fixed_states": fixed,
        "frame": frame,
        "H": H,
        "manifest": manifest,
        "metadata": metadata,
        "model": model,
        "planted": planted,
        "registry": registry,
        "syndrome": syndrome,
        "uniform_seed": int(uniform_seed),
    }


def _source_identity(args):
    _require(COMMIT_RE.fullmatch(args.source_commit) is not None,
             "source commit must be a full SHA")
    _require(SHA256_RE.fullmatch(args.archive_sha256) is not None
             and SHA256_RE.fullmatch(args.source_manifest_sha256) is not None,
             "source archive identity is invalid")
    env_commit = os.environ.get("EXP102_SOURCE_COMMIT")
    if env_commit is not None:
        _require(env_commit == args.source_commit, "verified-source commit mismatch")
    return {
        "archive_sha256": args.archive_sha256,
        "source_commit": args.source_commit,
        "source_manifest_sha256": args.source_manifest_sha256,
    }


def _task_rows(context, source):
    config = context["config"]
    resource = config["resource"]
    tasks = []
    for family in FAMILIES:
        for index in range(config["initialization"]["trajectories_per_family"]):
            identity = {
                "cell_fingerprint": context["metadata"]["cell_fingerprint"],
                "config_sha256": context["config_sha"],
                "control_content_sha256": context["metadata"]["control_content_sha256"],
                "family": family,
                "index": index,
                "method_id": RANDOM_FULL_COLUMN_METHOD_ID,
                "registry_sha256": context["registry"]["registry_sha256"],
                "resource_tier": "T1",
                "source_commit": source["source_commit"],
            }
            prefix = (
                config["seed_namespace"], CONTRACT_VERSION, source["source_commit"],
                context["config_sha"], context["registry"]["registry_sha256"],
                context["metadata"]["cell_fingerprint"], RANDOM_FULL_COLUMN_METHOD_ID,
                "T1", family, index,
            )
            row = {
                **identity,
                "burn_update_seed": derive_seed(*prefix, "burn", "update"),
                "initialization_seed": derive_seed(*prefix, "initialization", "state"),
                "measurement_update_seed": derive_seed(*prefix, "measurement", "update"),
                "observation_seed": derive_seed(*prefix, "measurement", "A_given_B"),
                "raw_version": RAW_VERSION,
            }
            row["task_fingerprint"] = sha256_json(row)
            tasks.append(row)
    _require(len(tasks) == 40
             and len({task["task_fingerprint"] for task in tasks}) == 40
             and len({task["burn_update_seed"] for task in tasks}) == 40
             and len({task["measurement_update_seed"] for task in tasks}) == 40
             and len({task["observation_seed"] for task in tasks}) == 40,
             "task seeds or identities collided")
    nodes = resource["allowed_nodes"]
    for position, task in enumerate(tasks):
        task["owner"] = nodes[position % len(nodes)]
    return tasks


def build_schedule(args):
    config, config_sha = _load_config()
    context = _load_control(SOURCE_CONTROL_DIR, config, config_sha)
    source = _source_identity(args)
    tasks = _task_rows(context, source)
    core = {
        "config_sha256": config_sha,
        "contract_version": CONTRACT_VERSION,
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "control_file_sha256": sha256_file(context["control_path"]),
        "control_manifest_sha256": context["manifest"]["manifest_sha256"],
        "ownership": {
            node: [task["task_fingerprint"] for task in tasks if task["owner"] == node]
            for node in config["resource"]["allowed_nodes"]
        },
        "resource": config["resource"],
        "schedule_version": SCHEDULE_VERSION,
        "scope": config["scope"],
        "source_identity": source,
        "tasks": tasks,
    }
    schedule = {**core, "schedule_sha256": sha256_json(core)}
    run_root = Path(args.run_root).resolve()
    _require(not run_root.exists(), "run root must be fresh")
    control_dir = run_root / "control"
    control_dir.mkdir(parents=True)
    shutil.copyfile(context["control_path"], control_dir / "control.npz")
    shutil.copyfile(SOURCE_CONTROL_DIR / "control_manifest.json",
                    control_dir / "control_manifest.json")
    _exclusive_json(control_dir / "schedule.json", schedule)
    print(canonical_json({
        "schedule_sha256": schedule["schedule_sha256"],
        "status": "SCHEDULE_FROZEN",
        "task_count": len(tasks),
    }))


def _load_schedule(run_root, args):
    run_root = Path(run_root).resolve()
    schedule = _load_canonical_json(run_root / "control/schedule.json")
    _verify_self_hash(schedule, "schedule_sha256")
    source = _source_identity(args)
    _require(schedule["contract_version"] == CONTRACT_VERSION
             and schedule["schedule_version"] == SCHEDULE_VERSION
             and schedule["source_identity"] == source,
             "schedule/source identity mismatch")
    config, config_sha = _load_config()
    context = _load_control(run_root / "control", config, config_sha)
    _require(schedule["config_sha256"] == config_sha
             and schedule["control_content_sha256"]
             == context["metadata"]["control_content_sha256"]
             and schedule["control_file_sha256"] == sha256_file(context["control_path"]),
             "schedule/control identity mismatch")
    return run_root, schedule, context


_PROBE_CONTEXT = None


def _probe_worker(index):
    context = _PROBE_CONTEXT
    config = context["config"]
    probe_updates = int(config["resource"]["runtime_probe_updates_per_worker"])
    sampler = RandomFullColumnConfig(
        p=0.04, burn_updates=max(1, probe_updates // 4),
        measurement_updates=probe_updates,
    )
    prefix = (
        config["seed_namespace"], context["config_sha"],
        context["metadata"]["control_content_sha256"], "preflight", index,
    )
    start = time.perf_counter()
    raw = run_random_full_column_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, context["planted"], derive_seed(*prefix, "burn"),
        derive_seed(*prefix, "measurement"), derive_seed(*prefix, "observation"),
        mass=context["mass"], cache=context["cache"],
        workspace=build_full_column_workspace(context["cache"]),
    )
    elapsed = time.perf_counter() - start
    discrete_names = (
        "burn__counters", "burn__final_b_columns", "burn__selected_columns",
        "burn__old_columns", "burn__new_columns", "measurement__counters",
        "measurement__selected_columns", "measurement__old_columns",
        "measurement__new_columns", "measurement__b_columns",
        "measurement__labels", "measurement__states_packed", "measurement__weights",
        "final_b_columns", "final_state_packed",
    )
    digest = hashlib.sha256()
    for name in discrete_names:
        value = np.ascontiguousarray(raw[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    total_updates = sampler.burn_updates + sampler.measurement_updates
    return {
        "elapsed_seconds": elapsed,
        "index": index,
        "portable_transcript_sha256": digest.hexdigest(),
        "seconds_per_weighted_update": elapsed / total_updates,
        "total_probe_updates": total_updates,
    }


def preflight_node(args):
    run_root, schedule, context = _load_schedule(args.run_root, args)
    _require(args.node in context["config"]["resource"]["allowed_nodes"],
             "preflight node is not allowed")
    global _PROBE_CONTEXT
    context["mass"] = build_classical_coset_mass(context["H"], 0.04, engine="numba")
    context["cache"] = build_full_column_candidate_cache(context["H"].shape[0], 0.04)
    _PROBE_CONTEXT = context
    workers = context["config"]["resource"]["runtime_probe_concurrency"]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        probes = list(executor.map(_probe_worker, range(workers)))
    _PROBE_CONTEXT = None
    worst = max(row["seconds_per_weighted_update"] for row in probes)
    resource = context["config"]["resource"]
    projection = worst * (
        resource["burn_updates"] + resource["measurement_updates"]
    ) * resource["safety_factor"]
    pass_runtime = projection <= resource["trajectory_wall_cap_seconds"]
    core = {
        "config_sha256": context["config_sha"],
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "mass_sha256": hashlib.sha256(
            np.ascontiguousarray(context["mass"], dtype=">f8").tobytes()
        ).hexdigest(),
        "node": args.node,
        "pass_runtime": pass_runtime,
        "preflight_version": PREFLIGHT_VERSION,
        "probes": probes,
        "projected_replay_inclusive_trajectory_seconds": projection,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": schedule["source_identity"],
        "status": "PASS" if pass_runtime else "RUNTIME_EXHAUSTED",
    }
    report = {**core, "preflight_sha256": sha256_json(core)}
    output = run_root / f"preflight/{args.node}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(output, report)
    print(canonical_json(report))


def combine_preflight(args):
    run_root, schedule, context = _load_schedule(args.run_root, args)
    reports = []
    for node in context["config"]["resource"]["allowed_nodes"]:
        report = _load_canonical_json(run_root / f"preflight/{node}.json")
        _verify_self_hash(report, "preflight_sha256")
        _require(report["node"] == node
                 and report["schedule_sha256"] == schedule["schedule_sha256"]
                 and report["source_identity"] == schedule["source_identity"],
                 "node preflight identity mismatch")
        reports.append(report)
    mass_hashes = {report["mass_sha256"] for report in reports}
    transcript_catalogs = {
        tuple(row["portable_transcript_sha256"] for row in report["probes"])
        for report in reports
    }
    exact_consensus = len(mass_hashes) == 1 and len(transcript_catalogs) == 1
    if not exact_consensus:
        status = "CONFLICT"
    elif not all(report["pass_runtime"] for report in reports):
        status = "RUNTIME_EXHAUSTED"
    else:
        status = "PASS"
    core = {
        "config_sha256": context["config_sha"],
        "exact_consensus": exact_consensus,
        "node_preflight_sha256": {
            report["node"]: report["preflight_sha256"] for report in reports
        },
        "node_status": {report["node"]: report["status"] for report in reports},
        "preflight_version": PREFLIGHT_VERSION,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": schedule["source_identity"],
        "status": status,
        "worst_projected_replay_inclusive_trajectory_seconds": max(
            report["projected_replay_inclusive_trajectory_seconds"]
            for report in reports
        ),
    }
    aggregate = {**core, "preflight_sha256": sha256_json(core)}
    _exclusive_json(run_root / "preflight/aggregate.json", aggregate)
    print(canonical_json(aggregate))


_MEASUREMENT_CONTEXT = None


def _initial_state(context, task):
    family = task["family"]
    if family == "P":
        return context["fixed_states"][0].copy()
    if family == "M0":
        return context["fixed_states"][1].copy()
    if family == "M1":
        return context["fixed_states"][2].copy()
    if family == "S":
        return context["fixed_states"][3 + int(task["index"])].copy()
    if family == "U":
        return uniform_hard_coset_state(
            context["model"], context["syndrome"], task["initialization_seed"],
        )
    raise T1ConflictError("unknown initialization family")


def _measurement_worker(task):
    context = _MEASUREMENT_CONTEXT
    config = context["config"]
    sampler = RandomFullColumnConfig(
        p=0.04, burn_updates=config["resource"]["burn_updates"],
        measurement_updates=config["resource"]["measurement_updates"],
    )
    initial = _initial_state(context, task)
    workspace = build_full_column_workspace(context["cache"])
    start = time.perf_counter()
    raw = run_random_full_column_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, initial, task["burn_update_seed"], task["measurement_update_seed"],
        task["observation_seed"], mass=context["mass"], cache=context["cache"],
        workspace=workspace,
    )
    sampling_seconds = time.perf_counter() - start
    replay_start = time.perf_counter()
    replay_random_full_column_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, initial, task["burn_update_seed"], task["measurement_update_seed"],
        task["observation_seed"], raw, mass=context["mass"], cache=context["cache"],
        workspace=workspace,
    )
    replay_seconds = time.perf_counter() - replay_start
    payload = {
        "archive_sha256": np.array(context["source_identity"]["archive_sha256"]),
        "config_sha256": np.array(context["config_sha"]),
        "contract_version": np.array(CONTRACT_VERSION),
        "control_content_sha256": np.array(context["metadata"]["control_content_sha256"]),
        "model_fingerprint": np.array(context["model"].fingerprint()),
        "raw_version": np.array(RAW_VERSION),
        "replay_seconds": np.array(replay_seconds, dtype=np.float64),
        "sampling_seconds": np.array(sampling_seconds, dtype=np.float64),
        "schedule_sha256": np.array(context["schedule_sha256"]),
        "source_commit": np.array(context["source_identity"]["source_commit"]),
        "source_manifest_sha256": np.array(
            context["source_identity"]["source_manifest_sha256"],
        ),
        "syndrome_packed": context["arrays"]["syndrome_packed"],
        "task_fingerprint": np.array(task["task_fingerprint"]),
        "task_json": np.array(canonical_json(task)),
        "version": np.array(RANDOM_FULL_COLUMN_VERSION),
        **raw,
    }
    output = context["raw_dir"] / f"{task['family']}_{task['index']:02d}.npz"
    _require(not output.exists(), f"raw already exists: {output}")
    atomic_npz(output, **payload)
    return {
        "family": task["family"],
        "file": output.name,
        "index": task["index"],
        "raw_sha256": sha256_file(output),
        "replay_seconds": replay_seconds,
        "sampling_seconds": sampling_seconds,
        "task_fingerprint": task["task_fingerprint"],
    }


def run_node(args):
    run_root, schedule, context = _load_schedule(args.run_root, args)
    _require(args.node in context["config"]["resource"]["allowed_nodes"],
             "measurement node is not allowed")
    aggregate = _load_canonical_json(run_root / "preflight/aggregate.json")
    _verify_self_hash(aggregate, "preflight_sha256")
    _require(aggregate["status"] == "PASS"
             and aggregate["exact_consensus"] is True
             and aggregate["schedule_sha256"] == schedule["schedule_sha256"],
             "aggregate preflight did not authorize measurement")
    tasks = [task for task in schedule["tasks"] if task["owner"] == args.node]
    expected_fingerprints = schedule["ownership"][args.node]
    _require([task["task_fingerprint"] for task in tasks] == expected_fingerprints,
             "node ownership changed")
    raw_dir = run_root / f"measurement/{args.node}/raw"
    _require(not raw_dir.exists(), "node raw directory already exists")
    raw_dir.mkdir(parents=True)
    context["mass"] = build_classical_coset_mass(context["H"], 0.04, engine="numba")
    context["cache"] = build_full_column_candidate_cache(context["H"].shape[0], 0.04)
    context["raw_dir"] = raw_dir
    context["schedule_sha256"] = schedule["schedule_sha256"]
    context["source_identity"] = schedule["source_identity"]
    global _MEASUREMENT_CONTEXT
    _MEASUREMENT_CONTEXT = context
    records = []
    workers = context["config"]["resource"]["fixed_workers_per_node"]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        future_to_task = {
            executor.submit(_measurement_worker, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            record = future.result()
            records.append(record)
            print(canonical_json(record), flush=True)
    _MEASUREMENT_CONTEXT = None
    records.sort(key=lambda row: (FAMILIES.index(row["family"]), row["index"]))
    _require(len(records) == len(tasks), "node task count changed")
    core = {
        "node": args.node,
        "node_report_version": NODE_REPORT_VERSION,
        "preflight_sha256": aggregate["preflight_sha256"],
        "raw_count": len(records),
        "raw_records": records,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": schedule["source_identity"],
        "status": "COMPLETE",
    }
    report = {**core, "node_report_sha256": sha256_json(core)}
    _exclusive_json(run_root / f"measurement/{args.node}/node_report.json", report)
    print(canonical_json({
        "node_report_sha256": report["node_report_sha256"],
        "raw_count": len(records),
        "status": "COMPLETE",
    }))


def _add_source_arguments(parser):
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    schedule = subparsers.add_parser("build-schedule")
    schedule.add_argument("--run-root", required=True)
    _add_source_arguments(schedule)
    preflight = subparsers.add_parser("preflight-node")
    preflight.add_argument("--run-root", required=True)
    preflight.add_argument("--node", required=True)
    _add_source_arguments(preflight)
    combine = subparsers.add_parser("combine-preflight")
    combine.add_argument("--run-root", required=True)
    _add_source_arguments(combine)
    measurement = subparsers.add_parser("run-node")
    measurement.add_argument("--run-root", required=True)
    measurement.add_argument("--node", required=True)
    _add_source_arguments(measurement)
    args = parser.parse_args()
    actions = {
        "build-schedule": build_schedule,
        "preflight-node": preflight_node,
        "combine-preflight": combine_preflight,
        "run-node": run_node,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
