"""Immutable V0 logical-transport diagnostic for LSI-IMH.

This module deliberately has narrow authority.  It can only run the frozen
``m08_c06, p=.04, d00`` diagnostic and can only conclude whether the new
label-stratified proposal produces observable logical transport.  It cannot
authorize a physics result, a tuning panel, or production.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from .q0_global import character_values, frozen_character_set, uniform_hard_coset_state
from .q0_logical_stratified import (
    LogicalStratifiedConfig,
    LogicalStratifiedConflictError,
    LogicalStratifiedSeedIdentity,
    STRATIFIED_METHOD_ID,
    _matrix_syndrome_sha256,
    build_hgp_signature_codebook,
    build_logical_stratified_frozen_artifact,
    build_logical_stratified_proposal,
    generate_bplsd_stratified_catalog,
    load_logical_stratified_frozen_artifact,
    replay_logical_stratified_trajectory,
    run_logical_stratified_trajectory,
    write_logical_stratified_frozen_artifact,
)
from .registry import load_frozen_code, load_registry
from .seeds import derive_seed
from .worker import build_model


V0_CONTRACT_VERSION = "exp102.q0_logical_stratified.v0.v1"
V0_MANIFEST_VERSION = "exp102.q0_logical_stratified.v0.manifest.v1"
V0_RAW_VERSION = "exp102.q0_logical_stratified.v0.raw.v1"
V0_PREFLIGHT_VERSION = "exp102.q0_logical_stratified.v0.preflight.v1"
V0_REPORT_VERSION = "exp102.q0_logical_stratified.v0.report.v1"
V0_METHOD_IDS = {0.5: "LSI-IMH-T05", 1.0: "LSI-IMH-T10"}
V0_CELL = {
    "code_id": "m08_c06",
    "p": 0.04,
    "disorder_index": 0,
    "disorder_source": "attempt022",
}
V0_NODES = ("nd-2", "nd-3")
V0_FAMILIES = ("P", "U", "L")
V0_TAU = (0.5, 1.0)
V0_CONFIG_FIELDS = {
    "artifact", "candidates", "cell", "contract_version", "execution",
    "frozen_nonbasis_characters", "gates", "init_families", "registry_sha256",
    "resource_tier", "scope", "seed_namespace", "trajectories_per_family",
}
V0_RAW_BASE_FIELDS = {
    "raw_version", "sampler_raw_version", "contract_version", "task_json", "task_fingerprint",
    "source_commit", "archive_sha256", "source_manifest_sha256",
    "v0_config_sha256", "registry_sha256", "artifact_file_sha256",
    "artifact_content_sha256", "core_seconds", "wall_seconds",
}


class LogicalStratifiedV0ConflictError(ValueError):
    pass


def _require_sha(value, length, name):
    if (not isinstance(value, str) or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _require_positive_int(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < int(minimum):
        raise ValueError(f"{name} is too small")
    return int(value)


def _cell_fingerprint(cell):
    return sha256_json(cell)


def _load_canonical_json(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    value = json.loads(serialized)
    if serialized != canonical_json(value) + "\n":
        raise LogicalStratifiedV0ConflictError("V0 config is not canonical JSON")
    return value


def load_v0_config(path, *, registry=None):
    """Load the single pre-registered V0 schedule and reject broader scope."""
    value = _load_canonical_json(path)
    if set(value) != V0_CONFIG_FIELDS or value.get("contract_version") != V0_CONTRACT_VERSION:
        raise LogicalStratifiedV0ConflictError("V0 config schema/version changed")
    if value.get("cell") != V0_CELL:
        raise LogicalStratifiedV0ConflictError("V0 cell changed")
    artifact = value.get("artifact")
    if artifact != {
            "codebook_combination_order": 3,
            "decoder_chunk_size": 128,
            "decoder_max_iter": 64,
            "decoder_workers": 8,
            "max_anchors": 128,
            "rank2_seed_count": 128,
    }:
        raise LogicalStratifiedV0ConflictError("V0 artifact construction changed")
    candidates = value.get("candidates")
    expected_candidates = [
        {"alpha_temperature": 0.5, "method_id": V0_METHOD_IDS[0.5]},
        {"alpha_temperature": 1.0, "method_id": V0_METHOD_IDS[1.0]},
    ]
    if candidates != expected_candidates:
        raise LogicalStratifiedV0ConflictError("V0 proposal candidates changed")
    if (value.get("init_families") != list(V0_FAMILIES)
            or value.get("trajectories_per_family") != 8
            or value.get("execution") != {
                "nodes": list(V0_NODES), "workers_per_node": 12,
            }
            or value.get("resource_tier") != {
                "burn_steps": 512, "measurement_steps": 4096, "name": "V0",
            }
            or value.get("frozen_nonbasis_characters") != 64
            or value.get("gates") != {
                "minimum_accepted_cross_label_changes_per_family": 32,
                "minimum_chains_with_two_cross_label_changes_per_family": 6,
                "minimum_distinct_catalog_sources_per_family": 4,
                "minimum_leave_return_character_chains_per_family": 4,
            }
            or value.get("scope") != {
                "formal_authorization": False,
                "maximum_terminal_status": "LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN",
                "production_authorization": False,
                "purpose": "diagnostic_only_not_a_posterior_or_physics_result",
            }
            or value.get("seed_namespace") != "exp102.q0_logical_stratified.v0.20260723"):
        raise LogicalStratifiedV0ConflictError("V0 schedule/gates/scope changed")
    _require_sha(value.get("registry_sha256"), 64, "V0 registry SHA")
    if registry is not None and value["registry_sha256"] != registry["registry_sha256"]:
        raise LogicalStratifiedV0ConflictError("V0 registry identity changed")
    result = dict(value)
    result["v0_config_sha256"] = sha256_file(path)
    result["config_path"] = str(Path(path).resolve())
    return result


def _uniform_seed(registry, code, cell):
    if cell["disorder_source"] != "attempt022":
        raise LogicalStratifiedV0ConflictError("V0 disorder source changed")
    return derive_seed(
        f"pilot_ladder_m{int(code['m'])}_attempt22",
        registry["registry_sha256"], code["code_id"],
        int(cell["disorder_index"]), "uniforms",
    )


def _context(registry_path, config):
    registry = load_registry(registry_path)
    if registry["registry_sha256"] != config["registry_sha256"]:
        raise LogicalStratifiedV0ConflictError("V0 registry bytes changed")
    _, code, H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed = _uniform_seed(registry, code, config["cell"])
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
    epsilon = (uniforms < float(config["cell"]["p"])).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    if not syndrome.any():
        raise LogicalStratifiedV0ConflictError("V0 planted syndrome unexpectedly vanishes")
    return registry, code, H, model, frame, uniform_seed, epsilon, syndrome


def _tail_indices(codebook, transcript, catalog, count):
    """Choose low-energy decoded candidates immediately outside the frozen cap."""
    selected = {int(index) for index in catalog.candidate_indices[1:]}
    ordered = sorted(
        np.flatnonzero(transcript.valid).tolist(),
        key=lambda index: (
            int(transcript.decoded_weights[index]),
            int(codebook.logical_move_weights[index]),
            int(codebook.signatures[index]),
            transcript.decoded_packed[index].tobytes(),
        ),
    )
    tail = [index for index in ordered if index not in selected][:int(count)]
    if len(tail) != int(count) or len(set(tail)) != len(tail):
        raise LogicalStratifiedV0ConflictError("V0 S-tail schedule cannot be formed")
    return np.asarray(tail, dtype=np.int32)


def _artifact_relpath(tau):
    return f"artifacts/lsi_imh_tau_{str(tau).replace('.', '')}.npz"


def _artifact_descriptor_file(path, content):
    return {
        "artifact_relpath": str(path),
        "artifact_file_sha256": content["artifact_file_sha256"],
        "artifact_content_sha256": content["artifact_content_sha256"],
        "descriptor": content["descriptor"],
    }


def prepare_v0_artifacts(registry_path, config_path, source_commit, archive_sha256,
                         source_manifest_sha256, artifact_root):
    """Generate one full transcript catalog and two immutable proposal artifacts."""
    _require_sha(source_commit, 40, "source commit")
    _require_sha(archive_sha256, 64, "archive SHA")
    _require_sha(source_manifest_sha256, 64, "source manifest SHA")
    registry = load_registry(registry_path)
    config = load_v0_config(config_path, registry=registry)
    artifact_root = Path(artifact_root)
    if artifact_root.exists():
        raise FileExistsError(f"V0 artifact root already exists: {artifact_root}")
    artifact_root.mkdir(parents=True, mode=0o700)
    registry, code, H, model, frame, uniform_seed, _, syndrome = _context(
        registry_path, config,
    )
    spec = config["artifact"]
    codebook = build_hgp_signature_codebook(
        model, frame, H,
        combination_order=spec["codebook_combination_order"],
        rank2_seed_count=spec["rank2_seed_count"],
    )
    catalog = generate_bplsd_stratified_catalog(
        model, frame, H, syndrome, config["cell"]["p"], codebook,
        max_anchors=spec["max_anchors"],
        decoder_max_iter=spec["decoder_max_iter"],
        chunk_size=spec["decoder_chunk_size"], num_workers=spec["decoder_workers"],
    )
    transcript = catalog.candidate_transcript
    tail = _tail_indices(codebook, transcript, catalog, config["trajectories_per_family"])
    tail_sha256 = hashlib.sha256(tail.astype(">i4").tobytes()).hexdigest()
    artifact_rows = []
    for candidate in config["candidates"]:
        tau = float(candidate["alpha_temperature"])
        identity = {
            "archive_sha256": archive_sha256,
            "cell_fingerprint": _cell_fingerprint(config["cell"]),
            "config_sha256": config["v0_config_sha256"],
            "method_id": candidate["method_id"],
            "registry_sha256": registry["registry_sha256"],
            "source_commit": source_commit,
            "source_manifest_sha256": source_manifest_sha256,
            "tail_indices_sha256": tail_sha256,
        }
        proposal = build_logical_stratified_proposal(
            model, frame, catalog, p=config["cell"]["p"],
            alpha_temperature=tau,
        )
        artifact = build_logical_stratified_frozen_artifact(
            model, frame, H, syndrome, codebook, catalog, proposal, identity=identity,
        )
        relative = _artifact_relpath(tau)
        written = write_logical_stratified_frozen_artifact(
            artifact_root / relative, model, frame, artifact,
        )
        artifact_rows.append({
            "alpha_temperature": tau,
            "method_id": candidate["method_id"],
            **_artifact_descriptor_file(relative, written),
        })
    identity = {
        "artifact_rows": artifact_rows,
        "cell": config["cell"],
        "codebook_sha256": codebook.codebook_sha256,
        "config_sha256": config["v0_config_sha256"],
        "contract_version": V0_CONTRACT_VERSION,
        "matrix_syndrome_sha256": _matrix_syndrome_sha256(model, syndrome),
        "registry_sha256": registry["registry_sha256"],
        "tail_candidate_indices": [int(value) for value in tail],
        "tail_indices_sha256": tail_sha256,
        "uniform_seed": int(uniform_seed),
    }
    artifact_manifest = {**identity, "artifact_manifest_sha256": sha256_json(identity)}
    atomic_json(artifact_root / "ARTIFACT_MANIFEST.json", artifact_manifest)
    return artifact_manifest


def _load_artifact_manifest(artifact_root):
    path = Path(artifact_root) / "ARTIFACT_MANIFEST.json"
    value = _load_canonical_json(path)
    identity = dict(value)
    digest = identity.pop("artifact_manifest_sha256", None)
    if digest != sha256_json(identity):
        raise LogicalStratifiedV0ConflictError("V0 artifact manifest SHA changed")
    return value


def _task_identity(config, source_commit, archive_sha256, source_manifest_sha256,
                   registry_sha256, artifact_row, family, trajectory, node):
    trajectory = _require_positive_int(trajectory + 1, "trajectory", minimum=1) - 1
    if family not in V0_FAMILIES or node not in V0_NODES:
        raise ValueError("V0 task family/node changed")
    task = {
        "artifact": artifact_row,
        "cell": config["cell"],
        "contract_version": V0_CONTRACT_VERSION,
        "init_family": family,
        "method_id": artifact_row["method_id"],
        "node": node,
        "resource_tier": config["resource_tier"]["name"],
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "config_sha256": config["v0_config_sha256"],
        "registry_sha256": registry_sha256,
        "trajectory_index": trajectory,
        "trajectory_namespace": (
            f"{config['seed_namespace']}.tau{artifact_row['alpha_temperature']:.1f}"
        ),
    }
    return task


def build_v0_manifest(registry_path, config_path, source_commit, archive_sha256,
                      source_manifest_sha256, artifact_root, output_path=None):
    _require_sha(source_commit, 40, "source commit")
    _require_sha(archive_sha256, 64, "archive SHA")
    _require_sha(source_manifest_sha256, 64, "source manifest SHA")
    registry = load_registry(registry_path)
    config = load_v0_config(config_path, registry=registry)
    artifacts = _load_artifact_manifest(artifact_root)
    expected_rows = [
        {"alpha_temperature": row["alpha_temperature"], "method_id": row["method_id"]}
        for row in artifacts["artifact_rows"]
    ]
    if expected_rows != config["candidates"]:
        raise LogicalStratifiedV0ConflictError("V0 artifacts/config candidate order changed")
    tasks = []
    for row in artifacts["artifact_rows"]:
        for family in V0_FAMILIES:
            for trajectory in range(config["trajectories_per_family"]):
                ordinal = len(tasks) % 24
                node = V0_NODES[0] if ordinal < 12 else V0_NODES[1]
                tasks.append(_task_identity(
                    config, source_commit, archive_sha256, source_manifest_sha256,
                    registry["registry_sha256"], row, family, trajectory, node,
                ))
    if len(tasks) != 48 or len({sha256_json(task) for task in tasks}) != len(tasks):
        raise AssertionError("V0 task schedule changed")
    identity = {
        "artifact_manifest_sha256": artifacts["artifact_manifest_sha256"],
        "artifact_root": str(Path(artifact_root).resolve()),
        "config_sha256": config["v0_config_sha256"],
        "contract_version": V0_CONTRACT_VERSION,
        "manifest_version": V0_MANIFEST_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "tasks": tasks,
    }
    manifest = {**identity, "manifest_sha256": sha256_json(identity)}
    if output_path is not None:
        path = Path(output_path)
        if path.exists():
            raise FileExistsError(f"V0 manifest already exists: {path}")
        atomic_json(path, manifest)
    return manifest


def load_v0_manifest(path):
    value = _load_canonical_json(path)
    identity = dict(value)
    digest = identity.pop("manifest_sha256", None)
    if (set(identity) != {
            "artifact_manifest_sha256", "artifact_root", "config_sha256",
            "contract_version", "manifest_version", "registry_sha256",
            "source_commit", "archive_sha256", "source_manifest_sha256", "tasks",
        }
            or identity["contract_version"] != V0_CONTRACT_VERSION
            or identity["manifest_version"] != V0_MANIFEST_VERSION
            or digest != sha256_json(identity)):
        raise LogicalStratifiedV0ConflictError("V0 manifest schema/SHA changed")
    return value


def validate_v0_manifest(manifest, registry_path, config_path):
    """Rebuild the fixed schedule and reject any hand-edited task or artifact row."""
    if not isinstance(manifest, dict):
        raise TypeError("V0 manifest must be a dictionary")
    expected = build_v0_manifest(
        registry_path, config_path, manifest.get("source_commit", ""),
        manifest.get("archive_sha256", ""),
        manifest.get("source_manifest_sha256", ""),
        manifest.get("artifact_root", ""), None,
    )
    if manifest != expected:
        raise LogicalStratifiedV0ConflictError("V0 manifest is noncanonical")
    return True


def _task_output_relpath(task):
    return f"raw/{sha256_json(task)}.npz"


def _artifact_for_task(artifact_root, task, model, frame):
    artifact_row = task["artifact"]
    path = Path(artifact_root) / artifact_row["artifact_relpath"]
    if sha256_file(path) != artifact_row["artifact_file_sha256"]:
        raise LogicalStratifiedV0ConflictError("V0 artifact file SHA changed")
    with np.load(path, allow_pickle=False) as data:
        content_sha256 = str(np.asarray(data["artifact_content_sha256"]).item())
    if content_sha256 != artifact_row["artifact_content_sha256"]:
        raise LogicalStratifiedV0ConflictError("V0 artifact content SHA changed")
    artifact = load_logical_stratified_frozen_artifact(path, model, frame)
    if artifact.descriptor != artifact_row["descriptor"]:
        raise LogicalStratifiedV0ConflictError("V0 artifact descriptor changed")
    return artifact


def _initial_state(task, artifact, model, syndrome, epsilon):
    family = task["init_family"]
    seed_identity = LogicalStratifiedSeedIdentity(
        source_commit=task["source_commit"], config_sha256=task["config_sha256"],
        registry_sha256=task["registry_sha256"],
        cell_fingerprint=_cell_fingerprint(task["cell"]), init_family=family,
        trajectory_index=task["trajectory_index"],
        resource_tier=task["resource_tier"],
        trajectory_namespace=task["trajectory_namespace"],
    )
    if family == "P":
        return epsilon.copy(), seed_identity
    if family == "U":
        return uniform_hard_coset_state(
            model, syndrome, seed_identity.seed("initialize_uniform"),
        ), seed_identity
    tail = artifact.descriptor["identity"].get("tail_indices_sha256")
    if not isinstance(tail, str):
        raise LogicalStratifiedV0ConflictError("V0 S-tail artifact identity changed")
    transcript = artifact.transcript
    catalog = artifact.catalog
    codebook = artifact.codebook
    indices = _tail_indices(codebook, transcript, catalog, 8)
    if hashlib.sha256(indices.astype(">i4").tobytes()).hexdigest() != tail:
        raise LogicalStratifiedV0ConflictError("V0 S-tail schedule digest changed")
    state = np.unpackbits(
        transcript.decoded_packed[indices[task["trajectory_index"]]],
        count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    return state.copy(), seed_identity


def _raw_payload(task, raw, *, config, artifact_row, core_seconds, wall_seconds):
    return {
        "raw_version": np.array(V0_RAW_VERSION),
        "sampler_raw_version": np.array(raw["raw_version"]),
        "contract_version": np.array(V0_CONTRACT_VERSION),
        "task_json": np.array(canonical_json(task)),
        "task_fingerprint": np.array(sha256_json(task)),
        "source_commit": np.array(task["source_commit"]),
        "archive_sha256": np.array(task["archive_sha256"]),
        "source_manifest_sha256": np.array(task["source_manifest_sha256"]),
        "v0_config_sha256": np.array(config["v0_config_sha256"]),
        "registry_sha256": np.array(task["registry_sha256"]),
        "artifact_file_sha256": np.array(artifact_row["artifact_file_sha256"]),
        "artifact_content_sha256": np.array(artifact_row["artifact_content_sha256"]),
        "core_seconds": np.float64(core_seconds),
        "wall_seconds": np.float64(wall_seconds),
        **{name: np.asarray(value) for name, value in raw.items()
           if name != "raw_version"},
    }


def _run_task_from_context(context, task, output_path=None):
    registry, code, H, model, frame, uniform_seed, epsilon, syndrome, config, artifacts = context
    artifact = artifacts[float(task["artifact"]["alpha_temperature"])]
    initial, seed = _initial_state(task, artifact, model, syndrome, epsilon)
    sampler = LogicalStratifiedConfig(
        p=config["cell"]["p"],
        burn_steps=config["resource_tier"]["burn_steps"],
        measurement_steps=config["resource_tier"]["measurement_steps"],
        alpha_temperature=float(task["artifact"]["alpha_temperature"]),
    )
    wall_start = time.monotonic()
    core_start = time.process_time()
    raw = run_logical_stratified_trajectory(
        model, frame, syndrome, sampler, seed, initial, artifact=artifact,
    )
    replay_logical_stratified_trajectory(
        model, frame, syndrome, sampler, seed, initial, raw, artifact=artifact,
    )
    payload = _raw_payload(
        task, raw, config=config, artifact_row=task["artifact"],
        core_seconds=time.process_time() - core_start,
        wall_seconds=time.monotonic() - wall_start,
    )
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(f"V0 raw already exists: {output_path}")
        atomic_npz(output_path, **payload)
    return payload


_NODE_CONTEXT = None


def _worker_task(task_and_path):
    task, output_path = task_and_path
    return _run_task_from_context(_NODE_CONTEXT, task, output_path)


def _node_context(registry_path, config_path, artifact_root, tasks):
    registry = load_registry(registry_path)
    config = load_v0_config(config_path, registry=registry)
    registry, code, H, model, frame, uniform_seed, epsilon, syndrome = _context(
        registry_path, config,
    )
    artifacts = {}
    for task in tasks:
        tau = float(task["artifact"]["alpha_temperature"])
        if tau not in artifacts:
            artifacts[tau] = _artifact_for_task(artifact_root, task, model, frame)
    return (registry, code, H, model, frame, uniform_seed, epsilon, syndrome,
            config, artifacts)


def run_v0_node(manifest_path, registry_path, config_path, node, raw_root,
                *, num_workers=12):
    """Run this node's fixed ownership slice with fork-shared frozen artifacts."""
    if node not in V0_NODES:
        raise ValueError("unknown V0 execution node")
    num_workers = _require_positive_int(num_workers, "num_workers")
    manifest = load_v0_manifest(manifest_path)
    validate_v0_manifest(manifest, registry_path, config_path)
    tasks = [task for task in manifest["tasks"] if task["node"] == node]
    if len(tasks) != 24:
        raise LogicalStratifiedV0ConflictError("V0 node ownership changed")
    context = _node_context(registry_path, config_path, manifest["artifact_root"], tasks)
    global _NODE_CONTEXT
    _NODE_CONTEXT = context
    raw_root = Path(raw_root)
    assignments = [
        (task, raw_root / _task_output_relpath(task)) for task in tasks
    ]
    if any(path.exists() for _, path in assignments):
        raise FileExistsError("V0 node raw path already exists")
    if num_workers == 1:
        for assignment in assignments:
            _worker_task(assignment)
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(processes=num_workers) as pool:
            list(pool.imap_unordered(_worker_task, assignments, chunksize=1))
    return {
        "node": node,
        "task_count": len(tasks),
        "raw_root": str(raw_root.resolve()),
        "task_sha256": sha256_json(tasks),
    }


def _portable_raw_sha(raw):
    digest = hashlib.sha256()
    for name in sorted(raw):
        if name in {"core_seconds", "wall_seconds"}:
            continue
        value = np.ascontiguousarray(np.asarray(raw[name]))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def preflight_v0(registry_path, config_path, manifest_path, node, output_path):
    """Validate artifacts and replay six short fixed-clock probes on a node."""
    manifest = load_v0_manifest(manifest_path)
    validate_v0_manifest(manifest, registry_path, config_path)
    tasks = [task for task in manifest["tasks"] if task["node"] == node]
    all_tasks = manifest["tasks"]
    context = _node_context(registry_path, config_path, manifest["artifact_root"], all_tasks)
    registry, code, H, model, frame, uniform_seed, epsilon, syndrome, config, artifacts = context
    rows = []
    for tau, artifact in sorted(artifacts.items()):
        for family in V0_FAMILIES:
            template = next(
                task for task in all_tasks
                if float(task["artifact"]["alpha_temperature"]) == tau
                and task["init_family"] == family
            )
            short = dict(config)
            short["resource_tier"] = {
                "name": "V0P", "burn_steps": 8, "measurement_steps": 16,
            }
            probe_context = (*context[:8], short, artifacts)
            payload = _run_task_from_context(probe_context, template)
            rows.append({
                "alpha_temperature": tau,
                "init_family": family,
                "portable_raw_sha256": _portable_raw_sha(payload),
                "artifact_content_sha256": template["artifact"]["artifact_content_sha256"],
            })
    identity = {
        "contract_version": V0_CONTRACT_VERSION,
        "preflight_version": V0_PREFLIGHT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "node": node,
        "rows": rows,
    }
    result = {**identity, "preflight_sha256": sha256_json(identity)}
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"V0 preflight already exists: {path}")
    atomic_json(path, result)
    return result


def _scalar(data, name):
    value = data[name]
    if np.asarray(value).shape != ():
        raise LogicalStratifiedV0ConflictError(f"V0 scalar {name} changed shape")
    return np.asarray(value).item()


def validate_v0_raw(path, manifest_path, registry_path, config_path):
    """Load a raw NPZ without pickle and deterministically replay all decisions."""
    manifest = load_v0_manifest(manifest_path)
    validate_v0_manifest(manifest, registry_path, config_path)
    try:
        with np.load(path, allow_pickle=False) as data:
            raw = {name: data[name].copy() for name in data.files}
            if any(value.dtype.hasobject for value in raw.values()):
                raise LogicalStratifiedV0ConflictError("V0 raw contains object array")
    except LogicalStratifiedV0ConflictError:
        raise
    except Exception as exc:
        raise LogicalStratifiedV0ConflictError(f"cannot load V0 raw: {exc}") from exc
    if not V0_RAW_BASE_FIELDS.issubset(raw):
        raise LogicalStratifiedV0ConflictError("V0 raw base fields missing")
    if (_scalar(raw, "raw_version") != V0_RAW_VERSION
            or _scalar(raw, "contract_version") != V0_CONTRACT_VERSION):
        raise LogicalStratifiedV0ConflictError("V0 raw version changed")
    task = json.loads(str(_scalar(raw, "task_json")))
    if canonical_json(task) != str(_scalar(raw, "task_json")):
        raise LogicalStratifiedV0ConflictError("V0 raw task is noncanonical")
    if task not in manifest["tasks"] or _scalar(raw, "task_fingerprint") != sha256_json(task):
        raise LogicalStratifiedV0ConflictError("V0 raw task changed")
    context = _node_context(
        registry_path, config_path, manifest["artifact_root"], [task],
    )
    expected = _run_task_from_context(context, task)
    expected_payload = _raw_payload(
        task, expected, config=context[8], artifact_row=task["artifact"],
        core_seconds=float(_scalar(raw, "core_seconds")),
        wall_seconds=float(_scalar(raw, "wall_seconds")),
    )
    if set(raw) != set(expected_payload):
        raise LogicalStratifiedV0ConflictError("V0 raw schema changed")
    for name, expected_value in expected_payload.items():
        if name in {"core_seconds", "wall_seconds"}:
            if not np.isfinite(float(_scalar(raw, name))) or float(_scalar(raw, name)) < 0.0:
                raise LogicalStratifiedV0ConflictError("V0 raw timing is invalid")
            continue
        if not np.array_equal(raw[name], np.asarray(expected_value)):
            raise LogicalStratifiedV0ConflictError(f"V0 raw replay changed: {name}")
    return task


def _chain_transport(raw, k, masks, num_qubits):
    labels = np.asarray(raw["measurement_labels"], dtype=np.uint64)
    previous = np.concatenate((
        np.asarray([np.uint64(_scalar(raw, "burn_label"))], dtype=np.uint64),
        labels[:-1],
    ))
    accepted_cross = (
        np.asarray(raw["measurement_accepted"], dtype=np.uint8).astype(bool)
        & np.asarray(raw["measurement_state_changed"], dtype=np.uint8).astype(bool)
        & np.asarray(raw["measurement_label_changed"], dtype=np.uint8).astype(bool)
    )
    deltas = labels ^ previous
    source_indices = np.asarray(raw["measurement_proposal_anchor_index"], dtype=np.int16)
    sources = np.unique(source_indices[accepted_cross & (source_indices >= 0)])
    all_labels = np.concatenate((
        np.asarray([np.uint64(_scalar(raw, "initial_label"))], dtype=np.uint64),
        np.asarray(raw["burn_labels"], dtype=np.uint64), labels,
    ))
    signs = character_values(all_labels, masks)
    leave_return = 0
    for column in range(signs.shape[1]):
        origin = signs[0, column]
        left = False
        for value in signs[1:, column]:
            if value != origin:
                left = True
            elif left:
                leave_return = 1
                break
        if leave_return:
            break
    basis_changed = np.asarray([
        int(np.count_nonzero(accepted_cross & ((deltas >> np.uint64(bit)) & 1).astype(bool)))
        for bit in range(k)
    ], dtype=np.int32)
    return {
        "measurement_cross_label_changes": int(accepted_cross.sum()),
        "burn_cross_label_changes": int(_scalar(raw, "burn_cross_label_changes")),
        "distinct_catalog_sources": [int(value) for value in sources],
        "leave_return_character": int(leave_return),
        "basis_character_changes": basis_changed.tolist(),
        "accepted_cross_label_deltas": [int(value) for value in deltas[accepted_cross]],
        "initial_weight": int(np.unpackbits(
            raw["initial_state_packed"], count=int(num_qubits),
            bitorder="little",
        ).sum()),
        "measurement_mean_weight": float(np.asarray(raw["measurement_weights"], dtype=np.float64).mean()),
    }


def analyze_v0(raw_root, manifest_path, registry_path, config_path, output_path=None):
    """Analyze only pre-registered transport observables; never estimate q_top."""
    manifest = load_v0_manifest(manifest_path)
    validate_v0_manifest(manifest, registry_path, config_path)
    registry = load_registry(registry_path)
    config = load_v0_config(config_path, registry=registry)
    _, code, H, model, frame, _, _, _ = _context(registry_path, config)
    character_seed = derive_seed(
        config["seed_namespace"], registry["registry_sha256"], code["code_id"],
        "diagnostic_characters",
    )
    characters = frozen_character_set(
        model.k, character_seed, config["frozen_nonbasis_characters"],
    )
    raw_root = Path(raw_root)
    rows = []
    missing = []
    for task in manifest["tasks"]:
        path = raw_root / _task_output_relpath(task)
        if not path.exists():
            missing.append(_task_output_relpath(task))
            continue
        validate_v0_raw(path, manifest_path, registry_path, config_path)
        with np.load(path, allow_pickle=False) as data:
            raw = {name: data[name].copy() for name in data.files}
        row = {
            "alpha_temperature": task["artifact"]["alpha_temperature"],
            "init_family": task["init_family"],
            "trajectory_index": task["trajectory_index"],
            "raw_relpath": _task_output_relpath(task),
            "raw_sha256": sha256_file(path),
            "wall_seconds": float(_scalar(raw, "wall_seconds")),
            "core_seconds": float(_scalar(raw, "core_seconds")),
            **_chain_transport(raw, model.k, characters.masks, model.num_qubits),
        }
        rows.append(row)
    grouped = {}
    gates = config["gates"]
    for tau in V0_TAU:
        grouped[str(tau)] = {}
        for family in V0_FAMILIES:
            subset = [
                row for row in rows
                if float(row["alpha_temperature"]) == tau
                and row["init_family"] == family
            ]
            changes = [
                row["measurement_cross_label_changes"] + row["burn_cross_label_changes"]
                for row in subset
            ]
            source_count = len({
                source for row in subset for source in row["distinct_catalog_sources"]
            })
            grouped[str(tau)][family] = {
                "chain_count": len(subset),
                "accepted_cross_label_changes": int(sum(changes)),
                "chains_with_two_cross_label_changes": int(sum(value >= 2 for value in changes)),
                "distinct_catalog_sources": source_count,
                "leave_return_character_chains": int(sum(
                    row["leave_return_character"] for row in subset
                )),
                "median_wall_seconds": float(np.median([
                    row["wall_seconds"] for row in subset
                ])) if subset else None,
            }
        values = grouped[str(tau)]
        for family, value in values.items():
            value["passes_transport_gate"] = bool(
                value["chain_count"] == config["trajectories_per_family"]
                and value["accepted_cross_label_changes"]
                >= gates["minimum_accepted_cross_label_changes_per_family"]
                and value["chains_with_two_cross_label_changes"]
                >= gates["minimum_chains_with_two_cross_label_changes_per_family"]
                and value["distinct_catalog_sources"]
                >= gates["minimum_distinct_catalog_sources_per_family"]
                and value["leave_return_character_chains"]
                >= gates["minimum_leave_return_character_chains_per_family"]
            )
    complete = not missing and len(rows) == len(manifest["tasks"])
    passing = [
        tau for tau in V0_TAU
        if complete and all(grouped[str(tau)][family]["passes_transport_gate"]
                            for family in V0_FAMILIES)
    ]
    status = (
        "LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN" if passing
        else "UNRESOLVED_LSI_IMH_V0_TRANSPORT"
    )
    identity = {
        "contract_version": V0_CONTRACT_VERSION,
        "report_version": V0_REPORT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "character_sha256": characters.character_sha256,
        "raw_rows": rows,
        "missing": missing,
        "groups": grouped,
        "status": status,
        "passing_alpha_temperatures": passing,
        "formal_authorization": False,
        "production_authorization": False,
    }
    report = {**identity, "report_sha256": sha256_json(identity)}
    if output_path is not None:
        path = Path(output_path)
        if path.exists():
            raise FileExistsError(f"V0 report already exists: {path}")
        atomic_json(path, report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-artifacts", "build-manifest"):
        item = subparsers.add_parser(name)
        item.add_argument("--registry", required=True)
        item.add_argument("--config", required=True)
        item.add_argument("--source-commit", required=True)
        item.add_argument("--archive-sha256", required=True)
        item.add_argument("--source-manifest-sha256", required=True)
        item.add_argument("--artifact-root", required=True)
        if name == "build-manifest":
            item.add_argument("--output", required=True)
    item = subparsers.add_parser("preflight")
    item.add_argument("--registry", required=True)
    item.add_argument("--config", required=True)
    item.add_argument("--manifest", required=True)
    item.add_argument("--node", required=True)
    item.add_argument("--output", required=True)
    item = subparsers.add_parser("run-node")
    item.add_argument("--registry", required=True)
    item.add_argument("--config", required=True)
    item.add_argument("--manifest", required=True)
    item.add_argument("--node", required=True)
    item.add_argument("--raw-root", required=True)
    item.add_argument("--num-workers", required=True, type=int)
    item = subparsers.add_parser("analyze")
    item.add_argument("--registry", required=True)
    item.add_argument("--config", required=True)
    item.add_argument("--manifest", required=True)
    item.add_argument("--raw-root", required=True)
    item.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "prepare-artifacts":
        result = prepare_v0_artifacts(
            args.registry, args.config, args.source_commit, args.archive_sha256,
            args.source_manifest_sha256, args.artifact_root,
        )
    elif args.command == "build-manifest":
        result = build_v0_manifest(
            args.registry, args.config, args.source_commit, args.archive_sha256,
            args.source_manifest_sha256, args.artifact_root, args.output,
        )
    elif args.command == "preflight":
        result = preflight_v0(
            args.registry, args.config, args.manifest, args.node, args.output,
        )
    elif args.command == "run-node":
        result = run_v0_node(
            args.manifest, args.registry, args.config, args.node, args.raw_root,
            num_workers=args.num_workers,
        )
    else:
        result = analyze_v0(
            args.raw_root, args.manifest, args.registry, args.config, args.output,
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
