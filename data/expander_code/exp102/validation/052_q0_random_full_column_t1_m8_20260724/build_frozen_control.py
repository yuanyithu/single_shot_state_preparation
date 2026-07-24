"""Build the immutable, outcome-free control artifact for the m8 T1 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
from data.expander_code.exp102.exp102_pipeline.q0_center_preserving import (
    build_dressed_logical_catalog,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    frozen_character_set,
    state_label,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import split_hgp_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import (
    frozen_b_character_set,
    _disorder,
)
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import (
    load_logical_stratified_frozen_artifact,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTROL_VERSION = "exp102.q0_random_full_column.t1_m8.control.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_config(path):
    path = Path(path).resolve()
    serialized = path.read_text(encoding="ascii")
    config = json.loads(serialized)
    _require(serialized == canonical_json(config) + "\n", "config is not canonical JSON")
    _require(config["version"] == config["contract_version"]
             == "exp102.q0_random_full_column.t1_m8.v0",
             "T1 contract version changed")
    return config, sha256_file(path)


def _content_sha(metadata, arrays):
    digest = hashlib.sha256(CONTROL_VERSION.encode("ascii") + b"\0")
    digest.update(canonical_json(metadata).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _b_block(state, H):
    return split_hgp_state(state, H)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, config_sha = _load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    _require(not output_dir.exists(), "control output directory already exists")

    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "registry identity changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600 and model.k == 64
             and int(syndrome.sum()) == 160,
             "m8 hard-cell identity changed")

    artifact_paths = {
        name: EXP102_ROOT / descriptor["relpath"]
        for name, descriptor in config["artifacts"].items()
    }
    for name, path in artifact_paths.items():
        _require(sha256_file(path) == config["artifacts"][name]["file_sha256"],
                 f"source artifact changed: {name}")
    geometry = json.loads(artifact_paths["initialization_geometry_report"].read_text(
        encoding="ascii",
    ))
    _require(geometry["report_sha256"] == sha256_json({
        key: value for key, value in geometry.items() if key != "report_sha256"
    }), "initialization geometry self-hash failed")
    selected = [int(value) for value in config["initialization"]["s_catalog_indices"]]
    _require(selected == geometry["selected_catalog_indices"],
             "S initialization selection changed")

    map_path = artifact_paths["map_source"]
    with np.load(map_path, allow_pickle=False) as archive:
        map_anchors = np.asarray(archive["anchors"], dtype=np.uint8)
    lsi_path = artifact_paths["logical_stratified_source"]
    lsi = load_logical_stratified_frozen_artifact(lsi_path, model, frame)
    catalog = build_dressed_logical_catalog(model, frame, lsi, max_moves=127)
    _require(catalog.catalog_sha256 == geometry["catalog_sha256"],
             "logical catalog changed after geometry freeze")
    catalog_anchors = catalog.unpack_anchors()
    s_states = np.ascontiguousarray(catalog_anchors[selected], dtype=np.uint8)
    fixed_states = np.ascontiguousarray(
        np.concatenate((planted[None, :], map_anchors, s_states), axis=0),
        dtype=np.uint8,
    )
    fixed_names = ["P", "M0", "M1", *(f"S{index}" for index in range(8))]
    _require(fixed_states.shape == (11, model.num_qubits), "fixed-state shape changed")
    recovered = (
        model.H_check.astype(np.int64) @ fixed_states.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    _require(np.array_equal(
        recovered, np.repeat(syndrome[None, :], fixed_states.shape[0], axis=0),
    ), "a fixed initialization leaves the hard coset")
    labels = np.asarray([state_label(frame, state) for state in fixed_states], dtype=np.uint64)
    weights = fixed_states.sum(axis=1).astype(np.int32)
    b_blocks = np.ascontiguousarray(
        np.stack([_b_block(state, H) for state in fixed_states]), dtype=np.uint8,
    )
    _require(np.unique(labels[3:]).size == 8
             and len({block.tobytes() for block in b_blocks[3:]}) == 8,
             "S starts lost logical or B diversity")
    _require(labels[0] == labels[1] == labels[2]
             and int((b_blocks[1] ^ b_blocks[2]).sum()) == 6,
             "MAP bridge identity changed")

    cell_fingerprint = sha256_json(config["cell"])
    logical_seed = derive_seed(
        config["seed_namespace"], config_sha, registry["registry_sha256"],
        cell_fingerprint, "logical_characters",
    )
    b_seed = derive_seed(
        config["seed_namespace"], config_sha, registry["registry_sha256"],
        cell_fingerprint, "b_characters",
    )
    logical_characters = frozen_character_set(
        model.k, logical_seed,
        num_nonbasis=config["statistics"]["logical_nonbasis_character_count"],
    )
    b_characters = frozen_b_character_set(
        H.shape[0], b_seed,
        dense_count=config["statistics"]["b_dense_character_count"],
    )
    arrays = {
        "H": np.ascontiguousarray(H, dtype=np.uint8),
        "b_character_masks_packed": b_characters.masks_packed,
        "fixed_b_blocks": b_blocks,
        "fixed_labels": labels,
        "fixed_states_packed": np.packbits(fixed_states, axis=1, bitorder="little"),
        "fixed_weights": weights,
        "logical_basis_positions": logical_characters.basis_positions,
        "logical_character_masks": logical_characters.masks,
        "syndrome_packed": np.packbits(syndrome, bitorder="little"),
    }
    metadata = {
        "artifact_file_sha256": {
            name: sha256_file(path) for name, path in artifact_paths.items()
        },
        "b_character_seed": int(b_seed),
        "b_character_sha256": b_characters.character_sha256,
        "catalog_sha256": catalog.catalog_sha256,
        "cell": config["cell"],
        "cell_fingerprint": cell_fingerprint,
        "config_sha256": config_sha,
        "control_version": CONTROL_VERSION,
        "fixed_names": fixed_names,
        "frame_fingerprint": frame.fingerprint(),
        "logical_character_seed": int(logical_seed),
        "logical_character_sha256": logical_characters.character_sha256,
        "model_fingerprint": model.fingerprint(),
        "registry_sha256": registry["registry_sha256"],
        "s_catalog_indices": selected,
        "uniform_disorder_seed": int(uniform_seed),
    }
    content_sha = _content_sha(metadata, arrays)
    metadata["control_content_sha256"] = content_sha
    output_dir.mkdir(parents=True)
    control_path = output_dir / "control.npz"
    atomic_npz(control_path, metadata_json=np.array(canonical_json(metadata)), **arrays)
    core = {
        "b_character_sha256": b_characters.character_sha256,
        "config_sha256": config_sha,
        "control_content_sha256": content_sha,
        "control_file_sha256": sha256_file(control_path),
        "control_version": CONTROL_VERSION,
        "fixed_b_pairwise_hamming": [
            {
                "distance": int((b_blocks[left] ^ b_blocks[right]).sum()),
                "left": fixed_names[left],
                "right": fixed_names[right],
            }
            for left in range(len(fixed_names))
            for right in range(left + 1, len(fixed_names))
        ],
        "fixed_labels": [int(value) for value in labels],
        "fixed_names": fixed_names,
        "fixed_weights": [int(value) for value in weights],
        "logical_character_sha256": logical_characters.character_sha256,
        "scope": config["scope"],
    }
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    atomic_json(output_dir / "control_manifest.json", manifest)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
