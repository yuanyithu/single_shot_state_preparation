"""Build a fresh v2 control from the fully audited validation-055 control."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path

import numpy as np


CONTRACT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.v2"
CONTROL_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.control.v2"
PREVIOUS_CONTROL_FILE_SHA256 = (
    "c84579cb2fcd593b176308610a5c69e0fe47f54136b61b9f70a7fff6d94c4168"
)
PREVIOUS_CONTROL_CONTENT_SHA256 = (
    "982fb9318fe423a1d642c118c4efccac247e446da07b6bdea4d8a64dab1b8421"
)

_previous = import_module(
    "data.expander_code.exp102.validation."
    "055_q0_random_full_column_direct_block_t1_m8_20260724.workflow"
)
_builder = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.build_frozen_control"
)


def _load_config(path):
    path = Path(path).resolve()
    serialized = path.read_text(encoding="ascii")
    config = json.loads(serialized)
    _builder._require(
        serialized == _builder.canonical_json(config) + "\n",
        "config is not canonical JSON",
    )
    _builder._require(
        config["version"] == config["contract_version"] == CONTRACT_VERSION,
        "direct-block T1 v2 contract version changed",
    )
    return config, _builder.sha256_file(path)


def _content_sha(metadata, arrays):
    digest = hashlib.sha256(CONTROL_VERSION.encode("ascii") + b"\0")
    digest.update(_builder.canonical_json(metadata).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, config_sha = _load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    _builder._require(not output_dir.exists(), "control output directory exists")

    previous_config, previous_config_sha = _previous._load_config()
    previous = _previous._load_control(
        _previous.SOURCE_CONTROL_DIR, previous_config, previous_config_sha,
    )
    _builder._require(
        _builder.sha256_file(previous["control_path"])
        == PREVIOUS_CONTROL_FILE_SHA256
        and previous["metadata"]["control_content_sha256"]
        == PREVIOUS_CONTROL_CONTENT_SHA256,
        "validation-055 source control identity changed",
    )
    _builder._require(
        config["artifacts"] == previous_config["artifacts"]
        and config["cell"] == previous_config["cell"]
        and config["initialization"] == previous_config["initialization"]
        and config["registry_sha256"] == previous_config["registry_sha256"]
        and config["statistics"] == previous_config["statistics"],
        "v2 changed scientific control geometry",
    )

    cell_fingerprint = _builder.sha256_json(config["cell"])
    logical_seed = _builder.derive_seed(
        config["seed_namespace"], config_sha, config["registry_sha256"],
        cell_fingerprint, "logical_characters",
    )
    b_seed = _builder.derive_seed(
        config["seed_namespace"], config_sha, config["registry_sha256"],
        cell_fingerprint, "b_characters",
    )
    logical_characters = _builder.frozen_character_set(
        previous["model"].k, logical_seed,
        num_nonbasis=config["statistics"]["logical_nonbasis_character_count"],
    )
    b_characters = _builder.frozen_b_character_set(
        previous["H"].shape[0], b_seed,
        dense_count=config["statistics"]["b_dense_character_count"],
    )
    arrays = {
        name: np.ascontiguousarray(value)
        for name, value in previous["arrays"].items()
    }
    arrays["logical_character_masks"] = logical_characters.masks
    arrays["logical_basis_positions"] = logical_characters.basis_positions
    arrays["b_character_masks_packed"] = b_characters.masks_packed
    _builder._require(
        not np.array_equal(
            arrays["logical_character_masks"],
            previous["arrays"]["logical_character_masks"],
        )
        and not np.array_equal(
            arrays["b_character_masks_packed"],
            previous["arrays"]["b_character_masks_packed"],
        ),
        "v2 character panels were not refreshed",
    )

    fixed = previous["fixed_states"]
    residuals = (
        previous["model"].H_check.astype(np.int64)
        @ fixed.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    _builder._require(
        np.array_equal(
            residuals,
            np.repeat(previous["syndrome"][None, :], fixed.shape[0], axis=0),
        )
        and int(previous["syndrome"].sum()) == 160
        and not np.array_equal(
            previous["syndrome"], np.zeros_like(previous["syndrome"])
        ),
        "v2 inherited an illegal hard-coset initialization",
    )

    metadata = dict(previous["metadata"])
    metadata.update({
        "b_character_seed": int(b_seed),
        "b_character_sha256": b_characters.character_sha256,
        "config_sha256": config_sha,
        "control_version": CONTROL_VERSION,
        "logical_character_seed": int(logical_seed),
        "logical_character_sha256": logical_characters.character_sha256,
        "predecessor_control_content_sha256": PREVIOUS_CONTROL_CONTENT_SHA256,
        "predecessor_control_file_sha256": PREVIOUS_CONTROL_FILE_SHA256,
    })
    metadata.pop("control_content_sha256", None)
    content_sha = _content_sha(metadata, arrays)
    metadata["control_content_sha256"] = content_sha

    output_dir.mkdir(parents=True)
    control_path = output_dir / "control.npz"
    _builder.atomic_npz(
        control_path,
        metadata_json=np.array(_builder.canonical_json(metadata)),
        **arrays,
    )
    core = {
        "b_character_sha256": b_characters.character_sha256,
        "config_sha256": config_sha,
        "control_content_sha256": content_sha,
        "control_file_sha256": _builder.sha256_file(control_path),
        "control_version": CONTROL_VERSION,
        "fixed_b_pairwise_hamming": previous["manifest"][
            "fixed_b_pairwise_hamming"
        ],
        "fixed_labels": [int(value) for value in arrays["fixed_labels"]],
        "fixed_names": metadata["fixed_names"],
        "fixed_weights": [int(value) for value in arrays["fixed_weights"]],
        "logical_character_sha256": logical_characters.character_sha256,
        "predecessor_control_content_sha256": PREVIOUS_CONTROL_CONTENT_SHA256,
        "predecessor_control_file_sha256": PREVIOUS_CONTROL_FILE_SHA256,
        "scope": config["scope"],
    }
    manifest = {**core, "manifest_sha256": _builder.sha256_json(core)}
    _builder.atomic_json(output_dir / "control_manifest.json", manifest)
    print(_builder.canonical_json(manifest))


if __name__ == "__main__":
    main()
