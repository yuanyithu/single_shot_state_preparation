"""Control, schedule, and analyzer tests for the fresh m8 T1 diagnostic."""

from importlib import import_module
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


workflow = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.workflow"
)
analyzer = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.analyze_t1"
)


def _source():
    return {
        "archive_sha256": "1" * 64,
        "source_commit": "2" * 40,
        "source_manifest_sha256": "3" * 64,
    }


def test_frozen_t1_control_and_schedule_are_complete():
    config, config_sha = workflow._load_config()
    context = workflow._load_control(workflow.SOURCE_CONTROL_DIR, config, config_sha)
    tasks = workflow._task_rows(context, _source())
    assert len(tasks) == 40
    assert {task["family"] for task in tasks} == set(workflow.FAMILIES)
    assert len({task["task_fingerprint"] for task in tasks}) == 40
    assert {node: sum(task["owner"] == node for task in tasks) for node in (
        "nd-1", "nd-2", "nd-3",
    )} == {"nd-1": 14, "nd-2": 13, "nd-3": 13}
    assert np.unique(context["arrays"]["fixed_labels"][3:]).size == 8
    assert len({row.tobytes() for row in context["arrays"]["fixed_b_blocks"][3:]}) == 8


def test_t1_control_rejects_tampered_npz(tmp_path):
    config, config_sha = workflow._load_config()
    control_dir = tmp_path / "control"
    shutil.copytree(workflow.SOURCE_CONTROL_DIR, control_dir)
    path = control_dir / "control.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["fixed_weights"][0] += 1
    np.savez_compressed(path, **payload)
    manifest_path = control_dir / "control_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["control_file_sha256"] = workflow.sha256_file(path)
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = workflow.sha256_json(core)
    workflow.atomic_json(manifest_path, manifest)
    with pytest.raises(workflow.T1ConflictError, match="content hash"):
        workflow._load_control(control_dir, config, config_sha)


def test_t1_b_column_packing_round_trip():
    rng = np.random.default_rng(17)
    bits = rng.integers(0, 2, size=(5, 24, 24), dtype=np.uint8)
    powers = np.left_shift(np.uint32(1), np.arange(24, dtype=np.uint32))
    columns = np.einsum("trc,r->tc", bits.astype(np.uint32), powers, optimize=False)
    packed = analyzer._columns_to_b_packed(columns, 24)
    observed = np.unpackbits(packed, axis=1, count=24 * 24, bitorder="little")
    assert np.array_equal(observed.reshape(5, 24, 24), bits)


def test_t1_bridge_reaction_coordinate_is_directional():
    config, config_sha = workflow._load_config()
    context = workflow._load_control(workflow.SOURCE_CONTROL_DIR, config, config_sha)
    map0, map1 = context["arrays"]["fixed_b_blocks"][1:3]
    assert np.count_nonzero(map0 ^ map1) == 6
    assert np.count_nonzero(map0 ^ map0) < np.count_nonzero(map0 ^ map1)
    assert np.count_nonzero(map1 ^ map1) < np.count_nonzero(map1 ^ map0)
