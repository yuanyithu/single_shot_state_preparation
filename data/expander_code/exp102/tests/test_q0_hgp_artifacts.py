import json
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import q0_hgp_screen as hs
from data.expander_code.exp102.exp102_pipeline import q0_map_mixture as mm
from data.expander_code.exp102.exp102_pipeline.io import atomic_json, atomic_npz
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    MAP_METHOD_ID,
    MapMixtureConfig,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_hgp_global.screen.v2.json"
SOURCE_COMMIT = "1" * 40
ARCHIVE_SHA256 = "a" * 64
SOURCE_MANIFEST_SHA256 = "b" * 64


@pytest.fixture(scope="module")
def frozen_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("hgp_map_artifact")
    cell = dict(hs.HARD_CELLS[0])
    descriptor = hs.build_hgp_map_artifact(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, root,
    )
    return root, cell, descriptor


def _protocol():
    registry = hs._registry_with_path(load_registry(REGISTRY_PATH), REGISTRY_PATH)
    config = hs.load_hgp_screen_config(CONFIG_PATH, registry)
    return registry, config


def test_artifact_is_complete_pickle_free_readonly_and_tamper_closed(
    tmp_path, frozen_artifact,
):
    root, cell, descriptor = frozen_artifact
    path = root / descriptor["artifact_relpath"]
    loaded = hs.load_hgp_map_artifact(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, root, descriptor,
    )
    assert loaded.descriptor == descriptor
    assert loaded.catalog.anchors.flags.writeable is False
    assert loaded.proposal.anchor_centers.flags.writeable is False
    assert loaded.proposal.coordinates.basis.flags.writeable is False
    assert descriptor["anchor_sha256"] == loaded.catalog.anchor_sha256
    assert descriptor["proposal_sha256"] == loaded.proposal.proposal_sha256

    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name].copy() for name in data.files}
        assert all(not value.dtype.hasobject for value in payload.values())
        assert {
            "anchors", "coordinate_H_check", "coordinate_basis",
            "coordinate_pivot_inverse", "proposal_anchor_centers",
            "proposal_theta_stabilizer", "proposal_component_weights",
        }.issubset(payload)

    with pytest.raises(FileExistsError):
        hs.build_hgp_map_artifact(
            REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, cell, root,
        )
    with pytest.raises(hs.HgpScreenConflictError):
        hs.load_hgp_map_artifact(
            REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, "c" * 64,
            SOURCE_MANIFEST_SHA256, cell, root,
        )

    tampered_root = tmp_path / "tampered"
    tampered_path = tampered_root / descriptor["artifact_relpath"]
    changed = {name: value.copy() for name, value in payload.items()}
    changed["proposal_theta_stabilizer"][0] += 0.0001
    atomic_npz(tampered_path, **changed)
    with pytest.raises(hs.HgpScreenConflictError, match="content SHA"):
        hs.load_hgp_map_artifact(
            REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, cell, tampered_root,
        )


def test_bound_artifact_load_is_independent_of_current_solver_version(
        frozen_artifact, monkeypatch):
    root, cell, descriptor = frozen_artifact
    monkeypatch.setattr(
        mm, "_solver_identity",
        lambda: "numpy=9.9;scipy=9.9;highs=9.9.9",
    )
    loaded = hs.load_hgp_map_artifact(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, root, descriptor,
    )
    assert loaded.descriptor == descriptor
    assert loaded.catalog.solver_identity == descriptor[
        "anchor_solver_identity"
    ]


def test_manifest_and_all_map_trajectories_bind_one_descriptor_per_cell(
    tmp_path, monkeypatch,
):
    registry, config = _protocol()
    descriptors = []
    for index, cell in enumerate(hs._map_cells(config)):
        descriptors.append({
            "artifact_version": hs.HGP_MAP_ARTIFACT_VERSION,
            "artifact_relpath": hs._map_artifact_relpath(cell),
            "artifact_file_sha256": format(index + 1, "064x"),
            "artifact_content_sha256": format(index + 11, "064x"),
            "source_commit": SOURCE_COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "registry_sha256": registry["registry_sha256"],
            "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
            "cell_fingerprint": hs._cell_fingerprint(cell),
            "model_fingerprint": format(index + 21, "064x"),
            "syndrome_sha256": format(index + 31, "064x"),
            "generation_wall_seconds": 1.0,
            "generation_core_seconds": 1.0,
            "requested_max_anchors": 8,
            "anchor_count": 1,
            "anchor_sha256": format(index + 41, "064x"),
            "anchor_solver_identity": "numpy=test;scipy=test;highs=test",
            "coordinate_sha256": format(index + 51, "064x"),
            "proposal_sha256": format(index + 61, "064x"),
        })
    monkeypatch.setattr(
        hs, "load_hgp_map_artifact_descriptors", lambda *args: descriptors,
    )
    manifest = hs.build_hgp_screen_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, "T1", tmp_path,
    )
    assert manifest["archive_sha256"] == ARCHIVE_SHA256
    assert manifest["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    assert manifest["importance_sampling"]["num_samples_per_cell"] == 50_000
    assert manifest["importance_sampling"]["used_for_gate_or_selection"] is False
    assert len(manifest["importance_sampling"]["outputs"]) == 2

    map_tasks = [
        row["task"] for row in manifest["tasks"]
        if row["task"]["method_id"] == MAP_METHOD_ID
    ]
    assert len(map_tasks) == 64
    for cell in hs._map_cells(config):
        values = [
            task["map_artifact"] for task in map_tasks
            if task["cell"] == cell
        ]
        assert len(values) == 32
        assert values == [values[0]] * 32
    assert all("map_artifact" not in row["task"] for row in manifest["tasks"]
               if row["task"]["method_id"] in hs.HP_METHODS)
    assert all(
        task["seed_identity"]["archive_sha256"] == ARCHIVE_SHA256
        and task["seed_identity"]["source_manifest_sha256"]
        == SOURCE_MANIFEST_SHA256
        for task in map_tasks
    )


def test_map_task_requires_and_replays_the_shared_artifact(
    tmp_path, monkeypatch, frozen_artifact,
):
    root, cell, descriptor = frozen_artifact
    registry, config = _protocol()
    task = hs._task_identity(
        config, registry, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, MAP_METHOD_ID, "T1", cell, "P", 0,
        map_artifact_descriptor=descriptor,
    )
    monkeypatch.setattr(
        hs, "_sampler_config",
        lambda method, p, tier, **kwargs: MapMixtureConfig(
            p, 8, 8, max_anchors=kwargs.get("max_anchors", 8),
        ),
    )

    def forbidden_rebuild(*args, **kwargs):
        raise AssertionError("trajectory attempted to rebuild its MAP artifact")

    monkeypatch.setattr(hs, "build_milp_map_anchors", forbidden_rebuild)
    monkeypatch.setattr(hs, "build_map_mixture_proposal", forbidden_rebuild)
    output = tmp_path / "map_raw.npz"
    hs.run_hgp_screen_task(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, task, root, output,
    )
    record = hs.validate_hgp_screen_raw(
        output, registry, config, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, root,
    )
    assert record["method_id"] == MAP_METHOD_ID
    with np.load(output, allow_pickle=False) as data:
        assert str(data["archive_sha256"].item()) == ARCHIVE_SHA256
        assert str(data["source_manifest_sha256"].item()) == SOURCE_MANIFEST_SHA256
        assert json.loads(str(data["map_artifact_descriptor_json"].item())) == descriptor
        assert str(data["sampler_anchor_sha256"].item()) == descriptor["anchor_sha256"]
        assert str(data["sampler_proposal_sha256"].item()) == descriptor["proposal_sha256"]

    changed = dict(task)
    changed["map_artifact"] = dict(descriptor)
    changed["map_artifact"]["artifact_file_sha256"] = "d" * 64
    with pytest.raises(hs.HgpScreenConflictError):
        hs.run_hgp_screen_task(
            REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, changed, root, tmp_path / "bad.npz",
        )


def test_frozen_50000_draw_is_is_replayable_and_auxiliary_only(
    tmp_path, monkeypatch, frozen_artifact,
):
    root, cell, descriptor = frozen_artifact
    output = tmp_path / "is.npz"
    result = hs.run_hgp_map_is_diagnostic(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, root, output,
    )
    assert result["used_for_gate_or_selection"] is False
    assert result["diagnostics"]["num_samples"] == 50_000
    assert np.isfinite(
        result["diagnostics"]["log_unnormalized_normalization_estimate"],
    )
    replay = hs.validate_hgp_map_is_diagnostic(
        output, REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, root,
    )
    for name in (
            "full_transcript_sha256", "portable_transcript_sha256",
            "nonportable_float_sha256", "field_manifest_sha256"):
        assert replay[name] == result[name]
    assert replay["used_for_gate_or_selection"] is False

    with np.load(output, allow_pickle=False) as data:
        payload = {name: data[name].copy() for name in data.files}
        assert payload["sample_states_packed"].shape[0] == 50_000
        assert payload["sample_coordinates_packed"].shape[0] == 50_000
        assert all(not value.dtype.hasobject for value in payload.values())
        identity = json.loads(str(payload["identity_json"].item()))
        diagnostics = json.loads(str(payload["diagnostics_json"].item()))
        assert identity["artifact_descriptor"] == descriptor
        assert identity["used_for_gate_or_selection"] is False
        replay_arrays = {
            name: value.copy() for name, value in payload.items()
            if name.startswith("sample_")
        }

    monkeypatch.setattr(
        hs, "_map_is_transcript",
        lambda proposal, p, num_samples, seed: (replay_arrays, diagnostics),
    )
    payload["sample_physical_weights"][0] += 1
    tampered = tmp_path / "is_tampered.npz"
    atomic_npz(tampered, **payload)
    with pytest.raises(hs.HgpScreenConflictError, match="stored evidence"):
        hs.validate_hgp_map_is_diagnostic(
            tampered, REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, cell, root,
        )


def test_analyzer_requires_the_exact_two_is_files_before_selection(
    tmp_path, monkeypatch,
):
    manifest = {
        "tasks": [],
        "importance_sampling": {
            "outputs": [
                f"importance_sampling/{index}.npz" for index in range(2)
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, manifest)
    monkeypatch.setattr(
        hs, "validate_hgp_screen_manifest", lambda *args: True,
    )
    with pytest.raises(hs.HgpScreenConflictError, match="IS raw set mismatch"):
        hs.analyze_hgp_screen(
            tmp_path, manifest_path, REGISTRY_PATH, CONFIG_PATH, tmp_path,
        )


def test_is_diagnostics_are_reported_but_cannot_change_pair_selection(
    tmp_path, monkeypatch,
):
    registry, config = _protocol()
    raw_root = tmp_path / "raw"
    tasks = []
    by_path = {}
    for method in hs.SCREEN_METHODS:
        for cell in hs._method_cells(config, method):
            for family in hs.INIT_FAMILIES:
                task = {
                    "method_id": method, "cell": cell,
                    "resource_tier": "T1", "init_family": family,
                    "trajectory_index": 0,
                }
                relative = f"trajectories/{len(tasks):03d}.npz"
                path = raw_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                by_path[str(path)] = task
                tasks.append({"task": task, "output_relpath": relative})
    is_outputs = []
    for index in range(2):
        relative = f"importance_sampling/{index}.npz"
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        is_outputs.append(relative)
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "resource_tier": "T1", "manifest_sha256": "c" * 64,
        "tasks": tasks,
        "importance_sampling": {"outputs": is_outputs},
    }
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, manifest)
    monkeypatch.setattr(
        hs, "validate_hgp_screen_manifest", lambda *args: True,
    )
    monkeypatch.setattr(
        hs, "_validate_raw_worker",
        lambda path: {
            **by_path[str(Path(path))], "algorithm_metrics": {},
            "core_seconds": 1.0,
        },
    )
    monkeypatch.setattr(hs, "_algorithm_failures", lambda *args: [])
    monkeypatch.setattr(
        hs._statistics, "_family_summary",
        lambda records, cfg: {"valid": True},
    )

    def fake_cell(records, cfg):
        method = records[0]["method_id"]
        return {
            "cell": records[0]["cell"], "method_id": method,
            "resource_tier": "T1", "num_qubits": 10,
            "core_seconds": 1.0 if method == "HP32" else 2.0,
            "valid": True, "failures": [],
        }

    monkeypatch.setattr(hs._statistics, "_cell_method_summary", fake_cell)
    monkeypatch.setattr(
        hs, "_b_family_summary",
        lambda records, cfg: {
            "init_family": records[0]["init_family"], "valid": True,
        },
    )
    monkeypatch.setattr(
        hs, "_b_cell_summary",
        lambda families, cfg: {
            "families": families, "initialization_comparison": {},
            "valid": True, "failures": [],
        },
    )
    monkeypatch.setattr(
        hs, "_compare_family_summaries",
        lambda left, right, cfg, family, n: {
            "init_family": family, "valid": True, "failures": [],
        },
    )
    diagnostic = {"value": 1.0}

    def fake_is(path, registry_path, config_path, source_commit, archive_sha,
                manifest_sha, cell, artifact_root):
        return {
            "sha256": "d" * 64, "cell": cell,
            "diagnostics": {"arbitrary_auxiliary_value": diagnostic["value"]},
            "full_transcript_sha256": "e" * 64,
            "portable_transcript_sha256": "f" * 64,
            "nonportable_float_sha256": "a" * 64,
            "field_manifest_sha256": "b" * 64,
            "used_for_gate_or_selection": False,
        }

    monkeypatch.setattr(hs, "validate_hgp_map_is_diagnostic", fake_is)
    first = hs.analyze_hgp_screen(
        raw_root, manifest_path, REGISTRY_PATH, CONFIG_PATH, tmp_path,
    )
    diagnostic["value"] = 1e300
    second = hs.analyze_hgp_screen(
        raw_root, manifest_path, REGISTRY_PATH, CONFIG_PATH, tmp_path,
    )
    assert first["status"] == second["status"] == "DIAGNOSTIC_HARD_PAIR_FOUND"
    assert first["selected_pair"] == second["selected_pair"]
    assert (
        first["importance_sampling_diagnostics"]
        != second["importance_sampling_diagnostics"]
    )
    assert all(
        row["used_for_gate_or_selection"] is False
        for row in first["importance_sampling_diagnostics"]
    )
