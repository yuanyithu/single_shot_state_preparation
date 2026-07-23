import json
import shutil

import pytest

from data.expander_code.exp102.exp102_pipeline.io import canonical_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified_v0 import (
    LogicalStratifiedV0ConflictError,
    V0_ARTIFACT_ROOT_RELPATH,
    V0_CONTRACT_VERSION,
    _manifest_artifact_root,
    _portable_raw_sha,
    build_v0_manifest,
    load_v0_manifest,
    load_v0_config,
    validate_v0_manifest,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


def _config_path():
    return "data/expander_code/exp102/config/q0_logical_stratified.v0.json"


def test_v0_config_is_narrow_diagnostic_contract():
    registry = load_registry("data/expander_code/exp102/registry/registry.json")
    config = load_v0_config(_config_path(), registry=registry)
    assert config["contract_version"] == V0_CONTRACT_VERSION
    assert config["cell"]["code_id"] == "m08_c06"
    assert config["cell"]["p"] == pytest.approx(0.04)
    assert config["scope"]["formal_authorization"] is False
    assert config["scope"]["production_authorization"] is False
    assert [item["alpha_temperature"] for item in config["candidates"]] == [0.5, 1.0]


def test_v0_config_rejects_scope_or_gate_edits(tmp_path):
    registry = load_registry("data/expander_code/exp102/registry/registry.json")
    original = json.loads(open(_config_path(), encoding="ascii").read())
    original["scope"]["formal_authorization"] = True
    path = tmp_path / "bad.json"
    path.write_text(canonical_json(original) + "\n", encoding="ascii")
    with pytest.raises(LogicalStratifiedV0ConflictError, match="schedule/gates/scope"):
        load_v0_config(path, registry=registry)


def test_portable_digest_excludes_only_timing_fields():
    first = {
        "labels": [1, 2], "core_seconds": 1.0, "wall_seconds": 2.0,
    }
    second = {
        "labels": [1, 2], "core_seconds": 9.0, "wall_seconds": 20.0,
    }
    changed = {
        "labels": [1, 3], "core_seconds": 1.0, "wall_seconds": 2.0,
    }
    assert _portable_raw_sha(first) == _portable_raw_sha(second)
    assert _portable_raw_sha(first) != _portable_raw_sha(changed)


def test_manifest_uses_portable_artifact_layout_and_replays_after_copy(tmp_path):
    registry_path = "data/expander_code/exp102/registry/registry.json"
    config_path = _config_path()
    registry = load_registry(registry_path)
    config = load_v0_config(config_path, registry=registry)
    run_root = tmp_path / "remote_run"
    artifact_root = run_root / "artifacts"
    artifact_root.mkdir(parents=True)
    rows = [
        {
            "alpha_temperature": candidate["alpha_temperature"],
            "method_id": candidate["method_id"],
            "artifact_relpath": f"artifacts/{candidate['method_id']}.npz",
            "artifact_file_sha256": "a" * 64,
            "artifact_content_sha256": "b" * 64,
            "descriptor": {},
        }
        for candidate in config["candidates"]
    ]
    artifact_identity = {"artifact_rows": rows}
    artifact_manifest = {
        **artifact_identity,
        "artifact_manifest_sha256": sha256_json(artifact_identity),
    }
    (artifact_root / "ARTIFACT_MANIFEST.json").write_text(
        canonical_json(artifact_manifest) + "\n", encoding="ascii",
    )
    manifest_path = run_root / "control" / "V0_MANIFEST.json"
    manifest = build_v0_manifest(
        registry_path, config_path, "1" * 40, "2" * 64, "3" * 64,
        artifact_root, manifest_path,
    )
    assert manifest["artifact_root_relpath"] == V0_ARTIFACT_ROOT_RELPATH
    assert str(run_root) not in manifest_path.read_text(encoding="ascii")
    assert _manifest_artifact_root(manifest_path, manifest) == artifact_root.resolve()
    assert validate_v0_manifest(
        load_v0_manifest(manifest_path), registry_path, config_path, artifact_root,
    )

    local_copy = tmp_path / "local_copy"
    shutil.copytree(run_root, local_copy)
    copied_manifest_path = local_copy / "control" / "V0_MANIFEST.json"
    copied_manifest = load_v0_manifest(copied_manifest_path)
    copied_artifact_root = local_copy / "artifacts"
    assert _manifest_artifact_root(copied_manifest_path, copied_manifest) == copied_artifact_root.resolve()
    assert validate_v0_manifest(
        copied_manifest, registry_path, config_path, copied_artifact_root,
    )
