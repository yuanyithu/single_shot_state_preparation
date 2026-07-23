import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import q0_logical_stratified_v0 as v0
from data.expander_code.exp102.exp102_pipeline.io import canonical_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified_v0 import (
    LogicalStratifiedV0ConflictError,
    V0_ARTIFACT_ROOT_RELPATH,
    V0_CONTRACT_VERSION,
    _chain_transport,
    _manifest_artifact_root,
    _portable_float_replay_matches,
    _portable_raw_sha,
    _transport_group_summary,
    audit_v0_artifacts,
    build_v0_manifest,
    load_v0_config,
    load_v0_manifest,
    prepare_v0_artifacts,
    validate_v0_manifest,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


REGISTRY_PATH = "data/expander_code/exp102/registry/registry.json"
EXP102_ROOT = Path(__file__).resolve().parents[1]


def _config_path():
    return "data/expander_code/exp102/config/q0_logical_stratified.v0.v2.json"


def _load_config():
    registry = load_registry(REGISTRY_PATH)
    return load_v0_config(_config_path(), registry=registry)


def _write_stub_artifact_manifest(artifact_root, config, *, authority=None):
    artifact_root.mkdir(parents=True)
    _, _, _, model, _, uniform_seed, _, syndrome = v0._context(
        REGISTRY_PATH, config,
    )
    tail = list(range(100, 100 + config["trajectories_per_family"]))
    tail_sha256 = hashlib.sha256(
        np.asarray(tail, dtype=">i4").tobytes(),
    ).hexdigest()
    rows = []
    for candidate in config["candidates"]:
        tau = candidate["alpha_temperature"]
        rows.append({
            "alpha_temperature": tau,
            "method_id": candidate["method_id"],
            "artifact_relpath": v0._artifact_relpath(tau),
            "artifact_file_sha256": "a" * 64,
            "artifact_content_sha256": "b" * 64,
            "descriptor": {},
        })
    identity = {
        "artifact_authority": (
            config["artifact_authority"] if authority is None else authority
        ),
        "artifact_rows": rows,
        "cell": config["cell"],
        "codebook_sha256": "c" * 64,
        "config_sha256": config["v0_config_sha256"],
        "contract_version": V0_CONTRACT_VERSION,
        "matrix_syndrome_sha256": v0._matrix_syndrome_sha256(model, syndrome),
        "registry_sha256": config["registry_sha256"],
        "tail_candidate_indices": tail,
        "tail_indices_sha256": tail_sha256,
        "uniform_seed": int(uniform_seed),
    }
    artifact_manifest = {
        **identity,
        "artifact_manifest_sha256": sha256_json(identity),
    }
    (artifact_root / "ARTIFACT_MANIFEST.json").write_text(
        canonical_json(artifact_manifest) + "\n", encoding="ascii",
    )
    return artifact_manifest


def _build_stub_manifest(tmp_path, monkeypatch):
    config = _load_config()
    monkeypatch.setattr(v0, "_hostname_short", lambda: "nd-1")
    run_root = tmp_path / "remote_run"
    artifact_root = run_root / "artifacts"
    _write_stub_artifact_manifest(artifact_root, config)
    manifest_path = run_root / "control" / "V0_MANIFEST.json"
    manifest = build_v0_manifest(
        REGISTRY_PATH, _config_path(), "1" * 40, "2" * 64, "3" * 64,
        artifact_root, manifest_path,
    )
    return config, run_root, artifact_root, manifest_path, manifest


def _transport_masks():
    basis = np.asarray([
        np.uint64(1) << np.uint64(bit) for bit in range(64)
    ], dtype=np.uint64)
    high = np.uint64(1) << np.uint64(63)
    nonbasis = [high | (np.uint64(1) << np.uint64(bit)) for bit in range(63)]
    nonbasis.append((np.uint64(1) << np.uint64(61)) | (
        np.uint64(1) << np.uint64(62)
    ))
    return np.asarray([*basis, *nonbasis], dtype=np.uint64)


def _full_rank_transport_raw():
    labels = []
    for bit in range(64):
        labels.extend((np.uint64(1) << np.uint64(bit), np.uint64(0)))
    steps = len(labels)
    return {
        "measurement_labels": np.asarray(labels, dtype=np.uint64),
        "measurement_accepted": np.ones(steps, dtype=np.uint8),
        "measurement_state_changed": np.ones(steps, dtype=np.uint8),
        "measurement_label_changed": np.ones(steps, dtype=np.uint8),
        "measurement_proposal_anchor_index": (
            np.arange(steps, dtype=np.int16) % np.int16(16)
        ),
        "burn_label": np.uint64(0),
        "burn_cross_label_changes": np.int64(0),
        "initial_state_packed": np.zeros(1, dtype=np.uint8),
        "measurement_weights": np.zeros(steps, dtype=np.int32),
    }


def _transport_row(trace):
    return {"wall_seconds": 1.0, "core_seconds": 1.0, **trace}


def test_v0_config_is_narrow_diagnostic_contract():
    config = _load_config()
    assert config["contract_version"] == V0_CONTRACT_VERSION
    assert config["cell"]["code_id"] == "m08_c06"
    assert config["cell"]["p"] == pytest.approx(0.04)
    assert config["artifact_authority"] == {
        "mode": "single_producer_algebraic_audit.v1",
        "producer_node": "nd-1",
    }
    assert config["scope"]["formal_authorization"] is False
    assert config["scope"]["production_authorization"] is False
    assert [item["alpha_temperature"] for item in config["candidates"]] == [0.5, 1.0]
    assert config["gates"]["minimum_measurement_label_delta_rank_per_family"] == 64
    assert config["gates"]["minimum_measurement_accepted_cross_label_changes_per_family"] == 128


def test_v0_config_rejects_scope_or_authority_edits(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    original = json.loads(open(_config_path(), encoding="ascii").read())
    original["scope"]["formal_authorization"] = True
    path = tmp_path / "bad-scope.json"
    path.write_text(canonical_json(original) + "\n", encoding="ascii")
    with pytest.raises(LogicalStratifiedV0ConflictError, match="schedule/gates/scope"):
        load_v0_config(path, registry=registry)

    original = json.loads(open(_config_path(), encoding="ascii").read())
    original["artifact_authority"]["producer_node"] = "nd-2"
    path = tmp_path / "bad-authority.json"
    path.write_text(canonical_json(original) + "\n", encoding="ascii")
    with pytest.raises(LogicalStratifiedV0ConflictError, match="artifact authority"):
        load_v0_config(path, registry=registry)


def test_nonproducer_cannot_construct_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(v0, "_hostname_short", lambda: "nd-2")
    artifact_root = tmp_path / "artifacts"
    with pytest.raises(LogicalStratifiedV0ConflictError, match="producer node"):
        prepare_v0_artifacts(
            REGISTRY_PATH, _config_path(), "1" * 40, "2" * 64, "3" * 64,
            artifact_root,
        )
    assert not artifact_root.exists()


def test_portable_digest_ignores_floats_but_not_discrete_trace():
    first = {
        "labels": np.asarray([1, 2], dtype=np.uint64),
        "proposal_log_q": np.asarray([1.0, 2.0]),
        "core_seconds": 1.0,
        "wall_seconds": 2.0,
    }
    changed_float = {
        **first,
        "proposal_log_q": np.asarray([1.0000000000001, 1.9999999999999]),
        "core_seconds": 9.0,
        "wall_seconds": 20.0,
    }
    changed_discrete = {**first, "labels": np.asarray([1, 3], dtype=np.uint64)}
    assert _portable_raw_sha(first) == _portable_raw_sha(changed_float)
    assert _portable_raw_sha(first) != _portable_raw_sha(changed_discrete)
    assert _portable_float_replay_matches(
        np.nextafter(np.ones(4), np.full(4, np.inf)), np.ones(4),
    )
    assert not _portable_float_replay_matches(np.ones(4) + 1e-8, np.ones(4))


def test_manifest_is_portable_and_balances_every_initialization_family(
        tmp_path, monkeypatch):
    config, run_root, artifact_root, manifest_path, manifest = _build_stub_manifest(
        tmp_path, monkeypatch,
    )
    assert manifest["artifact_root_relpath"] == V0_ARTIFACT_ROOT_RELPATH
    assert manifest["artifact_authority"] == config["artifact_authority"]
    assert str(run_root) not in manifest_path.read_text(encoding="ascii")
    assert _manifest_artifact_root(manifest_path, manifest) == artifact_root.resolve()
    assert validate_v0_manifest(
        load_v0_manifest(manifest_path), REGISTRY_PATH, _config_path(), artifact_root,
    )
    for candidate in config["candidates"]:
        for family in config["init_families"]:
            tasks = [
                task for task in manifest["tasks"]
                if (task["method_id"] == candidate["method_id"]
                    and task["init_family"] == family)
            ]
            assert len(tasks) == 8
            assert {node: sum(task["node"] == node for task in tasks)
                    for node in config["execution"]["nodes"]} == {"nd-2": 4, "nd-3": 4}

    local_copy = tmp_path / "local_copy"
    shutil.copytree(run_root, local_copy)
    copied_manifest_path = local_copy / "control" / "V0_MANIFEST.json"
    copied_manifest = load_v0_manifest(copied_manifest_path)
    copied_artifact_root = local_copy / "artifacts"
    assert _manifest_artifact_root(copied_manifest_path, copied_manifest) == copied_artifact_root.resolve()
    assert validate_v0_manifest(
        copied_manifest, REGISTRY_PATH, _config_path(), copied_artifact_root,
    )


def test_manifest_rejects_tampered_artifact_authority(tmp_path, monkeypatch):
    config = _load_config()
    monkeypatch.setattr(v0, "_hostname_short", lambda: "nd-1")
    artifact_root = tmp_path / "artifacts"
    _write_stub_artifact_manifest(
        artifact_root, config,
        authority={
            "mode": "single_producer_algebraic_audit.v1",
            "producer_node": "nd-2",
        },
    )
    with pytest.raises(LogicalStratifiedV0ConflictError, match="artifact manifest SHA"):
        build_v0_manifest(
            REGISTRY_PATH, _config_path(), "1" * 40, "2" * 64, "3" * 64,
            artifact_root, tmp_path / "control" / "V0_MANIFEST.json",
        )


def test_artifact_audit_never_regenerates_the_decoder_catalog(tmp_path, monkeypatch):
    config, _, _, manifest_path, _ = _build_stub_manifest(tmp_path, monkeypatch)
    tail_sha256 = hashlib.sha256(
        np.asarray(range(100, 108), dtype=">i4").tobytes(),
    ).hexdigest()
    _, _, _, model, _, _, _, syndrome = v0._context(REGISTRY_PATH, config)
    matrix_syndrome_sha256 = v0._matrix_syndrome_sha256(model, syndrome)

    def fake_artifact_for_task(_artifact_root, task, _model, _frame):
        identity = {
            "archive_sha256": task["archive_sha256"],
            "artifact_authority": config["artifact_authority"],
            "cell_fingerprint": v0._cell_fingerprint(config["cell"]),
            "config_sha256": task["config_sha256"],
            "method_id": task["method_id"],
            "registry_sha256": task["registry_sha256"],
            "source_commit": task["source_commit"],
            "source_manifest_sha256": task["source_manifest_sha256"],
            "tail_indices_sha256": tail_sha256,
        }
        return SimpleNamespace(
            descriptor={
                "identity": identity,
                "matrix_syndrome_sha256": matrix_syndrome_sha256,
            },
            codebook=SimpleNamespace(codebook_sha256="c" * 64),
            catalog=SimpleNamespace(catalog_sha256="c" * 64),
            proposal=SimpleNamespace(proposal_sha256="d" * 64),
            transcript=SimpleNamespace(
                transcript_sha256="e" * 64,
                decoder_identity="frozen-test-decoder",
            ),
        )

    monkeypatch.setattr(v0, "_artifact_for_task", fake_artifact_for_task)
    monkeypatch.setattr(
        v0, "_tail_indices",
        lambda _codebook, _transcript, _catalog, _count: np.asarray(
            range(100, 108), dtype=np.int32,
        ),
    )

    def decoder_must_not_run(*args, **kwargs):
        raise AssertionError("artifact audit must not rerun BpLSD/MILP")

    monkeypatch.setattr(v0, "generate_bplsd_stratified_catalog", decoder_must_not_run)
    audit = audit_v0_artifacts(REGISTRY_PATH, _config_path(), manifest_path)
    assert audit["artifact_authority"] == config["artifact_authority"]
    assert len(audit["artifact_rows"]) == 2
    assert len(audit["artifact_audit_sha256"]) == 64


def test_k64_measurement_transport_requires_full_rank_and_all_character_returns():
    config = _load_config()
    masks = _transport_masks()
    assert masks[63] == np.uint64(1) << np.uint64(63)
    trace = _chain_transport(_full_rank_transport_raw(), 64, masks, num_qubits=8)
    assert trace["measurement_cross_label_changes"] == 128
    assert trace["measurement_accepted_label_delta_rank"] == 64
    assert trace["basis_character_changes"][63] == 2
    assert all(trace["measurement_character_leave_return"])
    passing = _transport_group_summary(
        [_transport_row(trace) for _ in range(8)], k=64, num_nonbasis=64,
        gates=config["gates"], trajectories_per_family=8,
    )
    assert passing["measurement_accepted_label_delta_rank"] == 64
    assert passing["basis_characters_with_measurement_leave_return"] == 64
    assert passing["nonbasis_characters_with_measurement_leave_return"] == 64
    assert passing["passes_transport_gate"]

    low_rank_raw = _full_rank_transport_raw()
    high = np.uint64(1) << np.uint64(63)
    low_rank_raw["measurement_labels"] = np.asarray(
        [high if index % 2 == 0 else np.uint64(0) for index in range(128)],
        dtype=np.uint64,
    )
    low_rank = _chain_transport(low_rank_raw, 64, masks, num_qubits=8)
    failed = _transport_group_summary(
        [_transport_row(low_rank) for _ in range(8)], k=64, num_nonbasis=64,
        gates=config["gates"], trajectories_per_family=8,
    )
    assert low_rank["measurement_accepted_label_delta_rank"] == 1
    assert not failed["passes_transport_gate"]


def test_v0_wrapper_artifacts_handles_empty_prerequisites_under_nounset(tmp_path):
    """The root stage must reach its controlled failure marker, not expand ()."""
    wrapper = (
        EXP102_ROOT
        / "validation/015_q0_logical_stratified_v0b_20260723/run_v0_stage.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "[[ ${1:-} == -m && ${2:-} == "
        "data.expander_code.exp102.exp102_pipeline.q0_logical_stratified_v0 "
        "&& ${3:-} == prepare-artifacts ]] || exit 91\n"
        "exit 23\n",
        encoding="ascii",
    )
    fake_python.chmod(0o755)
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="ascii")
    fake_flock.chmod(0o755)
    environment = os.environ.copy()
    environment["EXP102_SOURCE_COMMIT"] = "1" * 40
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    stage_dir = tmp_path / "artifacts-stage"
    log_file = tmp_path / "artifacts.log"

    completed = subprocess.run(
        [
            "bash", str(wrapper), "artifacts", str(stage_dir), str(log_file),
            "--", "python", "-m", v0.__name__, "prepare-artifacts",
        ],
        check=False, capture_output=True, text=True, env=environment,
    )

    assert completed.returncode == 23, completed.stderr
    failed = json.loads((stage_dir / "FAILED").read_text(encoding="ascii"))
    assert failed["exit_code"] == 23
    assert failed["stage"] == "artifacts"
    assert not (stage_dir / "RUNNING").exists()
    assert "unbound variable" not in completed.stderr
