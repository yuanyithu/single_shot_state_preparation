import hashlib
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.expander_code.exp103.exp103_pipeline import identity, io, remote_cli
from data.expander_code.exp103.deployment.build_remote_deployment import (
    FROZEN_EXECUTION_PATHS, SOURCE_PATHS,
)
from data.expander_code.exp103.exp103_pipeline.config import (
    REMOTE_BPLSD_BINARY_SHA256, REMOTE_CONDA_PREFIX, REMOTE_LDPC_SOURCE,
    REMOTE_SUPPORT_PACKAGES, ensure_config,
)
from data.expander_code.exp103.exp103_pipeline.io import atomic_json, sha256_file
from data.expander_code.exp103.exp103_pipeline.seeds import derive_seed


def remote_config_from(local_config, *, source_tree_sha256="2" * 64):
    config = {
        key: value for key, value in local_config.items()
        if key not in {"config_sha256", "config_path"}
    }
    config = json.loads(json.dumps(config))
    config["schema_version"] = "exp103.config.remote.v2"
    config["source_commit"] = "1" * 40
    config["source_tree_sha256"] = source_tree_sha256
    remote_prefix = REMOTE_CONDA_PREFIX
    config["environment"] = {
        "device_name": "nd-3",
        "hostname": "nd-3",
        "conda_environment": remote_prefix,
        "conda_prefix_matches_python": True,
        "python": "3.12.12",
        "numpy": "2.4.1",
        "scipy": "1.17.0",
        "ldpc": "2.4.1",
    }
    config["bplsd_binary"] = {
        "module": "ldpc.bplsd_decoder._bplsd_decoder",
        "filename_suffix": ".cpython-312-x86_64-linux-gnu.so",
        "sha256": REMOTE_BPLSD_BINARY_SHA256,
    }
    config["execution_profile"] = {
        "profile_id": "exp103.remote_execution.v2",
        "entry_host": "yuany",
        "compute_host": "nd-3",
        "conda_environment": remote_prefix,
        "num_workers": 64,
        "omp_thread_count": 1,
        "run_root": "~/.single_shot/runs",
        "log_root": "~/.single_shot/logs",
        "reserve_multiplier": 2.0,
        "stage_core_hour_cap": 10000.0,
        "stage_wall_hour_cap": 96.0,
        "peak_rss_gib_cap": 128.0,
    }
    config["ldpc_source"] = dict(REMOTE_LDPC_SOURCE)
    config["support_packages"] = dict(REMOTE_SUPPORT_PACKAGES)
    return ensure_config(config)


@pytest.fixture
def remote_config(frozen_config):
    return remote_config_from(frozen_config)


def _plain_config(config):
    return json.loads(json.dumps({
        key: value for key, value in config.items()
        if key not in {"config_sha256", "config_path"}
    }))


def _canonical_local_config():
    path = Path("data/expander_code/exp103/config/decoder_mc.v1.json")
    return ensure_config(json.loads(path.read_text(encoding="ascii")))


def test_remote_schema_preserves_every_measurement_seed(remote_config, frozen_config):
    local_config = _canonical_local_config()
    assert remote_config["experiment_id"] == local_config["experiment_id"]
    assert remote_config["master_seed_hex"] == local_config["master_seed_hex"]
    assert remote_config["namespaces"] == local_config["namespaces"]
    for m in local_config["m_values"]:
        for code in range(8):
            code_id = f"m{m:02d}_c{code:02d}"
            for p_token in local_config["p_tokens"]:
                for shard in range(local_config["shards_per_code_p"]):
                    assert derive_seed(
                        remote_config, "measurement", code_id, p_token, shard,
                    ) == derive_seed(
                        local_config, "measurement", code_id, p_token, shard,
                    )


def test_prefix_style_conda_identity_is_verified_without_rewriting_environment(
    monkeypatch, remote_config,
):
    prefix = REMOTE_CONDA_PREFIX
    config = remote_config
    monkeypatch.setattr(identity.sys, "prefix", prefix)
    monkeypatch.setattr(identity.socket, "gethostname", lambda: "nd-3")
    monkeypatch.setattr(
        identity, "source_tree_sha256", lambda: config["source_tree_sha256"],
    )
    monkeypatch.setattr(
        identity, "bplsd_binary_path",
        lambda: Path("backend" + config["bplsd_binary"]["filename_suffix"]),
    )
    monkeypatch.setattr(
        identity, "sha256_file", lambda _path: config["bplsd_binary"]["sha256"],
    )
    actual_version = identity.importlib.metadata.version
    monkeypatch.setattr(
        identity.importlib.metadata, "version",
        lambda name: config["support_packages"].get(name, actual_version(name)),
    )
    monkeypatch.setenv("CONDA_DEFAULT_ENV", prefix)
    monkeypatch.setenv("CONDA_PREFIX", prefix)

    actual = identity.runtime_identity(config)

    assert actual["conda_environment"] == prefix
    assert actual["conda_prefix_matches_python"] is True


def test_selective_deployment_contains_every_qualification_dependency():
    required = {
        "src/build_toric_code_examples.py",
        "data/expander_code/exp102/config/production.v1.json",
        "data/expander_code/exp102/validation/002_numba_smoke_20260719/orchestrate_ladder.py",
        "data/expander_code/exp102/validation/002_numba_smoke_20260719/run_stage_wrapper.sh",
        "data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh",
        "data/expander_code/exp103/deployment/bootstrap_verified_archive.sh",
    }
    assert required <= set(SOURCE_PATHS)
    assert required - {"data/expander_code/exp103/deployment/bootstrap_verified_archive.sh"} <= set(
        FROZEN_EXECUTION_PATHS
    )

    bootstrap = Path(
        "data/expander_code/exp103/deployment/bootstrap_verified_archive.sh"
    ).read_text(encoding="ascii")
    wrapper = Path(
        "data/expander_code/exp103/deployment/run_verified_source.sh"
    ).read_text(encoding="ascii")
    assert bootstrap.index("sha256sum -c") < bootstrap.index("tar -xOf")
    assert "NUMBA_CACHE_DIR" in wrapper and "NUMBA_NUM_THREADS=1" in wrapper
    assert 'deployment.get("archive_sha256") != archive_sha' in wrapper
    assert 'deployment.get("source_manifest_sha256") != source_sha' in wrapper
    assert 'source.get("source_commit") != source_commit' in wrapper


def test_remote_publication_runs_all_gates_and_is_atomic(
    monkeypatch, tmp_path, remote_config,
):
    run_root = tmp_path / "run"
    aggregate = run_root / "final_results/decoder_crossing.npz"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"aggregate")
    calls = []
    monkeypatch.setattr(remote_cli, "load_config", lambda _path: remote_config)
    monkeypatch.setattr(
        remote_cli, "resolve_remote_run_root", lambda path, _config: Path(path),
    )
    monkeypatch.setattr(
        remote_cli, "_verify_remote_runtime",
        lambda *_args: calls.append("runtime") or ({}, {}),
    )
    monkeypatch.setattr(
        remote_cli, "_require_preflight",
        lambda _path, _root, _config, stage, _deployment: calls.append(stage),
    )
    monkeypatch.setattr(
        remote_cli, "_require_stage1_technical",
        lambda *_args: calls.append("technical"),
    )
    monkeypatch.setattr(
        remote_cli, "_require_live_aggregate_matches",
        lambda *_args: calls.append("live_aggregate"),
    )

    def report(_aggregate, output):
        calls.append("loader_report")
        Path(output, "report.json").write_text("{}", encoding="ascii")
        return {
            "terminal_status": "EXP103_DECODER_CROSSING_INCONCLUSIVE",
            "num_code_p": 624,
            "total_trials": 6_240_000,
        }

    monkeypatch.setattr(remote_cli, "generate_final_report", report)
    result = remote_cli.run_remote_publication(
        "config.json", run_root, "preflight.json", "deployment", "a" * 64,
    )

    assert result["total_trials"] == 6_240_000
    assert calls == [
        "runtime", "stage1", "stage2", "technical", "live_aggregate",
        "loader_report",
    ]
    assert (run_root / "final_results/publication/report.json").is_file()
    assert not list((run_root / "final_results").glob(".publication.partial-*"))
    with pytest.raises(FileExistsError, match="immutable"):
        remote_cli.run_remote_publication(
            "config.json", run_root, "preflight.json", "deployment", "a" * 64,
        )


def test_stage1_preliminary_is_published_only_after_committed_technical_gate(
    monkeypatch, tmp_path, remote_config,
):
    run_root = tmp_path / "run"
    aggregate = run_root / "final_results/stage1_aggregate.npz"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"stage1")
    calls = []
    monkeypatch.setattr(remote_cli, "load_config", lambda _path: remote_config)
    monkeypatch.setattr(
        remote_cli, "resolve_remote_run_root", lambda path, _config: Path(path),
    )
    monkeypatch.setattr(
        remote_cli, "_verify_remote_runtime",
        lambda *_args: calls.append("runtime") or ({}, {}),
    )
    monkeypatch.setattr(
        remote_cli, "_require_preflight",
        lambda *_args: calls.append("stage1_preflight"),
    )
    monkeypatch.setattr(
        remote_cli, "_require_stage1_technical",
        lambda *_args: calls.append("committed_technical"),
    )

    def report(_aggregate, output):
        calls.append("preliminary_report")
        Path(output, "report.json").write_text("{}", encoding="ascii")
        return {
            "stage1_status": "STAGE1_RESTRICTED_DECODER_CROSSING_INCONCLUSIVE",
            "reportable_code_p": 312,
            "total_trials": 3_120_000,
            "stage2_decision_uses_curves": False,
        }

    monkeypatch.setattr(remote_cli, "generate_stage1_preliminary_report", report)
    result = remote_cli.run_remote_stage1_preliminary(
        "config.json", run_root, "preflight.json", "deployment", "a" * 64,
    )

    assert result["stage2_decision_uses_curves"] is False
    assert calls == [
        "runtime", "stage1_preflight", "committed_technical", "preliminary_report",
    ]
    assert (run_root / "final_results/stage1_preliminary/report.json").is_file()


def test_publication_rejects_a_saved_aggregate_stale_against_live_raw(
    monkeypatch, tmp_path,
):
    stored = {"array": np.asarray([1.0]), "state": "same"}
    live = {"array": np.asarray([2.0]), "state": "same"}
    monkeypatch.setattr(remote_cli, "ARRAY_FIELDS", ("array",))
    monkeypatch.setattr(remote_cli, "SCALAR_FIELDS", ("state",))
    monkeypatch.setattr(remote_cli, "_load_aggregate", lambda _path: stored)
    monkeypatch.setattr(remote_cli, "aggregate_decoder_scan", lambda *_args: live)
    monkeypatch.setattr(
        remote_cli, "_require_scope_reportable", lambda aggregate, _scope: aggregate,
    )

    with pytest.raises(ValueError, match="stale for live field array"):
        remote_cli._require_live_aggregate_matches(
            tmp_path / "aggregate.npz", tmp_path / "raw", {}, "final",
        )

    stored = {"array": np.asarray([1.0]), "state": float("nan")}
    live = {"array": np.asarray([1.0]), "state": float("nan")}
    monkeypatch.setattr(remote_cli, "_load_aggregate", lambda _path: stored)
    monkeypatch.setattr(remote_cli, "aggregate_decoder_scan", lambda *_args: live)
    assert remote_cli._require_live_aggregate_matches(
        tmp_path / "aggregate.npz", tmp_path / "raw", {}, "stage1",
    ) is stored


def test_invalid_remote_aggregate_is_saved_as_evidence_but_exits_nonzero(
    monkeypatch, tmp_path, remote_config,
):
    result = {"replay_scope": "final_combined", "terminal_status": "EXP103_INVALID"}
    calls = []
    monkeypatch.setattr(remote_cli, "load_config", lambda _path: remote_config)
    monkeypatch.setattr(
        remote_cli, "resolve_remote_run_root", lambda path, _config: Path(path),
    )
    monkeypatch.setattr(remote_cli, "_verify_remote_runtime", lambda *_args: ({}, {}))
    monkeypatch.setattr(remote_cli, "_require_preflight", lambda *_args: None)
    monkeypatch.setattr(remote_cli, "_require_stage1_technical", lambda *_args: None)
    monkeypatch.setattr(remote_cli, "aggregate_decoder_scan", lambda *_args: result)
    monkeypatch.setattr(
        remote_cli, "save_aggregate",
        lambda path, aggregate: calls.append((Path(path), aggregate)),
    )

    def reject(aggregate, scope):
        assert calls and aggregate is result and scope == "final"
        raise ValueError("not formally reportable")

    monkeypatch.setattr(remote_cli, "_require_scope_reportable", reject)
    with pytest.raises(ValueError, match="not formally reportable"):
        remote_cli.run_remote_aggregate(
            "config.json", "final", tmp_path / "run", "preflight.json",
            tmp_path / "deployment", "a" * 64,
        )
    assert calls[0][0].name == "decoder_crossing.npz"


def test_remote_stage_driver_always_revalidates_completed_artifacts():
    driver = Path(
        "data/expander_code/exp103/deployment/run_remote_stage.sh"
    ).read_text(encoding="ascii")
    assert driver.count("verify-stage") == 2
    assert "--stage stage1" in driver
    assert "--stage stage2" in driver


def test_completed_report_manifest_rejects_tampered_output(tmp_path):
    report_dir = tmp_path / "publication"
    report_dir.mkdir()
    artifact = report_dir / "curve.csv"
    artifact.write_bytes(b"frozen curve\n")
    report = {
        "schema_version": "test.report.v1",
        "files": ["curve.csv"],
        "file_sha256": {"curve.csv": sha256_file(artifact)},
    }
    atomic_json(report_dir / "report.json", report)

    def build_reference(_result_path, output_dir):
        output_dir = Path(output_dir)
        reference_artifact = output_dir / "curve.csv"
        reference_artifact.write_bytes(b"frozen curve\n")
        atomic_json(output_dir / "report.json", {
            "schema_version": "test.report.v1",
            "files": ["curve.csv"],
            "file_sha256": {"curve.csv": sha256_file(reference_artifact)},
        })

    assert remote_cli._require_hashed_report(
        report_dir, ("curve.csv", "report.json"),
        {"schema_version": "test.report.v1"},
        tmp_path / "aggregate.npz", build_reference,
    ) == report

    artifact.write_bytes(b"tampered curve\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        remote_cli._require_hashed_report(
            report_dir, ("curve.csv", "report.json"),
            {"schema_version": "test.report.v1"},
        )

    report["file_sha256"]["curve.csv"] = sha256_file(artifact)
    atomic_json(report_dir / "report.json", report)
    with pytest.raises(ValueError, match="differs from the live aggregate"):
        remote_cli._require_hashed_report(
            report_dir, ("curve.csv", "report.json"),
            {"schema_version": "test.report.v1"},
            tmp_path / "aggregate.npz", build_reference,
        )


def test_remote_config_is_strictly_separate_from_local_v1(remote_config, frozen_config):
    local_with_profile = _plain_config(_canonical_local_config())
    local_with_profile["execution_profile"] = remote_config["execution_profile"]
    with pytest.raises(ValueError, match="unexpected exp103 config fields"):
        ensure_config(local_with_profile)

    remote_without_profile = _plain_config(remote_config)
    remote_without_profile.pop("execution_profile")
    with pytest.raises(ValueError, match="unexpected exp103 config fields"):
        ensure_config(remote_without_profile)

    placeholder = _plain_config(remote_config)
    placeholder["bplsd_binary"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not fully frozen"):
        ensure_config(placeholder)

    wrong_binary = _plain_config(remote_config)
    wrong_binary["bplsd_binary"]["sha256"] = "3" * 64
    with pytest.raises(ValueError, match="not fully frozen"):
        ensure_config(wrong_binary)

    wrong_prefix = _plain_config(remote_config)
    wrong_prefix["environment"]["conda_environment"] = "/tmp/alternate-prefix"
    wrong_prefix["execution_profile"]["conda_environment"] = "/tmp/alternate-prefix"
    with pytest.raises(ValueError, match="conda environment"):
        ensure_config(wrong_prefix)

    wrong_ldpc_source = _plain_config(remote_config)
    wrong_ldpc_source["ldpc_source"]["commit"] = "f" * 40
    with pytest.raises(ValueError, match="source provenance"):
        ensure_config(wrong_ldpc_source)

    wrong_support = _plain_config(remote_config)
    wrong_support["support_packages"]["numba"] = "0.65.1"
    with pytest.raises(ValueError, match="support package"):
        ensure_config(wrong_support)


def test_external_run_root_is_one_safe_direct_child(monkeypatch, tmp_path, remote_config):
    monkeypatch.setenv("HOME", str(tmp_path))
    base = tmp_path / ".single_shot" / "runs"
    base.mkdir(parents=True)
    run_root = base / "exp103_remote_001"
    assert remote_cli.resolve_remote_run_root(run_root, remote_config) == run_root

    with pytest.raises(ValueError, match="absolute"):
        remote_cli.resolve_remote_run_root("relative-run", remote_config)
    with pytest.raises(ValueError, match="outside"):
        remote_cli.resolve_remote_run_root(tmp_path / "elsewhere" / "run", remote_config)
    with pytest.raises(ValueError, match="outside"):
        remote_cli.resolve_remote_run_root(run_root / "nested", remote_config)

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = base / "linked"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside|symlink"):
        remote_cli.resolve_remote_run_root(symlink, remote_config)

    run_root.mkdir()
    internal = run_root / "raw"
    internal.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="contain symlinks"):
        remote_cli.resolve_remote_run_root(run_root, remote_config)


def _build_deployment(tmp_path, frozen_config):
    deployment = tmp_path / "deployment"
    source = deployment / "source"
    package = source / "data/expander_code/exp103/exp103_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="ascii")
    tree_sha = identity.source_tree_sha256(package)
    config = remote_config_from(frozen_config, source_tree_sha256=tree_sha)
    config_path = source / "data/expander_code/exp103/config/decoder_mc.remote.v2.json"
    atomic_json(config_path, _plain_config(config))

    archive = deployment / "SOURCE.tar"
    deployment.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w") as handle:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            handle.add(path, arcname=path.relative_to(source).as_posix())
    archive_sha = sha256_file(archive)
    source_commit = "a" * 40
    (deployment / "SOURCE_COMMIT").write_text(source_commit + "\n", encoding="ascii")
    (deployment / "ARCHIVE_SHA256").write_text(archive_sha + "\n", encoding="ascii")
    files = [
        {"path": path.relative_to(source).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(item for item in source.rglob("*") if item.is_file())
    ]
    source_manifest = {
        "source_identity_version": "exp102.source.v1",
        "source_commit": source_commit,
        "archive_sha256": archive_sha,
        "files": files,
    }
    atomic_json(deployment / "SOURCE_MANIFEST.json", source_manifest)
    deployment_manifest = {
        "schema_version": "exp103.remote_deployment.v1",
        "experiment_id": config["experiment_id"],
        "execution_profile_id": config["execution_profile"]["profile_id"],
        "source_commit": source_commit,
        "frozen_source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "archive_sha256": archive_sha,
        "source_manifest_sha256": sha256_file(deployment / "SOURCE_MANIFEST.json"),
    }
    atomic_json(deployment / "DEPLOYMENT_MANIFEST.json", deployment_manifest)
    return deployment, source, config


def test_archive_and_deployment_manifest_tamper_is_rejected(tmp_path, frozen_config):
    deployment, source, config = _build_deployment(tmp_path, frozen_config)
    manifest = deployment / "DEPLOYMENT_MANIFEST.json"
    manifest_sha = sha256_file(manifest)
    verified = identity.verify_remote_deployment(
        config, deployment, manifest_sha, _source_root=source,
    )
    assert verified["deployment_manifest_sha256"] == manifest_sha
    assert verified["config_sha256"] == config["config_sha256"]

    payload = json.loads(manifest.read_text(encoding="ascii"))
    payload["registry_sha256"] = "f" * 64
    atomic_json(manifest, payload)
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        identity.verify_remote_deployment(
            config, deployment, manifest_sha, _source_root=source,
        )


def test_old_blocked_local_preflight_cannot_authorize_remote(
    monkeypatch, tmp_path, remote_config,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    run_root = tmp_path / ".single_shot/runs/exp103_remote_001"
    path = run_root / remote_cli.REMOTE_PREFLIGHT_RELATIVE
    path.parent.mkdir(parents=True)
    local_report = json.loads(Path(
        "data/expander_code/exp103/validation/002_local_resource_preflight_20260804/"
        "resource_preflight.json"
    ).read_text(encoding="ascii"))
    atomic_json(path, local_report)
    with pytest.raises(ValueError, match="remote resource preflight schema mismatch"):
        remote_cli._require_preflight(
            path, run_root, remote_config, "stage1", tmp_path / "deployment",
        )


def _remote_report_from_local_timings(remote_config):
    local = json.loads(Path(
        "data/expander_code/exp103/validation/002_local_resource_preflight_20260804/"
        "resource_preflight.json"
    ).read_text(encoding="ascii"))
    tasks = local["tasks"]
    stages = remote_cli._remote_stage_estimates(tasks, remote_config)
    return {
        "schema_version": remote_cli.REMOTE_PREFLIGHT_SCHEMA,
        "status": "PASS_ALL_STAGES",
        "experiment_id": remote_config["experiment_id"],
        "execution_profile_id": remote_config["execution_profile"]["profile_id"],
        "config_sha256": remote_config["config_sha256"],
        "registry_sha256": remote_config["registry_sha256"],
        "source_commit": remote_config["source_commit"],
        "source_tree_sha256": remote_config["source_tree_sha256"],
        "bplsd_binary_sha256": remote_config["bplsd_binary"]["sha256"],
        "device_name": "nd-3",
        "hostname": "nd-3",
        "conda_environment": remote_config["environment"]["conda_environment"],
        "conda_prefix_matches_python": True,
        "num_workers": 64,
        "omp_thread_count": 1,
        "outcome_blind": True,
        "logical_outcomes_saved": False,
        "qualification_report_sha256": "7" * 64,
        "deployment": {
            "schema_version": "exp103.remote_deployment.v1",
            "deployment_manifest_sha256": "4" * 64,
            "source_commit": "a" * 40,
            "archive_sha256": "5" * 64,
            "source_manifest_sha256": "6" * 64,
            "source_tree_sha256": remote_config["source_tree_sha256"],
            "config_sha256": remote_config["config_sha256"],
        },
        "tasks": tasks,
        "stages": stages,
    }


def _qualification_report(remote_config):
    deployment = {
        "schema_version": "exp103.remote_deployment.v1",
        "deployment_manifest_sha256": "4" * 64,
        "source_commit": "a" * 40,
        "archive_sha256": "5" * 64,
        "source_manifest_sha256": "6" * 64,
        "source_tree_sha256": remote_config["source_tree_sha256"],
        "config_sha256": remote_config["config_sha256"],
    }
    groups = [
        {
            "name": name,
            "argv": list(argv),
            "exit_code": 0,
            "status": "PASS",
            "passed_count": remote_cli.QUALIFICATION_EXPECTED_PASSES[name],
            "expected_passed_count": remote_cli.QUALIFICATION_EXPECTED_PASSES[name],
            "skipped_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
            "deselected_count": 0,
            "stdout_sha256": "8" * 64,
            "stderr_sha256": "9" * 64,
        }
        for index, (name, argv) in enumerate(remote_cli._qualification_argv())
    ]
    return {
        "schema_version": remote_cli.REMOTE_QUALIFICATION_SCHEMA,
        "status": "PASS",
        "experiment_id": remote_config["experiment_id"],
        "execution_profile_id": remote_config["execution_profile"]["profile_id"],
        "config_sha256": remote_config["config_sha256"],
        "registry_sha256": remote_config["registry_sha256"],
        "source_commit": remote_config["source_commit"],
        "source_tree_sha256": remote_config["source_tree_sha256"],
        "bplsd_binary_sha256": remote_config["bplsd_binary"]["sha256"],
        "device_name": "nd-3",
        "hostname": "nd-3",
        "conda_environment": remote_config["environment"]["conda_environment"],
        "conda_prefix": str(Path(sys.prefix).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": remote_config["environment"]["python"],
        "numpy_version": remote_config["environment"]["numpy"],
        "scipy_version": remote_config["environment"]["scipy"],
        "ldpc_version": remote_config["environment"]["ldpc"],
        "support_packages": remote_config["support_packages"],
        "bplsd_binary_path": str(remote_cli.bplsd_binary_path()),
        "bplsd_binary_filename": remote_cli.bplsd_binary_path().name,
        "python_no_bytecode_flag": True,
        "pythondontwritebytecode": True,
        "pytest_cache_disabled": True,
        "bytecode_clean_before": True,
        "bytecode_clean_after": True,
        "deployment": deployment,
        "ldpc_source_provenance": {
            **remote_config["ldpc_source"], "project_version": "2.4.1",
        },
        "host_resources": {
            "platform_system": "Linux",
            "platform_release": "6.8.0-test",
            "platform_machine": "x86_64",
            "libc_name": "glibc",
            "libc_version": "2.39",
            "logical_cpu_count": 96,
            "physical_core_count": 48,
            "cpu_socket_count": 4,
            "memory_total_bytes": 256 * 1024 ** 3,
            "memory_available_bytes": 128 * 1024 ** 3,
            "run_disk_path": str(Path(
                remote_config["execution_profile"]["run_root"]
            ).expanduser().resolve()),
            "run_disk_total_bytes": 1024 ** 5,
            "run_disk_free_bytes": 1024 ** 4,
        },
        "groups": groups,
        "total_passed": sum(group["passed_count"] for group in groups),
    }


def test_remote_preflight_revalidates_task_namespace_fields_and_timing(remote_config):
    report = _remote_report_from_local_timings(remote_config)
    assert remote_cli.validate_remote_resource_preflight(
        report, remote_config,
    )["status"] == "PASS_ALL_STAGES"

    tampered = json.loads(json.dumps(report))
    tampered["tasks"][0]["seed_namespace"] = remote_config["namespaces"]["measurement"]
    with pytest.raises(ValueError, match="measurement namespace leaked"):
        remote_cli.validate_remote_resource_preflight(tampered, remote_config)

    tampered = json.loads(json.dumps(report))
    tampered["tasks"][0]["measurement_seconds_per_trial"] *= 2
    with pytest.raises(ValueError, match="measurement timing arithmetic"):
        remote_cli.validate_remote_resource_preflight(tampered, remote_config)


def test_qualification_runs_three_frozen_python_no_bytecode_groups(
    monkeypatch, tmp_path, remote_config,
):
    calls = []

    def completed(argv, cwd, env, capture_output):
        calls.append((argv, cwd, env, capture_output))
        name = ("exp103", "exp101", "exp102")[len(calls) - 1]
        count = remote_cli.QUALIFICATION_EXPECTED_PASSES[name]
        return SimpleNamespace(
            returncode=0,
            stdout=f"{count} passed in 0.01s\n".encode("ascii"),
            stderr=b"",
        )

    fake_binary = (
        Path(sys.prefix).resolve()
        / ("backend" + remote_config["bplsd_binary"]["filename_suffix"])
    )
    monkeypatch.setattr(remote_cli.subprocess, "run", completed)
    monkeypatch.setattr(
        remote_cli, "runtime_identity",
        lambda _config: {
            "python_version": remote_config["environment"]["python"],
            "numpy_version": remote_config["environment"]["numpy"],
            "scipy_version": remote_config["environment"]["scipy"],
            "ldpc_version": remote_config["environment"]["ldpc"],
            "support_packages": remote_config["support_packages"],
        },
    )
    monkeypatch.setattr(remote_cli, "bplsd_binary_path", lambda: fake_binary)
    monkeypatch.setattr(
        remote_cli, "_ldpc_source_provenance",
        lambda _config: {**remote_config["ldpc_source"], "project_version": "2.4.1"},
    )
    resources = _qualification_report(remote_config)["host_resources"]
    monkeypatch.setattr(remote_cli, "_host_resources", lambda _config: resources)
    report = remote_cli.run_environment_qualification(
        remote_config, _qualification_report(remote_config)["deployment"], tmp_path,
    )
    assert report["status"] == "PASS"
    assert len(calls) == 3
    assert [group["name"] for group in report["groups"]] == ["exp103", "exp101", "exp102"]
    assert all(call[0][0] == str(Path(sys.executable).resolve()) for call in calls)
    assert all(call[0][1:5] == ("-B", "-m", "pytest", "-q") for call in calls)
    assert all(call[2]["PYTHONDONTWRITEBYTECODE"] == "1" for call in calls)
    assert all(call[2]["PYTEST_ADDOPTS"] == "-p no:cacheprovider" for call in calls)
    assert all(group["skipped_count"] == 0 for group in report["groups"])
    assert remote_cli.validate_environment_qualification(report, remote_config) == report

    report["groups"][1]["exit_code"] = 1
    with pytest.raises(ValueError, match="failed for exp101"):
        remote_cli.validate_environment_qualification(report, remote_config)

    report = _qualification_report(remote_config)
    report["groups"][0]["skipped_count"] = 1
    with pytest.raises(ValueError, match="failed for exp103"):
        remote_cli.validate_environment_qualification(report, remote_config)

def test_formal_preflight_gate_requires_the_copy_in_current_deployment(
    tmp_path, remote_config,
):
    run_root = tmp_path / "run"
    run_qualification = run_root / remote_cli.REMOTE_QUALIFICATION_RELATIVE
    atomic_json(run_qualification, _qualification_report(remote_config))
    run_report = run_root / remote_cli.REMOTE_PREFLIGHT_RELATIVE
    report = _remote_report_from_local_timings(remote_config)
    report["qualification_report_sha256"] = sha256_file(run_qualification)
    atomic_json(run_report, report)
    deployment = tmp_path / "deployment"
    deployed_qualification = (
        deployment / "source" / remote_cli.COMMITTED_QUALIFICATION_RELATIVE
    )
    deployed_qualification.parent.mkdir(parents=True)
    deployed_qualification.write_bytes(run_qualification.read_bytes())
    with pytest.raises(ValueError, match="not present in the deployed pushed source"):
        remote_cli._require_preflight(
            run_report, run_root, remote_config, "stage1", deployment,
        )

    deployed = deployment / "source" / remote_cli.COMMITTED_PREFLIGHT_RELATIVE
    deployed.parent.mkdir(parents=True, exist_ok=True)
    deployed.write_bytes(run_report.read_bytes())
    assert remote_cli._require_preflight(
        run_report, run_root, remote_config, "stage1", deployment,
    )["status"] == "PASS_ALL_STAGES"


def test_formal_evidence_must_match_current_deployed_source_bytes(tmp_path):
    deployment = tmp_path / "deployment"
    source_evidence = deployment / "source" / remote_cli.COMMITTED_PREFLIGHT_RELATIVE
    run_evidence = tmp_path / "run/validation/remote_resource_preflight.json"
    source_evidence.parent.mkdir(parents=True)
    run_evidence.parent.mkdir(parents=True)
    source_evidence.write_bytes(b"frozen evidence\n")
    run_evidence.write_bytes(b"frozen evidence\n")
    digest = remote_cli._require_deployed_evidence_bytes(
        run_evidence, deployment, remote_cli.COMMITTED_PREFLIGHT_RELATIVE,
        "remote resource preflight",
    )
    assert digest == hashlib.sha256(b"frozen evidence\n").hexdigest()

    run_evidence.write_bytes(b"tampered evidence\n")
    with pytest.raises(ValueError, match="differs from the deployed pushed evidence"):
        remote_cli._require_deployed_evidence_bytes(
            run_evidence, deployment, remote_cli.COMMITTED_PREFLIGHT_RELATIVE,
            "remote resource preflight",
        )


def test_atomic_npz_partial_is_never_discoverable_as_raw(
    monkeypatch, tmp_path, remote_config,
):
    observed = {}

    def interrupted_replace(source, target):
        observed["source"] = Path(source)
        observed["target"] = Path(target)
        raise RuntimeError("simulated interruption before atomic replace")

    monkeypatch.setattr(io.os, "replace", interrupted_replace)
    output = tmp_path / "formal_raw.npz"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        io.atomic_npz(output, {"value": np.arange(4)})
    assert observed["source"].suffix == ".partial"
    assert observed["target"] == output
    assert not list(tmp_path.glob("*.npz"))

    stage_root = tmp_path / "stage1"
    stage_root.mkdir()
    orphan = stage_root / ".m03_c00__p0p02__s00.npz.dead.partial"
    orphan.write_bytes(b"interrupted")
    remote_cli._assert_no_unplanned_npz(
        stage_root, remote_config, "stage1", require_complete=False,
    )


def test_code_p_resume_never_redraws_existing_valid_shards(
    monkeypatch, tmp_path, remote_config,
):
    for shard in range(4):
        (tmp_path / remote_cli.raw_filename("m03_c00", "0.02", shard)).write_bytes(b"raw")
    monkeypatch.setattr(remote_cli, "load_config", lambda _path: remote_config)
    monkeypatch.setattr(remote_cli, "_cached_registry", lambda _path: {"m03_c00": {}})
    monkeypatch.setattr(remote_cli, "load_raw", lambda _path: {"status": "VALID"})
    monkeypatch.setattr(remote_cli, "_validate_raw", lambda *_args: None)
    monkeypatch.setattr(
        remote_cli, "run_decoder_shard",
        lambda *_args: pytest.fail("immutable resume redrew an existing shard"),
    )
    code_id, p_token, results = remote_cli._save_code_p_task(
        ("m03_c00", "0.02", "remote-config.json", tmp_path),
    )
    assert (code_id, p_token) == ("m03_c00", "0.02")
    assert [status for _, status in results] == ["RESUMED"] * 4


def test_remote_plan_is_code_p_granular_and_worker_count_is_frozen(
    monkeypatch, tmp_path, remote_config,
):
    assert len(remote_cli._planned_code_p(remote_config, "stage1")) == 312
    assert len(remote_cli._planned_code_p(remote_config, "stage2")) == 312
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(remote_cli, "load_config", lambda _path: remote_config)
    run_root = tmp_path / ".single_shot/runs/exp103_remote_001"
    with pytest.raises(ValueError, match="64"):
        remote_cli.run_remote_scan(
            "remote-config.json", "stage1", run_root,
            run_root / remote_cli.REMOTE_PREFLIGHT_RELATIVE, 8,
            tmp_path / "deployment", "f" * 64,
        )


def test_stage1_technical_report_is_an_exact_immutable_authorization(
    monkeypatch, tmp_path, remote_config,
):
    run_root = tmp_path / "run"
    path = run_root / remote_cli.REMOTE_TECHNICAL_RELATIVE
    path.parent.mkdir(parents=True)
    report = {
        "schema_version": "exp103.remote_stage1_technical.v1",
        "status": "TECHNICAL_PASS",
        "outcome_blind_stage2_authorization": True,
    }
    expected = dict(report)
    atomic_json(path, report)
    deployment = tmp_path / "deployment"
    deployed = deployment / "source" / remote_cli.COMMITTED_TECHNICAL_RELATIVE
    deployed.parent.mkdir(parents=True)
    atomic_json(deployed, report)
    monkeypatch.setattr(
        remote_cli, "build_remote_stage1_technical",
        lambda _root, _config: dict(expected),
    )
    assert remote_cli._require_stage1_technical(
        run_root, remote_config, deployment,
    ) == report
    report["outcome_blind_stage2_authorization"] = False
    atomic_json(path, report)
    with pytest.raises(ValueError, match="stale or tampered"):
        remote_cli._require_stage1_technical(run_root, remote_config, deployment)
