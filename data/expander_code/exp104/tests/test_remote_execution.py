"""Gates that stand between a config and a production run on nd-3."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp104.exp104_pipeline import remote_cli
from data.expander_code.exp104.exp104_pipeline.config import (
    CODES_PER_M,
    CODES_PER_TASK,
    M_VALUES,
    TASKS_PER_M,
)
from data.expander_code.exp104.exp104_pipeline.preflight import estimate_resources
from data.expander_code.exp104.exp104_pipeline.raw import raw_filename
from data.expander_code.exp104.exp104_pipeline.replay import (
    committed_replay_blocks,
    expected_replay_keys,
    validate_replay_report,
)


def _benchmark_tasks(per_trial=0.05, model_seconds=1.0, rss=0.4):
    tasks = []
    for m in M_VALUES:
        tasks.append({
            "m": m,
            "measurement_seconds_per_trial": per_trial * m,
            "replay_seconds_per_trial": per_trial * m,
            "model_seconds": model_seconds * m / 8.0,
            "identity_seconds": 0.001,
            "decoder_setup_seconds": 0.002,
            "replay_setup_seconds": 0.002,
            "raw_serialization_seconds": 0.01,
            "raw_load_seconds": 0.01,
            "raw_sha256_seconds": 0.001,
            "peak_rss_gib": rss,
        })
    return tasks


def test_committed_replay_subsample_is_deterministic(frozen_config):
    first = committed_replay_blocks(frozen_config)
    second = committed_replay_blocks(frozen_config)
    assert first == second


def test_committed_replay_subsample_is_about_ten_percent(frozen_config):
    blocks = committed_replay_blocks(frozen_config)
    assert set(blocks) == set(M_VALUES)
    for m in M_VALUES:
        selected = blocks[m]
        assert selected == sorted(set(selected))
        assert all(0 <= block < TASKS_PER_M[m] for block in selected)
        # Block zero is always included, so the count can exceed the ceiling by one.
        floor = math.ceil(0.10 * TASKS_PER_M[m])
        assert floor <= len(selected) <= floor + 1
        assert 0 in selected


def test_committed_replay_covers_every_m_and_therefore_every_p(frozen_config):
    keys = expected_replay_keys(frozen_config)
    assert {m for m, _ in keys} == set(M_VALUES)
    # Each task spans the whole p grid, so covering every m covers every cell type.
    assert len(keys) == sum(
        len(value) for value in committed_replay_blocks(frozen_config).values()
    )


def test_replay_subsample_depends_on_the_frozen_seed(frozen_config):
    shifted = copy.deepcopy(frozen_config)
    shifted.pop("config_sha256", None)
    shifted.pop("config_path", None)
    blocks = committed_replay_blocks(frozen_config)
    # A different registry would give a different subsample, which is why the
    # subsample is recorded in the preflight rather than recomputed later.
    shifted["registry_sha256"] = "f" * 64
    assert committed_replay_blocks(shifted) != blocks


def test_replay_report_validation_requires_full_coverage(frozen_config):
    from data.expander_code.exp104.exp104_pipeline.replay import (
        _results_manifest_sha256,
        REPLAY_SCHEMA,
    )
    from data.expander_code.exp104.exp104_pipeline.config import code_id
    from data.expander_code.exp104.exp104_pipeline.seeds import derive_seed
    from data.expander_code.exp104.exp104_pipeline.config import block_code_indices

    keys = sorted(expected_replay_keys(frozen_config))
    results = []
    for m, block in keys:
        indices = block_code_indices(m, block)
        results.append({
            "status": "PASS", "reason": "", "m": m, "block_index": block,
            "codes": len(indices),
            "trials": len(indices) * len(frozen_config["p_tokens"]) * 4,
            "replay_control_seed": derive_seed(
                frozen_config, "replay", code_id(m, indices[0]),
                frozen_config["p_tokens"][0], block,
            ),
            "raw_sha256": "a" * 64,
            "error_stream_sha256": "b" * 64,
            "correction_stream_sha256": "c" * 64,
            "label_stream_sha256": "d" * 64,
        })
    report = {
        "schema_version": REPLAY_SCHEMA,
        "config_sha256": frozen_config["config_sha256"],
        "registry_sha256": frozen_config["registry_sha256"],
        "source_commit": frozen_config["source_commit"],
        "source_tree_sha256": frozen_config["source_tree_sha256"],
        "decoder_binary_sha256": frozen_config["decoder_binary"]["sha256"],
        "device_name": frozen_config["environment"]["device_name"],
        "hostname": frozen_config["environment"]["hostname"],
        "conda_environment": frozen_config["environment"]["conda_environment"],
        "conda_prefix_matches_python": True,
        "scope": "committed_subsample",
        "replay_policy": "committed_random_subsample",
        "replay_fraction": 0.10,
        "expected_tasks": len(results),
        "tasks": len(results),
        "raw_manifest_sha256": _results_manifest_sha256(results),
        "status": "PASS",
        "results": results,
    }
    assert validate_replay_report(report, frozen_config) is report

    dropped = copy.deepcopy(report)
    dropped["results"] = dropped["results"][:-1]
    dropped["tasks"] -= 1
    with pytest.raises(ValueError):
        validate_replay_report(dropped, frozen_config)

    relabelled = copy.deepcopy(report)
    relabelled["status"] = "INVALID"
    with pytest.raises(ValueError):
        validate_replay_report(relabelled, frozen_config)


def test_resource_estimate_passes_under_the_frozen_caps(remote_config):
    profile = remote_config["execution_profile"]
    estimate = estimate_resources(
        _benchmark_tasks(), remote_config, profile["num_workers"], profile,
    )
    assert estimate["total_codes"] == CODES_PER_M * len(M_VALUES)
    assert estimate["total_tasks"] == sum(TASKS_PER_M[m] for m in M_VALUES)
    assert estimate["total_trials"] == (
        CODES_PER_M * len(M_VALUES) * len(remote_config["p_tokens"]) * 4
    )
    assert estimate["reserved_core_hours"] > 0.0
    assert estimate["predicted_wall_hours"] > 0.0


def test_resource_estimate_blocks_when_a_cap_binds(remote_config):
    profile = dict(remote_config["execution_profile"])
    profile["stage_core_hour_cap"] = 0.001
    estimate = estimate_resources(
        _benchmark_tasks(), remote_config, profile["num_workers"], profile,
    )
    assert estimate["status"] == "BLOCKED_RESOURCE_PREFLIGHT"
    assert estimate["checks"]["reserved_core_hours_le_cap"] is False


def test_resource_estimate_reserves_twice_the_projected_work(remote_config):
    profile = remote_config["execution_profile"]
    estimate = estimate_resources(
        _benchmark_tasks(), remote_config, profile["num_workers"], profile,
    )
    base = (
        estimate["measurement_generation_core_hours"]
        + estimate["committed_replay_core_hours"]
        + estimate["analysis_core_hours"]
        + estimate["fixed_overhead_core_hours"]
    )
    assert estimate["reserved_core_hours"] == pytest.approx(2.0 * base)


def test_replay_is_a_small_fraction_of_the_generation_cost(remote_config):
    profile = remote_config["execution_profile"]
    estimate = estimate_resources(
        _benchmark_tasks(), remote_config, profile["num_workers"], profile,
    )
    ratio = (
        estimate["committed_replay_core_hours"]
        / estimate["measurement_generation_core_hours"]
    )
    assert 0.05 < ratio < 0.20


def test_resource_estimate_requires_every_anchor(remote_config):
    profile = remote_config["execution_profile"]
    tasks = [task for task in _benchmark_tasks() if task["m"] != 8]
    with pytest.raises(ValueError, match="anchor"):
        estimate_resources(tasks, remote_config, profile["num_workers"], profile)


def test_run_root_must_sit_under_the_frozen_run_root(remote_config, tmp_path):
    resolved = remote_cli.resolve_remote_run_root(
        "~/.single_shot/runs/exp104_ensemble_v1_001", remote_config,
    )
    assert resolved.name == "exp104_ensemble_v1_001"
    with pytest.raises(ValueError):
        remote_cli.resolve_remote_run_root(tmp_path / "elsewhere", remote_config)
    with pytest.raises(ValueError):
        remote_cli.resolve_remote_run_root(
            "~/.single_shot/runs/nested/deeper", remote_config,
        )


def test_remote_commands_refuse_the_local_config(local_config):
    assert local_config["schema_version"] == "exp104.config.v1"
    with pytest.raises(ValueError):
        remote_cli._require_remote_config(local_config)


def test_unplanned_raw_evidence_is_refused_before_a_run(tmp_path, remote_config):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    remote_cli._assert_no_unplanned_npz(raw_root, remote_config)
    (raw_root / "stray.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="unplanned"):
        remote_cli._assert_no_unplanned_npz(raw_root, remote_config)


def test_incomplete_raw_evidence_is_refused_when_completeness_is_required(
    tmp_path, remote_config,
):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / raw_filename(8, 0)).write_bytes(b"")
    with pytest.raises(ValueError, match="incomplete"):
        remote_cli._assert_no_unplanned_npz(raw_root, remote_config, require_complete=True)


def test_bytecode_detection_catches_pycache(tmp_path):
    assert remote_cli._bytecode_clean(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    assert not remote_cli._bytecode_clean(tmp_path)


def test_qualification_groups_point_at_real_tests():
    root = Path(__file__).resolve().parents[4]
    for _, paths in remote_cli.QUALIFICATION_GROUPS:
        for path in paths:
            target = root / path
            assert target.is_dir() or target.is_file()
    assert set(remote_cli.QUALIFICATION_EXPECTED_PASSES) == {
        name for name, _ in remote_cli.QUALIFICATION_GROUPS
    }


def _qualification_report(config, **overrides):
    report = {
        "schema_version": remote_cli.REMOTE_QUALIFICATION_SCHEMA,
        "status": "PASS",
        "experiment_id": config["experiment_id"],
        "execution_profile_id": config["execution_profile"]["profile_id"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "decoder_binary_path_suffix": "x" + config["decoder_binary"]["filename_suffix"],
        "device_name": config["environment"]["device_name"],
        "hostname": config["environment"]["hostname"],
        "conda_environment": config["environment"]["conda_environment"],
        "conda_prefix_matches_python": True,
        "python_executable": "/opt/python",
        "bytecode_clean_before": True,
        "bytecode_clean_after": True,
        "ldpc_source": dict(config["ldpc_source"], project_version="2.4.1"),
        "support_packages": config["support_packages"],
        "host": {"logical_cpu_count": 96},
        "groups": [
            {
                "name": name,
                "argv": [],
                "exit_code": 0,
                "status": "PASS",
                "passed_count": remote_cli.QUALIFICATION_EXPECTED_PASSES[name],
                "expected_passed_count": remote_cli.QUALIFICATION_EXPECTED_PASSES[name],
                "skipped_count": 0,
                "xfailed_count": 0,
                "xpassed_count": 0,
                "deselected_count": 0,
                "stdout_sha256": "0" * 64,
                "stderr_sha256": "0" * 64,
            }
            for name, _ in remote_cli.QUALIFICATION_GROUPS
        ],
    }
    report.update(overrides)
    return report


def test_qualification_validation_accepts_a_clean_pass(remote_config):
    report = _qualification_report(remote_config)
    assert remote_cli.validate_environment_qualification(report, remote_config) is report


@pytest.mark.parametrize("override", [
    {"status": "FAIL"},
    {"bytecode_clean_before": False},
    {"bytecode_clean_after": False},
    {"hostname": "nd-2"},
    {"conda_prefix_matches_python": False},
    {"source_tree_sha256": "0" * 64},
])
def test_qualification_validation_refuses_a_degraded_environment(remote_config, override):
    report = _qualification_report(remote_config, **override)
    with pytest.raises(ValueError):
        remote_cli.validate_environment_qualification(report, remote_config)


def test_qualification_validation_refuses_skipped_or_missing_tests(remote_config):
    report = _qualification_report(remote_config)
    report["groups"][0]["skipped_count"] = 1
    with pytest.raises(ValueError, match="fully executed"):
        remote_cli.validate_environment_qualification(report, remote_config)

    report = _qualification_report(remote_config)
    report["groups"][0]["passed_count"] += 1
    with pytest.raises(ValueError, match="pass count"):
        remote_cli.validate_environment_qualification(report, remote_config)


def test_qualification_validation_refuses_too_few_cpus(remote_config):
    report = _qualification_report(remote_config, host={"logical_cpu_count": 8})
    with pytest.raises(ValueError, match="logical CPUs"):
        remote_cli.validate_environment_qualification(report, remote_config)


def test_preflight_validation_refuses_a_substituted_replay_subsample(remote_config):
    blocks = committed_replay_blocks(remote_config)
    report = {
        "schema_version": remote_cli.REMOTE_PREFLIGHT_SCHEMA,
        "status": "PASS",
        "experiment_id": remote_config["experiment_id"],
        "execution_profile_id": remote_config["execution_profile"]["profile_id"],
        "config_sha256": remote_config["config_sha256"],
        "registry_sha256": remote_config["registry_sha256"],
        "source_commit": remote_config["source_commit"],
        "source_tree_sha256": remote_config["source_tree_sha256"],
        "decoder_binary_sha256": remote_config["decoder_binary"]["sha256"],
        "hostname": remote_config["environment"]["hostname"],
        "num_workers": 64,
        "outcome_blind": True,
        "committed_replay_blocks": {str(m): blocks[m] for m in M_VALUES},
        "estimate": estimate_resources(
            _benchmark_tasks(), remote_config, 64,
            remote_config["execution_profile"],
        ),
    }
    assert remote_cli.validate_remote_resource_preflight(report, remote_config) is report

    substituted = copy.deepcopy(report)
    substituted["committed_replay_blocks"]["8"] = [0, 1, 2]
    with pytest.raises(ValueError, match="committed"):
        remote_cli.validate_remote_resource_preflight(substituted, remote_config)


def test_task_plan_covers_every_code_once(remote_config):
    tasks = remote_cli._planned_tasks(remote_config)
    assert len(tasks) == sum(TASKS_PER_M[m] for m in M_VALUES)
    assert len(set(tasks)) == len(tasks)
    total_codes = sum(CODES_PER_TASK[m] for m, _ in tasks)
    assert total_codes == CODES_PER_M * len(M_VALUES)
