from importlib import import_module
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


MODULE = (
    "data.expander_code.exp102.validation."
    "055_q0_random_full_column_direct_block_t1_m8_20260724.workflow"
)
OLD_MODULE = (
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.workflow"
)
SOURCE = "0123456789abcdef0123456789abcdef01234567"
ARCHIVE = "1" * 64
MANIFEST = "2" * 64


def test_import_does_not_mutate_frozen_v0_workflow():
    old = import_module(OLD_MODULE)
    old_contract = old.CONTRACT_VERSION
    old_method = old.RANDOM_FULL_COLUMN_METHOD_ID
    workflow = import_module(MODULE)
    assert old.CONTRACT_VERSION == old_contract
    assert old.RANDOM_FULL_COLUMN_METHOD_ID == old_method
    assert workflow.CONTRACT_VERSION.endswith("direct_block.t1_m8.v1")


def test_fresh_control_preserves_starts_but_refreshes_characters():
    workflow = import_module(MODULE)
    old = import_module(OLD_MODULE)
    config, config_sha = workflow._load_config()
    context = workflow._load_control(workflow.SOURCE_CONTROL_DIR, config, config_sha)
    old_config, old_config_sha = old._load_config()
    old_context = old._load_control(old.SOURCE_CONTROL_DIR, old_config, old_config_sha)

    assert context["metadata"]["control_version"] == workflow.CONTROL_VERSION
    assert np.array_equal(context["fixed_states"], old_context["fixed_states"])
    assert np.array_equal(context["arrays"]["fixed_b_blocks"],
                          old_context["arrays"]["fixed_b_blocks"])
    assert not np.array_equal(
        context["arrays"]["logical_character_masks"],
        old_context["arrays"]["logical_character_masks"],
    )
    assert not np.array_equal(
        context["arrays"]["b_character_masks_packed"],
        old_context["arrays"]["b_character_masks_packed"],
    )
    zero_syndrome = np.zeros(context["model"].num_checks, dtype=np.uint8)
    assert not np.array_equal(context["syndrome"], zero_syndrome)
    assert np.array_equal(
        context["model"].H_check.astype(np.int64)
        @ context["fixed_states"][0].astype(np.int64) % 2,
        context["syndrome"],
    )


def test_fresh_task_panel_and_seeds_are_disjoint_from_v0():
    workflow = import_module(MODULE)
    old = import_module(OLD_MODULE)
    config, config_sha = workflow._load_config()
    context = workflow._load_control(workflow.SOURCE_CONTROL_DIR, config, config_sha)
    old_config, old_config_sha = old._load_config()
    old_context = old._load_control(old.SOURCE_CONTROL_DIR, old_config, old_config_sha)
    source = {
        "archive_sha256": ARCHIVE,
        "source_commit": SOURCE,
        "source_manifest_sha256": MANIFEST,
    }
    # _task_rows resolves module globals, so configure only in this test after
    # all v0 tasks have been materialized.
    old_tasks = old._task_rows(old_context, source)
    rebound_names = (
        "CONTRACT_VERSION", "SCHEDULE_VERSION", "PREFLIGHT_VERSION", "RAW_VERSION",
        "NODE_REPORT_VERSION", "CONTROL_VERSION", "ROOT", "EXP102_ROOT",
        "CONFIG_PATH", "SOURCE_CONTROL_DIR", "FAMILIES",
        "RANDOM_FULL_COLUMN_METHOD_ID", "RANDOM_FULL_COLUMN_VERSION",
        "RandomFullColumnConfig", "build_classical_coset_mass",
        "build_full_column_candidate_cache", "build_full_column_workspace",
        "run_random_full_column_trajectory", "replay_random_full_column_trajectory",
        "_load_config", "_load_control",
    )
    original = {name: getattr(old, name) for name in rebound_names}
    try:
        workflow._configure_legacy()
        new_tasks = old._task_rows(context, source)
    finally:
        for name, value in original.items():
            setattr(old, name, value)
    assert len(new_tasks) == 40
    assert {task["method_id"] for task in new_tasks} == {"RFCG-C24-DPB12-S1"}
    assert {task["family"] for task in new_tasks} == {"P", "U", "M0", "M1", "S"}
    assert {node: sum(task["owner"] == node for task in new_tasks) for node in (
        "nd-1", "nd-2", "nd-3",
    )} == {"nd-1": 14, "nd-2": 13, "nd-3": 13}
    for field in (
        "initialization_seed", "burn_update_seed", "measurement_update_seed",
        "observation_seed",
    ):
        assert len({task[field] for task in new_tasks}) == 40
        assert not ({task[field] for task in old_tasks}
                    & {task[field] for task in new_tasks})


def test_schedule_cli_uses_fresh_direct_block_identities(tmp_path):
    run_root = tmp_path / "run"
    command = [
        sys.executable, "-m", MODULE, "build-schedule",
        "--run-root", str(run_root),
        "--source-commit", SOURCE,
        "--archive-sha256", ARCHIVE,
        "--source-manifest-sha256", MANIFEST,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    schedule = json.loads((run_root / "control/schedule.json").read_text())
    assert schedule["contract_version"].endswith("direct_block.t1_m8.v1")
    assert len(schedule["tasks"]) == 40
    assert {task["method_id"] for task in schedule["tasks"]} == {
        "RFCG-C24-DPB12-S1"
    }
    assert len({task["task_fingerprint"] for task in schedule["tasks"]}) == 40


def test_direct_analyzer_versions_are_fresh():
    analyzer = import_module(
        "data.expander_code.exp102.validation."
        "055_q0_random_full_column_direct_block_t1_m8_20260724.analyze_t1"
    )
    assert analyzer.RAW_VERSION.endswith("direct_block.t1_m8.raw.v1")
    assert analyzer.REPORT_VERSION.endswith("direct_block.t1_m8.report.v1")


def test_direct_raw_schema_and_independent_algebra_replay(tmp_path):
    workflow = import_module(MODULE)
    analyzer = import_module(
        "data.expander_code.exp102.validation."
        "055_q0_random_full_column_direct_block_t1_m8_20260724.analyze_t1"
    )
    direct = import_module(
        "data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column"
    )
    conditional = import_module(
        "data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs"
    )
    collapsed = import_module(
        "data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed"
    )
    io = import_module("data.expander_code.exp102.exp102_pipeline.io")
    config, config_sha = workflow._load_config()
    context = workflow._load_control(workflow.SOURCE_CONTROL_DIR, config, config_sha)
    context["config"] = copy.deepcopy(context["config"])
    context["config"]["resource"]["burn_updates"] = 1
    context["config"]["resource"]["measurement_updates"] = 8
    task = {
        "burn_update_seed": 101,
        "family": "P",
        "index": 0,
        "initialization_seed": 102,
        "measurement_update_seed": 103,
        "method_id": "RFCG-C24-DPB12-S1",
        "observation_seed": 104,
    }
    task["task_fingerprint"] = workflow.sha256_json(task)
    source = {
        "archive_sha256": ARCHIVE,
        "source_commit": SOURCE,
        "source_manifest_sha256": MANIFEST,
    }
    schedule = {"schedule_sha256": "3" * 64, "source_identity": source}
    sampler = direct.RandomFullColumnDirectBlockConfig(
        p=0.04, burn_updates=1, measurement_updates=8,
    )
    mass = np.ascontiguousarray(
        collapsed.build_classical_coset_mass(context["H"], 0.04, engine="numba"),
        dtype=np.float64,
    )
    cache = conditional.build_full_column_direct_block_cache(
        context["H"].shape[0], 0.04, mass,
    )
    workspace = conditional.build_full_column_direct_block_workspace(cache)
    raw = direct.run_random_full_column_direct_block_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, context["fixed_states"][0], task["burn_update_seed"],
        task["measurement_update_seed"], task["observation_seed"], mass=mass,
        cache=cache, workspace=workspace,
    )
    assert direct.replay_random_full_column_direct_block_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, context["fixed_states"][0], task["burn_update_seed"],
        task["measurement_update_seed"], task["observation_seed"], raw,
        mass=mass, cache=cache, workspace=workspace,
    )
    path = tmp_path / "raw.npz"
    io.atomic_npz(path, **{
        "archive_sha256": np.array(source["archive_sha256"]),
        "config_sha256": np.array(context["config_sha"]),
        "contract_version": np.array(workflow.CONTRACT_VERSION),
        "control_content_sha256": np.array(
            context["metadata"]["control_content_sha256"],
        ),
        "model_fingerprint": np.array(context["model"].fingerprint()),
        "raw_version": np.array(workflow.RAW_VERSION),
        "replay_seconds": np.array(1.0),
        "sampling_seconds": np.array(1.0),
        "schedule_sha256": np.array(schedule["schedule_sha256"]),
        "source_commit": np.array(source["source_commit"]),
        "source_manifest_sha256": np.array(source["source_manifest_sha256"]),
        "syndrome_packed": context["arrays"]["syndrome_packed"],
        "task_fingerprint": np.array(task["task_fingerprint"]),
        "task_json": np.array(workflow.canonical_json(task)),
        **raw,
    })
    verified = analyzer._load_and_verify_raw(
        path, task, context, schedule, np.log(mass),
    )
    assert verified["family"] == "P"
    assert verified["measurement_changes"] >= 0

    with np.load(path, allow_pickle=False) as archive:
        tampered = {name: archive[name].copy() for name in archive.files}
    tampered["conditional_engine"] = np.array("wrong_engine")
    bad_path = tmp_path / "bad.npz"
    io.atomic_npz(bad_path, **tampered)
    try:
        analyzer._load_and_verify_raw(
            bad_path, task, context, schedule, np.log(mass),
        )
    except analyzer.AnalysisConflictError as exc:
        assert "conditional_engine" in str(exc)
    else:
        raise AssertionError("tampered conditional engine was accepted")
