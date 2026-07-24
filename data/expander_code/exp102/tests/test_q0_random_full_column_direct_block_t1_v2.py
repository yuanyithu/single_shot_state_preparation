from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


MODULE = (
    "data.expander_code.exp102.validation."
    "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.workflow"
)
V1_MODULE = (
    "data.expander_code.exp102.validation."
    "055_q0_random_full_column_direct_block_t1_m8_20260724.workflow"
)
SOURCE = "0123456789abcdef0123456789abcdef01234567"
ARCHIVE = "1" * 64
MANIFEST = "2" * 64


def _schedule(module, run_root):
    subprocess.run([
        sys.executable, "-m", module, "build-schedule",
        "--run-root", str(run_root),
        "--source-commit", SOURCE,
        "--archive-sha256", ARCHIVE,
        "--source-manifest-sha256", MANIFEST,
    ], check=True, capture_output=True, text=True)
    return json.loads((run_root / "control/schedule.json").read_text())


def test_import_does_not_mutate_v1_workflow():
    v1 = import_module(V1_MODULE)
    old_contract = v1.CONTRACT_VERSION
    old_root = v1.ROOT
    workflow = import_module(MODULE)
    assert v1.CONTRACT_VERSION == old_contract
    assert v1.ROOT == old_root
    assert workflow.CONTRACT_VERSION.endswith("direct_block.t1_m8.v2")


def test_v2_config_binds_terminal_audit_and_runtime_design():
    workflow = import_module(MODULE)
    config, config_sha = workflow._load_config()
    assert len(config_sha) == 64
    assert config["resource"] == workflow.EXPECTED_RESOURCE
    assert config["implementation"]["validation_055_terminal_audit_sha256"] == (
        "00622194dc370a66e08a0b94a7108b324aa49322de648fda7656f2c6ed5fc665"
    )
    assert config["resource"]["burn_updates"] == 2048
    assert config["resource"]["measurement_updates"] == 8192
    assert config["resource"]["trajectory_wall_cap_seconds"] == 7200.0


def test_v2_control_preserves_legal_starts_and_refreshes_characters():
    workflow = import_module(MODULE)
    v1 = import_module(V1_MODULE)
    config, config_sha = workflow._load_config()
    context = workflow._load_control(workflow.SOURCE_CONTROL_DIR, config, config_sha)
    old_config, old_sha = v1._load_config()
    old = v1._load_control(v1.SOURCE_CONTROL_DIR, old_config, old_sha)
    assert np.array_equal(context["fixed_states"], old["fixed_states"])
    assert np.array_equal(
        context["arrays"]["fixed_b_blocks"], old["arrays"]["fixed_b_blocks"],
    )
    assert not np.array_equal(
        context["arrays"]["logical_character_masks"],
        old["arrays"]["logical_character_masks"],
    )
    assert not np.array_equal(
        context["arrays"]["b_character_masks_packed"],
        old["arrays"]["b_character_masks_packed"],
    )
    residuals = (
        context["model"].H_check.astype(np.int64)
        @ context["fixed_states"].T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    assert np.array_equal(
        residuals,
        np.repeat(context["syndrome"][None, :], 11, axis=0),
    )
    assert int(context["syndrome"].sum()) == 160


def test_v2_schedule_and_all_seed_fields_are_fresh(tmp_path):
    old = _schedule(V1_MODULE, tmp_path / "v1")
    new = _schedule(MODULE, tmp_path / "v2")
    assert new["contract_version"].endswith("direct_block.t1_m8.v2")
    assert len(new["tasks"]) == 40
    assert {task["family"] for task in new["tasks"]} == {
        "P", "U", "M0", "M1", "S",
    }
    assert {node: len(rows) for node, rows in new["ownership"].items()} == {
        "nd-1": 14, "nd-2": 13, "nd-3": 13,
    }
    for field in (
        "initialization_seed", "burn_update_seed", "measurement_update_seed",
        "observation_seed",
    ):
        old_values = {task[field] for task in old["tasks"]}
        new_values = {task[field] for task in new["tasks"]}
        assert len(new_values) == 40
        assert old_values.isdisjoint(new_values)


def test_two_length_fit_recovers_one_startup_and_slope():
    workflow = import_module(MODULE)
    fit = workflow._fit_runtime_component(
        10.0 + 136 * 0.2,
        10.0 + 272 * 0.2,
        136, 272, 10240,
    )
    assert fit["stable"] is True
    assert abs(fit["intercept_seconds"] - 10.0) < 1e-12
    assert abs(fit["slope_seconds_per_update"] - 0.2) < 1e-12
    assert abs(fit["target_seconds_before_safety"] - 2058.0) < 1e-9


def test_two_length_fit_fails_closed_on_nonpositive_slope():
    workflow = import_module(MODULE)
    assert workflow._fit_runtime_component(
        20.0, 19.0, 136, 272, 10240,
    ) == {"stable": False}
    assert workflow._fit_runtime_component(
        float("nan"), 30.0, 136, 272, 10240,
    ) == {"stable": False}


def test_v2_analyzer_and_shell_versions_are_fresh():
    analyzer = import_module(
        "data.expander_code.exp102.validation."
        "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.analyze_t1"
    )
    assert analyzer.RAW_VERSION.endswith("direct_block.t1_m8.raw.v2")
    assert analyzer.REPORT_VERSION.endswith("direct_block.t1_m8.report.v2")
    run_stage = Path(analyzer._workflow.ROOT) / "run_stage.sh"
    subprocess.run(["bash", "-n", str(run_stage)], check=True)


def test_constant_b_freeze_uses_signed_character_values():
    analyzer = import_module(
        "data.expander_code.exp102.validation."
        "052_q0_random_full_column_t1_m8_20260724.analyze_t1"
    )
    b_set = SimpleNamespace(
        masks_packed=np.asarray([[1]], dtype=np.uint8), size=1,
    )
    records = [{
        "b_packed": np.ones((4, 1), dtype=np.uint8),
        "burn_b_packed": np.zeros((4, 1), dtype=np.uint8),
        "family": "P",
        "index": 0,
        "initial_b_packed": np.ones(1, dtype=np.uint8),
    }]
    assert analyzer._constant_b_freeze_failures(records, b_set) == []
