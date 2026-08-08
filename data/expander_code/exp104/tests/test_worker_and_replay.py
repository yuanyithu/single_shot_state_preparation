"""End-to-end worker, storage and bit-exact replay on real codes."""

import numpy as np
import pytest

from data.expander_code.exp104.exp104_pipeline.audit_scorer import (
    pairing_score,
    row_echelon_basis,
    rowspace_score,
)
from data.expander_code.exp104.exp104_pipeline.model import load_model
from data.expander_code.exp104.exp104_pipeline.raw import (
    load_raw,
    raw_filename,
    save_raw,
)
from data.expander_code.exp104.exp104_pipeline.replay import replay_task
from data.expander_code.exp104.exp104_pipeline.seeds import derive_seed
from data.expander_code.exp104.exp104_pipeline.worker import (
    RAW_FIELDS,
    run_code_block,
    score_residual_pairing,
)


@pytest.fixture(scope="module")
def executed_task(request):
    """Run one real three-code m=3 task once and share it across the module."""
    from data.expander_code.exp104.exp104_pipeline import replay as replay_module
    from data.expander_code.exp104.exp104_pipeline import worker as worker_module

    config = request.getfixturevalue("frozen_config")
    rows = request.getfixturevalue("registry_rows")
    original = worker_module.block_code_indices

    def indices(m, block_index):
        return [0, 1, 2]

    worker_module.block_code_indices = indices
    replay_module.block_code_indices = indices
    try:
        raw = run_code_block(3, 0, config, rows)
        yield raw
    finally:
        worker_module.block_code_indices = original
        replay_module.block_code_indices = original


def test_task_completes_and_carries_the_frozen_identity(executed_task, frozen_config):
    raw = executed_task
    assert raw["status"] == "VALID"
    assert raw["invalid_reason"] == ""
    assert set(raw) == RAW_FIELDS
    assert raw["schema_version"] == "exp104.raw.v1"
    assert raw["experiment_id"] == "exp104.ensemble_mc.v1"
    assert raw["config_sha256"] == frozen_config["config_sha256"]
    assert raw["registry_sha256"] == frozen_config["registry_sha256"]
    assert raw["p_tokens"] == ",".join(frozen_config["p_tokens"])
    assert raw["trials_per_code_p"] == frozen_config["trials_per_code_p"]
    assert raw["completed_codes"] == raw["planned_codes"] == 3
    assert raw["n"] == 225 and raw["k"] == 9


def test_task_arrays_have_the_declared_shapes(executed_task, frozen_config):
    raw = executed_task
    codes = 3
    p_count = len(frozen_config["p_tokens"])
    trials = frozen_config["trials_per_code_p"]
    assert raw["failure_flags"].shape == (codes, p_count, trials)
    assert raw["logical_labels"].shape == (codes, p_count, trials, 9)
    assert raw["bp_iterations"].shape == (codes, p_count, trials)
    assert raw["trial_seed"].shape == (codes, p_count)
    assert raw["failure_flags"].dtype == np.bool_
    assert raw["bp_iterations"].dtype == np.int32
    # max_iter is n, so no trial may report more iterations than that.
    assert raw["bp_iterations"].max() <= raw["n"]


def test_seeds_are_derived_from_the_frozen_namespace(executed_task, frozen_config):
    raw = executed_task
    for slot, index in enumerate([0, 1, 2]):
        for p_slot, token in enumerate(frozen_config["p_tokens"]):
            expected = derive_seed(
                frozen_config, "measurement", f"m03_c{index:05d}", token, 0,
            )
            assert int(raw["trial_seed"][slot, p_slot]) == expected
    # Distinct code-p cells must not share a trial stream.
    assert len(set(raw["trial_seed"].ravel().tolist())) == raw["trial_seed"].size


def test_failure_rate_rises_with_p(executed_task):
    """A sanity direction check, not a physics claim: more noise cannot help."""
    per_p = executed_task["failure_flags"].mean(axis=(0, 2))
    assert per_p[0] <= per_p[-1]
    assert per_p[-1] > 0.0


def test_round_trip_through_storage_preserves_every_field(tmp_path, executed_task):
    path = tmp_path / raw_filename(3, 0)
    digest = save_raw(path, executed_task)
    assert len(digest) == 64
    loaded = load_raw(path)
    assert set(loaded) == RAW_FIELDS
    for field, value in executed_task.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(loaded[field], value)
            assert loaded[field].dtype == value.dtype
        else:
            assert loaded[field] == value


def test_storage_refuses_to_overwrite_immutable_evidence(tmp_path, executed_task):
    path = tmp_path / raw_filename(3, 0)
    save_raw(path, executed_task)
    with pytest.raises(FileExistsError):
        save_raw(path, executed_task)


def test_replay_reproduces_the_task_bit_for_bit(
    tmp_path, executed_task, frozen_config, registry_rows, short_block,
):
    path = tmp_path / raw_filename(3, 0)
    save_raw(path, executed_task)
    result = replay_task(path, frozen_config, registry_rows)
    assert result["status"] == "PASS", result.get("reason")
    assert result["reason"] == ""
    assert result["m"] == 3 and result["block_index"] == 0
    assert result["codes"] == 3
    assert result["trials"] == 3 * len(frozen_config["p_tokens"]) * 4
    for field in (
        "error_stream_sha256", "correction_stream_sha256", "label_stream_sha256",
    ):
        assert result[field] == executed_task[field]


@pytest.mark.parametrize("field", [
    "failure_flags", "logical_labels", "syndrome_match", "bp_converged",
    "bp_iterations",
])
def test_replay_rejects_a_single_flipped_outcome(
    executed_task, frozen_config, registry_rows, short_block, field,
):
    tampered = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in executed_task.items()
    }
    array = tampered[field]
    flat = array.reshape(-1)
    if array.dtype == np.bool_:
        flat[0] = not flat[0]
    else:
        flat[0] = flat[0] + 1
    result = replay_task(tampered, frozen_config, registry_rows)
    assert result["status"] == "INVALID"
    assert result["reason"].startswith("trial_replay_mismatch")


@pytest.mark.parametrize("field", [
    "error_stream_sha256", "correction_stream_sha256", "label_stream_sha256",
])
def test_replay_rejects_a_tampered_stream_digest(
    executed_task, frozen_config, registry_rows, short_block, field,
):
    tampered = dict(executed_task)
    tampered[field] = "0" * 64
    result = replay_task(tampered, frozen_config, registry_rows)
    assert result["status"] == "INVALID"
    assert result["reason"] == f"stream_hash_mismatch:{field}"


def test_replay_rejects_a_tampered_code_identity(
    executed_task, frozen_config, registry_rows, short_block,
):
    tampered = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in executed_task.items()
    }
    tampered["classical_distance"] = tampered["classical_distance"].copy()
    tampered["classical_distance"][0] += 2
    result = replay_task(tampered, frozen_config, registry_rows)
    assert result["status"] == "INVALID"
    assert result["reason"].startswith("code_identity_mismatch")


def test_replay_rejects_an_incomplete_task(
    executed_task, frozen_config, registry_rows, short_block,
):
    tampered = dict(executed_task)
    tampered["status"] = "INVALID"
    tampered["invalid_reason"] = "trial_infrastructure_error"
    result = replay_task(tampered, frozen_config, registry_rows)
    assert result["status"] == "INVALID"


def test_scorer_agrees_with_an_independent_rowspace_scorer(registry_rows):
    """Two different definitions of logical failure must agree exactly."""
    model = load_model(registry_rows["m03_c00000"])
    basis, pivots = row_echelon_basis(model.H_X)
    rng = np.random.Generator(np.random.PCG64(3))
    for _ in range(40):
        error = (rng.random(model.n) < 0.06).astype(np.uint8)
        correction = (rng.random(model.n) < 0.06).astype(np.uint8)
        failed, matched, _ = pairing_score(
            model.H_Z, model.logical_Z, error, correction,
        )
        other_failed, other_matched = rowspace_score(
            model.H_Z, basis, pivots, error, correction,
        )
        assert (failed, matched) == (other_failed, other_matched)


def test_worker_scorer_matches_the_audit_scorer(registry_rows):
    model = load_model(registry_rows["m03_c00001"])
    rng = np.random.Generator(np.random.PCG64(5))
    for _ in range(40):
        error = (rng.random(model.n) < 0.08).astype(np.uint8)
        correction = (rng.random(model.n) < 0.08).astype(np.uint8)
        mine = score_residual_pairing(model, error, correction)
        theirs = pairing_score(model.H_Z, model.logical_Z, error, correction)
        assert mine[0] == theirs[0] and mine[1] == theirs[1]
        assert np.array_equal(mine[2], theirs[2])


def test_worker_rejects_a_task_outside_the_frozen_plan(frozen_config, registry_rows):
    with pytest.raises(ValueError):
        run_code_block(9, 0, frozen_config, registry_rows)
    with pytest.raises(ValueError):
        run_code_block(3, 10 ** 6, frozen_config, registry_rows)
