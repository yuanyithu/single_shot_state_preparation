"""One real task, end to end, and the replay gate that has to catch tampering."""

import numpy as np
import pytest

from data.expander_code.exp105.exp105_pipeline.config import code_id
from data.expander_code.exp105.exp105_pipeline.model import (
    clear_model_cache,
    load_model,
    logical_label,
    parity_product,
)
from data.expander_code.exp105.exp105_pipeline.raw import (
    load_raw,
    raw_filename,
    save_raw,
)
from data.expander_code.exp105.exp105_pipeline.replay import (
    committed_replay_blocks,
    replay_task,
)
from data.expander_code.exp105.exp105_pipeline.seeds import derive_seed
from data.expander_code.exp105.exp105_pipeline.worker import (
    RAW_FIELDS,
    run_code_block,
)


@pytest.fixture
def task(frozen_config, registry_rows, short_block):
    """One real m=3 task, shortened to two codes.

    Function scoped because `short_block` patches the task blocking through
    monkeypatch; the m=3 path is cheap enough that rebuilding it per test costs
    about a second.
    """
    clear_model_cache()
    raw = run_code_block(3, 0, frozen_config, registry_rows)
    clear_model_cache()
    return raw


def test_task_is_valid_and_complete(task, frozen_config, short_block):
    assert task["status"] == "VALID", task["invalid_reason"]
    assert task["invalid_reason"] == ""
    assert set(task) == RAW_FIELDS
    assert task["completed_codes"] == short_block == task["planned_codes"]
    assert task["q_token"] == frozen_config["q_token"]
    assert task["p_tokens"] == ",".join(frozen_config["p_tokens"])
    assert task["n"] == 25 * 9 and task["n_checks"] == 12 * 9 and task["k"] == 9


def test_task_records_the_frame_identities(task, registry_rows):
    clear_model_cache()
    model = load_model(registry_rows[code_id(3, 0)])
    assert str(task["classical_H_sha256"][0]) == model.classical_H_sha256
    assert str(task["logical_frame_sha256"][0]) == model.logical_frame_sha256
    assert str(task["observable_frame_fingerprint"][0]) == (
        model.observable_frame_fingerprint
    )
    clear_model_cache()


def test_outcomes_are_in_range_and_stream_digests_are_populated(task, frozen_config):
    trials = int(task["trials_per_code_p"])
    n_p = len(frozen_config["p_tokens"])
    assert task["failure_flags"].shape == (task["planned_codes"], n_p, trials)
    assert task["logical_labels"].shape[-1] == task["k"]
    assert set(np.unique(task["logical_labels"])) <= {0, 1}
    assert np.all(task["bp_iterations"] >= 0)
    assert np.all(task["bp_iterations"] <= task["n"] + task["n_checks"])
    digests = {
        task["error_stream_sha256"], task["readout_stream_sha256"],
        task["correction_stream_sha256"], task["label_stream_sha256"],
    }
    assert len(digests) == 4, "stream digests must be distinct, not aliases"


def test_failure_flag_equals_nontrivial_label(task):
    """The stored verdict is exactly the stored class being nonzero."""
    labels_any = task["logical_labels"].any(axis=-1)
    assert np.array_equal(task["failure_flags"], labels_any)


def test_failure_rate_rises_with_p(task, frozen_config):
    """A weak physical sanity check, not a result: more noise, more failures."""
    rates = task["failure_flags"].mean(axis=(0, 2))
    assert rates[0] <= rates[-1]


def test_trials_reproduce_from_the_recorded_seed(task, frozen_config, registry_rows):
    """The recorded seed really is the stream that produced the outcomes."""
    clear_model_cache()
    model = load_model(registry_rows[code_id(3, 0)])
    q = float(frozen_config["q_token"])
    token = frozen_config["p_tokens"][0]
    seed = derive_seed(frozen_config, "measurement", model.code_id, token, 0)
    assert int(task["trial_seed"][0, 0]) == seed

    rng = np.random.Generator(np.random.PCG64(seed))
    error = (rng.random(model.n) < float(token)).astype(np.uint8)
    readout = (rng.random(model.n_checks) < q).astype(np.uint8)
    effective = np.bitwise_xor(parity_product(model.H_Z, error), readout)
    assert effective.shape == (model.n_checks,)
    clear_model_cache()


def test_replay_passes_on_an_untouched_task(tmp_path, task, frozen_config, registry_rows):
    path = tmp_path / raw_filename(frozen_config, 3, 0)
    save_raw(path, task)
    result = replay_task(path, frozen_config, registry_rows)
    assert result["status"] == "PASS", result.get("reason")
    assert result["error_stream_sha256"] == task["error_stream_sha256"]
    assert result["readout_stream_sha256"] == task["readout_stream_sha256"]
    assert result["correction_stream_sha256"] == task["correction_stream_sha256"]
    assert result["label_stream_sha256"] == task["label_stream_sha256"]


@pytest.mark.parametrize("field,index", [
    ("failure_flags", (0, 0, 0)),
    ("logical_labels", (0, 0, 0, 0)),
    ("readout_match", (0, 0, 0)),
    ("bp_iterations", (0, 0, 0)),
])
def test_replay_catches_a_single_flipped_outcome(
    tmp_path, task, frozen_config, registry_rows, field, index,
):
    tampered = {key: (value.copy() if hasattr(value, "copy") else value)
                for key, value in task.items()}
    if field == "bp_iterations":
        tampered[field][index] = int(tampered[field][index]) + 1
    else:
        tampered[field][index] = not bool(tampered[field][index])
    path = tmp_path / raw_filename(frozen_config, 3, 0)
    save_raw(path, tampered)
    result = replay_task(path, frozen_config, registry_rows)
    assert result["status"] == "INVALID"
    assert "mismatch" in result["reason"]


def test_replay_catches_a_tampered_stream_digest(
    tmp_path, task, frozen_config, registry_rows,
):
    tampered = dict(task)
    tampered["readout_stream_sha256"] = "0" * 64
    path = tmp_path / raw_filename(frozen_config, 3, 0)
    save_raw(path, tampered)
    result = replay_task(path, frozen_config, registry_rows)
    assert result["status"] == "INVALID"
    assert result["reason"] == "stream_hash_mismatch:readout_stream_sha256"


def test_raw_is_immutable(tmp_path, task, frozen_config):
    path = tmp_path / raw_filename(frozen_config, 3, 0)
    save_raw(path, task)
    with pytest.raises(FileExistsError):
        save_raw(path, task)
    assert set(load_raw(path)) == RAW_FIELDS


def test_committed_replay_covers_block_zero_of_every_m(frozen_config):
    blocks = committed_replay_blocks(frozen_config)
    assert set(blocks) == {int(m) for m in frozen_config["m_values"]}
    for m, selected in blocks.items():
        assert 0 in selected
        assert selected == sorted(set(selected))


def test_committed_replay_is_fixed_by_the_seed(frozen_config):
    assert committed_replay_blocks(frozen_config) == committed_replay_blocks(
        frozen_config
    )
