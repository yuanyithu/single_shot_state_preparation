import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

from data.expander_code.exp103.exp103_pipeline import replay, worker
from data.expander_code.exp103.exp103_pipeline.raw import load_raw, save_raw
from data.expander_code.exp103.exp103_pipeline.seeds import derive_seed


def _raw_signature(raw):
    return (
        raw["seed"],
        raw["error_stream_sha256"],
        raw["correction_stream_sha256"],
        raw["label_stream_sha256"],
        raw["failure_flags"].tobytes(),
        raw["logical_labels"].tobytes(),
        raw["syndrome_match"].tobytes(),
        raw["bp_converged"].tobytes(),
        raw["bp_iterations"].tobytes(),
    )


def _tiny_process_streams(task):
    """Independent, picklable reconstruction used by real worker processes."""
    config, code_id, p_token, shard_index = task
    seed = derive_seed(config, "measurement", code_id, p_token, shard_index)
    rng = np.random.Generator(np.random.PCG64(seed))
    error_digest = hashlib.sha256()
    correction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    for _ in range(config["trials_per_shard"]):
        error = (rng.random(4) < float(p_token)).astype(np.uint8)
        syndrome = int((error[0] + error[1]) & 1)
        correction = np.asarray([syndrome, 0, 0, 0], dtype=np.uint8)
        residual = np.bitwise_xor(error, correction)
        labels = np.asarray(
            [residual[0], (residual[2] + residual[3]) & 1], dtype=np.uint8,
        )
        error_digest.update(error.tobytes())
        correction_digest.update(correction.tobytes())
        label_digest.update(labels.tobytes())
    return code_id, p_token, shard_index, (
        error_digest.hexdigest(), correction_digest.hexdigest(), label_digest.hexdigest(),
    )


def test_seed_namespaces_are_disjoint_and_p_tokens_are_canonical(frozen_config):
    seeds = {
        namespace: derive_seed(
            frozen_config, namespace, "m03_c00", "0.08", 0,
        )
        for namespace in ("benchmark", "measurement", "replay", "bootstrap")
    }
    assert len(set(seeds.values())) == 4
    assert derive_seed(frozen_config, "measurement", "m03_c00", 0.08, 0) == (
        seeds["measurement"]
    )
    assert seeds["benchmark"] != seeds["measurement"]

    measurement_panel = {
        derive_seed(frozen_config, "measurement", f"m03_c{index:02d}", "0.08", 0)
        for index in range(8)
    }
    benchmark_panel = {
        derive_seed(frozen_config, "benchmark", f"m03_c{index:02d}", "0.08", 0)
        for index in range(8)
    }
    assert measurement_panel.isdisjoint(benchmark_panel)


def test_task_order_and_eight_worker_scheduling_preserve_raw_streams(
    frozen_config, install_tiny_runtime,
):
    install_tiny_runtime()
    tasks = [
        (f"m03_c{index:02d}", f"0.{2 + index:02d}", index % 4)
        for index in range(8)
    ]

    serial = {
        task: _raw_signature(worker.run_decoder_shard(*task, frozen_config))
        for task in tasks
    }
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            task: executor.submit(worker.run_decoder_shard, *task, frozen_config)
            for task in reversed(tasks)
        }
        scheduled = {task: _raw_signature(future.result()) for task, future in futures.items()}
    assert scheduled == serial


def test_actual_eight_process_error_correction_and_label_streams_match_serial_raw(
    frozen_config, install_tiny_runtime,
):
    install_tiny_runtime()
    tasks = [
        (f"m03_c{index:02d}", f"0.{2 + index:02d}", index % 4)
        for index in range(8)
    ]
    serial = {}
    for task in tasks:
        raw = worker.run_decoder_shard(*task, frozen_config)
        serial[task] = (
            raw["error_stream_sha256"], raw["correction_stream_sha256"],
            raw["label_stream_sha256"],
        )
    process_tasks = [
        (frozen_config, code_id, p_token, shard_index)
        for code_id, p_token, shard_index in reversed(tasks)
    ]
    with ProcessPoolExecutor(max_workers=8) as executor:
        parallel = {
            (code_id, p_token, shard_index): streams
            for code_id, p_token, shard_index, streams in executor.map(
                _tiny_process_streams, process_tasks,
            )
        }
    assert parallel == serial


def test_fresh_save_load_and_resume_refuse_to_change_raw(
    tmp_path, frozen_config, install_tiny_runtime,
):
    install_tiny_runtime()
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)
    path = tmp_path / "shard.npz"
    save_raw(path, raw)
    loaded = load_raw(path)
    assert _raw_signature(loaded) == _raw_signature(raw)
    with pytest.raises(FileExistsError):
        save_raw(path, worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config))
    save_raw(path, raw, refuse_overwrite=False)
    assert _raw_signature(load_raw(path)) == _raw_signature(raw)


def test_full_replay_uses_independent_scorer_and_matches_all_streams(
    monkeypatch, frozen_config, install_tiny_runtime,
):
    install_tiny_runtime()
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)

    def forbidden_worker_scorer(*args, **kwargs):
        raise AssertionError("replay imported the worker scorer")

    monkeypatch.setattr(worker, "score_residual_pairing", forbidden_worker_scorer)
    result = replay.replay_decoder_shard(raw, frozen_config)
    assert result["status"] == "PASS"
    assert result["trials"] == frozen_config["trials_per_shard"]
    for field in (
        "error_stream_sha256", "correction_stream_sha256", "label_stream_sha256",
    ):
        assert result[field] == raw[field]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("p", 0.03),
        ("m", 4),
        ("planned_trials", 2499),
        ("n", 5),
        ("k", 3),
        ("python_version", "3.12.11"),
        ("ldpc_version", "2.3.7"),
        ("device_name", "another-device"),
        ("hostname", "other.local"),
        ("conda_environment", "11"),
        ("conda_prefix_matches_python", False),
        ("conda_prefix_matches_python", 1),
    ],
)
def test_replay_rejects_every_raw_identity_drift(
    frozen_config, install_tiny_runtime, clone_payload, field, replacement,
):
    install_tiny_runtime()
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)
    tampered = clone_payload(raw)
    tampered[field] = replacement
    result = replay.replay_decoder_shard(tampered, frozen_config)
    assert result["status"] == "INVALID"
    assert "identity" in result["reason"] or "complete" in result["reason"]


def test_replay_rejects_trial_and_stream_tampering(
    frozen_config, install_tiny_runtime, clone_payload,
):
    install_tiny_runtime()
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)

    tampered_trial = clone_payload(raw)
    tampered_trial["logical_labels"][0, 0] ^= np.uint8(1)
    assert replay.replay_decoder_shard(tampered_trial, frozen_config)["status"] == "INVALID"

    tampered_hash = clone_payload(raw)
    tampered_hash["correction_stream_sha256"] = "f" * 64
    result = replay.replay_decoder_shard(tampered_hash, frozen_config)
    assert result == {
        "status": "INVALID",
        "reason": "stream_hash_mismatch:correction_stream_sha256",
    }


def test_replay_rejects_illegal_backend_correction(
    frozen_config, install_tiny_runtime,
):
    class BadShapeDecoder:
        def __init__(self, *args, **kwargs):
            self.converge = False
            self.iter = 0

        def decode(self, syndrome):
            return np.zeros((1, 4), dtype=np.uint8)

    install_tiny_runtime()
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)
    install_tiny_runtime(BadShapeDecoder)
    result = replay.replay_decoder_shard(raw, frozen_config)
    assert result["status"] == "INVALID"
    assert result["reason"] == "illegal_correction_at_trial:0"


def test_replay_report_covers_manifest_and_closes_on_post_replay_raw_change(
    monkeypatch, tmp_path, frozen_config, install_tiny_runtime, clone_payload,
):
    install_tiny_runtime()
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)
    raw_path = tmp_path / "m03_c00.npz"
    save_raw(raw_path, raw)
    result = replay.replay_decoder_shard(raw_path, frozen_config)
    key = {("m03_c00", "0.02", 0)}
    monkeypatch.setattr(replay, "expected_replay_keys", lambda _config, _scope: key)

    report = replay.build_replay_report(tmp_path, [result], frozen_config)
    assert report["scope"] == "stage1"
    assert report["status"] == "PASS"
    assert report["device_name"] == "macmini"
    assert report["hostname"] == "ymini.local"
    assert report["conda_environment"] == "12"
    assert report["conda_prefix_matches_python"] is True
    assert replay.validate_replay_report(
        report, tmp_path, frozen_config, required_scope="stage1",
    ) is report

    tampered_report = clone_payload(report)
    tampered_report["raw_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="result manifest"):
        replay.validate_replay_report_payload(
            tampered_report, frozen_config, required_scope="stage1",
        )

    wrong_host = clone_payload(report)
    wrong_host["hostname"] = "other.local"
    with pytest.raises(ValueError, match="hostname"):
        replay.validate_replay_report_payload(
            wrong_host, frozen_config, required_scope="stage1",
        )

    non_boolean_prefix = clone_payload(report)
    non_boolean_prefix["conda_prefix_matches_python"] = 1
    with pytest.raises(ValueError, match="boolean true"):
        replay.validate_replay_report_payload(
            non_boolean_prefix, frozen_config, required_scope="stage1",
        )

    changed = clone_payload(raw)
    changed["bp_iterations"][0] += 1
    save_raw(raw_path, changed, refuse_overwrite=False)
    with pytest.raises(ValueError, match="manifest|SHA"):
        replay.validate_replay_report(
            report, tmp_path, frozen_config, required_scope="stage1",
        )
