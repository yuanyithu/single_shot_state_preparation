import copy
import itertools
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.expander_code.exp103.exp103_pipeline import identity, preflight, replay, worker
from data.expander_code.exp103.exp103_pipeline.audit_scorer import (
    pairing_score,
    row_echelon_basis,
    rowspace_score,
)


def test_algebraic_golden_residuals(tiny_css_model):
    zero = np.zeros(4, dtype=np.uint8)
    stabilizer = tiny_css_model.H_X[0]
    logical_x = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    wrong_syndrome = np.asarray([1, 0, 0, 0], dtype=np.uint8)

    cases = (
        (zero, False, True, [0, 0]),
        (stabilizer, False, True, [0, 0]),
        (logical_x, True, True, [1, 0]),
        (wrong_syndrome, True, False, [1, 0]),
    )
    for residual, expected_failed, expected_match, expected_labels in cases:
        failed, matched, labels = pairing_score(
            tiny_css_model.H_Z, tiny_css_model.logical_Z, residual, zero,
        )
        assert (failed, matched) == (expected_failed, expected_match)
        assert labels.tolist() == expected_labels


def test_pairing_and_independent_rowspace_scorers_agree_exhaustively(tiny_css_model):
    basis, pivots = row_echelon_basis(tiny_css_model.H_X)
    correction = np.zeros(4, dtype=np.uint8)
    for bits in itertools.product((0, 1), repeat=4):
        error = np.asarray(bits, dtype=np.uint8)
        pairing_failed, pairing_match, _ = pairing_score(
            tiny_css_model.H_Z, tiny_css_model.logical_Z, error, correction,
        )
        rowspace_failed, rowspace_match = rowspace_score(
            tiny_css_model.H_Z, basis, pivots, error, correction,
        )
        assert (pairing_failed, pairing_match) == (rowspace_failed, rowspace_match)


@pytest.mark.parametrize("p", [0.0, 0.02, 0.13, 0.5])
def test_tiny_css_exact_bernoulli_enumeration_has_analytic_endpoints(
    tiny_css_model, p,
):
    weighted_failure = 0.0
    for bits in itertools.product((0, 1), repeat=4):
        error = np.asarray(bits, dtype=np.uint8)
        syndrome = int((tiny_css_model.H_Z @ error)[0] & 1)
        correction = np.asarray([syndrome, 0, 0, 0], dtype=np.uint8)
        failed, matched, _ = pairing_score(
            tiny_css_model.H_Z, tiny_css_model.logical_Z, error, correction,
        )
        assert matched
        weight = p ** sum(bits) * (1.0 - p) ** (4 - sum(bits))
        weighted_failure += weight * failed

    # Success requires e_1=0 and equality of the final stabilizer pair.
    exact_failure = 1.0 - (1.0 - p) * ((1.0 - p) ** 2 + p ** 2)
    assert weighted_failure == pytest.approx(exact_failure, abs=1e-15)
    if p == 0.0:
        assert weighted_failure == 0.0
    if p == 0.5:
        assert weighted_failure == 0.75  # Uniform block failure is 1 - 2^-k.


def _expected_decoder_kwargs(model, p):
    return {
        "error_rate": p,
        "bp_method": "product_sum",
        "max_iter": model.n,
        "schedule": "serial",
        "serial_schedule_order": list(range(model.n)),
        "osd_method": "osd_0",
        "osd_order": 0,
        "omp_thread_count": 1,
    }


@pytest.mark.parametrize(
    ("factory_module", "factory_name"),
    [(worker, "make_decoder"), (replay, "_decoder")],
)
@pytest.mark.parametrize("p", [0.02, 0.14])
def test_worker_and_replay_bind_every_decoder_parameter(
    monkeypatch, frozen_config, tiny_css_model, factory_module, factory_name, p,
):
    calls = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(factory_module, "BpOsdDecoder", Recorder)
    getattr(factory_module, factory_name)(tiny_css_model, p, frozen_config)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 1 and args[0] is tiny_css_model.H_Z_sparse
    assert kwargs == _expected_decoder_kwargs(tiny_css_model, p)


def test_preflight_uses_same_decoder_identity_and_benchmark_namespace(
    monkeypatch, frozen_config, tiny_css_model,
):
    calls = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.converge = True
            self.iter = 1

        def decode(self, syndrome):
            correction = np.zeros(4, dtype=np.uint8)
            correction[0] = np.asarray(syndrome, dtype=np.uint8)[0]
            return correction

    monkeypatch.setattr(preflight, "BpOsdDecoder", Recorder)
    monkeypatch.setattr(preflight, "load_model", lambda _config, _code_id: tiny_css_model)
    monkeypatch.setattr(preflight, "clear_model_cache", lambda: None)
    identity_calls = []

    def benchmark_identity(config, *args, **kwargs):
        identity_calls.append((args, kwargs))
        return {
            "device_name": config["environment"]["device_name"],
            "hostname": config["environment"]["hostname"],
            "conda_environment": config["environment"]["conda_environment"],
            "conda_prefix_matches_python": True,
            "python_version": config["environment"]["python"],
            "numpy_version": config["environment"]["numpy"],
            "scipy_version": config["environment"]["scipy"],
            "ldpc_version": config["environment"]["ldpc"],
            "decoder_binary_sha256": config["decoder_binary"]["sha256"],
            "source_tree_sha256": config["source_tree_sha256"],
            "source_commit": config["source_commit"],
        }

    monkeypatch.setattr(preflight, "runtime_identity", benchmark_identity)
    result = preflight.benchmark_task("m03_c00", "0.14", frozen_config)

    assert len(calls) == 2  # Separate measurement and full-replay decoder setup.
    assert len(identity_calls) == 2  # Formal workers check identity for every shard.
    for args, kwargs in calls:
        assert len(args) == 1 and args[0] is tiny_css_model.H_Z_sparse
        assert kwargs == _expected_decoder_kwargs(tiny_css_model, 0.14)
    assert result["seed_namespace"] == frozen_config["namespaces"]["benchmark"]
    assert "logical" not in result and "failure" not in result
    assert result["measurement_seconds"] >= 0.0
    assert result["replay_seconds"] >= 0.0
    for field in (
        "measurement_identity_seconds", "raw_serialization_seconds",
        "replay_identity_seconds", "raw_load_seconds",
        "replay_raw_sha256_seconds", "manifest_seconds",
    ):
        assert result[field] >= 0.0


def test_preflight_stage_arithmetic_uses_formal_overheads_and_stage_rss_anchors(
    frozen_config,
):
    config = copy.deepcopy(frozen_config)
    config["preflight"].update({
        "analysis_core_hours": 0.0,
        "fixed_overhead_core_hours": 0.0,
        "reserve_multiplier": 1.0,
    })

    def task(m, rss):
        return {
            "m": m,
            "measurement_seconds_per_trial": 0.0,
            "replay_seconds_per_trial": 0.0,
            "model_seconds": 1.0,
            "measurement_identity_seconds": float(m),
            "decoder_setup_seconds": float(m),
            "raw_serialization_seconds": float(m),
            "replay_identity_seconds": float(m),
            "replay_setup_seconds": float(m),
            "raw_load_seconds": float(m),
            "replay_raw_sha256_seconds": float(m),
            "manifest_seconds": float(m),
            "peak_rss_gib": rss,
        }

    tasks = [task(3, 1.0), task(5, 2.0), task(8, 99.0)]
    stage1 = preflight._stage_estimate("stage1", [3, 4, 5], tasks, config)
    stage2 = preflight._stage_estimate("stage2", [6, 7, 8], tasks, config)
    shards_per_m = 8 * len(config["p_tokens"]) * config["shards_per_code_p"]
    code_p_tasks_per_m = 8 * len(config["p_tokens"])

    assert stage1["generation_per_shard_seconds_upper_by_m"] == {
        "3": 9.0, "4": 15.0, "5": 15.0,
    }
    assert stage1["replay_per_shard_seconds_upper_by_m"] == {
        "3": 15.0, "4": 25.0, "5": 25.0,
    }
    assert stage1["measurement_generation_core_hours"] == pytest.approx(
        (shards_per_m * (9.0 + 15.0 + 15.0) + 3 * code_p_tasks_per_m)
        / 3600.0,
    )
    assert stage1["full_replay_core_hours"] == pytest.approx(
        (shards_per_m * (15.0 + 25.0 + 25.0) + 3 * code_p_tasks_per_m)
        / 3600.0,
    )
    assert stage1["model_loads_upper_by_m"] == {
        "3": code_p_tasks_per_m,
        "4": code_p_tasks_per_m,
        "5": code_p_tasks_per_m,
    }
    assert stage1["rss_anchor_m_values"] == [3, 5]
    assert stage1["projected_peak_rss_gib"] == 2.0 * 8 * 3.0
    assert stage2["rss_anchor_m_values"] == [8]
    assert stage2["projected_peak_rss_gib"] == 99.0 * 8 * 3.0


def test_preflight_replay_timing_never_gates_on_benchmark_outcomes(tiny_css_model):
    class Decoder:
        converge = False
        iter = 1

        def decode(self, syndrome):
            correction = np.zeros(tiny_css_model.n, dtype=np.uint8)
            correction[0] = np.asarray(syndrome, dtype=np.uint8)[0]
            return correction

    trials = 2
    deliberately_wrong = {
        "failure_flags": np.ones(trials, dtype=np.bool_),
        "logical_labels": np.ones((trials, tiny_css_model.k), dtype=np.uint8),
        "syndrome_match": np.zeros(trials, dtype=np.bool_),
        "bp_converged": np.ones(trials, dtype=np.bool_),
        "bp_iterations": np.zeros(trials, dtype=np.int32),
        "error_stream_sha256": "f" * 64,
        "correction_stream_sha256": "f" * 64,
        "label_stream_sha256": "f" * 64,
    }
    elapsed, _ = preflight._time_full_trial_path(
        tiny_css_model, Decoder(), "0.02", 7, trials,
        expected=deliberately_wrong,
    )
    assert elapsed >= 0.0


def test_decoder_exception_invalidates_shard_without_fallback(
    frozen_config, install_tiny_runtime,
):
    class RaisingDecoder:
        constructions = 0

        def __init__(self, *args, **kwargs):
            type(self).constructions += 1
            self.converge = False
            self.iter = 0

        def decode(self, syndrome):
            raise RuntimeError("deliberate backend failure")

    install_tiny_runtime(RaisingDecoder)
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)
    assert raw["status"] == "INVALID"
    assert raw["invalid_reason"] == "trial_infrastructure_error"
    assert raw["completed_trials"] == 0
    assert raw["exception_type"] == "RuntimeError"
    assert RaisingDecoder.constructions == 1


@pytest.mark.parametrize(
    "correction",
    [
        [0, 0, 0],
        np.zeros(4, dtype=np.int64),
        np.zeros((1, 4), dtype=np.uint8),
        np.asarray([0, 1, 0, 2], dtype=np.uint8),
    ],
)
def test_illegal_correction_shape_dtype_or_value_is_rejected(correction):
    with pytest.raises((TypeError, ValueError)):
        worker._validate_correction(correction, 4)


def test_bp_nonconvergence_with_legal_correction_remains_valid(
    frozen_config, install_tiny_runtime,
):
    install_tiny_runtime()
    raw = worker.run_decoder_shard("m03_c00", "0.02", 0, frozen_config)
    assert raw["status"] == "VALID"
    assert raw["completed_trials"] == frozen_config["trials_per_shard"]
    assert not raw["bp_converged"].any()
    assert np.all(raw["bp_iterations"] == 1)
    assert raw["device_name"] == frozen_config["environment"]["device_name"]
    assert raw["hostname"] == frozen_config["environment"]["hostname"]
    assert raw["conda_environment"] == frozen_config["environment"]["conda_environment"]
    assert raw["conda_prefix_matches_python"] is True


def test_runtime_identity_rejects_wrong_host_or_noncanonical_conda(
    monkeypatch, frozen_config,
):
    monkeypatch.setattr(
        identity, "source_tree_sha256", lambda: frozen_config["source_tree_sha256"],
    )
    monkeypatch.setattr(
        identity, "decoder_binary_path",
        lambda: Path("backend" + frozen_config["decoder_binary"]["filename_suffix"]),
    )
    monkeypatch.setattr(
        identity, "sha256_file", lambda _path: frozen_config["decoder_binary"]["sha256"],
    )
    expected_environment = frozen_config["environment"]
    monkeypatch.setattr(
        identity.socket, "gethostname", lambda: expected_environment["hostname"],
    )
    monkeypatch.setenv("CONDA_DEFAULT_ENV", expected_environment["conda_environment"])
    monkeypatch.setenv("CONDA_PREFIX", str(Path(identity.sys.prefix).resolve()))
    actual = identity.runtime_identity(frozen_config)
    assert actual["device_name"] == expected_environment["device_name"]
    assert actual["conda_prefix_matches_python"] is True

    monkeypatch.setattr(identity.socket, "gethostname", lambda: "other.local")
    with pytest.raises(ValueError, match="device_name|hostname"):
        identity.runtime_identity(frozen_config)

    monkeypatch.setattr(
        identity.socket, "gethostname", lambda: expected_environment["hostname"],
    )
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "not-" + expected_environment["conda_environment"])
    with pytest.raises(ValueError, match="conda_environment"):
        identity.runtime_identity(frozen_config)

    monkeypatch.setenv("CONDA_DEFAULT_ENV", expected_environment["conda_environment"])
    monkeypatch.setenv(
        "CONDA_PREFIX",
        "/tmp/not-the-running-python/envs/" + expected_environment["conda_environment"],
    )
    with pytest.raises(ValueError, match="conda_prefix_matches_python"):
        identity.runtime_identity(frozen_config)


def test_contract_byte_binding_rejects_later_committed_or_worktree_edits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    contract = repo / "EXPERIMENT_CONTRACT.md"
    contract.write_bytes(b"frozen contract\n")
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "add", contract.name), check=True)
    subprocess.run(
        (
            "git", "-C", str(repo), "-c", "user.name=exp103-test",
            "-c", "user.email=exp103@example.invalid", "commit", "-q", "-m", "freeze",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    identity._require_file_matches_commit(repo, contract, commit)

    contract.write_bytes(b"later contract\n")
    subprocess.run(("git", "-C", str(repo), "add", contract.name), check=True)
    subprocess.run(
        (
            "git", "-C", str(repo), "-c", "user.name=exp103-test",
            "-c", "user.email=exp103@example.invalid", "commit", "-q", "-m", "later",
        ),
        check=True,
    )
    with pytest.raises(ValueError, match="differs from source commit"):
        identity._require_file_matches_commit(repo, contract, commit)


def test_frozen_runtime_and_decoder_binary_identity(frozen_config):
    assert frozen_config["source_commit"] != "0" * 40
    assert frozen_config["source_tree_sha256"] != "0" * 64
    # Both canonical configs are frozen at the same v2 decoder source, so each
    # must match the machine that runs the formal suite in its own environment.
    actual = identity.runtime_identity(frozen_config)
    assert actual["source_tree_sha256"] == identity.source_tree_sha256()
    assert actual["decoder_binary_sha256"] == frozen_config["decoder_binary"]["sha256"]
    assert identity.decoder_binary_path().name.endswith(
        frozen_config["decoder_binary"]["filename_suffix"],
    )
