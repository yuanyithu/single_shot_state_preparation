import copy
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.expander_code.exp102.exp102_pipeline.registry import load_registry
from data.expander_code.exp103.exp103_pipeline.aggregate import (
    ARRAY_FIELDS,
    SCALAR_FIELDS,
    _blank_arrays,
)
from data.expander_code.exp103.exp103_pipeline.config import load_config
from data.expander_code.exp103.exp103_pipeline.crossing import (
    classify_final_crossing,
    simultaneous_bootstrap,
    wilson_interval,
)
from data.expander_code.exp103.exp103_pipeline.io import (
    arrays_sha256,
    canonical_json,
    sha256_json,
)
from data.expander_code.exp103.exp103_pipeline.seeds import derive_seed
from data.expander_code.exp103.exp103_pipeline.worker import RAW_FIELDS


CONFIG_PATH = Path(os.environ.get(
    "EXP103_TEST_CONFIG_PATH",
    REPO_ROOT / "data/expander_code/exp103/config/decoder_mc.v1.json",
))
REMOTE_CONFIG_PATH = (
    REPO_ROOT / "data/expander_code/exp103/config/decoder_mc.remote.v1.json"
)


@pytest.fixture(scope="session")
def frozen_config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def publication_config():
    return load_config(REMOTE_CONFIG_PATH)


@pytest.fixture(scope="session")
def registry_rows(frozen_config):
    registry = load_registry(REPO_ROOT / frozen_config["registry_path"])
    return {row["code_id"]: row for row in registry["codes"]}


@pytest.fixture(scope="session")
def tiny_css_model():
    # [[4, 2]] CSS with X stabilizer 0011 and logical-X basis 1100, 0010.
    h_z = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
    h_x = np.asarray([[0, 0, 1, 1]], dtype=np.uint8)
    logical_z = np.asarray([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    return SimpleNamespace(
        code_id="m03_c00",
        m=3,
        n=4,
        k=2,
        classical_distance=1,
        H_Z=h_z,
        H_X=h_x,
        H_Z_sparse=csr_matrix(h_z),
        logical_Z=logical_z,
    )


class TinySyndromeDecoder:
    """Deterministic matching-syndrome decoder for the tiny CSS fixture."""

    def __init__(self, pcm, **kwargs):
        self.pcm = pcm
        self.kwargs = kwargs
        self.converge = False
        self.iter = 0

    def decode(self, syndrome):
        correction = np.zeros(4, dtype=np.uint8)
        correction[0] = np.asarray(syndrome, dtype=np.uint8)[0]
        self.iter = 1
        return correction


def runtime_identity_for(config):
    return {
        "device_name": config["environment"]["device_name"],
        "hostname": config["environment"]["hostname"],
        "conda_environment": config["environment"]["conda_environment"],
        "conda_prefix_matches_python": True,
        "python_version": config["environment"]["python"],
        "numpy_version": config["environment"]["numpy"],
        "scipy_version": config["environment"]["scipy"],
        "ldpc_version": config["environment"]["ldpc"],
        "bplsd_binary_sha256": config["bplsd_binary"]["sha256"],
        "source_tree_sha256": config["source_tree_sha256"],
        "source_commit": config["source_commit"],
    }


@pytest.fixture
def install_tiny_runtime(monkeypatch, tiny_css_model):
    from data.expander_code.exp103.exp103_pipeline import replay, worker

    def install(decoder_class=TinySyndromeDecoder):
        monkeypatch.setattr(worker, "BpLsdDecoder", decoder_class)
        monkeypatch.setattr(replay, "BpLsdDecoder", decoder_class)
        monkeypatch.setattr(worker, "load_model", lambda _config, _code_id: tiny_css_model)
        monkeypatch.setattr(replay, "load_model", lambda _config, _code_id: tiny_css_model)
        monkeypatch.setattr(
            worker,
            "runtime_identity",
            lambda config, *args, **kwargs: runtime_identity_for(config),
        )
        monkeypatch.setattr(
            replay,
            "runtime_identity",
            lambda config, *args, **kwargs: runtime_identity_for(config),
        )
        return decoder_class

    return install


@pytest.fixture
def raw_factory(frozen_config, registry_rows):
    def make(code_id="m03_c00", p_token="0.02", shard_index=0, failures=0):
        row = registry_rows[code_id]
        trials = frozen_config["trials_per_shard"]
        labels = np.zeros((trials, row["k"]), dtype=np.uint8)
        labels[:failures, 0] = 1
        failure_flags = np.zeros(trials, dtype=np.bool_)
        failure_flags[:failures] = True
        seed = derive_seed(
            frozen_config, "measurement", code_id, p_token, shard_index,
        )
        stream_tag = f"{code_id}:{p_token}:{shard_index}".encode("ascii")
        raw = {
            "schema_version": "exp103.raw.v1",
            "status": "VALID",
            "invalid_reason": "",
            "exception_type": "",
            "exception_message": "",
            "experiment_id": "exp103.decoder_mc.v1",
            "code_id": code_id,
            "m": row["m"],
            "p_token": p_token,
            "p": float(p_token),
            "shard_index": int(shard_index),
            "planned_trials": trials,
            "completed_trials": trials,
            "seed": seed,
            "seed_namespace": frozen_config["namespaces"]["measurement"],
            "config_sha256": frozen_config["config_sha256"],
            "registry_sha256": frozen_config["registry_sha256"],
            "source_commit": frozen_config["source_commit"],
            "source_tree_sha256": frozen_config["source_tree_sha256"],
            "bplsd_binary_sha256": frozen_config["bplsd_binary"]["sha256"],
            "python_version": frozen_config["environment"]["python"],
            "numpy_version": frozen_config["environment"]["numpy"],
            "scipy_version": frozen_config["environment"]["scipy"],
            "ldpc_version": frozen_config["environment"]["ldpc"],
            "device_name": frozen_config["environment"]["device_name"],
            "hostname": frozen_config["environment"]["hostname"],
            "conda_environment": frozen_config["environment"]["conda_environment"],
            "conda_prefix_matches_python": True,
            "n": row["n"],
            "k": row["k"],
            "classical_distance": row["classical_distance"],
            "error_stream_sha256": hashlib.sha256(b"error:" + stream_tag).hexdigest(),
            "correction_stream_sha256": hashlib.sha256(b"correction:" + stream_tag).hexdigest(),
            "label_stream_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "failure_flags": failure_flags,
            "logical_labels": labels,
            "syndrome_match": np.ones(trials, dtype=np.bool_),
            "bp_converged": np.zeros(trials, dtype=np.bool_),
            "bp_iterations": np.ones(trials, dtype=np.int32),
        }
        assert set(raw) == RAW_FIELDS
        return raw

    return make


@pytest.fixture(scope="session")
def complete_aggregate_factory(publication_config, registry_rows):
    aggregate = _blank_arrays(registry_rows)
    trials = 10_000
    for m_index in range(6):
        for p_index in range(13):
            step = -300 if p_index < 6 else 300
            for local_code in range(8):
                code_index = 8 * m_index + local_code
                failures = 3000 + 50 * p_index + m_index * step + 2 * local_code
                rate = failures / trials
                aggregate["code_status"][code_index, p_index] = "REPORTABLE"
                aggregate["failure_counts"][code_index, p_index] = failures
                aggregate["trial_counts"][code_index, p_index] = trials
                aggregate["code_rates"][code_index, p_index] = rate
                low, high = wilson_interval(failures, trials)
                aggregate["wilson_low"][code_index, p_index] = low
                aggregate["wilson_high"][code_index, p_index] = high
                aggregate["bp_convergence_rate"][code_index, p_index] = 0.9
                aggregate["mean_bp_iterations"][code_index, p_index] = 2.0
                aggregate["syndrome_mismatch_rate"][code_index, p_index] = 0.0
                aggregate["mean_logical_weight"][code_index, p_index] = rate

            code_slice = slice(8 * m_index, 8 * (m_index + 1))
            rates = aggregate["code_rates"][code_slice, p_index]
            aggregate["m_status"][m_index, p_index] = "REPORTABLE"
            aggregate["primary_mean"][m_index, p_index] = rates.mean()
            aggregate["primary_median"][m_index, p_index] = np.median(rates)
            aggregate["fixed_panel_mc_se"][m_index, p_index] = (
                np.sqrt(np.sum(rates * (1.0 - rates) / trials)) / 8.0
            )
            aggregate["between_code_std"][m_index, p_index] = np.std(rates, ddof=1)
            aggregate["between_code_sem"][m_index, p_index] = (
                aggregate["between_code_std"][m_index, p_index] / np.sqrt(8.0)
            )

    failures_3d = aggregate["failure_counts"].reshape(6, 8, 13)
    trials_3d = aggregate["trial_counts"].reshape(6, 8, 13)
    stage1 = simultaneous_bootstrap(
        failures_3d, trials_3d, (0, 1, 2), publication_config, "stage1_m3_m5",
    )
    final = simultaneous_bootstrap(
        failures_3d, trials_3d, tuple(range(6)), publication_config, "final_m3_m8",
    )
    aggregate["stage1_primary_band_low"] = stage1["point_low"]
    aggregate["stage1_primary_band_high"] = stage1["point_high"]
    aggregate["stage1_delta35"] = stage1["endpoint"]
    aggregate["stage1_band_low"] = stage1["endpoint_low"]
    aggregate["stage1_band_high"] = stage1["endpoint_high"]
    aggregate["stage1_adjacent_delta"] = stage1["adjacent"]
    aggregate["stage1_adjacent_band_low"] = stage1["adjacent_low"]
    aggregate["stage1_adjacent_band_high"] = stage1["adjacent_high"]
    aggregate["primary_band_low"] = final["point_low"]
    aggregate["primary_band_high"] = final["point_high"]
    aggregate["delta38"] = final["endpoint"]
    aggregate["delta38_band_low"] = final["endpoint_low"]
    aggregate["delta38_band_high"] = final["endpoint_high"]
    aggregate["adjacent_delta"] = final["adjacent"]
    aggregate["adjacent_band_low"] = final["adjacent_low"]
    aggregate["adjacent_band_high"] = final["adjacent_high"]

    padded_adjacent = np.full((5, 13), np.nan)
    padded_low = np.full((5, 13), np.nan)
    padded_high = np.full((5, 13), np.nan)
    padded_adjacent[:2] = stage1["adjacent"]
    padded_low[:2] = stage1["adjacent_low"]
    padded_high[:2] = stage1["adjacent_high"]
    stage1_decision = classify_final_crossing(
        aggregate["p_values"], stage1["endpoint"], stage1["endpoint_low"],
        stage1["endpoint_high"], padded_adjacent, padded_low, padded_high,
    )
    final_decision = classify_final_crossing(
        aggregate["p_values"], final["endpoint"], final["endpoint_low"],
        final["endpoint_high"], final["adjacent"], final["adjacent_low"],
        final["adjacent_high"],
    )
    def make_replay_report(scope, m_values):
        replay_results = []
        for m in m_values:
            for code_index in range(8):
                code_id = f"m{m:02d}_c{code_index:02d}"
                for p_token in publication_config["p_tokens"]:
                    for shard_index in range(publication_config["shards_per_code_p"]):
                        tag = f"{code_id}:{p_token}:{shard_index}".encode("ascii")
                        replay_results.append({
                            "status": "PASS",
                            "reason": "",
                            "code_id": code_id,
                            "p_token": p_token,
                            "shard_index": shard_index,
                            "trials": publication_config["trials_per_shard"],
                            "replay_control_seed": derive_seed(
                                publication_config, "replay", code_id, p_token, shard_index,
                            ),
                            "raw_sha256": hashlib.sha256(b"raw:" + tag).hexdigest(),
                            "error_stream_sha256": hashlib.sha256(b"error:" + tag).hexdigest(),
                            "correction_stream_sha256": hashlib.sha256(b"correction:" + tag).hexdigest(),
                            "label_stream_sha256": hashlib.sha256(b"label:" + tag).hexdigest(),
                        })
        manifest_entries = [
            {
                "code_id": item["code_id"],
                "p_token": item["p_token"],
                "shard_index": item["shard_index"],
                "raw_sha256": item["raw_sha256"],
            }
            for item in replay_results
        ]
        manifest_entries.sort(key=lambda item: (
            item["code_id"], item["p_token"], item["shard_index"],
        ))
        return {
            "schema_version": "exp103.replay.v1",
            "config_sha256": publication_config["config_sha256"],
            "registry_sha256": publication_config["registry_sha256"],
            "source_commit": publication_config["source_commit"],
            "source_tree_sha256": publication_config["source_tree_sha256"],
            "bplsd_binary_sha256": publication_config["bplsd_binary"]["sha256"],
            "device_name": publication_config["environment"]["device_name"],
            "hostname": publication_config["environment"]["hostname"],
            "conda_environment": publication_config["environment"]["conda_environment"],
            "conda_prefix_matches_python": True,
            "scope": scope,
            "expected_shards": len(replay_results),
            "shards": len(replay_results),
            "raw_manifest_sha256": hashlib.sha256(
                canonical_json(manifest_entries).encode("ascii")
            ).hexdigest(),
            "status": "PASS",
            "results": replay_results,
        }

    replay_report = {
        "schema_version": "exp103.replay_bundle.v1",
        "stage1": make_replay_report("stage1", [3, 4, 5]),
        "stage2": make_replay_report("stage2", [6, 7, 8]),
    }
    combined_entries = []
    for scope in ("stage1", "stage2"):
        for item in replay_report[scope]["results"]:
            combined_entries.append({
                "code_id": item["code_id"], "p_token": item["p_token"],
                "shard_index": item["shard_index"], "raw_sha256": item["raw_sha256"],
            })
    combined_entries.sort(key=lambda item: (
        item["code_id"], item["p_token"], item["shard_index"],
    ))
    raw_manifest_sha256 = hashlib.sha256(
        canonical_json(combined_entries).encode("ascii")
    ).hexdigest()
    replay_report["raw_manifest_sha256"] = raw_manifest_sha256
    aggregate.update({
            "schema_version": "exp103.aggregate.v1",
            "experiment_id": "exp103.decoder_mc.v1",
            "config_sha256": publication_config["config_sha256"],
            "registry_sha256": publication_config["registry_sha256"],
            "source_commit": publication_config["source_commit"],
            "source_tree_sha256": publication_config["source_tree_sha256"],
            "bplsd_binary_sha256": publication_config["bplsd_binary"]["sha256"],
            "overall_status": "COMPLETE",
            "terminal_status": final_decision["status"],
            "crossing_bracket_low": final_decision["bracket"][0],
            "crossing_bracket_high": final_decision["bracket"][1],
            "compatible_triple_json": canonical_json(final_decision["compatible_triple"]),
            "stage1_status": "STAGE1_RESTRICTED_" + stage1_decision["status"].removeprefix("EXP103_"),
            "stage1_bracket_low": stage1_decision["bracket"][0],
            "stage1_bracket_high": stage1_decision["bracket"][1],
            "bootstrap_half_width": final["half_width"],
            "stage1_bootstrap_half_width": stage1["half_width"],
            "stage1_compatible_triple_json": canonical_json(stage1_decision["compatible_triple"]),
            "unexpected_raw_errors_json": "[]",
            "replay_status": "PASS",
            "replay_scope": "final_combined",
            "replay_report_sha256": sha256_json(replay_report),
            "raw_manifest_sha256": raw_manifest_sha256,
            "replay_report_json": canonical_json(replay_report),
        })
    aggregate["payload_sha256"] = arrays_sha256(aggregate, ARRAY_FIELDS)
    assert set(aggregate) == set(ARRAY_FIELDS) | set(SCALAR_FIELDS)

    def make():
        return {
            key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for key, value in aggregate.items()
        }

    return make


@pytest.fixture
def clone_payload():
    def clone(payload):
        return {
            key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for key, value in payload.items()
        }

    return clone


@pytest.fixture
def rehash_aggregate():
    def rehash(aggregate):
        aggregate["payload_sha256"] = arrays_sha256(aggregate, ARRAY_FIELDS)
        return aggregate

    return rehash
