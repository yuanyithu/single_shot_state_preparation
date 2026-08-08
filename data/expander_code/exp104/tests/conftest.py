import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.expander_code.exp104.exp104_pipeline import replay as replay_module
from data.expander_code.exp104.exp104_pipeline import worker as worker_module
from data.expander_code.exp104.exp104_pipeline.aggregate import (
    ARRAY_FIELDS,
    DISTANCE_STRATA,
    SCALAR_FIELDS,
    _blank_arrays,
)
from data.expander_code.exp104.exp104_pipeline.config import (
    CODES_PER_M,
    M_VALUES,
    load_config,
)
from data.expander_code.exp104.exp104_pipeline.crossing import (
    classify_crossing,
    cluster_bootstrap,
    crossing_location,
    wilson_interval,
)
from data.expander_code.exp104.exp104_pipeline.ensemble import (
    load_registry,
    registry_index,
)
from data.expander_code.exp104.exp104_pipeline.io import arrays_sha256, canonical_json


CONFIG_PATH = Path(os.environ.get(
    "EXP104_TEST_CONFIG_PATH",
    REPO_ROOT / "data/expander_code/exp104/config/ensemble_mc.v1.json",
))
REMOTE_CONFIG_PATH = (
    REPO_ROOT / "data/expander_code/exp104/config/ensemble_mc.remote.v1.json"
)
# Three real m=3 codes: small enough for a unit test, but the physics, the
# decoder and the seeds are the frozen production ones.
SLICE_CODES = 3


@pytest.fixture(scope="session")
def frozen_config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def remote_config():
    return load_config(REMOTE_CONFIG_PATH)


@pytest.fixture(scope="session")
def registry(frozen_config):
    return load_registry(REPO_ROOT / frozen_config["registry_path"])


@pytest.fixture(scope="session")
def registry_rows(registry):
    return registry_index(registry)


@pytest.fixture
def short_block(monkeypatch):
    """Shrink one task to three codes so the real decoder path stays testable."""
    def indices(m, block_index):
        return list(range(SLICE_CODES))

    monkeypatch.setattr(worker_module, "block_code_indices", indices)
    monkeypatch.setattr(replay_module, "block_code_indices", indices)
    return SLICE_CODES


@pytest.fixture(scope="session")
def complete_aggregate_factory(frozen_config, registry):
    """A COMPLETE aggregate built directly, with a real certified crossing."""
    arrays = _blank_arrays(frozen_config)
    trials = int(frozen_config["trials_per_code_p"])
    n_p = len(frozen_config["p_tokens"])
    distance_by_code = {row["code_id"]: row["classical_distance"] for row in registry["codes"]}
    rng = np.random.Generator(np.random.PCG64(4242))

    for m_index, m in enumerate(M_VALUES):
        for local in range(CODES_PER_M):
            slot = m_index * CODES_PER_M + local
            arrays["classical_distance"][slot] = distance_by_code[
                f"m{m:02d}_c{local:05d}"
            ]
        for p_index in range(n_p):
            # A curve that is lower for large m below p=0.06 and higher above,
            # so the frozen classifier has a genuine crossing to certify.
            centre = 0.06
            p = arrays["p_values"][p_index]
            base = 0.10 + 4.0 * (p - 0.02)
            slope = (p - centre) * 6.0
            rate = float(np.clip(base + slope * (m - 3) / 5.0, 0.01, 0.99))
            counts = rng.binomial(trials, rate, size=CODES_PER_M)
            code_slice = slice(m_index * CODES_PER_M, (m_index + 1) * CODES_PER_M)
            arrays["code_status"][code_slice, p_index] = "REPORTABLE"
            arrays["failure_counts"][code_slice, p_index] = counts
            arrays["trial_counts"][code_slice, p_index] = trials
            arrays["code_rates"][code_slice, p_index] = counts / trials
            low = np.empty(CODES_PER_M)
            high = np.empty(CODES_PER_M)
            for index, value in enumerate(counts):
                low[index], high[index] = wilson_interval(int(value), trials)
            arrays["wilson_low"][code_slice, p_index] = low
            arrays["wilson_high"][code_slice, p_index] = high
            arrays["bp_convergence_rate"][code_slice, p_index] = 0.5
            arrays["mean_bp_iterations"][code_slice, p_index] = 12.0
            arrays["syndrome_mismatch_rate"][code_slice, p_index] = 0.0
            arrays["mean_logical_weight"][code_slice, p_index] = counts / trials

            total = int(counts.sum())
            total_trials = CODES_PER_M * trials
            mean = total / total_trials
            arrays["m_status"][m_index, p_index] = "REPORTABLE"
            arrays["primary_failures"][m_index, p_index] = total
            arrays["primary_trials"][m_index, p_index] = total_trials
            arrays["primary_mean"][m_index, p_index] = mean
            arrays["pooled_binomial_se"][m_index, p_index] = np.sqrt(
                mean * (1.0 - mean) / total_trials
            )
            rates = counts / trials
            std = float(np.std(rates, ddof=1))
            arrays["between_code_std"][m_index, p_index] = std
            arrays["cluster_se"][m_index, p_index] = std / np.sqrt(CODES_PER_M)

    distances = arrays["classical_distance"]
    for m_index in range(len(M_VALUES)):
        code_slice = slice(m_index * CODES_PER_M, (m_index + 1) * CODES_PER_M)
        block_d = distances[code_slice]
        block_counts = arrays["failure_counts"][code_slice]
        block_status = arrays["code_status"][code_slice]
        for d_index, distance in enumerate(DISTANCE_STRATA):
            member = block_d == distance
            arrays["strata_code_counts"][m_index, d_index] = int(member.sum())
            if not member.any():
                continue
            eligible = member[:, None] & (block_status == "REPORTABLE")
            failures = np.where(eligible, block_counts, 0).sum(axis=0)
            stratum_trials = eligible.sum(axis=0) * trials
            arrays["strata_failures"][m_index, d_index] = failures
            arrays["strata_trials"][m_index, d_index] = stratum_trials
            with np.errstate(invalid="ignore", divide="ignore"):
                arrays["strata_rate"][m_index, d_index] = np.where(
                    stratum_trials > 0, failures / stratum_trials, np.nan,
                )

    failures_by_m = [
        arrays["failure_counts"][m_index * CODES_PER_M:(m_index + 1) * CODES_PER_M]
        for m_index in range(len(M_VALUES))
    ]
    bootstrap = cluster_bootstrap(failures_by_m, trials, frozen_config, "final_m3_m8")
    arrays["primary_band_low"] = bootstrap["point_low"]
    arrays["primary_band_high"] = bootstrap["point_high"]
    arrays["delta38"] = bootstrap["endpoint"]
    arrays["delta38_band_low"] = bootstrap["endpoint_low"]
    arrays["delta38_band_high"] = bootstrap["endpoint_high"]
    arrays["adjacent_delta"] = bootstrap["adjacent"]
    arrays["adjacent_band_low"] = bootstrap["adjacent_low"]
    arrays["adjacent_band_high"] = bootstrap["adjacent_high"]
    decision = classify_crossing(
        arrays["p_values"], bootstrap["endpoint"],
        bootstrap["endpoint_low"], bootstrap["endpoint_high"],
    )
    location = crossing_location(
        arrays["p_values"], bootstrap["endpoint"],
        bootstrap["endpoint_replicates"], decision,
    )
    arrays.update({
        "schema_version": "exp104.aggregate.v1",
        "experiment_id": "exp104.ensemble_mc.v1",
        "config_sha256": frozen_config["config_sha256"],
        "registry_sha256": frozen_config["registry_sha256"],
        "source_commit": frozen_config["source_commit"],
        "source_tree_sha256": frozen_config["source_tree_sha256"],
        "decoder_binary_sha256": frozen_config["decoder_binary"]["sha256"],
        "overall_status": "COMPLETE",
        "terminal_status": decision["status"],
        "crossing_bracket_low": decision["bracket"][0],
        "crossing_bracket_high": decision["bracket"][1],
        "certified_negative_p_json": canonical_json(decision["certified_negative_p"]),
        "certified_positive_p_json": canonical_json(decision["certified_positive_p"]),
        "bootstrap_half_width": bootstrap["half_width"],
        "p_cross": location["p_cross"],
        "p_cross_low": location["p_cross_low"],
        "p_cross_high": location["p_cross_high"],
        "p_cross_defined_fraction": location["defined_fraction"],
        "p_cross_reason": location["reason"],
        "codes_per_m": CODES_PER_M,
        "trials_per_code_p": trials,
        "replay_status": "PASS",
        "replay_scope": "committed_subsample",
        "replay_report_sha256": "b" * 64,
        "raw_manifest_sha256": "c" * 64,
        "replay_report_json": canonical_json({"status": "PASS"}),
        "unexpected_raw_errors_json": "[]",
    })
    arrays["payload_sha256"] = arrays_sha256(arrays, ARRAY_FIELDS)
    assert set(arrays) == set(ARRAY_FIELDS) | set(SCALAR_FIELDS)

    def make():
        return {
            key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for key, value in arrays.items()
        }

    return make


@pytest.fixture
def rehash_aggregate():
    def rehash(aggregate):
        aggregate["payload_sha256"] = arrays_sha256(aggregate, ARRAY_FIELDS)
        return aggregate

    return rehash
