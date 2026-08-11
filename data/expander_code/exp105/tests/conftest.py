import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.expander_code.exp105.exp105_pipeline import replay as replay_module
from data.expander_code.exp105.exp105_pipeline import worker as worker_module
from data.expander_code.exp105.exp105_pipeline.aggregate import (
    ARRAY_FIELDS,
    DISTANCE_STRATA,
    SCALAR_FIELDS,
    _blank_arrays,
    panel_layout,
    qtop_lower_bound,
)
from data.expander_code.exp105.exp105_pipeline.config import load_config
from data.expander_code.exp105.exp105_pipeline.crossing import (
    classify_crossing,
    cluster_bootstrap,
    crossing_location,
    wilson_interval,
)
from data.expander_code.exp105.exp105_pipeline.ensemble import (
    load_registry,
    registry_index,
)
from data.expander_code.exp105.exp105_pipeline.io import arrays_sha256, canonical_json


CONFIG_PATH = Path(os.environ.get(
    "EXP105_TEST_CONFIG_PATH",
    REPO_ROOT / "data/expander_code/exp105/config/noisy_mc.pilot.v1.json",
))
PILOT_CONFIG_PATH = (
    REPO_ROOT / "data/expander_code/exp105/config/noisy_mc.pilot.v1.json"
)
EXP104_REGISTRY_PATH = (
    REPO_ROOT / "data/expander_code/exp104/config/ensemble_registry.v1.json"
)
# Two real codes per size: small enough for a unit test, but the physics, the
# decoder and the seeds are the frozen ones.
SLICE_CODES = 2


@pytest.fixture(scope="session")
def frozen_config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def pilot_config():
    """Always the pilot config, whichever one the environment points at."""
    return load_config(PILOT_CONFIG_PATH)


@pytest.fixture(scope="session")
def registry(frozen_config):
    return load_registry(REPO_ROOT / frozen_config["registry_path"])


@pytest.fixture(scope="session")
def registry_rows(registry):
    return registry_index(registry)


@pytest.fixture(scope="session")
def exp104_registry_rows():
    from data.expander_code.exp104.exp104_pipeline.ensemble import (
        load_registry as load_exp104_registry,
    )

    registry = load_exp104_registry(EXP104_REGISTRY_PATH)
    return {row["code_id"]: row for row in registry["codes"]}


@pytest.fixture
def short_block(monkeypatch):
    """Shrink one task to two codes so the real decoder path stays testable."""
    def indices(config, m, block_index):
        return list(range(SLICE_CODES))

    monkeypatch.setattr(worker_module, "block_code_indices", indices)
    monkeypatch.setattr(replay_module, "block_code_indices", indices)
    return SLICE_CODES


@pytest.fixture(scope="session")
def complete_aggregate_factory(frozen_config, registry):
    """A COMPLETE aggregate built directly, with a real certified crossing.

    Built on the pilot plan, whose two sizes are exactly the primary pair, so
    the crossing classifier is exercised on the panel it will actually decide.
    """
    arrays = _blank_arrays(frozen_config)
    m_values, codes_per_m, trials_by_m, offsets, _ = panel_layout(frozen_config)
    n_p = len(frozen_config["p_tokens"])
    distance_by_code = {
        row["code_id"]: row["classical_distance"] for row in registry["codes"]
    }
    rng = np.random.Generator(np.random.PCG64(4242))

    for m_index, m in enumerate(m_values):
        trials = trials_by_m[m]
        codes = codes_per_m[m]
        code_slice = slice(offsets[m], offsets[m] + codes)
        for local in range(codes):
            arrays["classical_distance"][offsets[m] + local] = distance_by_code[
                f"m{m:02d}_c{local:06d}"
            ]
        for p_index in range(n_p):
            # A curve that is lower for large m below the centre and higher
            # above it, so the frozen classifier has a genuine crossing to
            # certify rather than a synthetic constant.
            centre = 0.03
            p = arrays["p_values"][p_index]
            base = 0.10 + 4.0 * (p - 0.005)
            slope = (p - centre) * 12.0
            rate = float(np.clip(base + slope * (m - 3) / 5.0, 0.01, 0.99))
            counts = rng.binomial(trials, rate, size=codes)
            arrays["code_status"][code_slice, p_index] = "REPORTABLE"
            arrays["failure_counts"][code_slice, p_index] = counts
            arrays["trial_counts"][code_slice, p_index] = trials
            arrays["code_rates"][code_slice, p_index] = counts / trials
            low = np.empty(codes)
            high = np.empty(codes)
            for index, value in enumerate(counts):
                low[index], high[index] = wilson_interval(int(value), trials)
            arrays["wilson_low"][code_slice, p_index] = low
            arrays["wilson_high"][code_slice, p_index] = high
            arrays["bp_convergence_rate"][code_slice, p_index] = 0.5
            arrays["mean_bp_iterations"][code_slice, p_index] = 12.0
            arrays["readout_mismatch_rate"][code_slice, p_index] = 0.0
            arrays["mean_logical_weight"][code_slice, p_index] = counts / trials

            total = int(counts.sum())
            total_trials = codes * trials
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
            arrays["cluster_se"][m_index, p_index] = std / np.sqrt(codes)

    distances = arrays["classical_distance"]
    for m_index, m in enumerate(m_values):
        trials = trials_by_m[m]
        code_slice = slice(offsets[m], offsets[m] + codes_per_m[m])
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
        arrays["failure_counts"][offsets[m]:offsets[m] + codes_per_m[m]]
        for m in m_values
    ]
    trials_list = [trials_by_m[m] for m in m_values]
    bootstrap = cluster_bootstrap(
        failures_by_m, trials_list, frozen_config, "final_m3_m8",
    )
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
        "schema_version": "exp105.aggregate.v1",
        "experiment_id": "exp105.noisy_syndrome_mc.v1",
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
        "codes_per_m_json": canonical_json(
            {str(m): codes_per_m[m] for m in m_values}
        ),
        "trials_per_code_p_json": canonical_json(
            {str(m): trials_by_m[m] for m in m_values}
        ),
        "q_token": str(frozen_config["q_token"]),
        "qtop_lower_bound_json": canonical_json(
            qtop_lower_bound(m_values, arrays["p_values"], arrays["primary_mean"])
        ),
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


@pytest.fixture(scope="session")
def foreign_config(frozen_config):
    """A valid config that is definitely not the one under test.

    Only `source_commit` differs, which is enough to change the config hash and
    is legal for a local phase, so the loader is tested against a config that
    passes validation rather than one that fails it for an unrelated reason.
    """
    from data.expander_code.exp105.exp105_pipeline.config import ensure_config

    payload = {
        key: copy.deepcopy(value) for key, value in frozen_config.items()
        if key not in {"config_sha256", "config_path"}
    }
    payload["source_commit"] = "f" * 40
    return ensure_config(payload)


@pytest.fixture
def rehash_aggregate():
    def rehash(aggregate):
        aggregate["payload_sha256"] = arrays_sha256(aggregate, ARRAY_FIELDS)
        return aggregate

    return rehash
