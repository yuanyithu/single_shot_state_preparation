"""The compute-host cost benchmark, exercised before it ever reaches that host.

This command exists to break a circularity exp105 could not: its only path to
remote costs was `preflight`, which already needs the frozen production plan, and
the allocation rule that produces that plan needs the costs. exp105 guessed with
macmini numbers and its resource gate blocked at 5,368 core-hours against a cap
of 800.

Being new code that runs exactly once, on a machine reachable only through a
deployment round trip, it is worth testing here rather than discovering a typo
after `git archive`. The decoder work is real; only the host identity is
substituted, because a macmini cannot claim to be the compute host.
"""

import json

import pytest

from data.expander_code.exp106.exp106_pipeline import preflight as preflight_module
from data.expander_code.exp106.exp106_pipeline import remote_cli
from data.expander_code.exp106.exp106_pipeline.config import (
    COMPUTE_HOST,
    COST_BENCHMARK_TRIALS,
    M_VALUES,
    load_config,
)
from data.expander_code.exp106.exp106_pipeline.ensemble import (
    load_registry,
    registry_index,
)
from data.expander_code.exp106.exp106_pipeline.io import sha256_json


PILOT_REMOTE_CONFIG = (
    "data/expander_code/exp106/config/noisy_mc.pilot.remote.v1.json"
)


@pytest.fixture(scope="module")
def pilot_remote_config():
    return load_config(PILOT_REMOTE_CONFIG)


@pytest.fixture(scope="module")
def pilot_rows(pilot_remote_config):
    return registry_index(load_registry(pilot_remote_config["registry_path"]))


@pytest.fixture(scope="module")
def report(pilot_remote_config, pilot_rows):
    """One real benchmark, shared by every assertion below.

    The decode work is genuine and costs about a minute, so it runs once. Only
    the host identity is substituted: `runtime_identity` compares hostname,
    conda prefix and decoder binary hash against the config, and a macmini is
    none of those things. Everything downstream -- the model build, the decoder,
    the trials, the timing -- is the real path.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            preflight_module, "runtime_identity", lambda config, **kwargs: {},
        )
        return remote_cli.run_remote_cost_benchmark(pilot_remote_config, pilot_rows)


def test_the_benchmark_covers_every_size_and_the_frozen_grid_points(
    pilot_remote_config, report,
):
    assert report["schema_version"] == "exp106.cost_benchmark.v1"
    assert report["outcome_blind"] is True
    assert report["q_token"] == "0.01"
    assert report["trials_per_batch"] == COST_BENCHMARK_TRIALS

    # first, middle and last of the pilot grid, chosen by position because the
    # production grid does not exist yet
    tokens = list(pilot_remote_config["p_tokens"])
    assert report["benchmark_p_tokens"] == [
        tokens[0], tokens[len(tokens) // 2], tokens[-1],
    ]

    assert set(report["kappa_seconds_upper"]) == {str(m) for m in M_VALUES}
    assert set(report["c_seconds_upper"]) == {str(m) for m in M_VALUES}
    expected = len(M_VALUES) * len(report["code_indices"]) * 3
    assert len(report["tasks"]) == expected


def test_costs_rise_with_the_code_size(report):
    """A sanity floor: if these came out flat, the benchmark timed the wrong thing."""
    per_trial = [report["c_seconds_upper"][str(m)] for m in M_VALUES]
    kappa = [report["kappa_seconds_upper"][str(m)] for m in M_VALUES]
    assert all(value > 0 for value in per_trial + kappa)
    assert per_trial == sorted(per_trial)
    assert kappa == sorted(kappa)
    assert per_trial[-1] > 10 * per_trial[0], (
        "an m=8 trial costs tens of times an m=3 trial; a flat profile means "
        "the timing is dominated by something other than the decode"
    )


def test_the_report_is_self_verifying(report):
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == sha256_json(core)
    # and it is the shape `pilot allocate` will read
    assert json.loads(json.dumps(report, sort_keys=True))["device"] == COMPUTE_HOST


def test_the_benchmark_refuses_a_production_config(pilot_rows):
    """The pilot remote config is the only one this command accepts."""
    from data.expander_code.exp106.exp106_pipeline.config import (
        ProductionPlanNotFrozen,
    )

    with pytest.raises((ValueError, FileNotFoundError, ProductionPlanNotFrozen)):
        remote_cli.run_remote_cost_benchmark(
            load_config("data/expander_code/exp106/config/noisy_mc.pilot.v1.json"),
            pilot_rows,
        )


def test_no_outcome_leaves_the_benchmark(report):
    """Outcome-blind means nothing about which trials failed is reachable.

    `benchmark_task` decodes and replays and compares, then discards the
    comparison. If a failure count, a label or a verdict ever appeared in this
    report, the resource gate would become a place to look at results before the
    plan is frozen.

    Checked on the field names rather than on the serialized text: the
    experiment id is `exp106.noisy_syndrome_mc.q001.v1`, so a substring scan
    would flag the report's own identity.
    """
    expected_top_level = {
        "schema_version", "experiment_id", "config_sha256", "registry_sha256",
        "source_commit", "source_tree_sha256", "decoder_binary_sha256",
        "device", "hostname", "q_token", "outcome_blind", "benchmark_p_tokens",
        "code_indices", "trials_per_batch", "seed_namespace",
        "kappa_seconds_upper", "c_seconds_upper", "peak_rss_gib", "tasks",
        "report_sha256",
    }
    assert set(report) == expected_top_level

    forbidden = ("fail", "label", "verdict", "logical", "correction", "residual")
    for task in report["tasks"]:
        for key in task:
            assert not any(word in key.lower() for word in forbidden), (
                f"benchmark task exposes {key!r}"
            )
        # everything a task reports is a duration, a count or an identifier
        for key, value in task.items():
            assert isinstance(value, (int, float, str)), (key, type(value))
