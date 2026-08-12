"""The section 6 freezing rules are arithmetic, so they can be checked now.

Running the pilot is meant to be an *evaluation*, not a decision: given measured
inputs the grid and the allocation follow with no free parameters. That is only
true if the functions behave the same way for every input they might see, which
is what this file pins down -- before the pilot runs, so that no result can be
argued into existence afterwards.

exp105 shipped these rules untested and had to patch the allocation rule in
flight when the pilot handed it a degenerate input it had never been run on.
"""

import json
import math

import pytest

from data.expander_code.exp106.exp106_pipeline.config import (
    COMPUTE_HOST,
    DIAGNOSTIC_M_VALUES,
    FALLBACK_P_TOKENS,
    GENERATION_BUDGET_CORE_HOURS,
    M_VALUES,
    PILOT_P_TOKENS,
    PRIMARY_M_VALUES,
    PRODUCTION_GRID_CLIP,
    PRODUCTION_GRID_POINTS,
    TARGET_TASK_SECONDS,
    TRIALS_PER_CODE_P_RANGE,
)
from data.expander_code.exp106.exp106_pipeline.pilot import (
    PILOT_SCHEMA,
    allocation_rule,
    anchor_index,
    block_size_rule,
    effective_sigma,
    grid_rule,
    main,
)


PILOT_P = [float(token) for token in PILOT_P_TOKENS]

# Shaped like the nd-3 numbers exp105's Validation 004 measured, scaled up by
# about a factor two because exp106's grid lives entirely in the expensive half
# of the p range. Invented numbers would test nothing about the real regime.
ND3_KAPPA = {3: 0.186, 4: 0.420, 5: 0.980, 6: 2.050, 7: 4.520, 8: 7.722}
ND3_C = {3: 0.19, 4: 0.61, 5: 1.50, 6: 3.10, 7: 5.75, 8: 9.76}


def _measurements(sigma_c_3, sigma_c_8, rate_3=0.2, rate_8=0.3):
    points = len(PILOT_P)
    return {
        3: {
            "sigma_c": [sigma_c_3] * points,
            "sigma_w2": [rate_3 * (1 - rate_3)] * points,
            "pooled_rate": [rate_3] * points,
        },
        8: {
            "sigma_c": [sigma_c_8] * points,
            "sigma_w2": [rate_8 * (1 - rate_8)] * points,
            "pooled_rate": [rate_8] * points,
        },
    }


# ---------------------------------------------------------------------------
# Grid rule
# ---------------------------------------------------------------------------

def test_no_sign_change_falls_back_to_the_frozen_window():
    tokens, bracket, reason = grid_rule(PILOT_P, [0.05] * len(PILOT_P))
    assert tokens == FALLBACK_P_TOKENS
    assert bracket is None
    assert reason == "no_sign_change_fallback_grid"


def test_the_fallback_covers_the_window_exp104_measured_negative():
    """The fallback is the claim a no-crossing terminal has to support.

    exp104 certified `Delta38 < 0` at p = 0.03, 0.04 and 0.05 with q = 0. If
    exp106 reports no crossing, it must be able to say so across that window,
    which means the window has to be inside the grid it measured.
    """
    values = [float(token) for token in FALLBACK_P_TOKENS]
    assert min(values) <= 0.03 and max(values) >= 0.055
    for certified_negative_at_q0 in (0.03, 0.04, 0.05):
        assert min(values) < certified_negative_at_q0 < max(values)


def test_a_sign_change_brackets_it_with_a_uniform_grid():
    delta = [-1.0] * len(PILOT_P)
    change = PILOT_P.index(0.035)
    for index in range(change, len(delta)):
        delta[index] = 1.0
    tokens, bracket, reason = grid_rule(PILOT_P, delta)

    assert reason == "pilot_bracket"
    assert bracket == (0.03, 0.035)
    assert len(tokens) == PRODUCTION_GRID_POINTS
    values = [float(token) for token in tokens]
    assert values == sorted(values)
    # [p_lo - 2h, p_hi + 2h] with h = 0.005
    assert values[0] == pytest.approx(0.02)
    assert values[-1] == pytest.approx(0.045)
    # Uniform up to the frozen four-decimal rounding, which is the only thing
    # that perturbs the spacing: every point is within half a unit in the last
    # place of the exact linear spacing.
    step = (values[-1] - values[0]) / (len(values) - 1)
    for index, value in enumerate(values):
        assert abs(value - (values[0] + index * step)) <= 5e-5


def test_the_innermost_sign_change_wins():
    """Two crossings must give the first negative-to-positive one, not the last."""
    delta = [-1.0, 1.0, -1.0, 1.0] + [1.0] * (len(PILOT_P) - 4)
    _, bracket, reason = grid_rule(PILOT_P, delta)
    assert reason == "pilot_bracket"
    assert bracket == (PILOT_P[0], PILOT_P[1])


def test_the_grid_is_clipped_and_deduplicated():
    low, high = (float(value) for value in PRODUCTION_GRID_CLIP)
    tokens, _, _ = grid_rule([0.0006, 0.0007], [-1.0, 1.0])
    values = [float(token) for token in tokens]
    assert len(set(tokens)) == len(tokens)
    assert min(values) >= low and max(values) <= high


def test_a_positive_to_negative_change_is_not_a_bracket():
    """The contract's direction is negative first. The reverse is not a crossing."""
    delta = [1.0] * 7 + [-1.0] * (len(PILOT_P) - 7)
    _, bracket, reason = grid_rule(PILOT_P, delta)
    assert reason == "no_sign_change_fallback_grid"
    assert bracket is None


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------

def test_the_anchor_is_the_pilot_point_nearest_the_bracket():
    index = anchor_index(PILOT_P, FALLBACK_P_TOKENS, (0.03, 0.035))
    assert PILOT_P[index] in (0.03, 0.035)


def test_without_a_bracket_the_anchor_is_the_geometric_centre():
    index = anchor_index(PILOT_P, FALLBACK_P_TOKENS, None)
    values = [float(token) for token in FALLBACK_P_TOKENS]
    centre = math.exp(sum(math.log(v) for v in values) / len(values))
    assert PILOT_P[index] == min(PILOT_P, key=lambda p: abs(p - centre))
    # Geometric, not arithmetic: the fallback spans more than a decade, and the
    # arithmetic centre would sit at the expensive end.
    assert centre < sum(values) / len(values)


# ---------------------------------------------------------------------------
# Effective sigma and the allocation rule
# ---------------------------------------------------------------------------

def test_effective_sigma_reduces_to_each_regime():
    # between-code spread dominates
    assert effective_sigma(0.3, 1e-9, 6) == pytest.approx(0.3, rel=1e-6)
    # shot noise dominates -- the case that broke exp105's rule
    assert effective_sigma(0.0, 0.24, 6) == pytest.approx(math.sqrt(0.24 / 6))
    assert effective_sigma(0.0, 0.24, 6) > 0.0
    # monotone in trials
    assert effective_sigma(0.1, 0.24, 3) > effective_sigma(0.1, 0.24, 6)


def test_the_allocation_rule_survives_a_vanishing_between_code_spread():
    """The exact input that forced exp105 to patch this rule mid-flight."""
    codes, trials, unit_cost, s_effective = allocation_rule(
        _measurements(0.0, 0.0), ND3_KAPPA, ND3_C, len(FALLBACK_P_TOKENS), 5,
    )
    assert set(codes) == set(M_VALUES)
    assert all(value > 0 for value in codes.values())
    assert all(s > 0 for s in s_effective.values()), "no degenerate 0/0 split"
    low, high = TRIALS_PER_CODE_P_RANGE
    assert all(trials[m] == high for m in M_VALUES), (
        "with no between-code spread there is nothing to buy by spreading the "
        "same trials over more codes, so trials go to the cap"
    )
    assert all(unit_cost[m] > 0 for m in M_VALUES)


def test_the_primary_split_minimises_the_contrast_variance():
    """C_3/C_8 = (s_3/s_8) sqrt(u_8/u_3) is the constrained optimum, so perturbing
    the split at fixed cost must make Var(Delta38) larger, not smaller."""
    codes, trials, unit_cost, s = allocation_rule(
        _measurements(0.08, 0.12), ND3_KAPPA, ND3_C, len(FALLBACK_P_TOKENS), 5,
    )
    a, b = PRIMARY_M_VALUES
    budget = codes[a] * unit_cost[a] + codes[b] * unit_cost[b]

    def variance(codes_a):
        codes_b = (budget - codes_a * unit_cost[a]) / unit_cost[b]
        return s[a] ** 2 / codes_a + s[b] ** 2 / codes_b

    best = variance(codes[a])
    for shift in (0.85, 0.95, 1.05, 1.15):
        assert variance(codes[a] * shift) >= best * (1 - 1e-9)


def test_larger_codes_get_smaller_panels_and_the_budget_is_respected():
    codes, trials, unit_cost, _ = allocation_rule(
        _measurements(0.08, 0.12), ND3_KAPPA, ND3_C, len(FALLBACK_P_TOKENS), 5,
    )
    assert codes[3] > codes[8], "an m=8 code costs about fifty times an m=3 code"
    for smaller, larger in zip(DIAGNOSTIC_M_VALUES, DIAGNOSTIC_M_VALUES[1:]):
        assert codes[smaller] > codes[larger]
    spent = sum(codes[m] * unit_cost[m] for m in M_VALUES)
    assert spent <= GENERATION_BUDGET_CORE_HOURS * 3600.0 * 1.001


def test_diagnostic_sizes_get_their_frozen_share():
    codes, _, unit_cost, _ = allocation_rule(
        _measurements(0.08, 0.12), ND3_KAPPA, ND3_C, len(FALLBACK_P_TOKENS), 5,
    )
    budget = GENERATION_BUDGET_CORE_HOURS * 3600.0
    for m in DIAGNOSTIC_M_VALUES:
        assert codes[m] * unit_cost[m] == pytest.approx(0.06 * budget, rel=0.01)


# ---------------------------------------------------------------------------
# Block size
# ---------------------------------------------------------------------------

def test_block_sizes_keep_a_task_under_the_frozen_target():
    _, _, unit_cost, _ = allocation_rule(
        _measurements(0.08, 0.12), ND3_KAPPA, ND3_C, len(FALLBACK_P_TOKENS), 5,
    )
    blocks = block_size_rule(unit_cost)
    for m in M_VALUES:
        assert blocks[m] >= 1
        if blocks[m] > 1:
            assert blocks[m] * unit_cost[m] <= TARGET_TASK_SECONDS
            assert (blocks[m] + 1) * unit_cost[m] > TARGET_TASK_SECONDS
    assert blocks[3] >= blocks[8], "cheaper codes belong in bigger blocks"


def test_a_code_more_expensive_than_the_target_still_gets_one_per_task():
    assert block_size_rule({8: 10 * TARGET_TASK_SECONDS}) == {8: 1}


# ---------------------------------------------------------------------------
# The allocate command
# ---------------------------------------------------------------------------

def _write_report(path, payload):
    from data.expander_code.exp106.exp106_pipeline.io import atomic_json, sha256_json

    atomic_json(path, dict(payload, report_sha256=sha256_json(payload)))


def _pilot_plan(**overrides):
    points = len(PILOT_P_TOKENS)
    plan = {
        "schema_version": PILOT_SCHEMA,
        "q_token": "0.01",
        "p_tokens": list(PILOT_P_TOKENS),
        "production_grid_tokens": list(FALLBACK_P_TOKENS),
        "pilot_bracket": None,
        "grid_rule_reason": "no_sign_change_fallback_grid",
        "sigma_c": {"3": [0.08] * points, "8": [0.12] * points},
        "sigma_w2": {"3": [0.16] * points, "8": [0.21] * points},
        "pooled_rate": {"3": [0.2] * points, "8": [0.3] * points},
    }
    plan.update(overrides)
    return plan


def _cost_benchmark(**overrides):
    report = {
        "schema_version": "exp106.cost_benchmark.v1",
        "device": COMPUTE_HOST,
        "q_token": "0.01",
        "kappa_seconds_upper": {str(m): ND3_KAPPA[m] for m in M_VALUES},
        "c_seconds_upper": {str(m): ND3_C[m] for m in M_VALUES},
    }
    report.update(overrides)
    return report


def test_allocate_writes_a_plan_that_the_config_could_adopt(tmp_path):
    _write_report(tmp_path / "pilot_plan.json", _pilot_plan())
    _write_report(tmp_path / "cost_benchmark.json", _cost_benchmark())
    output = tmp_path / "allocation_plan.json"
    assert main([
        "allocate",
        "--pilot-plan", str(tmp_path / "pilot_plan.json"),
        "--cost-benchmark", str(tmp_path / "cost_benchmark.json"),
        "--output", str(output),
    ]) == 0

    plan = json.loads(output.read_text(encoding="ascii"))
    assert plan["status"] == "EVALUATED_NOT_APPLIED", (
        "evaluating the rule is not applying it; the freeze is a separate act"
    )
    assert plan["cost_device"] == COMPUTE_HOST
    codes = {int(m): v for m, v in plan["codes_per_m"].items()}
    blocks = {int(m): v for m, v in plan["codes_per_task"].items()}
    for m in M_VALUES:
        assert codes[m] % blocks[m] == 0, "blocking must partition every panel"
        assert codes[m] >= blocks[m]
    assert plan["generation_core_hours"] <= GENERATION_BUDGET_CORE_HOURS
    assert plan["predicted_pointwise_sd_delta38"] > 0


def test_allocate_refuses_costs_from_the_wrong_machine(tmp_path):
    """The single most expensive process failure in exp105, made unrepeatable."""
    _write_report(tmp_path / "pilot_plan.json", _pilot_plan())
    _write_report(tmp_path / "cost_benchmark.json", _cost_benchmark(device="macmini"))
    with pytest.raises(ValueError, match="must be evaluated on .* costs"):
        main([
            "allocate",
            "--pilot-plan", str(tmp_path / "pilot_plan.json"),
            "--cost-benchmark", str(tmp_path / "cost_benchmark.json"),
            "--output", str(tmp_path / "out.json"),
        ])


def test_allocate_refuses_a_cost_report_from_another_q(tmp_path):
    _write_report(tmp_path / "pilot_plan.json", _pilot_plan())
    _write_report(tmp_path / "cost_benchmark.json", _cost_benchmark(q_token="0.05"))
    with pytest.raises(ValueError, match="disagree about q"):
        main([
            "allocate",
            "--pilot-plan", str(tmp_path / "pilot_plan.json"),
            "--cost-benchmark", str(tmp_path / "cost_benchmark.json"),
            "--output", str(tmp_path / "out.json"),
        ])


def test_allocate_refuses_a_tampered_report(tmp_path):
    _write_report(tmp_path / "pilot_plan.json", _pilot_plan())
    _write_report(tmp_path / "cost_benchmark.json", _cost_benchmark())
    path = tmp_path / "cost_benchmark.json"
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["c_seconds_upper"]["8"] = 0.001
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="report SHA256 mismatch"):
        main([
            "allocate",
            "--pilot-plan", str(tmp_path / "pilot_plan.json"),
            "--cost-benchmark", str(path),
            "--output", str(tmp_path / "out.json"),
        ])


def test_a_bracket_plan_anchors_on_the_bracket(tmp_path):
    _write_report(tmp_path / "pilot_plan.json", _pilot_plan(
        production_grid_tokens=[
            "0.02", "0.0228", "0.0256", "0.0283", "0.0311",
            "0.0339", "0.0367", "0.0394", "0.0422", "0.045",
        ],
        pilot_bracket=[0.03, 0.035],
        grid_rule_reason="pilot_bracket",
    ))
    _write_report(tmp_path / "cost_benchmark.json", _cost_benchmark())
    output = tmp_path / "allocation_plan.json"
    main([
        "allocate",
        "--pilot-plan", str(tmp_path / "pilot_plan.json"),
        "--cost-benchmark", str(tmp_path / "cost_benchmark.json"),
        "--output", str(output),
    ])
    plan = json.loads(output.read_text(encoding="ascii"))
    assert plan["anchor_p_token"] in ("0.03", "0.035")
    assert plan["grid_rule_reason"] == "pilot_bracket"
