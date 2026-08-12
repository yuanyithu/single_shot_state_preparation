"""The locating pilot: a local scan, and the frozen rules it feeds.

The pilot exists to evaluate two rules that were frozen before it ran
(EXPERIMENT_CONTRACT.md section 6): where the production p grid goes, and how
many codes and trials each m gets. It is not evidence about physics. Its raw is
never merged into production, its codes come from a separate ensemble namespace
so that no code which helps choose the grid is later measured on it, and nothing
here may be published.

Both rules are implemented as functions of measured inputs with no free
parameters, so running this module is an evaluation rather than a decision.
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np

from .aggregate import panel_layout
from .config import (
    COMPUTE_HOST,
    DIAGNOSTIC_BUDGET_SHARE,
    DIAGNOSTIC_M_VALUES,
    FALLBACK_P_TOKENS,
    GENERATION_BUDGET_CORE_HOURS,
    M_VALUES,
    PRIMARY_M_VALUES,
    PRODUCTION_GRID_CLIP,
    PRODUCTION_GRID_DECIMALS,
    PRODUCTION_GRID_POINTS,
    TARGET_TASK_SECONDS,
    TRIALS_PER_CODE_P_RANGE,
    ensure_config,
    load_config,
    tasks_per_m,
)
from .ensemble import load_registry, registry_index
from .io import atomic_json, sha256_json
from .raw import load_raw, raw_filename, save_raw
from .worker import run_code_block


REPO_ROOT = Path(__file__).resolve().parents[4]
PILOT_SCHEMA = "exp106.pilot_plan.v1"

_ROWS_CACHE = {}


def _cached_rows(config_path):
    key = str(config_path)
    if key not in _ROWS_CACHE:
        config = load_config(config_path)
        _ROWS_CACHE[key] = registry_index(
            load_registry(REPO_ROOT / config["registry_path"])
        )
    return _ROWS_CACHE[key]


def _run_one(payload):
    m, block_index, config_path, raw_root = payload
    config = load_config(config_path)
    rows = _cached_rows(config_path)
    output = Path(raw_root) / raw_filename(config, m, block_index)
    if output.exists():
        return m, block_index, "EXISTS"
    raw = run_code_block(m, block_index, config, rows)
    save_raw(output, raw)
    return m, block_index, raw["status"]


def planned_tasks(config):
    counts = {int(k): int(v) for k, v in config["codes_per_m"].items()}
    sizes = {int(k): int(v) for k, v in config["codes_per_task"].items()}
    per_m = tasks_per_m(counts, sizes)
    return [
        (int(m), block)
        for m in config["m_values"]
        for block in range(per_m[int(m)])
    ]


def run_local_scan(config_path, raw_root, num_workers):
    """Run every planned task of the given config locally, in parallel."""
    config = load_config(config_path)
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        (m, block, str(config_path), str(raw_root))
        for m, block in planned_tasks(config)
    ]
    results = []
    with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        for result in executor.map(_run_one, payloads):
            results.append(result)
            print(
                f"m={result[0]} block={result[1]} {result[2]}",
                flush=True,
            )
    invalid = [item for item in results if item[2] not in ("VALID", "EXISTS")]
    return {
        "tasks": len(results),
        "invalid": invalid,
        "status": "PASS" if not invalid else "INVALID_TASKS_PRESENT",
    }


# ---------------------------------------------------------------------------
# Measurement of the rule inputs
# ---------------------------------------------------------------------------

def measure_pilot(raw_root, config):
    """Per (m, p): the pooled rate, the between-code and within-code spreads.

    `sigma_c` is the standard deviation of the per-code failure rates and
    `sigma_w2` the mean within-code trial variance. With `T` trials per code the
    per-code rate has variance `sigma_true^2 + sigma_w2 / T`, so the raw spread
    of the observed rates overstates the between-code spread; the estimate below
    subtracts that inflation and floors at zero, because the allocation rule
    wants the true between-code term.
    """
    config = ensure_config(config)
    m_values, counts, trials_by_m, _, _ = panel_layout(config)
    tokens = list(config["p_tokens"])
    per_m = {}
    for m in m_values:
        trials = trials_by_m[m]
        rates = np.full((counts[m], len(tokens)), np.nan)
        seen = np.zeros(counts[m], dtype=bool)
        block = int(config["codes_per_task"][str(m)])
        for _, block_index in [t for t in planned_tasks(config) if t[0] == m]:
            path = Path(raw_root) / raw_filename(config, m, block_index)
            if not path.exists():
                continue
            raw = load_raw(path)
            if raw["status"] != "VALID":
                continue
            start = block_index * block
            flags = raw["failure_flags"]
            rates[start:start + flags.shape[0]] = flags.mean(axis=2)
            seen[start:start + flags.shape[0]] = True
        if not seen.all():
            raise ValueError(f"pilot raw is incomplete at m={m}")
        pooled = rates.mean(axis=0)
        raw_std = rates.std(axis=0, ddof=1)
        within = pooled * (1.0 - pooled)
        between = np.sqrt(np.maximum(raw_std ** 2 - within / trials, 0.0))
        per_m[m] = {
            "pooled_rate": pooled,
            "sigma_c": between,
            "sigma_c_raw": raw_std,
            "sigma_w2": within,
            "trials": trials,
            "codes": counts[m],
        }
    return per_m


def grid_rule(p_values, delta):
    """The frozen grid rule. Returns (tokens, bracket, reason)."""
    p_values = [float(value) for value in p_values]
    delta = [float(value) for value in delta]
    low_index = None
    for index in range(len(delta) - 1):
        if delta[index] < 0.0 <= delta[index + 1]:
            low_index = index
            break
    if low_index is None:
        return list(FALLBACK_P_TOKENS), None, "no_sign_change_fallback_grid"

    p_lo = p_values[low_index]
    p_hi = p_values[low_index + 1]
    spacing = p_hi - p_lo
    lower = max(float(Decimal(PRODUCTION_GRID_CLIP[0])), p_lo - 2.0 * spacing)
    upper = min(float(Decimal(PRODUCTION_GRID_CLIP[1])), p_hi + 2.0 * spacing)
    points = np.linspace(lower, upper, PRODUCTION_GRID_POINTS)
    tokens = []
    for value in points:
        token = f"{round(float(value), PRODUCTION_GRID_DECIMALS):.{PRODUCTION_GRID_DECIMALS}f}"
        token = token.rstrip("0").rstrip(".") if "." in token else token
        if token not in tokens:
            tokens.append(token)
    return tokens, (p_lo, p_hi), "pilot_bracket"


def anchor_index(pilot_p_values, production_grid, bracket):
    """The pilot grid point whose statistics feed the allocation rule.

    The contract says "at the pilot grid point nearest the bracket". When the
    grid rule finds a bracket that is its midpoint; when it falls back, there is
    no bracket, and the anchor is the pilot point nearest the geometric centre of
    the production grid -- geometric rather than arithmetic because the fallback
    grid spans more than a decade and its arithmetic centre would sit at the
    expensive end.
    """
    pilot = [float(value) for value in pilot_p_values]
    if bracket is not None:
        target = 0.5 * (float(bracket[0]) + float(bracket[1]))
    else:
        grid = [float(token) for token in production_grid]
        target = math.exp(sum(math.log(value) for value in grid) / len(grid))
    return min(range(len(pilot)), key=lambda index: abs(pilot[index] - target))


def effective_sigma(sigma_c, sigma_w2, trials):
    """The standard deviation of one code's *observed* rate at `trials` trials.

    This is the quantity that actually enters the variance of a panel mean:
    `Var(mean) = (sigma_c^2 + sigma_w2 / T) / C`. It reduces to `sigma_c` when
    the between-code spread dominates and to the shot-noise term when it does
    not.

    exp106 preregisters this form rather than raw `sigma_c` because exp105 was
    forced to substitute it mid-flight: at `q = 0.05` the readout channel is
    common to every code, the pilot measured `sigma_c` at or below its own
    resolution at almost every grid point, and the primary split degenerated
    into 0/0. At `q = 0.01` the channel is five times weaker and `sigma_c` may
    well be recoverable, but which branch applies must not be a choice made
    after seeing the pilot.
    """
    return math.sqrt(max(float(sigma_c), 0.0) ** 2 + float(sigma_w2) / float(trials))


def allocation_rule(measurements, cost_per_code_seconds, cost_per_trial_seconds,
                    grid_points, bracket_index):
    """The frozen allocation rule.

    Returns `(codes_per_m, trials_per_m, unit_cost_per_m, effective_sigma_per_m)`.
    """
    low, high = TRIALS_PER_CODE_P_RANGE
    trials = {}
    sigma_inputs = {}
    for m in M_VALUES:
        # The pilot runs at the primary pair only, so the diagnostic sizes take
        # the largest measured m's row. They receive a fixed budget share either
        # way, so this only sets their trials per code.
        source = measurements[m if m in measurements else max(measurements)]
        sigma_c = float(source["sigma_c"][bracket_index])
        sigma_w2 = float(source["sigma_w2"][bracket_index])
        sigma_inputs[m] = (sigma_c, sigma_w2)
        kappa = float(cost_per_code_seconds[m])
        cost = float(cost_per_trial_seconds[m])
        if sigma_c <= 0.0 or cost <= 0.0:
            # The T -> infinity limit of the line below, not a special case:
            # with no between-code spread there is nothing to buy by spreading
            # the same trials over more codes.
            trials[m] = high
            continue
        value = math.sqrt(kappa * sigma_w2 / (grid_points * cost * sigma_c ** 2))
        trials[m] = int(min(high, max(low, round(value))))

    unit_cost = {
        m: cost_per_code_seconds[m] + grid_points * trials[m] * cost_per_trial_seconds[m]
        for m in M_VALUES
    }
    s_effective = {
        m: effective_sigma(sigma_inputs[m][0], sigma_inputs[m][1], trials[m])
        for m in M_VALUES
    }
    budget_seconds = GENERATION_BUDGET_CORE_HOURS * 3600.0
    share = {m: DIAGNOSTIC_BUDGET_SHARE * budget_seconds for m in DIAGNOSTIC_M_VALUES}
    primary_budget = budget_seconds - sum(share.values())

    a, b = PRIMARY_M_VALUES
    # Minimising s_a^2/C_a + s_b^2/C_b subject to C_a u_a + C_b u_b = B gives
    # C_a / C_b = (s_a / s_b) sqrt(u_b / u_a).
    ratio = (s_effective[a] / s_effective[b]) * math.sqrt(unit_cost[b] / unit_cost[a])
    codes_b = primary_budget / (ratio * unit_cost[a] + unit_cost[b])
    codes_a = ratio * codes_b

    codes = {a: codes_a, b: codes_b}
    for m in DIAGNOSTIC_M_VALUES:
        codes[m] = share[m] / unit_cost[m]
    return (
        {m: int(codes[m]) for m in M_VALUES},
        trials,
        {m: unit_cost[m] for m in M_VALUES},
        s_effective,
    )


def block_size_rule(unit_cost_seconds):
    """Codes per task: the largest block that still fits the frozen target."""
    return {
        m: max(1, int(TARGET_TASK_SECONDS / float(cost)))
        for m, cost in unit_cost_seconds.items()
    }


def _checked_report(path, expected_schema=None):
    report = json.loads(Path(path).read_text(encoding="ascii"))
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_json(core):
        raise ValueError(f"report SHA256 mismatch: {path}")
    if expected_schema and report.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected schema in {path}: {report.get('schema_version')}")
    return report


def _command_allocate(args):
    """Evaluate the frozen allocation rule. No free parameters, no choices."""
    plan = _checked_report(args.pilot_plan, PILOT_SCHEMA)
    costs = _checked_report(args.cost_benchmark)
    if costs.get("device") != COMPUTE_HOST:
        # exp105 evaluated this rule on macmini costs and its nd-3 resource gate
        # blocked at 5,368 core-hours against a cap of 800. The rule spends a
        # budget of core-hours on the machine that runs it, so the check names
        # the frozen compute host rather than a literal -- the run has already
        # moved once.
        raise ValueError(
            f"the allocation rule must be evaluated on {COMPUTE_HOST} costs, not "
            f"{costs.get('device')!r}"
        )
    if costs.get("q_token") != plan["q_token"]:
        raise ValueError("cost benchmark and pilot plan disagree about q")

    grid = list(plan["production_grid_tokens"])
    bracket = plan["pilot_bracket"]
    index = anchor_index(plan["p_tokens"], grid, bracket)
    measurements = {
        int(m): {
            "sigma_c": plan["sigma_c"][m],
            "sigma_w2": plan["sigma_w2"][m],
            "pooled_rate": plan["pooled_rate"][m],
        }
        for m in plan["sigma_c"]
    }
    kappa = {int(m): float(value) for m, value in costs["kappa_seconds_upper"].items()}
    per_trial = {int(m): float(value) for m, value in costs["c_seconds_upper"].items()}

    codes, trials, unit_cost, s_effective = allocation_rule(
        measurements, kappa, per_trial, len(grid), index,
    )
    blocks = block_size_rule(unit_cost)
    codes = {m: (codes[m] // blocks[m]) * blocks[m] for m in M_VALUES}
    for m in M_VALUES:
        if codes[m] < blocks[m]:
            raise ValueError(f"the allocation rule leaves m={m} with no complete task")

    a, b = PRIMARY_M_VALUES
    predicted_sd = math.sqrt(
        s_effective[a] ** 2 / codes[a] + s_effective[b] ** 2 / codes[b]
    )
    generation_seconds = sum(codes[m] * unit_cost[m] for m in M_VALUES)
    core = {
        "schema_version": "exp106.allocation_plan.v1",
        "status": "EVALUATED_NOT_APPLIED",
        "pilot_plan_sha256": plan["report_sha256"],
        "cost_benchmark_sha256": costs["report_sha256"],
        "cost_device": costs["device"],
        "q_token": plan["q_token"],
        "grid_rule_reason": plan["grid_rule_reason"],
        "production_grid_tokens": grid,
        "anchor_p_token": plan["p_tokens"][index],
        "anchor_index": index,
        "codes_per_m": {str(m): int(codes[m]) for m in M_VALUES},
        "trials_per_code_p": {str(m): int(trials[m]) for m in M_VALUES},
        "codes_per_task": {str(m): int(blocks[m]) for m in M_VALUES},
        "unit_cost_seconds": {str(m): float(unit_cost[m]) for m in M_VALUES},
        "s_effective": {str(m): float(s_effective[m]) for m in M_VALUES},
        "generation_core_hours": generation_seconds / 3600.0,
        "generation_budget_core_hours": GENERATION_BUDGET_CORE_HOURS,
        "total_codes": int(sum(codes.values())),
        "total_tasks": int(sum(codes[m] // blocks[m] for m in M_VALUES)),
        "total_trials": int(sum(codes[m] * len(grid) * trials[m] for m in M_VALUES)),
        "predicted_pointwise_sd_delta38": predicted_sd,
    }
    report = dict(core, report_sha256=sha256_json(core))
    atomic_json(args.output, report)
    print(json.dumps({
        key: report[key] for key in (
            "codes_per_m", "trials_per_code_p", "codes_per_task",
            "generation_core_hours", "total_trials",
            "predicted_pointwise_sd_delta38", "report_sha256",
        )
    }, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp106 locating pilot")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="run every pilot task locally")
    scan.add_argument("--config", required=True)
    scan.add_argument("--raw-root", required=True)
    scan.add_argument("--num-workers", type=int, required=True)

    measure = sub.add_parser("measure", help="measure the rule inputs")
    measure.add_argument("--config", required=True)
    measure.add_argument("--raw-root", required=True)
    measure.add_argument("--output", required=True)

    allocate = sub.add_parser(
        "allocate", help="evaluate the frozen allocation rule on compute-host costs",
    )
    allocate.add_argument("--pilot-plan", required=True)
    allocate.add_argument(
        "--cost-benchmark", required=True,
        help="cost_benchmark.json from `remote_cli cost-benchmark` on the compute host",
    )
    allocate.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "scan":
        result = run_local_scan(args.config, args.raw_root, args.num_workers)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0 if result["status"] == "PASS" else 1
    if args.command == "allocate":
        return _command_allocate(args)

    config = load_config(args.config)
    measurements = measure_pilot(args.raw_root, config)
    tokens = list(config["p_tokens"])
    delta = (
        measurements[8]["pooled_rate"] - measurements[3]["pooled_rate"]
    )
    grid, bracket, reason = grid_rule([float(t) for t in tokens], delta)
    core = {
        "schema_version": PILOT_SCHEMA,
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "p_tokens": tokens,
        "q_token": config["q_token"],
        "delta38": [float(value) for value in delta],
        "pooled_rate": {
            str(m): [float(value) for value in row["pooled_rate"]]
            for m, row in measurements.items()
        },
        "sigma_c": {
            str(m): [float(value) for value in row["sigma_c"]]
            for m, row in measurements.items()
        },
        "sigma_c_raw": {
            str(m): [float(value) for value in row["sigma_c_raw"]]
            for m, row in measurements.items()
        },
        "sigma_w2": {
            str(m): [float(value) for value in row["sigma_w2"]]
            for m, row in measurements.items()
        },
        "production_grid_tokens": grid,
        "pilot_bracket": list(bracket) if bracket else None,
        "grid_rule_reason": reason,
    }
    report = dict(core, report_sha256=sha256_json(core))
    atomic_json(args.output, report)
    print(json.dumps({
        "grid_rule_reason": reason,
        "pilot_bracket": core["pilot_bracket"],
        "production_grid_tokens": grid,
        "report_sha256": report["report_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
