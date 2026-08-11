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
    DIAGNOSTIC_BUDGET_SHARE,
    DIAGNOSTIC_M_VALUES,
    FALLBACK_P_TOKENS,
    GENERATION_BUDGET_CORE_HOURS,
    M_VALUES,
    PRIMARY_M_VALUES,
    PRODUCTION_GRID_CLIP,
    PRODUCTION_GRID_DECIMALS,
    PRODUCTION_GRID_POINTS,
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
PILOT_SCHEMA = "exp105.pilot_plan.v1"

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


def allocation_rule(measurements, cost_per_code_seconds, cost_per_trial_seconds,
                    grid_points, bracket_index):
    """The frozen allocation rule. Returns (codes_per_m, trials_per_m)."""
    low, high = TRIALS_PER_CODE_P_RANGE
    trials = {}
    for m in M_VALUES:
        source = measurements[m if m in measurements else max(measurements)]
        sigma_c = float(source["sigma_c"][bracket_index])
        sigma_w2 = float(source["sigma_w2"][bracket_index])
        kappa = float(cost_per_code_seconds[m])
        cost = float(cost_per_trial_seconds[m])
        if sigma_c <= 0.0 or cost <= 0.0:
            trials[m] = high
            continue
        value = math.sqrt(kappa * sigma_w2 / (grid_points * cost * sigma_c ** 2))
        trials[m] = int(min(high, max(low, round(value))))

    unit_cost = {
        m: cost_per_code_seconds[m] + grid_points * trials[m] * cost_per_trial_seconds[m]
        for m in M_VALUES
    }
    budget_seconds = GENERATION_BUDGET_CORE_HOURS * 3600.0
    share = {m: DIAGNOSTIC_BUDGET_SHARE * budget_seconds for m in DIAGNOSTIC_M_VALUES}
    primary_budget = budget_seconds - sum(share.values())

    a, b = PRIMARY_M_VALUES
    sigma_a = float(measurements[a]["sigma_c"][bracket_index]) if a in measurements else 1.0
    sigma_b = float(measurements[b]["sigma_c"][bracket_index]) if b in measurements else 1.0
    sigma_a = max(sigma_a, 1e-9)
    sigma_b = max(sigma_b, 1e-9)
    # Minimising sigma_a^2/C_a + sigma_b^2/C_b subject to C_a u_a + C_b u_b = B
    # gives C_a / C_b = (sigma_a / sigma_b) sqrt(u_b / u_a).
    ratio = (sigma_a / sigma_b) * math.sqrt(unit_cost[b] / unit_cost[a])
    codes_b = primary_budget / (ratio * unit_cost[a] + unit_cost[b])
    codes_a = ratio * codes_b

    codes = {a: codes_a, b: codes_b}
    for m in DIAGNOSTIC_M_VALUES:
        codes[m] = share[m] / unit_cost[m]
    return (
        {m: int(codes[m]) for m in M_VALUES},
        trials,
        {m: unit_cost[m] for m in M_VALUES},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp105 locating pilot")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="run every pilot task locally")
    scan.add_argument("--config", required=True)
    scan.add_argument("--raw-root", required=True)
    scan.add_argument("--num-workers", type=int, required=True)

    measure = sub.add_parser("measure", help="measure the rule inputs")
    measure.add_argument("--config", required=True)
    measure.add_argument("--raw-root", required=True)
    measure.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "scan":
        result = run_local_scan(args.config, args.raw_root, args.num_workers)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0 if result["status"] == "PASS" else 1

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
