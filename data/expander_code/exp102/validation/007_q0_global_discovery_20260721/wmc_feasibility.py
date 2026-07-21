"""Bounded exact weighted-model-counting feasibility for m3/m4 sentinels.

The solver performs binary factor elimination only when a deterministic
min-degree order stays below the explicit width/memory limit.  Otherwise it
returns diagnostic ``INCONCLUSIVE_WIDTH`` and no numerical estimate.  It never
turns a heuristic approximation into discovery evidence.
"""

import argparse
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    load_global_discovery_config,
    uniform_seed_for_cell,
)
from data.expander_code.exp102.exp102_pipeline.io import atomic_json
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


class WidthLimit(Exception):
    pass


class SolverTimeout(Exception):
    pass


@dataclass
class Factor:
    scope: tuple
    values: np.ndarray
    log_scale: float = 0.0


def _factor(scope, values):
    scope = tuple(int(value) for value in scope)
    values = np.asarray(values, dtype=np.float64)
    if len(scope) != len(set(scope)) or values.shape != (2,) * len(scope):
        raise ValueError("invalid binary factor")
    order = np.argsort(scope)
    ordered_scope = tuple(scope[index] for index in order)
    if tuple(order) != tuple(range(len(scope))):
        values = np.transpose(values, axes=order)
    return Factor(ordered_scope, np.ascontiguousarray(values), 0.0)


def _xor_factor(scope, parity):
    values = np.zeros((2,) * len(scope), dtype=np.float64)
    for assignment in np.ndindex(values.shape):
        if (sum(assignment) & 1) == int(parity):
            values[assignment] = 1.0
    return _factor(scope, values)


def _xor_chain(variables, parity, next_variable):
    variables = [int(value) for value in variables]
    if not variables:
        if parity:
            return [_factor((), np.array(0.0))], next_variable
        return [], next_variable
    if len(variables) <= 2:
        return [_xor_factor(variables, parity)], next_variable
    factors = []
    previous = next_variable
    next_variable += 1
    factors.append(_xor_factor((variables[0], variables[1], previous), 0))
    for variable in variables[2:-1]:
        current = next_variable
        next_variable += 1
        factors.append(_xor_factor((previous, variable, current), 0))
        previous = current
    factors.append(_xor_factor((previous, variables[-1]), parity))
    return factors, next_variable


def posterior_factors(model, syndrome, p, replicas=1, logical_collision=False):
    n = model.num_qubits
    next_variable = replicas * n
    factors = []
    K = math.log((1.0 - p) / p)
    unary = np.asarray([1.0, math.exp(-K)], dtype=np.float64)
    for replica in range(replicas):
        offset = replica * n
        for qubit in range(n):
            factors.append(_factor((offset + qubit,), unary))
        for check, target in enumerate(syndrome):
            support = np.flatnonzero(model.H_check[check]) + offset
            chain, next_variable = _xor_chain(support, int(target), next_variable)
            factors.extend(chain)
    if logical_collision:
        if replicas != 2:
            raise ValueError("logical collision factors require two replicas")
        for row in model._wmc_frame_W:
            support = np.flatnonzero(row)
            variables = [*support.tolist(), *(support + n).tolist()]
            chain, next_variable = _xor_chain(variables, 0, next_variable)
            factors.extend(chain)
    return factors, next_variable


def min_degree_order(factors, num_variables, max_width, deadline):
    adjacency = {variable: set() for variable in range(num_variables)}
    for factor in factors:
        for variable in factor.scope:
            adjacency[variable].update(value for value in factor.scope if value != variable)
    heap = [(len(neighbors), variable) for variable, neighbors in adjacency.items()]
    heapq.heapify(heap)
    alive = set(adjacency)
    order = []
    width = 0
    while alive:
        if time.monotonic() > deadline:
            raise SolverTimeout
        while heap:
            degree, variable = heapq.heappop(heap)
            if variable in alive:
                neighbors = adjacency[variable] & alive
                if degree == len(neighbors):
                    break
                heapq.heappush(heap, (len(neighbors), variable))
        else:
            raise AssertionError("min-degree heap exhausted")
        width = max(width, len(neighbors))
        if width > max_width:
            raise WidthLimit((width, len(alive), order))
        neighbors = sorted(neighbors)
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        alive.remove(variable)
        for neighbor in neighbors:
            adjacency[neighbor].discard(variable)
            heapq.heappush(heap, (len(adjacency[neighbor] & alive), neighbor))
        order.append(variable)
    return order, width


def _combine(factors, max_width, deadline):
    if time.monotonic() > deadline:
        raise SolverTimeout
    scope = tuple(sorted({value for factor in factors for value in factor.scope}))
    if len(scope) > max_width + 1:
        raise WidthLimit((len(scope) - 1, None, None))
    values = np.ones((2,) * len(scope), dtype=np.float64)
    position = {variable: index for index, variable in enumerate(scope)}
    log_scale = 0.0
    for factor in factors:
        shape = [1] * len(scope)
        for variable in factor.scope:
            shape[position[variable]] = 2
        values *= factor.values.reshape(shape)
        log_scale += factor.log_scale
    return Factor(scope, values, log_scale)


def exact_log_partition(factors, num_variables, max_width, deadline):
    order, width = min_degree_order(factors, num_variables, max_width, deadline)
    active = list(factors)
    for variable in order:
        selected = [factor for factor in active if variable in factor.scope]
        if not selected:
            continue
        active = [factor for factor in active if variable not in factor.scope]
        combined = _combine(selected, max_width, deadline)
        axis = combined.scope.index(variable)
        values = combined.values.sum(axis=axis)
        scope = tuple(value for value in combined.scope if value != variable)
        maximum = float(values.max())
        if maximum == 0.0:
            return float("-inf"), width
        active.append(Factor(scope, values / maximum, combined.log_scale + math.log(maximum)))
    final = _combine(active, max_width, deadline)
    total = float(final.values.sum())
    return final.log_scale + math.log(total), width


def solve_cell(model, frame, syndrome, p, max_width, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    object.__setattr__(model, "_wmc_frame_W", frame.W_basis)
    started = time.monotonic()
    try:
        denominator, variables_den = posterior_factors(model, syndrome, p)
        log_z, width_den = exact_log_partition(
            denominator, variables_den, max_width, deadline,
        )
        numerator, variables_num = posterior_factors(
            model, syndrome, p, replicas=2, logical_collision=True,
        )
        log_collision_z, width_num = exact_log_partition(
            numerator, variables_num, max_width, deadline,
        )
        collision = math.exp(log_collision_z - 2.0 * log_z)
        uniform = 2.0 ** (-model.k)
        q_top = (collision - uniform) / (1.0 - uniform)
        return {
            "status": "EXACT",
            "q_top": q_top,
            "collision_mass": collision,
            "log_partition": log_z,
            "denominator_width": width_den,
            "collision_width": width_num,
            "wall_seconds": time.monotonic() - started,
            "evidence_kind": "exact_weighted_model_count",
        }
    except WidthLimit as exc:
        width, remaining, _ = exc.args[0]
        return {
            "status": "INCONCLUSIVE_WIDTH",
            "observed_width": int(width),
            "remaining_variables": None if remaining is None else int(remaining),
            "max_exact_width": int(max_width),
            "wall_seconds": time.monotonic() - started,
            "evidence_kind": "diagnostic_only_no_bound",
        }
    except SolverTimeout:
        return {
            "status": "INCONCLUSIVE_TIMEOUT",
            "max_exact_width": int(max_width),
            "wall_seconds": time.monotonic() - started,
            "evidence_kind": "diagnostic_only_no_bound",
        }
    finally:
        object.__delattr__(model, "_wmc_frame_W")


def run_panel(registry_path, config_path, max_width=20, timeout_seconds=7200.0):
    registry = load_registry(registry_path)
    config = load_global_discovery_config(config_path, registry)
    cells = config["panels"]["SMALL6"]["cells"]
    records = []
    for cell in cells:
        _, code, H = load_frozen_code(registry_path, cell["code_id"])
        model, frame = build_model(H)
        seed = uniform_seed_for_cell(registry, code, cell)
        uniforms = np.random.Generator(np.random.PCG64(seed)).random(model.num_qubits)
        epsilon = (uniforms < cell["p"]).astype(np.uint8)
        syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
        result = solve_cell(
            model, frame, syndrome, cell["p"], int(max_width),
            min(float(timeout_seconds), 7200.0),
        )
        records.append({"cell": cell, "uniform_seed": seed, **result})
    return {
        "report_version": "exp102.q0_global.wmc_feasibility.v1",
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "max_exact_width": int(max_width),
        "timeout_seconds_per_cell": min(float(timeout_seconds), 7200.0),
        "records": records,
        "status": "EXACT" if all(value["status"] == "EXACT" for value in records) else "INCONCLUSIVE",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--max-width", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args(argv)
    if not 1 <= args.max_width <= 24:
        raise ValueError("exact WMC max width must lie in [1,24]")
    if not 0 < args.timeout_seconds <= 7200.0:
        raise ValueError("WMC timeout must lie in (0,7200]")
    result = run_panel(
        args.registry, args.config, args.max_width, args.timeout_seconds,
    )
    atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "cell_status": [value["status"] for value in result["records"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
