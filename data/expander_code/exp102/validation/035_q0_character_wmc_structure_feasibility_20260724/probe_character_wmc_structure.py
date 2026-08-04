"""Bounded structural probe for one-copy character weighted-model counting.

The probe intentionally stops before numerical factor contraction.  Its only
question is whether avoiding the two-copy purity constraint makes an exact
single-copy character route structurally plausible on the m8 hard sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_character_wmc_structure.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
WMC_SOURCE = EXP102_ROOT / "validation/007_q0_global_discovery_20260721/wmc_feasibility.py"


class StructureProbeError(RuntimeError):
    pass


class StructureTimeout(RuntimeError):
    pass


class StructureResourceLimit(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise StructureProbeError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise StructureProbeError("character-WMC structural config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "character-WMC structural config is not canonical")
    expected = {
        "cell", "config_version", "contract_version", "encodings",
        "max_directed_adjacency_edges", "max_seconds_per_order", "orders",
        "registry_sha256", "scope", "version",
    }
    _require(set(config) == expected and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"]
             == "exp102.q0_character_wmc_structure.feasibility.config.v0",
             "character-WMC structural config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "character-WMC structural cell changed")
    _require(config["encodings"] == ["raw_hz_checks", "ternary_xor_chain"],
             "character-WMC structural encodings changed")
    _require(config["orders"] == [
        "min_degree_then_variable_index", "min_fill_then_degree_then_variable_index",
    ], "character-WMC structural orders changed")
    _require(config["max_seconds_per_order"] == 120
             and config["max_directed_adjacency_edges"] == 10000000,
             "character-WMC structural resource limits changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "character-WMC structural registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "single_copy_character_wmc_structural_feasibility_only",
        "remote_authorization": False,
    }, "character-WMC structural scope changed")
    return config, sha256_file(path)


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "config": sha256_file(config_path),
        "probe": sha256_file(Path(__file__)),
        "registry": sha256_file(REGISTRY_PATH),
        "wmc_engine": sha256_file(WMC_SOURCE),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _load_wmc_module():
    spec = importlib.util.spec_from_file_location("exp102_character_wmc_engine", WMC_SOURCE)
    _require(spec is not None and spec.loader is not None,
             "cannot load existing strict WMC factor encoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalise_scopes(scopes, variable_count):
    result = []
    for scope in scopes:
        normalized = tuple(sorted({int(value) for value in scope}))
        _require(normalized, "structural factor scope is empty")
        _require(all(0 <= value < int(variable_count) for value in normalized),
                 "structural factor scope has an invalid variable")
        result.append(normalized)
    _require(result, "structural encoding has no factors")
    return tuple(result)


def _adjacency(scopes, variable_count):
    adjacency = [set() for _ in range(int(variable_count))]
    for scope in _normalise_scopes(scopes, variable_count):
        for index, left in enumerate(scope):
            adjacency[left].update(scope[:index])
            adjacency[left].update(scope[index + 1:])
    return adjacency


def _directed_edge_count(adjacency, alive):
    return int(sum(len(adjacency[variable] & alive) for variable in alive))


def _order_hash(order):
    packed = np.asarray(order, dtype=">u4").tobytes()
    return hashlib.sha256(packed).hexdigest()


def _min_degree(scopes, variable_count, deadline, edge_cap):
    adjacency = _adjacency(scopes, variable_count)
    alive = set(range(int(variable_count)))
    heap = [(len(adjacency[variable]), variable) for variable in alive]
    heapq.heapify(heap)
    order = []
    width = 0
    peak_directed_edges = _directed_edge_count(adjacency, alive)
    while alive:
        if time.monotonic() > deadline:
            raise StructureTimeout((len(alive), width, order, peak_directed_edges))
        while heap:
            degree, variable = heapq.heappop(heap)
            if variable not in alive:
                continue
            neighbors = adjacency[variable] & alive
            if degree == len(neighbors):
                break
            heapq.heappush(heap, (len(neighbors), variable))
        else:  # pragma: no cover - defensive invariant
            raise StructureProbeError("min-degree heap exhausted")
        width = max(width, len(neighbors))
        neighbors = sorted(neighbors)
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        alive.remove(variable)
        for neighbor in neighbors:
            adjacency[neighbor].discard(variable)
            heapq.heappush(heap, (len(adjacency[neighbor] & alive), neighbor))
        peak_directed_edges = max(peak_directed_edges, _directed_edge_count(adjacency, alive))
        if peak_directed_edges > int(edge_cap):
            raise StructureResourceLimit((len(alive), width, order, peak_directed_edges))
        order.append(variable)
    return tuple(order), int(width), int(peak_directed_edges)


def _fill_count(adjacency, alive, variable):
    neighbors = sorted(adjacency[variable] & alive)
    fill = 0
    for index, left in enumerate(neighbors):
        left_adjacency = adjacency[left]
        for right in neighbors[index + 1:]:
            if right not in left_adjacency:
                fill += 1
    return fill, neighbors


def _min_fill(scopes, variable_count, deadline, edge_cap):
    """Run literal greedy min-fill, not a stale-score approximation."""
    adjacency = _adjacency(scopes, variable_count)
    alive = set(range(int(variable_count)))
    order = []
    width = 0
    peak_directed_edges = _directed_edge_count(adjacency, alive)
    while alive:
        if time.monotonic() > deadline:
            raise StructureTimeout((len(alive), width, order, peak_directed_edges))
        selected = None
        for scanned, variable in enumerate(sorted(alive), start=1):
            if scanned % 32 == 0 and time.monotonic() > deadline:
                raise StructureTimeout((len(alive), width, order, peak_directed_edges))
            fill, neighbors = _fill_count(adjacency, alive, variable)
            key = (fill, len(neighbors), variable)
            if selected is None or key < selected[0]:
                selected = (key, variable, neighbors)
        _require(selected is not None, "min-fill lost every live variable")
        _key, variable, neighbors = selected
        width = max(width, len(neighbors))
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        alive.remove(variable)
        for neighbor in neighbors:
            adjacency[neighbor].discard(variable)
        peak_directed_edges = max(peak_directed_edges, _directed_edge_count(adjacency, alive))
        if peak_directed_edges > int(edge_cap):
            raise StructureResourceLimit((len(alive), width, order, peak_directed_edges))
        order.append(variable)
    return tuple(order), int(width), int(peak_directed_edges)


def _run_order(method, scopes, variable_count, config):
    started = time.monotonic()
    deadline = started + float(config["max_seconds_per_order"])
    solver = _min_degree if method == "min_degree_then_variable_index" else _min_fill
    try:
        order, width, peak_edges = solver(
            scopes, variable_count, deadline, config["max_directed_adjacency_edges"],
        )
    except StructureTimeout as exc:
        remaining, width, order, peak_edges = exc.args[0]
        return {
            "method": method,
            "status": "INCONCLUSIVE_TIMEOUT",
            "remaining_variables": int(remaining),
            "max_observed_width": int(width),
            "completed_order_entries": int(len(order)),
            "completed_prefix_sha256": _order_hash(order),
            "peak_directed_adjacency_edges": int(peak_edges),
            "wall_seconds": time.monotonic() - started,
        }
    except StructureResourceLimit as exc:
        remaining, width, order, peak_edges = exc.args[0]
        return {
            "method": method,
            "status": "INCONCLUSIVE_EDGE_CAP",
            "remaining_variables": int(remaining),
            "max_observed_width": int(width),
            "completed_order_entries": int(len(order)),
            "completed_prefix_sha256": _order_hash(order),
            "peak_directed_adjacency_edges": int(peak_edges),
            "wall_seconds": time.monotonic() - started,
        }
    return {
        "method": method,
        "status": "COMPLETE",
        "induced_width": int(width),
        "order_sha256": _order_hash(order),
        "peak_directed_adjacency_edges": int(peak_edges),
        "wall_seconds": time.monotonic() - started,
    }


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "character-WMC structural registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, _frame = build_model(H)
    uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, config["cell"])
    _require(model.num_qubits == 1600 and model.k == 64 and syndrome.any(),
             "character-WMC structural m8 context changed")
    return registry, code, model, int(uniform_seed), syndrome


def run_probe(config):
    registry, code, model, uniform_seed, syndrome = _context(config)
    wmc = _load_wmc_module()
    encodings = {
        "raw_hz_checks": (
            tuple(tuple(np.flatnonzero(row).tolist()) for row in model.H_check),
            int(model.num_qubits),
        ),
    }
    chain_factors, chain_variables = wmc.posterior_factors(
        model, syndrome, config["cell"]["p"], replicas=1, logical_collision=False,
    )
    encodings["ternary_xor_chain"] = (
        tuple(tuple(factor.scope) for factor in chain_factors), int(chain_variables),
    )
    records = []
    for encoding in config["encodings"]:
        scopes, variable_count = encodings[encoding]
        records.append({
            "encoding": encoding,
            "variable_count": variable_count,
            "factor_count": len(scopes),
            "maximum_initial_factor_arity": max(len(scope) for scope in scopes),
            "orders": [
                _run_order(order, scopes, variable_count, config)
                for order in config["orders"]
            ],
        })
    return {
        "cell": config["cell"],
        "registry_sha256": registry["registry_sha256"],
        "disorder_uniform_seed": uniform_seed,
        "physical_dimensions": {
            "qubits": int(model.num_qubits),
            "checks": int(model.H_check.shape[0]),
            "logical_dimension": int(model.k),
            "syndrome_weight": int(syndrome.sum()),
        },
        "identity": {
            "Z_u": "sum_{H_Z e=y} b**|e| * (-1)**<w_u,e>",
            "m_u": "Z_u / Z_0",
            "scope_invariance": "unary character signs do not alter factor scopes",
        },
        "records": records,
        "does_not_establish": [
            "A numerical Z_0, Z_u, posterior, purity, or q_top value.",
            "Signed numerical stability, outward rounding, or an estimator contract.",
            "Any MCMC, remote, formal, held-out, or production authorization.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "character_wmc_structure.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace character-WMC report: {args.output}")
    config, config_sha256 = _load_config(args.config)
    core = {
        "probe_version": PROBE_VERSION,
        "config_sha256": config_sha256,
        "scope": config["scope"],
        "source_binding": _source_binding(args.config),
        "probe": run_probe(config),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
