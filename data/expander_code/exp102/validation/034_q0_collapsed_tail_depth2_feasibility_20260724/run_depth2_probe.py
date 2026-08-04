"""Bounded feasibility probe for a two-row collapsed-B tail envelope.

This is not a sampler and does not estimate a posterior or q_top.  It asks
whether retaining deterministic MAP-derived B marginals plus a strict,
outward-rounded depth-two upper envelope can bound the remaining B mass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_collapsed_tail_bound import (
    binary_elimination_plan,
    classical_coset_mass_interval,
    prefix_max_upper_tables,
    row_prefix_factor_scopes,
    row_prefix_partition_upper,
    scaled_state_weight_interval,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import split_hgp_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import build_milp_map_anchors
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_collapsed_tail.depth2.feasibility.v0"
EXP102_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "depth2_probe.json"


class Depth2ProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise Depth2ProbeError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise Depth2ProbeError("depth-two feasibility config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "depth-two feasibility config is not canonical")
    expected = {
        "anchor", "cell", "classical_p_rational", "config_version", "contract_version",
        "envelope", "registry_sha256", "scope", "version",
    }
    _require(set(config) == expected and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"] == "exp102.q0_collapsed_tail.depth2.feasibility.config.v0",
             "depth-two feasibility config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "depth-two feasibility cell changed")
    _require(config["classical_p_rational"] == [1, 25],
             "depth-two feasibility p rational changed")
    _require(config["registry_sha256"] == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "depth-two feasibility registry SHA changed")
    _require(config["anchor"] == {
        "anchor_count": 2,
        "anchor_sha256": "b0ad56f2cd3ec7815c5acb989260ed841276c69d33fc193bc9322a6de96549e5",
        "requested_max_anchors": 8,
        "selection": "all_unique_B_marginals",
    }, "depth-two feasibility anchor contract changed")
    _require(config["envelope"] == {
        "depth": 2,
        "expected_induced_width": 25,
        "expected_max_combined_variables": 26,
        "memory_cap_bytes": 6442450944,
        "max_runtime_seconds": 900,
        "tail_fraction_goal": 0.01,
    }, "depth-two feasibility envelope contract changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "strict_depth2_collapsed_B_upper_envelope_feasibility_only",
        "remote_authorization": False,
    }, "depth-two feasibility scope changed")
    return config, sha256_file(path)


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "config": sha256_file(config_path),
        "probe": sha256_file(Path(__file__)),
        "tail_bound": sha256_file(EXP102_ROOT / "exp102_pipeline/q0_collapsed_tail_bound.py"),
        "collapsed": sha256_file(EXP102_ROOT / "exp102_pipeline/q0_hgp_collapsed.py"),
        "map_mixture": sha256_file(EXP102_ROOT / "exp102_pipeline/q0_map_mixture.py"),
        "registry": sha256_file(EXP102_ROOT / "registry/registry.json"),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _max_combined_variables(scopes, variable_count):
    active = [set(int(value) for value in scope) for scope in scopes if scope]
    remaining = set(range(int(variable_count)))
    largest = 0
    while remaining:
        choices = []
        for variable in sorted(remaining):
            touching = [scope for scope in active if variable in scope]
            union = set().union(*touching) if touching else {variable}
            choices.append((len(union) - 1, variable, touching, union))
        _, variable, _touching, union = min(choices, key=lambda item: (item[0], item[1]))
        largest = max(largest, len(union))
        active = [scope for scope in active if variable not in scope]
        reduced = union - {variable}
        if reduced:
            active.append(reduced)
        remaining.remove(variable)
    return int(largest)


def _round_down_sum(values):
    result = 0.0
    for value in values:
        result = math.nextafter(result + float(value), -math.inf)
    return result


def _round_up_tail_fraction(partition_upper, retained_lower):
    _require(math.isfinite(partition_upper) and partition_upper > 0.0,
             "partition upper envelope is invalid")
    _require(math.isfinite(retained_lower) and retained_lower > 0.0,
             "retained lower mass is invalid")
    _require(partition_upper >= retained_lower,
             "partition upper envelope fell below the retained lower mass")
    tail_upper = math.nextafter(float(partition_upper) - float(retained_lower), math.inf)
    return tail_upper, math.nextafter(tail_upper / float(retained_lower), math.inf)


def _context(config):
    registry_path = EXP102_ROOT / "registry/registry.json"
    registry = load_registry(registry_path)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "depth-two feasibility registry bytes changed")
    _unused, code, H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, _frame = build_model(H)
    uniform_seed, _epsilon, syndrome_flat = _disorder(registry, code, model, config["cell"])
    H = np.ascontiguousarray(H, dtype=np.uint8)
    _require(H.shape == (24, 32) and model.num_qubits == 1600 and model.k == 64,
             "depth-two feasibility m8 dimensions changed")
    _require(int(syndrome_flat.sum()) == 160, "depth-two feasibility syndrome changed")
    return registry, code, H, model, int(uniform_seed), syndrome_flat.reshape(H.shape[0], H.shape[1])


def _retained_anchor_rows(config, model, H, syndrome_flat, syndrome_matrix,
                          lower_mass, upper_mass, scale_lower):
    catalog = build_milp_map_anchors(
        model.H_check, syndrome_flat, config["cell"]["p"],
        max_anchors=config["anchor"]["requested_max_anchors"],
    )
    _require(catalog.size == config["anchor"]["anchor_count"]
             and catalog.anchor_sha256 == config["anchor"]["anchor_sha256"],
             "depth-two feasibility MAP anchor identity changed")
    unique = {}
    for index, state in enumerate(catalog.anchors):
        _A, B = split_hgp_state(state, H)
        packed = np.packbits(B, bitorder="little").tobytes()
        lower, upper, masks = scaled_state_weight_interval(
            H, syndrome_matrix, B, lower_mass, upper_mass, scale_lower,
            *config["classical_p_rational"],
        )
        record = {
            "source_anchor_index": int(index),
            "full_state_weight": int(state.sum()),
            "B_weight": int(B.sum()),
            "B_sha256": hashlib.sha256(packed).hexdigest(),
            "scaled_marginal_lower": lower,
            "scaled_marginal_upper": upper,
            "classical_syndrome_masks": [int(value) for value in masks],
        }
        existing = unique.get(packed)
        if existing is None or float(record["scaled_marginal_lower"]) > float(existing["scaled_marginal_lower"]):
            unique[packed] = record
    rows = sorted(unique.values(), key=lambda item: item["B_sha256"])
    _require(rows, "depth-two feasibility retained no B anchors")
    return catalog, rows, _round_down_sum(item["scaled_marginal_lower"] for item in rows)


def run_probe(config):
    registry, code, H, model, uniform_seed, syndrome = _context(config)
    depth = config["envelope"]["depth"]
    scopes = row_prefix_factor_scopes(H, depth)
    order, induced_width = binary_elimination_plan(scopes, depth * H.shape[0])
    max_combined = _max_combined_variables(scopes, depth * H.shape[0])
    largest_table_bytes = (1 << max_combined) * np.dtype(np.float64).itemsize
    _require(induced_width == config["envelope"]["expected_induced_width"]
             and max_combined == config["envelope"]["expected_max_combined_variables"],
             "depth-two feasibility elimination structure changed")
    _require(largest_table_bytes <= config["envelope"]["memory_cap_bytes"],
             "depth-two feasibility preflight exceeds its memory cap")

    numerator, denominator = config["classical_p_rational"]
    lower_mass, upper_mass = classical_coset_mass_interval(H, numerator, denominator)
    scale_lower = float(np.max(lower_mass))
    _require(scale_lower > 0.0, "depth-two feasibility scale vanished")
    catalog, retained_rows, retained_lower = _retained_anchor_rows(
        config, model, H, syndrome.reshape(-1), syndrome,
        lower_mass, upper_mass, scale_lower,
    )
    tables = prefix_max_upper_tables(upper_mass, (depth,))
    started = time.monotonic()
    partition_upper, observed_order, observed_width = row_prefix_partition_upper(
        H, syndrome, depth, tables[depth], scale_lower, numerator, denominator,
        config["envelope"]["expected_induced_width"],
    )
    elapsed = time.monotonic() - started
    _require(tuple(observed_order) == tuple(order) and observed_width == induced_width,
             "depth-two feasibility elimination order drifted")
    tail_upper, tail_fraction_upper = _round_up_tail_fraction(partition_upper, retained_lower)
    runtime_pass = elapsed <= config["envelope"]["max_runtime_seconds"]
    tightness_pass = tail_fraction_upper <= config["envelope"]["tail_fraction_goal"]
    status = (
        "DEPTH2_ENVELOPE_TIGHTNESS_PASS_BUT_NO_QTOP"
        if runtime_pass and tightness_pass
        else "DEPTH2_ENVELOPE_NOT_TIGHT_ENOUGH"
    )
    return {
        "status": status,
        "registry_sha256": registry["registry_sha256"],
        "cell": config["cell"],
        "disorder_uniform_seed": uniform_seed,
        "dimensions": {
            "classical_rows": int(H.shape[0]),
            "classical_columns": int(H.shape[1]),
            "collapsed_B_bits": int(H.shape[0] * H.shape[0]),
            "physical_qubits": int(model.num_qubits),
            "logical_dimension": int(model.k),
            "syndrome_weight": int(syndrome.sum()),
        },
        "preflight": {
            "factor_count": int(len(scopes)),
            "induced_width": int(induced_width),
            "max_combined_variables": max_combined,
            "largest_float64_table_bytes": int(largest_table_bytes),
            "memory_cap_bytes": config["envelope"]["memory_cap_bytes"],
        },
        "interval_mass": {
            "sum_lower": float(lower_mass.sum(dtype=np.float64)),
            "sum_upper": float(upper_mass.sum(dtype=np.float64)),
            "max_lower_scale": scale_lower,
            "lower_sha256": hashlib.sha256(lower_mass.astype(">f8").tobytes()).hexdigest(),
            "upper_sha256": hashlib.sha256(upper_mass.astype(">f8").tobytes()).hexdigest(),
        },
        "retained_B_marginals": {
            "anchor_catalog_sha256": catalog.anchor_sha256,
            "rows": retained_rows,
            "scaled_mass_lower": retained_lower,
        },
        "partition_upper": {
            "scaled_mass_upper": partition_upper,
            "scaled_tail_upper": tail_upper,
            "tail_fraction_upper": tail_fraction_upper,
            "tail_fraction_goal": config["envelope"]["tail_fraction_goal"],
            "tightness_pass": tightness_pass,
        },
        "runtime": {
            "envelope_seconds": elapsed,
            "runtime_cap_seconds": config["envelope"]["max_runtime_seconds"],
            "runtime_pass": runtime_pass,
            "ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "ru_maxrss_unit": "bytes_on_darwin",
        },
        "does_not_establish": [
            "A posterior, q_top, purity, or physical parameter-point result.",
            "A complete logical-sector mass decomposition or character interval.",
            "Any remote, formal, held-out, or production authorization.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace depth-two feasibility report: {args.output}")
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
