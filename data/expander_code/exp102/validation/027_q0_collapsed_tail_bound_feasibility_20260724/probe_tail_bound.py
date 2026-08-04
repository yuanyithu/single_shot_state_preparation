"""Bounded feasibility probe for a certified collapsed-B tail envelope.

This is intentionally narrower than a posterior calculation.  It measures
whether a conservative, factorized upper envelope has enough resolution on the
m8 hard sentinel to justify building a complete branch-and-bound certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_collapsed_tail_bound import (
    binary_elimination_plan,
    classical_coset_mass_interval,
    partial_rows_scaled_upper,
    prefix_max_upper_tables,
    row_prefix_factor_scopes,
    row_prefix_partition_upper,
    scaled_state_weight_interval,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import split_hgp_state
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_collapsed_tail_bound.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
DEFAULT_OUTPUT = ROOT / "tail_bound_probe.json"
P_NUMERATOR = 1
P_DENOMINATOR = 25
PREFIX_DEPTHS = (0, 1, 2)
WIDTH_CAP = 18


class ProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ProbeError(message)


def _context():
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, _ = build_model(H)
    uniform_seed = derive_seed(
        "pilot_ladder_m8_attempt22", registry["registry_sha256"],
        code["code_id"], 0, "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < P_NUMERATOR / P_DENOMINATOR
    ).astype(np.uint8)
    syndrome_flat = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    r, n = H.shape
    syndrome = syndrome_flat.reshape(r, n)
    _require(syndrome.any(), "m8 hard syndrome unexpectedly vanished")
    return registry, code, np.ascontiguousarray(H), epsilon, syndrome


def _source_binding():
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    paths = {
        "probe": Path(__file__),
        "tail_bound": EXP102_ROOT / "exp102_pipeline" / "q0_collapsed_tail_bound.py",
        "collapsed": EXP102_ROOT / "exp102_pipeline" / "q0_hgp_collapsed.py",
        "registry": REGISTRY_PATH,
    }
    files = {name: sha256_file(path) for name, path in paths.items()}
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _log10_ratio(upper, lower):
    if not math.isfinite(upper) or not math.isfinite(lower) or upper <= 0.0 or lower <= 0.0:
        return None
    return math.log10(upper) - math.log10(lower)


def _anchor_record(name, B, H, syndrome, lower_mass, upper_mass, scale_lower, tables):
    lower, upper, masks = scaled_state_weight_interval(
        H, syndrome, B, lower_mass, upper_mass, scale_lower,
        P_NUMERATOR, P_DENOMINATOR,
    )
    prefix_records = []
    for depth in PREFIX_DEPTHS:
        node_upper = partial_rows_scaled_upper(
            H, syndrome, B[:depth], tables[depth], scale_lower,
            P_NUMERATOR, P_DENOMINATOR,
        )
        prefix_records.append({
            "depth": int(depth),
            "node_upper_scaled": node_upper,
            "log10_node_upper_over_anchor_lower": _log10_ratio(node_upper, lower),
        })
    return {
        "name": name,
        "B_weight": int(B.sum()),
        "scaled_weight_lower": lower,
        "scaled_weight_upper": upper,
        "classical_syndrome_masks": [int(value) for value in masks],
        "prefix_nodes": prefix_records,
    }


def run_probe():
    registry, code, H, epsilon, syndrome = _context()
    r, n = H.shape
    lower_mass, upper_mass = classical_coset_mass_interval(H, P_NUMERATOR, P_DENOMINATOR)
    scale_lower = float(np.max(lower_mass))
    _require(scale_lower > 0.0, "classical mass lower envelope has no positive scale")
    tables = prefix_max_upper_tables(upper_mass, PREFIX_DEPTHS)
    _, planted_B = split_hgp_state(epsilon, H)
    anchors = [
        _anchor_record("zero_B_truth_free", np.zeros((r, r), dtype=np.uint8), H, syndrome,
                       lower_mass, upper_mass, scale_lower, tables),
        _anchor_record("planted_B_diagnostic_only", planted_B, H, syndrome,
                       lower_mass, upper_mass, scale_lower, tables),
    ]
    partition_records = []
    for depth in PREFIX_DEPTHS:
        scopes = row_prefix_factor_scopes(H, depth)
        order, induced_width = binary_elimination_plan(scopes, depth * r)
        record = {
            "depth": int(depth),
            "variable_count": int(depth * r),
            "factor_count": int(len(scopes)),
            "induced_width": int(induced_width),
            "width_cap": WIDTH_CAP,
            "status": "WIDTH_EXCEEDED",
        }
        if induced_width <= WIDTH_CAP:
            upper, observed_order, observed_width = row_prefix_partition_upper(
                H, syndrome, depth, tables[depth], scale_lower,
                P_NUMERATOR, P_DENOMINATOR, WIDTH_CAP,
            )
            _require(tuple(order) == tuple(observed_order) and induced_width == observed_width,
                     "row-envelope contraction drifted from its planned order")
            record.update({
                "status": "COMPLETE_UPPER_ENVELOPE",
                "partition_upper_scaled": upper,
                "log10_upper_over_zero_B_lower": _log10_ratio(upper, anchors[0]["scaled_weight_lower"]),
            })
        partition_records.append(record)
    return {
        "cell": {
            "code_id": code["code_id"], "p": P_NUMERATOR / P_DENOMINATOR,
            "p_rational": [P_NUMERATOR, P_DENOMINATOR], "disorder_index": 0,
            "disorder_source": "attempt022",
        },
        "registry_sha256": registry["registry_sha256"],
        "dimensions": {
            "classical_rows": int(r), "classical_columns": int(n),
            "collapsed_B_bits": int(r * r),
            "physical_qubits": int(epsilon.size),
            "syndrome_weight": int(syndrome.sum()),
        },
        "interval_mass": {
            "sum_lower": float(lower_mass.sum(dtype=np.float64)),
            "sum_upper": float(upper_mass.sum(dtype=np.float64)),
            "max_lower_scale": scale_lower,
            "max_upper": float(np.max(upper_mass)),
            "lower_sha256": hashlib.sha256(lower_mass.astype(">f8").tobytes()).hexdigest(),
            "upper_sha256": hashlib.sha256(upper_mass.astype(">f8").tobytes()).hexdigest(),
        },
        "anchors": anchors,
        "partition_envelopes": partition_records,
        "interpretation": {
            "result": "NO_QTOP_OR_POSTERIOR_ESTIMATE",
            "what_is_bounded": (
                "Each completed partition value is an outward-rounded upper envelope "
                "for the collapsed B normalizer divided by max_lower**n."
            ),
            "what_is_not_bounded": [
                "The mass of an omitted set relative to a certified retained set.",
                "The logical-character contribution of retained B modes.",
                "q_top, posterior purity, or a formal experimental result.",
            ],
            "next_requirement": (
                "A successor must partition every omitted B branch, add their upper "
                "bounds, and propagate the resulting tail bound through character or "
                "sector-mass intervals before it can estimate q_top."
            ),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    core = {
        "probe_version": PROBE_VERSION,
        "status": "EXPLORATORY_INTERVAL_ENVELOPE_NO_QTOP_OR_READINESS_CLAIM",
        "scope": {
            "posterior_estimation": False,
            "formal_authorization": False,
            "remote_authorization": False,
            "production_authorization": False,
        },
        "source_binding": _source_binding(),
        "probe": run_probe(),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
