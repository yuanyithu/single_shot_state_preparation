"""Structural feasibility probe for a one-copy affine linear-code trellis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

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
from data.expander_code.exp102.exp102_pipeline.q0_trellis_structure import (
    tanner_min_degree_order,
    trellis_state_profile,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_trellis_wmc_structure.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"


class TrellisProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise TrellisProbeError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise TrellisProbeError("trellis-WMC config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n", "trellis-WMC config is not canonical")
    expected = {
        "actionability", "cell", "config_version", "contract_version", "expected_hz_rank",
        "orderings", "registry_sha256", "scope", "version",
    }
    _require(set(config) == expected and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"] == "exp102.q0_trellis_wmc_structure.feasibility.config.v0",
             "trellis-WMC config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "trellis-WMC cell changed")
    _require(config["expected_hz_rank"] == 768, "trellis-WMC H_Z rank changed")
    _require(config["orderings"] == [
        "native_a_then_b", "native_b_then_a", "column_major_a_then_b",
        "column_major_b_then_a", "interleaved_rows", "interleaved_columns",
        "tanner_min_degree",
    ], "trellis-WMC ordering set changed")
    _require(config["actionability"] == {
        "max_state_exponent": 24,
        "max_transition_states": 500000000,
        "two_layer_bytes_per_state": 32,
    }, "trellis-WMC actionability gate changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "trellis-WMC registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "linear_code_trellis_single_copy_structural_feasibility_only",
        "remote_authorization": False,
    }, "trellis-WMC scope changed")
    return config, sha256_file(path)


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "config": sha256_file(config_path),
        "probe": sha256_file(Path(__file__)),
        "registry": sha256_file(REGISTRY_PATH),
        "trellis_structure": sha256_file(EXP102_ROOT / "exp102_pipeline/q0_trellis_structure.py"),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _orders(rows, columns, orderings):
    a_size = columns * columns
    b_size = rows * rows
    a_row = [row * columns + column for row in range(columns) for column in range(columns)]
    b_row = [a_size + row * rows + column for row in range(rows) for column in range(rows)]
    a_column = [row * columns + column for column in range(columns) for row in range(columns)]
    b_column = [a_size + row * rows + column for column in range(rows) for row in range(rows)]
    interleaved_rows = []
    for row in range(max(rows, columns)):
        if row < columns:
            interleaved_rows.extend(row * columns + column for column in range(columns))
        if row < rows:
            interleaved_rows.extend(a_size + row * rows + column for column in range(rows))
    interleaved_columns = []
    for column in range(max(rows, columns)):
        if column < columns:
            interleaved_columns.extend(row * columns + column for row in range(columns))
        if column < rows:
            interleaved_columns.extend(a_size + row * rows + column for row in range(rows))
    table = {
        "native_a_then_b": tuple(a_row + b_row),
        "native_b_then_a": tuple(b_row + a_row),
        "column_major_a_then_b": tuple(a_column + b_column),
        "column_major_b_then_a": tuple(b_column + a_column),
        "interleaved_rows": tuple(interleaved_rows),
        "interleaved_columns": tuple(interleaved_columns),
    }
    return {name: table[name] for name in orderings if name in table}


def _order_sha256(order):
    return hashlib.sha256(np.asarray(order, dtype=">u4").tobytes()).hexdigest()


def _transition_states(exponents):
    return int(sum(1 << int(value) for value in np.asarray(exponents, dtype=np.int64)))


def _record(name, matrix, order, config):
    profile = trellis_state_profile(matrix, order)
    exponents = profile["state_exponents"]
    maximum = int(exponents.max())
    maximum_cuts = np.flatnonzero(exponents == maximum).astype(int).tolist()
    transitions = _transition_states(exponents)
    actionability = config["actionability"]
    bytes_upper = 2 * int(actionability["two_layer_bytes_per_state"]) * (1 << maximum)
    return {
        "ordering": name,
        "order_sha256": _order_sha256(order),
        "rank": int(profile["rank"]),
        "max_state_exponent": maximum,
        "max_state_count": 1 << maximum,
        "max_state_cuts": maximum_cuts,
        "transition_state_upper": transitions,
        "two_layer_working_bytes_at_max": bytes_upper,
        "actionable": bool(
            maximum <= int(actionability["max_state_exponent"])
            and transitions <= int(actionability["max_transition_states"])
        ),
    }


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "trellis-WMC registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, _frame = build_model(H)
    uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600 and model.k == 64,
             "trellis-WMC HGP dimensions changed")
    _require(syndrome.any(), "trellis-WMC hard syndrome vanished")
    return registry, code, H, model, int(uniform_seed), syndrome


def run_probe(config):
    registry, code, H, model, uniform_seed, syndrome = _context(config)
    rows, columns = H.shape
    orders = _orders(rows, columns, config["orderings"])
    orders["tanner_min_degree"] = tanner_min_degree_order(model.H_check)
    _require(set(orders) == set(config["orderings"]), "trellis-WMC order construction changed")
    records = [_record(name, model.H_check, orders[name], config) for name in config["orderings"]]
    _require(all(record["rank"] == config["expected_hz_rank"] for record in records),
             "trellis-WMC H_Z rank differs across orders")
    return {
        "cell": config["cell"],
        "registry_sha256": registry["registry_sha256"],
        "disorder_uniform_seed": uniform_seed,
        "dimensions": {
            "h_rows": int(model.H_check.shape[0]),
            "h_columns": int(model.H_check.shape[1]),
            "classical_rows": int(rows),
            "classical_columns": int(columns),
            "logical_dimension": int(model.k),
            "syndrome_weight": int(syndrome.sum()),
        },
        "identity": {
            "state_exponent": "rank(H_prefix)+rank(H_suffix)-rank(H_Z)",
            "single_copy_character": "Z_u=sum_{H_Z e=y} b**|e|*(-1)**<w_u,e>",
            "scope_invariance": "syndrome and unary character signs do not change trellis width",
        },
        "records": records,
        "any_actionable": any(record["actionable"] for record in records),
        "does_not_establish": [
            "A constructed trellis, numerical partition, posterior, purity, or q_top.",
            "Signed numeric stability or an exact character-estimator implementation.",
            "Any MCMC, remote, formal, held-out, or production authorization.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "trellis_wmc_structure.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace trellis-WMC report: {args.output}")
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
