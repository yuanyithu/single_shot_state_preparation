"""Frozen structural preflight for a sparse-coordinate Houdayer kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from data.expander_code.exp102.exp102_pipeline.q0_global import qubit_signatures
from data.expander_code.exp102.exp102_pipeline.q0_houdayer import (
    build_sparse_hgp_coordinate_basis,
    component_logical_mask,
    coordinate_factor_scopes,
    coordinates_to_state,
    houdayer_components,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_houdayer_structure.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"


class HoudayerStructureError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise HoudayerStructureError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise HoudayerStructureError("Houdayer structure config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "Houdayer structure config is not canonical")
    expected = {
        "cell", "config_version", "contract_version", "coordinate_basis",
        "pair_definitions", "registry_sha256", "scope", "seed_namespace", "version",
    }
    _require(set(config) == expected and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"] == "exp102.q0_houdayer_structure.feasibility.config.v0",
             "Houdayer structure config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "Houdayer structure cell changed")
    _require(config["coordinate_basis"] == "h_x_plus_code_only_tensor_logical_complement.v0",
             "Houdayer coordinate basis changed")
    _require(config["pair_definitions"] == [
        "p_vs_lowest_tensor", "p_vs_lowest_tensor_rank8",
        "p_vs_lowest_stabilizer_rank64", "uniform_coordinate_pair",
    ], "Houdayer pair definitions changed")
    _require(config["seed_namespace"] == "q0_houdayer_structure_feasibility_v0",
             "Houdayer structure seed namespace changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "Houdayer structure registry SHA changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "houdayer_coordinate_component_feasibility_only",
        "remote_authorization": False,
    }, "Houdayer structure scope changed")
    return config, sha256_file(path)


def _source_binding(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "config": sha256_file(config_path),
        "probe": sha256_file(Path(__file__)),
        "houdayer": sha256_file(EXP102_ROOT / "exp102_pipeline/q0_houdayer.py"),
        "registry": sha256_file(REGISTRY_PATH),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _rank_uint64(values):
    basis = {}
    for entry in values:
        value = int(entry)
        while value:
            pivot = value.bit_length() - 1
            previous = basis.get(pivot)
            if previous is None:
                basis[pivot] = value
                break
            value ^= previous
    return len(basis)


def _uniform_coordinates(length, seed):
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return (generator.integers(0, 2, size=int(length), dtype=np.uint8)).astype(np.uint8)


def _pair_coordinates(name, basis, config, registry, code):
    generators = basis["generators"]
    stabilizer_count = basis["stabilizer_count"]
    logical_count = basis["logical_count"]
    logical_weights = np.asarray(basis["tensor_logicals"].sum(axis=1), dtype=np.int64)
    tensor_order = np.lexsort((np.arange(logical_count, dtype=np.int64), logical_weights))
    stabilizer_weights = np.asarray(generators[:stabilizer_count].sum(axis=1), dtype=np.int64)
    stabilizer_order = np.lexsort((np.arange(stabilizer_count, dtype=np.int64), stabilizer_weights))
    left = np.zeros(generators.shape[0], dtype=np.uint8)
    right = left.copy()
    if name == "p_vs_lowest_tensor":
        right[stabilizer_count + int(tensor_order[0])] = 1
    elif name == "p_vs_lowest_tensor_rank8":
        right[stabilizer_count + tensor_order[:8]] = 1
    elif name == "p_vs_lowest_stabilizer_rank64":
        right[stabilizer_order[:64]] = 1
    elif name == "uniform_coordinate_pair":
        left = _uniform_coordinates(
            generators.shape[0],
            derive_seed(config["seed_namespace"], registry["registry_sha256"], code["code_id"], name, 0),
        )
        right = _uniform_coordinates(
            generators.shape[0],
            derive_seed(config["seed_namespace"], registry["registry_sha256"], code["code_id"], name, 1),
        )
    else:  # pragma: no cover - config validation keeps this unreachable
        raise HoudayerStructureError("unknown Houdayer pair definition")
    return left, right


def _component_record(components, difference, basis):
    logical_masks = basis["logical_masks"]
    stabilizer_count = basis["stabilizer_count"]
    component_masks = [
        component_logical_mask(component, difference, logical_masks)
        for component in components
    ]
    details = []
    for component, mask in zip(components, component_masks, strict=True):
        details.append({
            "coordinate_count": int(component.size),
            "stabilizer_coordinate_count": int(np.count_nonzero(component < stabilizer_count)),
            "logical_coordinate_count": int(np.count_nonzero(component >= stabilizer_count)),
            "logical_delta_mask": int(mask),
            "logical_delta_rank": _rank_uint64([mask]),
        })
    return {
        "component_count": int(len(components)),
        "largest_component_coordinates": int(max((component.size for component in components), default=0)),
        "component_logical_delta_rank": int(_rank_uint64(component_masks)),
        "components": details,
    }


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "Houdayer structure registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, epsilon, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600 and model.k == 64,
             "Houdayer structure HGP dimensions changed")
    return registry, code, H, model, frame, int(uniform_seed), epsilon, syndrome


def run_probe(config):
    registry, code, H, model, frame, uniform_seed, planted, syndrome = _context(config)
    basis = build_sparse_hgp_coordinate_basis(H, model, frame)
    factors = coordinate_factor_scopes(basis["generators"])
    generator_weights = basis["generators"].sum(axis=1)
    factor_degrees = np.asarray([len(scope) for scope in factors], dtype=np.int64)
    records = []
    for name in config["pair_definitions"]:
        left, right = _pair_coordinates(name, basis, config, registry, code)
        difference = left ^ right
        components = houdayer_components(left, right, factors)
        left_state = coordinates_to_state(planted, basis["generators"], left)
        right_state = coordinates_to_state(planted, basis["generators"], right)
        _require(np.array_equal(
            model.H_check.astype(np.int64) @ left_state.astype(np.int64) % 2,
            syndrome,
        ) and np.array_equal(
            model.H_check.astype(np.int64) @ right_state.astype(np.int64) % 2,
            syndrome,
        ), "Houdayer structural pair left the hard coset")
        records.append({
            "pair_definition": name,
            "differing_coordinates": int(difference.sum()),
            "left_weight_diagnostic_only": int(left_state.sum()),
            "right_weight_diagnostic_only": int(right_state.sum()),
            **_component_record(components, difference, basis),
        })
    return {
        "cell": config["cell"],
        "registry_sha256": registry["registry_sha256"],
        "disorder_uniform_seed": uniform_seed,
        "coordinate_basis": {
            "version": basis["version"],
            "coordinate_count": int(basis["generators"].shape[0]),
            "stabilizer_coordinate_count": basis["stabilizer_count"],
            "tensor_logical_coordinate_count": basis["logical_count"],
            "generator_weight_min_max": [int(generator_weights.min()), int(generator_weights.max())],
            "physical_factor_degree_min_max": [int(factor_degrees.min()), int(factor_degrees.max())],
            "physical_factor_degree_mean": float(factor_degrees.mean()),
            "qubit_signature_sha256": hashlib.sha256(qubit_signatures(frame).tobytes()).hexdigest(),
        },
        "identity": {
            "pair_target": "pi(e_left|y)*pi(e_right|y)",
            "move": "uniformly select one complete disagreement component and swap it",
            "invariant": "each touched physical-bit factor exchanges its two replica inputs",
        },
        "records": records,
        "does_not_establish": [
            "A constructed or run MCMC trajectory.",
            "A posterior, q_top, logical-sector probability, or convergence claim.",
            "Any remote, formal, held-out, or production authorization.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "houdayer_structure.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace Houdayer structure report: {args.output}")
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
