"""Outcome-blind algebra and runtime preflight for BP-systematic proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    build_bp_systematic_proposal,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_bp_systematic.preflight.v0"


class PreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise PreflightError(message)


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise PreflightError("BP-systematic preflight config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "BP-systematic preflight config is not canonical")
    expected = {
        "bp", "cell", "component_weights", "config_version", "contract_version", "orders",
        "preflight_schedule", "registry_sha256", "seed_namespace",
    }
    _require(set(config) == expected
             and config["config_version"] == "exp102.q0_bp_systematic.preflight.config.v0"
             and config["contract_version"] == CONTRACT_VERSION,
             "BP-systematic preflight config schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "BP-systematic preflight cell changed")
    _require(config["registry_sha256"] == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "BP-systematic registry identity changed")
    _require(config["bp"] == {
        "damping": 0.5, "iterations": 64, "llr_cap": 30.0, "min_probability": 1e-5,
    }, "BP-systematic BP settings changed")
    _require(config["component_weights"] == [0.9, 0.09, 0.01],
             "BP-systematic component weights changed")
    _require(config["orders"] == ["forward", "reverse"],
             "BP-systematic order panel changed")
    _require(config["preflight_schedule"] == {
        "draws_per_order": 128, "no_estimator": True, "no_target_weight_read": True,
    }, "BP-systematic preflight schedule changed")
    _require(config["seed_namespace"] == "exp102.q0_bp_systematic.preflight.v0.20260724",
             "BP-systematic seed namespace changed")
    return config, sha256_file(path)


def _source_identity(root):
    paths = {
        "config": root / "config/q0_bp_systematic.preflight.v0.json",
        "module": root / "exp102_pipeline/q0_bp_systematic.py",
        "runner": Path(__file__),
        "registry": root / "registry/registry.json",
    }
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    core = {"source_commit": source_commit,
            "files": {name: sha256_file(path) for name, path in paths.items()}}
    return {**core, "source_identity_sha256": sha256_json(core)}


def _order(name, size):
    if name == "forward":
        return np.arange(size, dtype=np.int32)
    if name == "reverse":
        return np.arange(size - 1, -1, -1, dtype=np.int32)
    raise PreflightError("unknown frozen systematic order")


def run_preflight(config, config_sha256):
    root = Path(__file__).resolve().parents[2]
    registry_path = root / "registry/registry.json"
    registry = load_registry(registry_path)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "BP-systematic registry bytes changed")
    _unused, code, _H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, _frame = build_model(_H)
    uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, config["cell"])
    _require(model.num_qubits == 1600 and model.num_checks == 768 and model.k == 64
             and int(syndrome.sum()) == 160,
             "BP-systematic m8 model identity changed")
    result = []
    for name in config["orders"]:
        started = time.perf_counter()
        proposal = build_bp_systematic_proposal(
            model, syndrome, config["cell"]["p"], column_order=_order(name, model.num_qubits),
            bp_iterations=config["bp"]["iterations"], bp_damping=config["bp"]["damping"],
            bp_llr_cap=config["bp"]["llr_cap"],
            min_probability=config["bp"]["min_probability"],
            component_weights=config["component_weights"],
        )
        construction_seconds = time.perf_counter() - started
        load_exp101()
        from exp101_certified_src.prng import PortablePrng

        seed = derive_seed(
            config["seed_namespace"], config_sha256, registry["registry_sha256"], name,
            proposal.proposal_sha256, "draws",
        )
        rng = PortablePrng(seed)
        started = time.perf_counter()
        component_counts = np.zeros(proposal.num_components, dtype=np.int64)
        for _ in range(config["preflight_schedule"]["draws_per_order"]):
            draw = proposal.sample(rng)
            state = np.asarray(draw["state"], dtype=np.uint8)
            _require(np.array_equal(
                (model.H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8),
                syndrome,
            ), "BP-systematic draw escaped the hard coset")
            _require(np.isfinite(float(draw["log_q"]))
                     and abs(float(draw["log_q"]) - float(proposal.log_probability_state(state))) <= 1e-12,
                     "BP-systematic proposal density did not replay")
            component_counts[int(draw["component_index"])] += 1
        draw_seconds = time.perf_counter() - started
        result.append({
            "order": name,
            "coordinate_sha256": proposal.coordinates.coordinate_sha256,
            "proposal_sha256": proposal.proposal_sha256,
            "construction_seconds": construction_seconds,
            "draw_seconds_total": draw_seconds,
            "draw_seconds_per_sample": draw_seconds / config["preflight_schedule"]["draws_per_order"],
            "component_counts": component_counts.tolist(),
            "bp_final_max_message_delta": proposal.bp_diagnostics.final_max_message_delta,
            "bp_marginal_min": float(proposal.bp_diagnostics.marginal_probability_one.min()),
            "bp_marginal_max": float(proposal.bp_diagnostics.marginal_probability_one.max()),
            "free_bp_probability_mean": float(proposal.component_probabilities[0].mean()),
            "hard_coset_draw_replay": "PASS",
        })
    return {
        "authority": "outcome_blind_local_runtime_and_algebra_only",
        "cell": config["cell"],
        "config_sha256": config_sha256,
        "registry_sha256": registry["registry_sha256"],
        "m8_disorder_uniform_seed": int(uniform_seed),
        "model": {"num_qubits": model.num_qubits, "num_checks": model.num_checks,
                  "logical_dimension": model.k, "syndrome_weight": int(syndrome.sum())},
        "orders": result,
        "does_not_establish": [
            "A target weight, posterior estimate, purity, or q_top.",
            "Importance-weight stability or coverage of unobserved modes.",
            "Any remote, formal, held-out, or production authority.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace BP-systematic preflight: {args.output}")
    config, config_sha256 = _load_config(args.config)
    core = {
        "contract_version": CONTRACT_VERSION,
        "source_identity": _source_identity(Path(__file__).resolve().parents[2]),
        "preflight": run_preflight(config, config_sha256),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])


if __name__ == "__main__":
    main()
