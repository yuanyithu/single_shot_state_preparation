"""Frozen local witness screen for BP-mixture rejection-envelope feasibility."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys

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
from data.expander_code.exp102.exp102_pipeline.q0_bp_dominance import (
    BP_DOMINANCE_VERSION,
    canonical_rank_complete_logical_witnesses,
    deterministic_witness_panel,
    dominance_record,
)
from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    build_bp_systematic_proposal,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_bp_dominance_witness.feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"


class DominanceProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DominanceProbeError(message)


def _order(name, size):
    if name == "forward":
        return np.arange(int(size), dtype=np.int32)
    if name == "reverse":
        return np.arange(int(size) - 1, -1, -1, dtype=np.int32)
    raise DominanceProbeError("unknown frozen systematic order")


def _load_config(path):
    serialized = Path(path).read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise DominanceProbeError("dominance witness config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "dominance witness config is not canonical")
    expected = {
        "bp", "cell", "component_weights", "config_version", "contract_version", "orders",
        "rejection_envelope", "registry_sha256", "scope", "version", "witness_panel",
    }
    _require(set(config) == expected
             and config["version"] == PROBE_VERSION
             and config["contract_version"] == PROBE_VERSION
             and config["config_version"]
             == "exp102.q0_bp_dominance_witness.feasibility.config.v0",
             "dominance witness config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "dominance witness cell changed")
    _require(config["registry_sha256"]
             == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "dominance witness registry identity changed")
    _require(config["bp"] == {
        "damping": 0.5, "iterations": 64, "llr_cap": 30.0, "min_probability": 1e-5,
    }, "dominance witness BP settings changed")
    _require(config["component_weights"] == [0.9, 0.09, 0.01],
             "dominance witness mixture weights changed")
    _require(config["orders"] == ["forward", "reverse"],
             "dominance witness order panel changed")
    _require(config["witness_panel"] == {
        "candidate_orders": [1, 2, 3],
        "include_planted": True,
        "include_systematic_coordinate_neighbors": True,
        "logical_rank_count": 64,
        "selection_order": "state_weight,move_weight,signature,packed_move",
    }, "dominance witness panel changed")
    _require(config["rejection_envelope"] == {
        "max_expected_proposal_calls_per_exact_draw": 1_000_000,
        "planned_exact_draw_count": 128,
    }, "dominance witness rejection cap changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "posterior_estimation": False,
        "production_authorization": False,
        "purpose": "proposal_dominance_witness_feasibility_only",
    }, "dominance witness scope changed")
    return config, sha256_file(path)


def _source_identity(config_path):
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    paths = {
        "config": Path(config_path),
        "bp_dominance": EXP102_ROOT / "exp102_pipeline/q0_bp_dominance.py",
        "bp_systematic": EXP102_ROOT / "exp102_pipeline/q0_bp_systematic.py",
        "q0_global": EXP102_ROOT / "exp102_pipeline/q0_global.py",
        "q0_hgp_screen": EXP102_ROOT / "exp102_pipeline/q0_hgp_screen.py",
        "registry": REGISTRY_PATH,
        "runner": Path(__file__),
        "worker": EXP102_ROOT / "exp102_pipeline/worker.py",
    }
    core = {
        "source_commit": source_commit,
        "files": {name: sha256_file(path) for name, path in paths.items()},
    }
    return {**core, "source_identity_sha256": sha256_json(core)}


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "dominance witness registry bytes changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600
             and model.num_checks == 768 and model.k == 64
             and int(syndrome.sum()) == 160,
             "dominance witness m8 model identity changed")
    residual = (model.H_check.astype(np.int64) @ planted.astype(np.int64) % 2).astype(np.uint8)
    _require(np.array_equal(residual, syndrome), "planted witness is outside the hard coset")
    return registry, model, frame, int(uniform_seed), planted, syndrome


def _build_proposals(model, syndrome, config):
    proposals = {}
    for order in config["orders"]:
        proposal = build_bp_systematic_proposal(
            model, syndrome, config["cell"]["p"],
            column_order=_order(order, model.num_qubits),
            bp_iterations=config["bp"]["iterations"],
            bp_damping=config["bp"]["damping"],
            bp_llr_cap=config["bp"]["llr_cap"],
            min_probability=config["bp"]["min_probability"],
            component_weights=config["component_weights"],
        )
        proposals[order] = proposal
    _require(tuple(proposals) == ("forward", "reverse"),
             "dominance witness proposal order changed")
    return proposals


def _run_probe(config, config_sha256):
    registry, model, frame, uniform_seed, planted, syndrome = _context(config)
    proposals = _build_proposals(model, syndrome, config)
    rank_witnesses = canonical_rank_complete_logical_witnesses(
        model, frame, planted, candidate_orders=config["witness_panel"]["candidate_orders"],
    )
    _require(len(rank_witnesses) == config["witness_panel"]["logical_rank_count"],
             "dominance witness logical rank panel changed")
    witnesses = deterministic_witness_panel(
        model, frame, planted, proposals,
        candidate_orders=config["witness_panel"]["candidate_orders"],
    )
    cap = Decimal(str(config["rejection_envelope"]["max_expected_proposal_calls_per_exact_draw"]))
    planned_draws = Decimal(str(config["rejection_envelope"]["planned_exact_draw_count"]))
    best = {order: None for order in proposals}
    report_witnesses = []
    for witness in witnesses:
        state = np.asarray(witness["state"], dtype=np.uint8)
        residual = (model.H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8)
        _require(np.array_equal(residual, syndrome), "deterministic witness escaped the hard coset")
        scores = {}
        for order, proposal in proposals.items():
            record = dominance_record(state, proposal, config["cell"]["p"])
            lower = Decimal(record["posterior_to_proposal_lower"])
            scores[order] = record
            candidate = {
                "state_sha256": witness["state_sha256"],
                "origins": list(witness["origins"]),
                **record,
            }
            if best[order] is None or lower > Decimal(best[order]["posterior_to_proposal_lower"]):
                best[order] = candidate
        report_witnesses.append({
            "origins": list(witness["origins"]),
            "state_sha256": witness["state_sha256"],
            "scores": scores,
        })
    _require(all(value is not None for value in best.values()),
             "dominance witness panel has no proposal scores")
    summaries = {}
    for order, record in best.items():
        lower = Decimal(record["posterior_to_proposal_lower"])
        summaries[order] = {
            "best_witness": record,
            "cap_exceeded": bool(lower > cap),
            "minimum_expected_proposal_calls_per_exact_draw": record[
                "posterior_to_proposal_lower"
            ],
            "minimum_expected_proposal_calls_for_planned_draw_count": format(
                lower * planned_draws, "e",
            ),
            "proposal_sha256": proposals[order].proposal_sha256,
            "coordinate_sha256": proposals[order].coordinates.coordinate_sha256,
        }
    exceeded_orders = [order for order, summary in summaries.items() if summary["cap_exceeded"]]
    if len(exceeded_orders) == len(summaries):
        terminal = "BP_MIXTURE_REJECTION_ENVELOPE_NOT_VIABLE_FOR_ALL_FROZEN_ORDERS"
    elif exceeded_orders:
        terminal = "BP_MIXTURE_REJECTION_ENVELOPE_PARTIAL_WITNESS_REJECTION"
    else:
        terminal = "BP_MIXTURE_REJECTION_ENVELOPE_WITNESS_INCONCLUSIVE"
    return {
        "authority": "local_structural_witness_only_not_a_posterior_estimator",
        "bp_dominance_version": BP_DOMINANCE_VERSION,
        "cell": config["cell"],
        "config_sha256": config_sha256,
        "does_not_establish": [
            "A global upper dominance bound, a rejection sampler, or bounded importance coverage.",
            "A posterior, purity, q_top, MCMC convergence, or an initialization verdict.",
            "Any remote, formal, held-out, or production authorization.",
        ],
        "m8_disorder_uniform_seed": uniform_seed,
        "model": {
            "logical_dimension": int(model.k),
            "num_checks": int(model.num_checks),
            "num_qubits": int(model.num_qubits),
            "syndrome_weight": int(syndrome.sum()),
        },
        "normalizer_bound": {
            "derivation": "Z <= (1-p)^(-n) from Pr_p(H_Z e=y) <= 1",
            "p": config["cell"]["p"],
        },
        "proposal_summaries": summaries,
        "registry_sha256": registry["registry_sha256"],
        "terminal_status": terminal,
        "witness_count": len(report_witnesses),
        "witness_panel": report_witnesses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace dominance witness report")
    config, config_sha256 = _load_config(args.config)
    core = {
        "contract_version": PROBE_VERSION,
        "source_identity": _source_identity(args.config),
        "probe": _run_probe(config, config_sha256),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])


if __name__ == "__main__":
    main()
