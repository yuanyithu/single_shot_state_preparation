"""Run the frozen center-preserving logical XOR structural feasibility probe."""

from __future__ import annotations

import argparse
import json
import math
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
from data.expander_code.exp102.exp102_pipeline.q0_center_preserving import (
    CENTER_PRESERVING_VERSION,
    build_dressed_logical_catalog,
    validate_dressed_logical_catalog,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    _signature_rank_masks,
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    deterministic_low_energy_logical_starts,
)
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import (
    load_logical_stratified_frozen_artifact,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_center_preserving.structure.v0"
REPORT_VERSION = "exp102.q0_center_preserving.structure.report.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"


class StructureProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise StructureProbeError(message)


def _load_config(path):
    path = Path(path).resolve()
    serialized = path.read_text(encoding="ascii")
    config = json.loads(serialized)
    _require(serialized == canonical_json(config) + "\n", "config is not canonical JSON")
    _require(set(config) == {
        "artifact", "candidate_rule", "cell", "config_version", "contract_version",
        "gates", "initialization_diagnostics", "registry_sha256", "scope",
        "seed_namespace", "version",
    }, "config schema changed")
    _require(config["version"] == config["contract_version"] == CONTRACT_VERSION
             and config["config_version"]
             == "exp102.q0_center_preserving.structure.config.v0",
             "config version changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "hard sentinel changed")
    _require(config["candidate_rule"] == {
        "max_moves": 127,
        "selection": "rank_first_then_anchor_weight_fill",
        "sources": ["base_xor_codebook_move", "decoded_absolute_representative"],
        "tie_break": ["anchor_weight", "packed_anchor"],
    }, "candidate rule changed")
    _require(config["gates"] == {
        "expected_accept_threshold": 4.0,
        "min_base_accessible_signature_rank": 8,
        "min_base_total_expected_accepts": 16.0,
        "min_l_exact_signature_coverage": 8,
        "min_p_accessible_signature_rank": 8,
        "require_each_l_nonuphill_exact_route": True,
        "required_catalog_signature_rank": 64,
        "t3_catalog_sweeps": 32768,
    }, "structural gates changed")
    _require(config["initialization_diagnostics"] == {
        "families": ["BASE", "P", "U", "L"], "l_count": 8, "u_count": 8,
        "u_role": "ungated_adversarial_structure_only",
    }, "initialization diagnostics changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "maximum_terminal_status": "LOCAL_CENTER_PRESERVING_STRUCTURE_VIABLE",
        "posterior_estimation": False,
        "production_authorization": False,
        "remote_authorization": False,
    }, "scope changed")
    _require(config["seed_namespace"]
             == "exp102.q0_center_preserving.structure.v0.20260724",
             "seed namespace changed")
    return config, sha256_file(path)


def _source_identity(config_path):
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    paths = {
        "config": Path(config_path).resolve(),
        "module": EXP102_ROOT / "exp102_pipeline/q0_center_preserving.py",
        "runner": Path(__file__).resolve(),
        "review": EXP102_ROOT / "reviews/CENTER_PRESERVING_REVIEW.md",
        "registry": REGISTRY_PATH,
    }
    core = {
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "source_commit": commit,
    }
    return {**core, "source_identity_sha256": sha256_json(core)}


def _profile_state(state, moves, signatures, p, sweeps, threshold):
    proposals = state[None, :] ^ moves
    current_weight = int(state.sum())
    proposal_weights = proposals.sum(axis=1).astype(np.int32)
    deltas = proposal_weights - current_weight
    log_ratios = deltas.astype(np.float64) * math.log(p / (1.0 - p))
    acceptance = np.exp(np.minimum(0.0, log_ratios))
    expected = float(sweeps) * acceptance
    accessible = expected >= float(threshold)
    rank = (
        _signature_rank_masks(signatures[accessible], 64)
        if np.any(accessible) else 0
    )
    return {
        "acceptance_max": float(acceptance.max()),
        "acceptance_median": float(np.median(acceptance)),
        "accessible_count": int(accessible.sum()),
        "accessible_signature_rank": int(rank),
        "current_weight": current_weight,
        "endpoint_weight_max": int(proposal_weights.max()),
        "endpoint_weight_min": int(proposal_weights.min()),
        "expected_accepts_max": float(expected.max()),
        "expected_accepts_median": float(np.median(expected)),
        "total_expected_cross_signature_accepts": float(expected.sum()),
    }


def _minimum_cost_rank_basis(signatures, acceptance, k, sweeps):
    order = sorted(range(len(signatures)), key=lambda index: (
        1.0 / float(acceptance[index]), int(signatures[index]),
    ))
    selected = []
    pivots = {}
    for index in order:
        residue = int(signatures[index])
        while residue:
            pivot = residue.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = residue
                selected.append(index)
                break
            residue ^= pivots[pivot]
        if len(pivots) == int(k):
            break
    _require(len(selected) == int(k), "catalog lost full rank during bottleneck audit")
    inverse_sum = sum(1.0 / float(acceptance[index]) for index in selected)
    equalized = float(sweeps) / inverse_sum
    return {
        "basis_indices": [int(index) for index in selected],
        "basis_max_inverse_acceptance": float(max(
            1.0 / float(acceptance[index]) for index in selected
        )),
        "best_equalized_expected_accepts_per_direction": float(equalized),
        "interpretation": "optimistic state-independent rank-basis scheduler upper diagnostic",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, config_sha = _load_config(args.config)
    source_identity = _source_identity(args.config)
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "registry identity changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600
             and model.num_checks == 768 and model.k == 64
             and int(syndrome.sum()) == 160, "hard sentinel dimensions changed")

    artifact_path = EXP102_ROOT / config["artifact"]["relpath"]
    _require(sha256_file(artifact_path) == config["artifact"]["file_sha256"],
             "historical proposal artifact bytes changed")
    artifact = load_logical_stratified_frozen_artifact(
        artifact_path, model, frame,
    )
    _require(np.array_equal(artifact.syndrome, syndrome),
             "artifact syndrome is not the frozen hard sentinel")
    catalog = build_dressed_logical_catalog(
        model, frame, artifact, max_moves=config["candidate_rule"]["max_moves"],
    )
    validate_dressed_logical_catalog(model, frame, syndrome, catalog)
    moves = catalog.unpack_moves()
    signatures = np.asarray(catalog.signatures, dtype=np.uint64)
    gates = config["gates"]
    sweeps = int(gates["t3_catalog_sweeps"])
    threshold = float(gates["expected_accept_threshold"])

    base = np.asarray(catalog.base_anchor, dtype=np.uint8)
    logical_starts = deterministic_low_energy_logical_starts(
        model, frame, planted, count=config["initialization_diagnostics"]["l_count"],
        orders=(1, 2, 3),
    )
    profiles = {
        "BASE": [_profile_state(base, moves, signatures, 0.04, sweeps, threshold)],
        "P": [_profile_state(planted, moves, signatures, 0.04, sweeps, threshold)],
        "U": [],
        "L": [],
    }
    for index in range(config["initialization_diagnostics"]["u_count"]):
        seed = derive_seed(
            config["seed_namespace"], config_sha, registry["registry_sha256"],
            "U", index, "initialization",
        )
        state = uniform_hard_coset_state(model, syndrome, seed)
        row = _profile_state(state, moves, signatures, 0.04, sweeps, threshold)
        row.update({"index": index, "seed": int(seed), "label": int(state_label(frame, state))})
        profiles["U"].append(row)

    signature_to_index = {
        int(signature): index for index, signature in enumerate(signatures)
    }
    exact_routes = []
    for record in logical_starts:
        state = np.asarray(record["state"], dtype=np.uint8)
        row = _profile_state(state, moves, signatures, 0.04, sweeps, threshold)
        row.update({
            "index": int(record["index"]),
            "label": int(state_label(frame, state)),
            "logical_start_signature": int(record["signature"]),
            "logical_start_move_weight": int(record["move_weight"]),
        })
        profiles["L"].append(row)
        catalog_index = signature_to_index.get(int(record["signature"]))
        if catalog_index is None:
            exact_routes.append({
                "catalog_index": None, "covered": False,
                "logical_start_signature": int(record["signature"]),
            })
            continue
        endpoint = state ^ moves[catalog_index]
        delta = int(endpoint.sum()) - int(state.sum())
        acceptance = math.exp(min(0.0, delta * math.log(0.04 / 0.96)))
        exact_routes.append({
            "catalog_index": int(catalog_index),
            "covered": True,
            "endpoint_label": int(state_label(frame, endpoint)),
            "endpoint_weight": int(endpoint.sum()),
            "expected_accepts": float(sweeps * acceptance),
            "logical_start_signature": int(record["signature"]),
            "nonuphill": bool(delta <= 0),
            "start_weight": int(state.sum()),
            "weight_delta": delta,
        })

    base_proposals = base[None, :] ^ moves
    base_delta = base_proposals.sum(axis=1).astype(np.int32) - int(base.sum())
    base_acceptance = np.exp(np.minimum(
        0.0, base_delta.astype(np.float64) * math.log(0.04 / 0.96),
    ))
    bottleneck = _minimum_cost_rank_basis(
        signatures, base_acceptance, model.k, sweeps,
    )
    catalog_rank = _signature_rank_masks(signatures, model.k)
    covered_routes = [row for row in exact_routes if row["covered"]]
    base_profile = profiles["BASE"][0]
    p_profile = profiles["P"][0]
    checks = {
        "base_accessible_rank": (
            base_profile["accessible_signature_rank"]
            >= gates["min_base_accessible_signature_rank"]
        ),
        "base_total_expected_accepts": (
            base_profile["total_expected_cross_signature_accepts"]
            >= gates["min_base_total_expected_accepts"]
        ),
        "catalog_rank": catalog_rank == gates["required_catalog_signature_rank"],
        "l_exact_signature_coverage": (
            len(covered_routes) >= gates["min_l_exact_signature_coverage"]
        ),
        "l_exact_routes_nonuphill": (
            not gates["require_each_l_nonuphill_exact_route"]
            or (len(covered_routes) == len(exact_routes)
                and all(row["nonuphill"] for row in covered_routes))
        ),
        "l_exact_routes_expected": (
            len(covered_routes) == len(exact_routes)
            and all(row["expected_accepts"] >= threshold for row in covered_routes)
        ),
        "p_accessible_rank": (
            p_profile["accessible_signature_rank"]
            >= gates["min_p_accessible_signature_rank"]
        ),
    }
    status = (
        "LOCAL_CENTER_PRESERVING_STRUCTURE_VIABLE"
        if all(checks.values())
        else "LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE"
    )
    core = {
        "artifact": {
            "artifact_content_sha256": str(artifact.descriptor["proposal_sha256"]),
            "file_sha256": sha256_file(artifact_path),
            "relpath": config["artifact"]["relpath"],
        },
        "catalog": {
            "anchor_weight_max": int(catalog.anchor_weights.max()),
            "anchor_weight_min": int(catalog.anchor_weights.min()),
            "catalog_sha256": catalog.catalog_sha256,
            "candidate_source_counts_code_decoded": [
                int(value) for value in catalog.candidate_source_counts
            ],
            "move_weight_max": int(catalog.move_weights.max()),
            "move_weight_min": int(catalog.move_weights.min()),
            "selected_source_counts_code_decoded": [
                int(np.count_nonzero(catalog.source_kind == source)) for source in (0, 1)
            ],
            "signature_rank": int(catalog_rank),
            "size": int(catalog.size),
        },
        "cell": config["cell"],
        "checks": checks,
        "config_sha256": config_sha,
        "contract_version": CONTRACT_VERSION,
        "exact_l_routes": exact_routes,
        "logical_labels": {
            "base": int(state_label(frame, base)),
            "p": int(state_label(frame, planted)),
        },
        "optimistic_full_rank_bottleneck": bottleneck,
        "profiles": profiles,
        "registry_sha256": registry["registry_sha256"],
        "report_version": REPORT_VERSION,
        "scope": config["scope"],
        "source_identity": source_identity,
        "status": status,
        "uniform_disorder_seed": int(uniform_seed),
        "version": CENTER_PRESERVING_VERSION,
    }
    report = {**core, "report_sha256": sha256_json(core)}
    output = ROOT / "structure_report.json"
    _require(not output.exists(), "structure report already exists")
    atomic_json(output, report)
    print(canonical_json({
        "catalog_sha256": catalog.catalog_sha256,
        "checks": checks,
        "report_sha256": report["report_sha256"],
        "status": status,
    }))


if __name__ == "__main__":
    main()
