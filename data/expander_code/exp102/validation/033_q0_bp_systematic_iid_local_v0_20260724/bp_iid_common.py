"""Frozen-input helpers for the local BP-systematic IID-MIS diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import canonical_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    build_bp_systematic_proposal,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    build_map_mixture_proposal,
    build_milp_map_anchors,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


LOCAL_CONTRACT_VERSION = "exp102.q0_bp_systematic_iid.local.v0"
LOCAL_RAW_FIELDS = {
    "metadata_json", "content_sha256", "states_packed", "coordinates_packed", "labels",
    "physical_weights", "block_indices", "source_indices", "anchor_indices",
    "component_indices", "source_log_q", "mixture_log_q", "source_log_importance",
    "mixture_log_importance",
}
PRELIGHT_REPORT = (
    "validation/032_q0_bp_systematic_preflight_20260724/bp_systematic_runtime_probe.json"
)


class BpIidConflictError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise BpIidConflictError(message)


def exp102_root():
    return Path(__file__).resolve().parents[2]


def _order(name, size):
    if name == "forward":
        return np.arange(size, dtype=np.int32)
    if name == "reverse":
        return np.arange(size - 1, -1, -1, dtype=np.int32)
    raise BpIidConflictError("unknown frozen BP systematic order")


def load_config(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise BpIidConflictError("BP-systematic IID config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "BP-systematic IID config is not canonical")
    expected = {
        "bp", "cell", "component_weights", "config_version", "contract_version", "gates",
        "preflight_report_sha256", "proposal_mixture_weights", "proposals", "raw_version",
        "registry_sha256", "sample_schedule", "scope", "seed_namespace",
    }
    _require(set(config) == expected
             and config["config_version"] == "exp102.q0_bp_systematic_iid.local.config.v0"
             and config["contract_version"] == LOCAL_CONTRACT_VERSION
             and config["raw_version"] == "exp102.q0_bp_systematic_iid.local.raw.v0",
             "BP-systematic IID config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "BP-systematic IID cell changed")
    _require(config["registry_sha256"] == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b",
             "BP-systematic IID registry SHA changed")
    _require(config["bp"] == {
        "damping": 0.5, "iterations": 64, "llr_cap": 30.0, "min_probability": 1e-5,
    } and config["component_weights"] == [0.9, 0.09, 0.01],
             "BP-systematic IID proposal parameters changed")
    _require(config["proposal_mixture_weights"] == [
        0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
    ], "BP-systematic IID mixture changed")
    _require(config["sample_schedule"] == {
        "block_count": 16, "draws_per_proposal_per_block": 1024,
        "layout": "block_major_then_proposal_then_draw", "no_mcmc_initial_state": True,
        "no_resampling": True, "no_result_dependent_extension": True,
    }, "BP-systematic IID schedule changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "maximum_terminal_status": "LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_ONLY",
        "production_authorization": False,
        "purpose": "fresh_bp_systematic_iid_diagnostic_on_m8_sentinel_only",
    }, "BP-systematic IID authority scope changed")
    _require(config["seed_namespace"] == "exp102.q0_bp_systematic_iid.local.v0.20260724",
             "BP-systematic IID seed namespace changed")
    gates = config["gates"]
    _require(gates == {
        "agreement_sigma_multiple": 3.0, "agreement_slack": 0.005,
        "max_abs_primary_q_top_delta": 0.04,
        "max_mixture_block_normalized_weight": 0.1,
        "max_mixture_q_top_jackknife_se": 0.03,
        "max_primary_block_normalized_weight": 0.1,
        "max_primary_d2_upper": 0.04,
        "min_component_draws_per_block_total": 32,
        "min_mixture_block_effective_sample_size": 50.0,
        "min_primary_block_effective_sample_size": 50.0,
        "primary_proposal_ids": ["BP-SYS-F64", "BP-SYS-R64"],
        "stress_proposal_id": "MAM-IMH8",
        "stress_role": "mandatory_report_only_not_a_pass_gate",
    }, "BP-systematic IID gates changed")
    expected_bp = [
        ("BP-SYS-F64", "forward",
         "c3282b154306577d860275b8ec66008d4b567e2b12ddd92aba50c5ef2a29f37a",
         "6eda36a3f3b7e7f4503b8e2f65949c4e11cf94b2d35768bb15b3d303d430e4c9"),
        ("BP-SYS-R64", "reverse",
         "c253c964ea5ac83cf6588d08078cb27e9bfed488c207865ae452f665102bec14",
         "f6af143909b89ac0b0c851d0ee2d5cc75178e3e92439eb91fd381fc77d80d531"),
    ]
    for value, (identifier, order, coordinate_sha, proposal_sha) in zip(
            config["proposals"][:2], expected_bp, strict=True):
        _require(value == {
            "column_order": order, "coordinate_sha256": coordinate_sha, "id": identifier,
            "kind": "bp_systematic", "proposal_sha256": proposal_sha,
        }, "BP-systematic IID BP proposal identity changed")
    _require(config["proposals"][2] == {
        "anchor_count": 2,
        "anchor_sha256": "b0ad56f2cd3ec7815c5acb989260ed841276c69d33fc193bc9322a6de96549e5",
        "component_weights": [0.35, 0.3, 0.2, 0.1, 0.045, 0.005],
        "id": "MAM-IMH8", "kind": "rebuilt_map_mixture",
        "proposal_sha256": "4356da8b6289e628ee89f78f5664710a24a1e0e1b05b2f8dcaf8585fe25aed68",
        "requested_max_anchors": 8,
        "theta_logical": [0.001, 0.003, 0.02, 0.08, 0.25, 0.5],
        "theta_stabilizer": [0.001, 0.003, 0.01, 0.04, 0.15, 0.5],
    }, "BP-systematic IID MAM proposal identity changed")
    root = exp102_root()
    preflight_path = root / PRELIGHT_REPORT
    _require(preflight_path.is_file(), "BP-systematic preflight report is missing")
    preflight = json.loads(preflight_path.read_text(encoding="ascii"))
    _require(preflight.get("report_sha256") == config["preflight_report_sha256"],
             "BP-systematic preflight report identity changed")
    return config, sha256_file(path)


def build_context(config):
    root = exp102_root()
    registry = load_registry(root / "registry/registry.json")
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "BP-systematic IID registry bytes changed")
    _unused, code, _H = load_frozen_code(root / "registry/registry.json", config["cell"]["code_id"])
    model, frame = build_model(_H)
    uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, config["cell"])
    _require(model.num_qubits == 1600 and model.num_checks == 768 and model.k == 64
             and int(syndrome.sum()) == 160,
             "BP-systematic IID m8 model/syndrome identity changed")
    return root, registry, code, _H, model, frame, uniform_seed, syndrome


def build_proposals(config, model, syndrome):
    rows = []
    for spec in config["proposals"][:2]:
        proposal = build_bp_systematic_proposal(
            model, syndrome, config["cell"]["p"],
            column_order=_order(spec["column_order"], model.num_qubits),
            bp_iterations=config["bp"]["iterations"], bp_damping=config["bp"]["damping"],
            bp_llr_cap=config["bp"]["llr_cap"],
            min_probability=config["bp"]["min_probability"],
            component_weights=config["component_weights"],
        )
        _require(proposal.coordinates.coordinate_sha256 == spec["coordinate_sha256"]
                 and proposal.proposal_sha256 == spec["proposal_sha256"],
                 "BP-systematic proposal replay changed")
        rows.append((spec, proposal, {
            "id": spec["id"], "kind": spec["kind"], "column_order": spec["column_order"],
            "coordinate_sha256": proposal.coordinates.coordinate_sha256,
            "proposal_sha256": proposal.proposal_sha256,
            "bp_final_max_message_delta": proposal.bp_diagnostics.final_max_message_delta,
        }))
    spec = config["proposals"][2]
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, config["cell"]["p"], max_anchors=spec["requested_max_anchors"],
    )
    proposal = build_map_mixture_proposal(
        model, catalog, theta_stabilizer=spec["theta_stabilizer"],
        theta_logical=spec["theta_logical"], component_weights=spec["component_weights"],
    )
    _require(catalog.size == spec["anchor_count"] and catalog.anchor_sha256 == spec["anchor_sha256"]
             and proposal.proposal_sha256 == spec["proposal_sha256"],
             "BP-systematic IID MAM proposal replay changed")
    rows.append((spec, proposal, {
        "id": spec["id"], "kind": spec["kind"], "anchor_sha256": catalog.anchor_sha256,
        "proposal_sha256": proposal.proposal_sha256,
    }))
    dimensions = {int(proposal.coordinates.dimension) for _spec, proposal, _identity in rows}
    _require(dimensions == {832}, "BP-systematic IID coordinate dimensions changed")
    return rows


def seed_schedule(config, config_sha256, registry, proposal_rows):
    blocks = config["sample_schedule"]["block_count"]
    cell_sha = sha256_json(config["cell"])
    values = np.empty((blocks, len(proposal_rows)), dtype=np.uint64)
    for block in range(blocks):
        for source, (spec, _proposal, identity) in enumerate(proposal_rows):
            values[block, source] = np.uint64(derive_seed(
                config["seed_namespace"], config_sha256, registry["registry_sha256"], cell_sha,
                spec["id"], identity["proposal_sha256"], block, "iid_draw",
            ))
    _require(np.unique(values).size == values.size, "BP-systematic IID seed collision")
    return values


def raw_content_sha256(metadata_json, arrays):
    digest = hashlib.sha256(b"exp102.q0_bp_systematic_iid.local.raw_content.v0\0")
    digest.update(str(metadata_json).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        _require(not value.dtype.hasobject, "BP-systematic IID raw contains an object array")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def raw_metadata(config, config_sha256, registry, model, uniform_seed, syndrome,
                 proposal_rows, seeds):
    return {
        "contract_version": LOCAL_CONTRACT_VERSION,
        "raw_version": config["raw_version"],
        "config_sha256": config_sha256,
        "registry_sha256": registry["registry_sha256"],
        "cell": config["cell"],
        "model_fingerprint": model.fingerprint(),
        "syndrome_packed_sha256": hashlib.sha256(
            np.packbits(syndrome, bitorder="little").tobytes(),
        ).hexdigest(),
        "disorder_uniform_seed": int(uniform_seed),
        "seed_namespace": config["seed_namespace"],
        "seed_schedule_sha256": hashlib.sha256(seeds.astype(">u8").tobytes()).hexdigest(),
        "proposal_identities": [identity for _spec, _proposal, identity in proposal_rows],
        "sample_schedule": config["sample_schedule"],
        "proposal_mixture_weights": config["proposal_mixture_weights"],
        "preflight_report_sha256": config["preflight_report_sha256"],
        "internal_component_provenance": True,
        "raw_reuse_prohibited": True,
    }
