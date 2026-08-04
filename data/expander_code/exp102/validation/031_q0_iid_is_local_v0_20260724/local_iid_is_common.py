"""Shared immutable-input helpers for the local q=0 iid-MIS diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import (
    load_logical_stratified_frozen_artifact,
)
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    build_map_mixture_proposal,
    build_milp_map_anchors,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


LOCAL_CONTRACT_VERSION = "exp102.q0_iid_is.local.v0"
LOCAL_RAW_FIELDS = {
    "metadata_json", "content_sha256", "states_packed", "labels", "physical_weights",
    "block_indices", "source_indices", "source_log_q", "mixture_log_q",
    "source_log_importance", "mixture_log_importance",
}


class LocalIidIsConflictError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise LocalIidIsConflictError(message)


def exp102_root():
    return Path(__file__).resolve().parents[2]


def load_config(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise LocalIidIsConflictError("iid-MIS config is not JSON") from exc
    _require(serialized == canonical_json(config) + "\n",
             "iid-MIS config is not canonical JSON")
    expected = {
        "cell", "config_version", "contract_version", "gates", "proposal_mixture_weights",
        "proposals", "raw_version", "registry_sha256", "sample_schedule", "scope",
        "seed_namespace",
    }
    _require(set(config) == expected
             and config["config_version"] == "exp102.q0_iid_is.local.config.v0"
             and config["contract_version"] == LOCAL_CONTRACT_VERSION
             and config["raw_version"] == "exp102.q0_iid_importance.local.raw.v0",
             "iid-MIS config version/schema changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "iid-MIS sentinel cell changed")
    _require(isinstance(config["registry_sha256"], str)
             and len(config["registry_sha256"]) == 64,
             "iid-MIS registry SHA is invalid")
    schedule = config["sample_schedule"]
    _require(schedule == {
        "block_count": 16,
        "draws_per_proposal_per_block": 1024,
        "layout": "block_major_then_proposal_then_draw",
        "no_mcmc_initial_state": True,
        "no_resampling": True,
        "no_result_dependent_extension": True,
    }, "iid-MIS sample schedule changed")
    _require(config["proposal_mixture_weights"] == [
        0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
    ], "iid-MIS equal mixture changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "maximum_terminal_status": "LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_ONLY",
        "production_authorization": False,
        "purpose": "fresh_iid_proposal_draw_diagnostic_on_m8_sentinel_only",
    }, "iid-MIS authority boundary changed")
    _require(config["seed_namespace"] == "exp102.q0_iid_is.local.v0.20260724",
             "iid-MIS seed namespace changed")
    proposals = config["proposals"]
    _require(isinstance(proposals, list) and len(proposals) == 3
             and [item.get("id") for item in proposals]
             == ["MAM-IMH8", "LSI-IMH-T05", "LSI-IMH-T10"],
             "iid-MIS proposal order changed")
    mam, lsi05, lsi10 = proposals
    _require(mam == {
        "anchor_count": 2,
        "anchor_sha256": "b0ad56f2cd3ec7815c5acb989260ed841276c69d33fc193bc9322a6de96549e5",
        "component_weights": [0.35, 0.3, 0.2, 0.1, 0.045, 0.005],
        "id": "MAM-IMH8",
        "kind": "rebuilt_map_mixture",
        "proposal_sha256": "4356da8b6289e628ee89f78f5664710a24a1e0e1b05b2f8dcaf8585fe25aed68",
        "requested_max_anchors": 8,
        "theta_logical": [0.001, 0.003, 0.02, 0.08, 0.25, 0.5],
        "theta_stabilizer": [0.001, 0.003, 0.01, 0.04, 0.15, 0.5],
    }, "iid-MIS MAM proposal freeze changed")
    expected_lsi = [
        ("LSI-IMH-T05", "lsi_imh_tau_05.npz",
         "2e1d2888002cce9ed3730268c0ffa6b884fed1132942107df9bedcdda0217fd8",
         "b9cd926113478ad8e51b2391f8c764446564bfbe6f14fe56f3e51844825412eb",
         "primary_independent_proposal"),
        ("LSI-IMH-T10", "lsi_imh_tau_10.npz",
         "d22b6f9ce71a094600894d4c64ef4e81b53210380c66db3aee3658cdc747268c",
         "fb8fedd2182b97c193beb0f15321260e4b7884e99f9e43a689a05dc166f83f29",
         "mandatory_concentrated_stress_diagnostic"),
    ]
    prefix = (
        "validation/015_q0_logical_stratified_v0b_20260723/remote_run/"
        "exp102_q0_lsi_v0d_20260723_9f0c473/pulled_run/artifacts/artifacts/"
    )
    for value, (identifier, name, file_sha, proposal_sha, role) in zip(
            (lsi05, lsi10), expected_lsi):
        _require(value == {
            "artifact_relpath": prefix + name,
            "artifact_sha256": file_sha,
            "id": identifier,
            "kind": "historical_frozen_lsi_artifact",
            "proposal_sha256": proposal_sha,
            "role": role,
        }, "iid-MIS LSI proposal freeze changed")
    return config, sha256_file(path)


def build_context(config):
    root = exp102_root()
    registry_path = root / "registry/registry.json"
    registry = load_registry(registry_path)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "iid-MIS registry bytes changed")
    _, code, H = load_frozen_code(registry_path, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, _epsilon, syndrome = _disorder(registry, code, model, config["cell"])
    _require(model.k == 64 and model.num_qubits == 1600 and model.num_checks == 768
             and int(syndrome.sum()) == 160,
             "iid-MIS m8 model/syndrome identity changed")
    return root, registry, code, H, model, frame, uniform_seed, syndrome


def build_proposals(config, root, model, frame, syndrome):
    rows = []
    mam_spec = config["proposals"][0]
    catalog = build_milp_map_anchors(
        model.H_check, syndrome, config["cell"]["p"],
        max_anchors=mam_spec["requested_max_anchors"],
    )
    proposal = build_map_mixture_proposal(
        model, catalog, theta_stabilizer=mam_spec["theta_stabilizer"],
        theta_logical=mam_spec["theta_logical"],
        component_weights=mam_spec["component_weights"],
    )
    _require(catalog.size == mam_spec["anchor_count"]
             and catalog.anchor_sha256 == mam_spec["anchor_sha256"]
             and proposal.proposal_sha256 == mam_spec["proposal_sha256"],
             "iid-MIS rebuilt MAM artifact changed")
    rows.append((mam_spec, proposal, {
        "id": mam_spec["id"], "kind": mam_spec["kind"],
        "anchor_sha256": catalog.anchor_sha256,
        "proposal_sha256": proposal.proposal_sha256,
    }))
    for spec in config["proposals"][1:]:
        path = root / spec["artifact_relpath"]
        _require(path.is_file() and sha256_file(path) == spec["artifact_sha256"],
                 "iid-MIS frozen LSI artifact bytes changed")
        artifact = load_logical_stratified_frozen_artifact(path, model, frame)
        _require(np.array_equal(artifact.syndrome, syndrome)
                 and artifact.proposal.proposal_sha256 == spec["proposal_sha256"],
                 "iid-MIS frozen LSI artifact algebra changed")
        rows.append((spec, artifact.proposal, {
            "id": spec["id"], "kind": spec["kind"], "role": spec["role"],
            "artifact_relpath": spec["artifact_relpath"],
            "artifact_sha256": spec["artifact_sha256"],
            "proposal_sha256": artifact.proposal.proposal_sha256,
        }))
    return rows


def seed_schedule(config, config_sha256, registry, proposal_rows):
    blocks = config["sample_schedule"]["block_count"]
    cell_sha = sha256_json(config["cell"])
    values = np.empty((blocks, len(proposal_rows)), dtype=np.uint64)
    for block in range(blocks):
        for source, (spec, _proposal, identity) in enumerate(proposal_rows):
            values[block, source] = np.uint64(derive_seed(
                config["seed_namespace"], config_sha256, registry["registry_sha256"],
                cell_sha, spec["id"], identity["proposal_sha256"], block, "iid_draw",
            ))
    _require(np.unique(values).size == values.size, "iid-MIS derived seed collision")
    return values


def raw_content_sha256(metadata_json, arrays):
    digest = hashlib.sha256(b"exp102.q0_iid_importance.local.raw_content.v0\0")
    digest.update(str(metadata_json).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        _require(not value.dtype.hasobject, "iid-MIS raw contains an object array")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def raw_metadata(config, config_sha256, registry, model, uniform_seed, syndrome,
                 proposal_rows, seeds):
    identity = {
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
        "raw_reuse_prohibited": True,
    }
    return identity
