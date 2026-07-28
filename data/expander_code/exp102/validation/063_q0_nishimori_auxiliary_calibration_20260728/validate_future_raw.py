"""Fail-closed validator for future Nishimori auxiliary-audit raw."""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np


RAW_VERSION = "exp102.q0_nishimori_auxiliary.raw.v1"
MANIFEST_VERSION = "exp102.q0_nishimori_auxiliary.planned_disorders.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
WEIGHTING_RULE = "[sum_basis+(2^k-1-k)*mean_sampled_nonbasis]/(2^k-1)"


class FutureRawConflictError(RuntimeError):
    pass


class EnsembleAuditNotComputable(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def build_planned_manifest(core):
    core = dict(core)
    required = {
        "code_id", "code_sha256", "generation_contract", "generation_config_sha256",
        "p", "planned_count", "planned_disorders", "registry_sha256", "source_commit",
        "source_tree_sha256", "version",
    }
    if set(core) != required:
        raise FutureRawConflictError("planned manifest fields changed")
    if core["version"] != MANIFEST_VERSION:
        raise FutureRawConflictError("planned manifest version changed")
    if core["generation_contract"] != "fresh_iid_bernoulli_truth.v1":
        raise FutureRawConflictError("disorders are not fresh iid Bernoulli truth")
    if not HEX40.fullmatch(str(core["source_commit"])):
        raise FutureRawConflictError("planned source commit is invalid")
    for field in (
        "code_sha256", "generation_config_sha256", "registry_sha256", "source_tree_sha256",
    ):
        if not HEX64.fullmatch(str(core[field])):
            raise FutureRawConflictError(f"planned {field} is invalid")
    planned = core["planned_disorders"]
    if not isinstance(planned, list) or int(core["planned_count"]) != len(planned) or not planned:
        raise FutureRawConflictError("planned disorder count changed")
    expected_indices = list(range(len(planned)))
    actual_indices = [int(row.get("disorder_index", -1)) for row in planned]
    if actual_indices != expected_indices:
        raise FutureRawConflictError("planned disorder order is not canonical")
    seed_identities = [row.get("disorder_seed_identity") for row in planned]
    if any(not isinstance(value, str) or not value for value in seed_identities):
        raise FutureRawConflictError("planned seed identity is empty")
    if len(seed_identities) != len(set(seed_identities)):
        raise FutureRawConflictError("planned disorder seeds are not unique")
    result = dict(core)
    result["manifest_sha256"] = sha256_json(core)
    return result


def validate_planned_manifest(manifest):
    manifest = dict(manifest)
    expected_hash = manifest.pop("manifest_sha256", None)
    rebuilt = build_planned_manifest(manifest)
    if expected_hash != rebuilt["manifest_sha256"]:
        raise FutureRawConflictError("planned manifest self-hash changed")
    return rebuilt


def _array_digest(arrays):
    digest = hashlib.sha256(b"exp102.q0_nishimori_auxiliary.arrays.v1\0")
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(canonical(list(value.shape)).encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def compute_raw_self_hash(identity, arrays):
    unsigned = dict(identity)
    unsigned.pop("raw_self_sha256", None)
    return sha256_json({
        "array_sha256": _array_digest(arrays),
        "identity": unsigned,
    })


def _require_sha(identity, field, pattern=HEX64):
    if not pattern.fullmatch(str(identity.get(field, ""))):
        raise FutureRawConflictError(f"invalid identity hash: {field}")


def _require_array(arrays, name, dtype, shape):
    if name not in arrays:
        raise FutureRawConflictError(f"missing raw array: {name}")
    value = np.asarray(arrays[name])
    if value.dtype != np.dtype(dtype) or value.shape != tuple(shape):
        raise FutureRawConflictError(f"raw array schema changed: {name}")
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise FutureRawConflictError(f"nonfinite raw array: {name}")
    return value


def validate_raw_record(identity, arrays, manifest, schema):
    identity = dict(identity)
    arrays = {str(name): np.asarray(value) for name, value in arrays.items()}
    manifest = validate_planned_manifest(manifest)
    if schema["raw_version"] != RAW_VERSION or identity.get("raw_version") != RAW_VERSION:
        raise FutureRawConflictError("raw version changed")
    if schema["allow_pickle"] is not False:
        raise FutureRawConflictError("pickle was enabled")
    if identity.get("physics_contract_version") != "exp101.physics.v2":
        raise FutureRawConflictError("physics contract changed")
    if identity.get("posterior_ensemble") != "true_posterior":
        raise FutureRawConflictError("posterior ensemble changed")
    if identity.get("sector") != "x_error" or identity.get("H_check_role") != "H_Z":
        raise FutureRawConflictError("sector/H_check wiring changed")
    if float(identity.get("q", math.nan)) != 0.0:
        raise FutureRawConflictError("Nishimori raw is not q=0")
    if identity.get("planned_manifest_sha256") != manifest["manifest_sha256"]:
        raise FutureRawConflictError("raw planned-manifest binding changed")
    if identity.get("code_id") != manifest["code_id"] or float(identity.get("p", math.nan)) != float(manifest["p"]):
        raise FutureRawConflictError("raw code/p changed from planned manifest")
    disorder_index = int(identity.get("disorder_index", -1))
    if not 0 <= disorder_index < int(manifest["planned_count"]):
        raise FutureRawConflictError("raw disorder index is not planned")
    planned = manifest["planned_disorders"][disorder_index]
    if identity.get("disorder_seed_identity") != planned["disorder_seed_identity"]:
        raise FutureRawConflictError("raw disorder seed identity changed")
    _require_sha(identity, "source_commit", HEX40)
    for field in schema["required_sha256_identity_fields"]:
        _require_sha(identity, field)
    if identity.get("character_weighting_rule") != WEIGHTING_RULE:
        raise FutureRawConflictError("character finite-population weighting changed")

    k = int(identity.get("k", -1))
    character_count = int(identity.get("character_count", -1))
    chain_count = int(identity.get("truth_blind_scoring_chain_count", -1))
    batch_count = int(identity.get("finite_population_batch_count", -1))
    if not 1 <= k <= 64 or character_count < k or chain_count < int(schema["minimum_truth_blind_scoring_chains"]):
        raise FutureRawConflictError("logical/character/chain dimensions are invalid")
    if batch_count < 0:
        raise FutureRawConflictError("finite-population batch count is invalid")
    masks = _require_array(arrays, "character_masks_uint64", np.uint64, (character_count,))
    basis = _require_array(arrays, "basis_character_mask", np.bool_, (character_count,))
    truth_signs = _require_array(arrays, "truth_character_signs", np.int8, (character_count,))
    means = _require_array(
        arrays, "scoring_chain_character_means", np.float64, (chain_count, character_count)
    )
    _require_array(arrays, "m2_debiased_per_character", np.float64, (character_count,))
    _require_array(arrays, "planted_cross_moment_per_character", np.float64, (character_count,))
    _require_array(arrays, "trajectory_jackknife_values", np.float64, (chain_count, character_count))
    _require_array(arrays, "finite_population_batch_means", np.float64, (batch_count,))
    _require_array(arrays, "scoring_chain_seed_ids", np.uint64, (chain_count,))
    truth_blind = _require_array(arrays, "truth_blind_scoring_chain_mask", np.bool_, (chain_count,))
    gate_pass = _require_array(arrays, "per_disorder_sampler_gate_pass", np.bool_, ())
    failures = _require_array(arrays, "per_disorder_sampler_gate_failures_json", np.dtype("<U4096"), ())
    _require_array(arrays, "collision_mass_u_statistic", np.float64, ())
    _require_array(arrays, "posterior_mass_on_planted_class_estimate", np.float64, ())
    if not np.all(truth_blind):
        raise FutureRawConflictError("a scoring chain is not truth blind")
    if not np.all(np.isin(truth_signs, np.asarray([-1, 1], dtype=np.int8))):
        raise FutureRawConflictError("truth character signs are not +/-1")
    if not np.all(np.isfinite(means)):
        raise FutureRawConflictError("scoring means are nonfinite")
    if any(int(value) == 0 for value in masks) or len(set(int(value) for value in masks)) != character_count:
        raise FutureRawConflictError("character masks are zero or duplicated")
    if any(int(value) >= (1 << k) for value in masks):
        raise FutureRawConflictError("character mask exceeds logical dimension")
    expected_basis = np.asarray(
        [(int(value) & (int(value) - 1)) == 0 for value in masks], dtype=np.bool_
    )
    if not np.array_equal(basis, expected_basis):
        raise FutureRawConflictError("basis-character mask changed")
    required_basis = {1 << index for index in range(k)}
    if not required_basis.issubset({int(value) for value in masks}):
        raise FutureRawConflictError("not all basis characters are present")
    nonbasis_sampled = int((~basis).sum())
    if int(identity.get("nonbasis_sampled_count", -1)) != nonbasis_sampled:
        raise FutureRawConflictError("nonbasis sample count changed")
    if int(identity.get("nonbasis_population_size", -1)) != (1 << k) - 1 - k:
        raise FutureRawConflictError("nonbasis population size changed")
    families = identity.get("scoring_chain_initialization_families")
    initial_hashes = identity.get("scoring_chain_initial_state_sha256")
    seed_identities = identity.get("scoring_chain_seed_identities")
    if not all(isinstance(value, list) and len(value) == chain_count for value in (
        families, initial_hashes, seed_identities,
    )):
        raise FutureRawConflictError("scoring-chain provenance length changed")
    if any(not isinstance(value, str) or not value for value in families + seed_identities):
        raise FutureRawConflictError("scoring-chain provenance is empty")
    if any(not HEX64.fullmatch(str(value)) for value in initial_hashes):
        raise FutureRawConflictError("initial-state SHA is invalid")
    parsed_failures = json.loads(str(failures.item()))
    if not isinstance(parsed_failures, list):
        raise FutureRawConflictError("sampler gate failures are not a list")
    if bool(gate_pass.item()) != (len(parsed_failures) == 0):
        raise FutureRawConflictError("sampler gate boolean/failures disagree")
    if identity.get("raw_self_sha256") != compute_raw_self_hash(identity, arrays):
        raise FutureRawConflictError("raw self-hash changed")
    return {
        "disorder_index": disorder_index,
        "eligible_for_ensemble_audit": bool(gate_pass.item()),
        "raw_self_sha256": identity["raw_self_sha256"],
    }


def validate_complete_planned_ensemble(records, manifest, schema):
    manifest = validate_planned_manifest(manifest)
    if len(records) != int(manifest["planned_count"]):
        raise EnsembleAuditNotComputable("not every planned disorder has raw")
    validated = [
        validate_raw_record(identity, arrays, manifest, schema)
        for identity, arrays in records
    ]
    indices = [row["disorder_index"] for row in validated]
    if indices != list(range(int(manifest["planned_count"]))):
        raise EnsembleAuditNotComputable("planned disorder order/coverage changed")
    if not all(row["eligible_for_ensemble_audit"] for row in validated):
        raise EnsembleAuditNotComputable("at least one planned disorder failed its sampler gate")
    return {
        "planned_count": int(manifest["planned_count"]),
        "planned_manifest_sha256": manifest["manifest_sha256"],
        "status": "COMPLETE_PLANNED_IID_ENSEMBLE_ELIGIBLE_FOR_AUXILIARY_AUDIT",
    }
