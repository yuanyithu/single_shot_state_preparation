import hashlib
from pathlib import Path

import numpy as np
from ldpc import BpOsdDecoder

from .audit_scorer import pairing_score
from .config import ensure_config, normalize_p_token
from .identity import runtime_identity
from .model import load_model
from .raw import load_raw
from .io import canonical_json, sha256_file
from .seeds import derive_seed


def _decoder(model, p, config):
    spec = config["decoder"]
    return BpOsdDecoder(
        model.H_Z_sparse,
        error_rate=float(p),
        bp_method="product_sum",
        max_iter=model.n,
        schedule="serial",
        serial_schedule_order=list(range(model.n)),
        osd_method="osd_0",
        osd_order=0,
        omp_thread_count=1,
    )


def replay_decoder_shard(raw_or_path, config):
    config = ensure_config(config)
    raw_path = Path(raw_or_path) if isinstance(raw_or_path, (str, bytes, Path)) else None
    raw = load_raw(raw_path) if raw_path is not None else raw_or_path
    identity = runtime_identity(config)
    token = normalize_p_token(raw["p_token"])
    expected_seed = derive_seed(
        config, "measurement", raw["code_id"], token, int(raw["shard_index"]),
    )
    identity_fields = {
        "schema_version": "exp103.raw.v2",
        "experiment_id": "exp103.decoder_mc.v2",
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "decoder_binary_sha256": identity["decoder_binary_sha256"],
        "seed": expected_seed,
        "seed_namespace": config["namespaces"]["measurement"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
    }
    if raw.get("conda_prefix_matches_python") is not True:
        return {
            "status": "INVALID",
            "reason": "raw_identity_mismatch:conda_prefix_matches_python",
        }
    for field, expected in identity_fields.items():
        if raw[field] != expected:
            return {"status": "INVALID", "reason": f"raw_identity_mismatch:{field}"}
    if type(raw["conda_prefix_matches_python"]) is not bool:
        return {"status": "INVALID", "reason": "raw_identity_mismatch:conda_prefix_matches_python"}
    if raw["status"] != "VALID" or raw["completed_trials"] != config["trials_per_shard"]:
        return {"status": "INVALID", "reason": "raw_not_complete_valid"}
    model = load_model(config, raw["code_id"])
    expected_fields = {
        "invalid_reason": "",
        "exception_type": "",
        "exception_message": "",
        "m": model.m,
        "p": float(token),
        "planned_trials": config["trials_per_shard"],
        "python_version": config["environment"]["python"],
        "numpy_version": config["environment"]["numpy"],
        "scipy_version": config["environment"]["scipy"],
        "ldpc_version": config["environment"]["ldpc"],
        "n": model.n,
        "k": model.k,
        "classical_distance": model.classical_distance,
    }
    for field, expected in expected_fields.items():
        if raw[field] != expected:
            return {"status": "INVALID", "reason": f"raw_identity_mismatch:{field}"}
    trials = config["trials_per_shard"]
    expected_arrays = {
        "failure_flags": ((trials,), np.dtype(np.bool_)),
        "logical_labels": ((trials, model.k), np.dtype(np.uint8)),
        "syndrome_match": ((trials,), np.dtype(np.bool_)),
        "bp_converged": ((trials,), np.dtype(np.bool_)),
        "bp_iterations": ((trials,), np.dtype(np.int32)),
    }
    for field, (shape, dtype) in expected_arrays.items():
        if raw[field].shape != shape or raw[field].dtype != dtype:
            return {"status": "INVALID", "reason": f"raw_array_identity_mismatch:{field}"}
    decoder = _decoder(model, float(token), config)
    rng = np.random.Generator(np.random.PCG64(expected_seed))
    error_digest = hashlib.sha256()
    correction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    for trial in range(config["trials_per_shard"]):
        error = (rng.random(model.n) < float(token)).astype(np.uint8)
        syndrome = np.asarray(model.H_Z @ error, dtype=np.uint8) & np.uint8(1)
        correction = decoder.decode(syndrome)
        if (
            not isinstance(correction, np.ndarray)
            or correction.dtype != np.uint8
            or correction.shape != (model.n,)
            or np.any(correction > 1)
        ):
            return {"status": "INVALID", "reason": f"illegal_correction_at_trial:{trial}"}
        failed, matched, labels = pairing_score(model.H_Z, model.logical_Z, error, correction)
        if (
            failed != bool(raw["failure_flags"][trial])
            or matched != bool(raw["syndrome_match"][trial])
            or not np.array_equal(labels, raw["logical_labels"][trial])
            or bool(decoder.converge) != bool(raw["bp_converged"][trial])
            or int(decoder.iter) != int(raw["bp_iterations"][trial])
        ):
            return {"status": "INVALID", "reason": f"trial_replay_mismatch:{trial}"}
        error_digest.update(error.tobytes())
        correction_digest.update(correction.tobytes())
        label_digest.update(labels.tobytes())
    hashes = {
        "error_stream_sha256": error_digest.hexdigest(),
        "correction_stream_sha256": correction_digest.hexdigest(),
        "label_stream_sha256": label_digest.hexdigest(),
    }
    for field, value in hashes.items():
        if raw[field] != value:
            return {"status": "INVALID", "reason": f"stream_hash_mismatch:{field}"}
    return {
        "status": "PASS",
        "reason": "",
        "code_id": raw["code_id"],
        "p_token": token,
        "shard_index": int(raw["shard_index"]),
        "trials": int(raw["completed_trials"]),
        "replay_control_seed": derive_seed(
            config, "replay", raw["code_id"], token, int(raw["shard_index"]),
        ),
        "raw_sha256": sha256_file(raw_path) if raw_path is not None else "",
        **hashes,
    }


def expected_replay_keys(config, scope):
    config = ensure_config(config)
    if scope == "stage1":
        m_values = config["stage_m_values"]["stage1"]
    elif scope == "stage2":
        m_values = config["stage_m_values"]["stage2"]
    elif scope == "final":
        m_values = config["m_values"]
    else:
        raise ValueError("replay scope must be stage1, stage2, or final")
    return {
        (f"m{m:02d}_c{code:02d}", p_token, shard)
        for m in m_values
        for code in range(8)
        for p_token in config["p_tokens"]
        for shard in range(config["shards_per_code_p"])
    }


def raw_manifest(raw_root):
    entries = []
    seen = set()
    paths = {}
    for path in sorted(Path(raw_root).rglob("*.npz")):
        raw = load_raw(path)
        key = (str(raw["code_id"]), str(raw["p_token"]), int(raw["shard_index"]))
        if key in seen:
            raise ValueError(f"duplicate raw identity in replay manifest: {key!r}")
        seen.add(key)
        paths[key] = path
        entries.append({
            "code_id": key[0], "p_token": key[1], "shard_index": key[2],
            "raw_sha256": sha256_file(path),
        })
    entries.sort(key=lambda item: (item["code_id"], item["p_token"], item["shard_index"]))
    digest = hashlib.sha256(canonical_json(entries).encode("ascii")).hexdigest()
    return entries, digest, paths


def build_replay_report(raw_root, results, config):
    config = ensure_config(config)
    identity = runtime_identity(config)
    entries, manifest_sha256, _ = raw_manifest(raw_root)
    keys = {(item["code_id"], item["p_token"], item["shard_index"]) for item in entries}
    if keys == expected_replay_keys(config, "stage1"):
        scope = "stage1"
    elif keys == expected_replay_keys(config, "stage2"):
        scope = "stage2"
    else:
        scope = "invalid"
    result_keys = {
        (item.get("code_id"), item.get("p_token"), item.get("shard_index"))
        for item in results if item.get("status") == "PASS"
    }
    status = "PASS" if scope != "invalid" and result_keys == keys and len(results) == len(keys) and all(
        item.get("status") == "PASS" for item in results
    ) else "INVALID"
    return {
        "schema_version": "exp103.replay.v1",
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
        "scope": scope,
        "expected_shards": len(expected_replay_keys(config, scope)) if scope != "invalid" else 0,
        "shards": len(results),
        "raw_manifest_sha256": manifest_sha256,
        "status": status,
        "results": sorted(results, key=lambda item: (
            item.get("code_id", ""), item.get("p_token", ""), item.get("shard_index", -1),
        )),
    }


def validate_replay_report(report, raw_root, config, required_scope=None):
    config = ensure_config(config)
    validate_replay_report_payload(report, config, required_scope)
    scope = report["scope"]
    expected_keys = expected_replay_keys(config, scope)
    entries, manifest_sha256, paths = raw_manifest(raw_root)
    actual = {
        (entry["code_id"], entry["p_token"], entry["shard_index"]): entry
        for entry in entries
    }
    if set(actual) != expected_keys or report["raw_manifest_sha256"] != manifest_sha256:
        raise ValueError("replay report raw manifest mismatch")
    seen = set()
    for item in report["results"]:
        key = (item["code_id"], item["p_token"], int(item["shard_index"]))
        if key in seen or key not in actual:
            raise ValueError("replay result identity is duplicate or unplanned")
        seen.add(key)
        if item["raw_sha256"] != actual[key]["raw_sha256"]:
            raise ValueError("replay result raw SHA mismatch")
        raw = load_raw(paths[key])
        for field in ("error_stream_sha256", "correction_stream_sha256", "label_stream_sha256"):
            if item[field] != raw[field]:
                raise ValueError(f"replay stream mismatch for {field}")
    if seen != expected_keys:
        raise ValueError("replay report does not cover every planned shard")
    return report


def validate_replay_report_payload(report, config, required_scope=None):
    config = ensure_config(config)
    if set(report) != {
        "schema_version", "config_sha256", "registry_sha256", "source_commit",
        "source_tree_sha256", "decoder_binary_sha256", "device_name", "hostname",
        "conda_environment", "conda_prefix_matches_python", "scope",
        "expected_shards", "shards", "raw_manifest_sha256", "status", "results",
    } or report["schema_version"] != "exp103.replay.v1":
        raise ValueError("replay report schema mismatch")
    if report["conda_prefix_matches_python"] is not True:
        raise ValueError("replay report conda prefix attestation is not boolean true")
    for field, expected in (
        ("config_sha256", config["config_sha256"]),
        ("registry_sha256", config["registry_sha256"]),
        ("source_commit", config["source_commit"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
        ("decoder_binary_sha256", config["decoder_binary"]["sha256"]),
        ("device_name", config["environment"]["device_name"]),
        ("hostname", config["environment"]["hostname"]),
        ("conda_environment", config["environment"]["conda_environment"]),
        ("conda_prefix_matches_python", True),
        ("status", "PASS"),
    ):
        if report[field] != expected:
            raise ValueError(f"replay report identity mismatch for {field}")
    scope = report["scope"]
    if required_scope is not None and scope != required_scope:
        raise ValueError("replay report scope mismatch")
    expected_keys = expected_replay_keys(config, scope)
    if report["expected_shards"] != len(expected_keys) or report["shards"] != len(expected_keys):
        raise ValueError("replay report shard count mismatch")
    results = report["results"]
    if len(results) != len(expected_keys):
        raise ValueError("replay report result count mismatch")
    seen = set()
    for item in results:
        if set(item) != {
            "status", "reason", "code_id", "p_token", "shard_index", "trials",
            "replay_control_seed", "raw_sha256", "error_stream_sha256",
            "correction_stream_sha256", "label_stream_sha256",
        } or item["status"] != "PASS" or item["reason"] != "":
            raise ValueError("replay result schema or status mismatch")
        key = (item["code_id"], item["p_token"], int(item["shard_index"]))
        if key in seen or key not in expected_keys:
            raise ValueError("replay result identity is duplicate or unplanned")
        seen.add(key)
        if item["trials"] != config["trials_per_shard"]:
            raise ValueError("replay result trial count mismatch")
        if item["replay_control_seed"] != derive_seed(config, "replay", *key):
            raise ValueError("replay control namespace mismatch")
        for field in (
            "raw_sha256", "error_stream_sha256", "correction_stream_sha256", "label_stream_sha256",
        ):
            value = str(item[field])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"replay result contains an invalid SHA for {field}")
    if seen != expected_keys:
        raise ValueError("replay report does not cover every planned shard")
    manifest_entries = [
        {
            "code_id": item["code_id"],
            "p_token": item["p_token"],
            "shard_index": int(item["shard_index"]),
            "raw_sha256": item["raw_sha256"],
        }
        for item in results
    ]
    manifest_entries.sort(key=lambda item: (
        item["code_id"], item["p_token"], item["shard_index"],
    ))
    manifest_sha256 = hashlib.sha256(
        canonical_json(manifest_entries).encode("ascii")
    ).hexdigest()
    if report["raw_manifest_sha256"] != manifest_sha256:
        raise ValueError("replay report result manifest mismatch")
    return report
