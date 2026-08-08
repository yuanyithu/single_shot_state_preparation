"""Bit-exact replay against a committed subsample of the production tasks.

exp104 replays a preregistered ten percent of tasks rather than all of them.
That is only defensible because the decoder's determinism is measured rather
than assumed: `tests/test_decoder_determinism.py` is a resident regression gate
that runs in the local suite and again during nd-3 qualification, exp103
reproduced 2496 of 2496 shards bit for bit with this same decoder identity on
this same node, and Validation 002 replays frozen exp103 shards through this
package. The subsample is fixed by a seed derived before any production task
runs, and any single mismatch invalidates the whole run.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
from ldpc import BpOsdDecoder

from . import RAW_SCHEMA
from .audit_scorer import pairing_score
from .config import (
    M_VALUES,
    TASKS_PER_M,
    block_code_indices,
    code_id as make_code_id,
    ensure_config,
)
from .identity import runtime_identity
from .io import canonical_json, sha256_file
from .model import clear_model_cache, load_model
from .raw import load_raw
from .seeds import derive_seed, replay_selection_seed


REPLAY_SCHEMA = "exp104.replay.v1"
RESULT_FIELDS = {
    "status", "reason", "m", "block_index", "codes", "trials",
    "replay_control_seed", "raw_sha256", "error_stream_sha256",
    "correction_stream_sha256", "label_stream_sha256",
}
REPORT_FIELDS = {
    "schema_version", "config_sha256", "registry_sha256", "source_commit",
    "source_tree_sha256", "decoder_binary_sha256", "device_name", "hostname",
    "conda_environment", "conda_prefix_matches_python", "scope",
    "replay_policy", "replay_fraction", "expected_tasks", "tasks",
    "raw_manifest_sha256", "status", "results",
}


def committed_replay_blocks(config):
    """The replay subsample, fixed before production by a frozen seed."""
    config = ensure_config(config)
    policy = config["replay"]
    if policy["policy"] != "committed_random_subsample":
        raise ValueError("unexpected exp104 replay policy")
    fraction = float(policy["fraction"])
    always = int(policy["always_include_block_index"])
    rng = np.random.Generator(np.random.PCG64(replay_selection_seed(config)))
    selected = {}
    for m in config["m_values"]:
        total = TASKS_PER_M[m]
        size = int(math.ceil(fraction * total))
        drawn = np.asarray(rng.permutation(total)[:size], dtype=np.int64)
        blocks = sorted({int(value) for value in drawn} | {always})
        if any(not 0 <= block < total for block in blocks):
            raise ValueError("committed replay block is outside the frozen plan")
        selected[int(m)] = blocks
    return selected


def expected_replay_keys(config):
    return {
        (int(m), int(block))
        for m, blocks in committed_replay_blocks(config).items()
        for block in blocks
    }


def _decoder(model, p):
    """Independently constructed decoder, not shared with the worker path."""
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


def _invalid(reason):
    return {"status": "INVALID", "reason": reason}


def replay_task(raw_or_path, config, registry_rows):
    config = ensure_config(config)
    raw_path = Path(raw_or_path) if isinstance(raw_or_path, (str, bytes, Path)) else None
    raw = load_raw(raw_path) if raw_path is not None else raw_or_path
    identity = runtime_identity(config)
    m = int(raw["m"])
    block_index = int(raw["block_index"])
    if m not in M_VALUES or not 0 <= block_index < TASKS_PER_M[m]:
        return _invalid("raw_identity_mismatch:task_outside_frozen_plan")
    indices = block_code_indices(m, block_index)
    tokens = list(config["p_tokens"])
    trials = int(config["trials_per_code_p"])
    k = m ** 2

    if raw.get("conda_prefix_matches_python") is not True:
        return _invalid("raw_identity_mismatch:conda_prefix_matches_python")
    identity_fields = {
        "schema_version": RAW_SCHEMA,
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "decoder_binary_sha256": identity["decoder_binary_sha256"],
        "seed_namespace": config["namespaces"]["measurement"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "p_tokens": ",".join(tokens),
        "trials_per_code_p": trials,
        "planned_codes": len(indices),
        "invalid_reason": "",
        "exception_type": "",
        "exception_message": "",
        "n": 25 * m ** 2,
        "k": k,
        "python_version": config["environment"]["python"],
        "numpy_version": config["environment"]["numpy"],
        "scipy_version": config["environment"]["scipy"],
        "ldpc_version": config["environment"]["ldpc"],
    }
    for field, expected in identity_fields.items():
        if raw[field] != expected:
            return _invalid(f"raw_identity_mismatch:{field}")
    if raw["status"] != "VALID" or int(raw["completed_codes"]) != len(indices):
        return _invalid("raw_not_complete_valid")

    codes = len(indices)
    expected_arrays = {
        "code_index": ((codes,), np.dtype(np.int32)),
        "graph_seed": ((codes,), np.dtype(np.int64)),
        "classical_distance": ((codes,), np.dtype(np.int16)),
        "trial_seed": ((codes, len(tokens)), np.dtype(np.int64)),
        "failure_flags": ((codes, len(tokens), trials), np.dtype(np.bool_)),
        "logical_labels": ((codes, len(tokens), trials, k), np.dtype(np.uint8)),
        "syndrome_match": ((codes, len(tokens), trials), np.dtype(np.bool_)),
        "bp_converged": ((codes, len(tokens), trials), np.dtype(np.bool_)),
        "bp_iterations": ((codes, len(tokens), trials), np.dtype(np.int32)),
    }
    for field, (shape, dtype) in expected_arrays.items():
        if raw[field].shape != shape or raw[field].dtype != dtype:
            return _invalid(f"raw_array_identity_mismatch:{field}")

    error_digest = hashlib.sha256()
    correction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    try:
        for slot, index in enumerate(indices):
            identifier = make_code_id(m, index)
            row = registry_rows[identifier]
            clear_model_cache()
            model = load_model(row)
            if (
                int(raw["code_index"][slot]) != index
                or int(raw["graph_seed"][slot]) != int(row["graph_seed"])
                or int(raw["classical_distance"][slot]) != model.classical_distance
                or str(raw["classical_H_sha256"][slot]) != model.classical_H_sha256
                or str(raw["logical_frame_sha256"][slot]) != model.logical_frame_sha256
            ):
                return _invalid(f"code_identity_mismatch:{identifier}")
            for p_slot, token in enumerate(tokens):
                seed = derive_seed(config, "measurement", identifier, token, 0)
                if int(raw["trial_seed"][slot, p_slot]) != seed:
                    return _invalid(f"seed_mismatch:{identifier}:{token}")
                decoder = _decoder(model, float(token))
                rng = np.random.Generator(np.random.PCG64(seed))
                for trial in range(trials):
                    error = (rng.random(model.n) < float(token)).astype(np.uint8)
                    syndrome = np.asarray(model.H_Z @ error, dtype=np.uint8) & np.uint8(1)
                    correction = decoder.decode(syndrome)
                    if (
                        not isinstance(correction, np.ndarray)
                        or correction.dtype != np.uint8
                        or correction.shape != (model.n,)
                        or np.any(correction > 1)
                    ):
                        return _invalid(f"illegal_correction:{identifier}:{token}:{trial}")
                    failed, matched, labels = pairing_score(
                        model.H_Z, model.logical_Z, error, correction,
                    )
                    if (
                        failed != bool(raw["failure_flags"][slot, p_slot, trial])
                        or matched != bool(raw["syndrome_match"][slot, p_slot, trial])
                        or not np.array_equal(labels, raw["logical_labels"][slot, p_slot, trial])
                        or bool(decoder.converge) != bool(raw["bp_converged"][slot, p_slot, trial])
                        or int(decoder.iter) != int(raw["bp_iterations"][slot, p_slot, trial])
                    ):
                        return _invalid(f"trial_replay_mismatch:{identifier}:{token}:{trial}")
                    error_digest.update(error.tobytes())
                    correction_digest.update(correction.tobytes())
                    label_digest.update(labels.tobytes())
    finally:
        clear_model_cache()

    hashes = {
        "error_stream_sha256": error_digest.hexdigest(),
        "correction_stream_sha256": correction_digest.hexdigest(),
        "label_stream_sha256": label_digest.hexdigest(),
    }
    for field, value in hashes.items():
        if raw[field] != value:
            return _invalid(f"stream_hash_mismatch:{field}")
    return {
        "status": "PASS",
        "reason": "",
        "m": m,
        "block_index": block_index,
        "codes": codes,
        "trials": codes * len(tokens) * trials,
        "replay_control_seed": derive_seed(
            config, "replay", make_code_id(m, indices[0]), tokens[0], block_index,
        ),
        "raw_sha256": sha256_file(raw_path) if raw_path is not None else "",
        **hashes,
    }


def raw_manifest(raw_root):
    entries = []
    seen = set()
    paths = {}
    for path in sorted(Path(raw_root).rglob("*.npz")):
        raw = load_raw(path)
        key = (int(raw["m"]), int(raw["block_index"]))
        if key in seen:
            raise ValueError(f"duplicate raw identity in manifest: {key!r}")
        seen.add(key)
        paths[key] = path
        entries.append({
            "m": key[0], "block_index": key[1], "raw_sha256": sha256_file(path),
        })
    entries.sort(key=lambda item: (item["m"], item["block_index"]))
    digest = hashlib.sha256(canonical_json(entries).encode("ascii")).hexdigest()
    return entries, digest, paths


def _results_manifest_sha256(results):
    entries = sorted(
        (
            {
                "m": int(item["m"]),
                "block_index": int(item["block_index"]),
                "raw_sha256": item["raw_sha256"],
            }
            for item in results
        ),
        key=lambda item: (item["m"], item["block_index"]),
    )
    return hashlib.sha256(canonical_json(entries).encode("ascii")).hexdigest()


def build_replay_report(results, config):
    config = ensure_config(config)
    identity = runtime_identity(config)
    expected = expected_replay_keys(config)
    keys = [
        (int(item.get("m", -1)), int(item.get("block_index", -1)))
        for item in results
        if item.get("status") == "PASS"
    ]
    complete = set(keys) == expected and len(keys) == len(expected)
    status = "PASS" if complete and all(
        item.get("status") == "PASS" for item in results
    ) else "INVALID"
    return {
        "schema_version": REPLAY_SCHEMA,
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
        "scope": "committed_subsample",
        "replay_policy": config["replay"]["policy"],
        "replay_fraction": float(config["replay"]["fraction"]),
        "expected_tasks": len(expected),
        "tasks": len(results),
        "raw_manifest_sha256": _results_manifest_sha256(results) if status == "PASS" else "",
        "status": status,
        "results": sorted(
            results, key=lambda item: (item.get("m", -1), item.get("block_index", -1)),
        ),
    }


def validate_replay_report(report, config):
    """Re-derive the committed subsample and require exact, complete coverage."""
    config = ensure_config(config)
    if set(report) != REPORT_FIELDS or report["schema_version"] != REPLAY_SCHEMA:
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
        ("scope", "committed_subsample"),
        ("replay_policy", config["replay"]["policy"]),
        ("replay_fraction", float(config["replay"]["fraction"])),
        ("status", "PASS"),
    ):
        if report[field] != expected:
            raise ValueError(f"replay report identity mismatch for {field}")
    expected_keys = expected_replay_keys(config)
    if report["expected_tasks"] != len(expected_keys) or report["tasks"] != len(expected_keys):
        raise ValueError("replay report task count mismatch")
    results = report["results"]
    if len(results) != len(expected_keys):
        raise ValueError("replay report result count mismatch")
    seen = set()
    for item in results:
        if set(item) != RESULT_FIELDS or item["status"] != "PASS" or item["reason"] != "":
            raise ValueError("replay result schema or status mismatch")
        key = (int(item["m"]), int(item["block_index"]))
        if key in seen or key not in expected_keys:
            raise ValueError("replay result identity is duplicate or unplanned")
        seen.add(key)
        indices = block_code_indices(key[0], key[1])
        if item["codes"] != len(indices):
            raise ValueError("replay result code count mismatch")
        if item["trials"] != len(indices) * len(config["p_tokens"]) * int(
            config["trials_per_code_p"]
        ):
            raise ValueError("replay result trial count mismatch")
        if item["replay_control_seed"] != derive_seed(
            config, "replay", make_code_id(key[0], indices[0]), config["p_tokens"][0], key[1],
        ):
            raise ValueError("replay control namespace mismatch")
        for field in (
            "raw_sha256", "error_stream_sha256", "correction_stream_sha256",
            "label_stream_sha256",
        ):
            value = str(item[field])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"replay result contains an invalid SHA for {field}")
    if seen != expected_keys:
        raise ValueError("replay report does not cover the committed subsample")
    if report["raw_manifest_sha256"] != _results_manifest_sha256(results):
        raise ValueError("replay report result manifest mismatch")
    return report


def validate_replay_against_raw(report, raw_root, config):
    """Require the replayed bytes to be the bytes that were actually stored."""
    validate_replay_report(report, config)
    entries, _, paths = raw_manifest(raw_root)
    actual = {(entry["m"], entry["block_index"]): entry for entry in entries}
    for item in report["results"]:
        key = (int(item["m"]), int(item["block_index"]))
        if key not in actual:
            raise ValueError("replay result refers to a task that is not on disk")
        if item["raw_sha256"] != actual[key]["raw_sha256"]:
            raise ValueError("replay result raw SHA mismatch")
        raw = load_raw(paths[key])
        for field in (
            "error_stream_sha256", "correction_stream_sha256", "label_stream_sha256",
        ):
            if item[field] != raw[field]:
                raise ValueError(f"replay stream mismatch for {field}")
    return report
