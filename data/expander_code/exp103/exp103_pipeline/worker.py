import hashlib
import traceback
from numbers import Integral

import numpy as np
from ldpc import BpOsdDecoder

from . import EXPERIMENT_ID, RAW_SCHEMA
from .config import ensure_config, normalize_p_token
from .identity import runtime_identity
from .model import load_model, parity_product
from .seeds import derive_seed


RAW_FIELDS = {
    "schema_version", "status", "invalid_reason", "exception_type",
    "exception_message", "experiment_id", "code_id", "m", "p_token", "p",
    "shard_index", "planned_trials", "completed_trials", "seed",
    "seed_namespace", "config_sha256", "registry_sha256", "source_commit",
    "source_tree_sha256", "decoder_binary_sha256", "python_version",
    "numpy_version", "scipy_version", "ldpc_version", "device_name",
    "hostname", "conda_environment", "conda_prefix_matches_python", "n", "k",
    "classical_distance", "error_stream_sha256", "correction_stream_sha256",
    "label_stream_sha256", "failure_flags", "logical_labels", "syndrome_match",
    "bp_converged", "bp_iterations",
}


def make_decoder(model, p, config):
    decoder = config["decoder"]
    return BpOsdDecoder(
        model.H_Z_sparse,
        error_rate=float(p),
        bp_method=decoder["bp_method"],
        max_iter=model.n,
        schedule=decoder["schedule"],
        serial_schedule_order=list(range(model.n)),
        osd_method=decoder["osd_method"],
        osd_order=decoder["osd_order"],
        omp_thread_count=decoder["omp_thread_count"],
    )


def score_residual_pairing(model, error, correction):
    residual = np.bitwise_xor(error, correction)
    syndrome_match = not parity_product(model.H_Z, residual).any()
    labels = parity_product(model.logical_Z, residual)
    failed = (not syndrome_match) or bool(labels.any())
    return failed, syndrome_match, labels


def _base_raw(code_id, token, shard_index, config, identity, seed):
    return {
        "schema_version": RAW_SCHEMA,
        "status": "INVALID",
        "invalid_reason": "not_completed",
        "exception_type": "",
        "exception_message": "",
        "experiment_id": EXPERIMENT_ID,
        "code_id": str(code_id),
        "m": int(str(code_id)[1:3]) if len(str(code_id)) >= 3 else -1,
        "p_token": token,
        "p": float(token),
        "shard_index": int(shard_index),
        "planned_trials": int(config["trials_per_shard"]),
        "completed_trials": 0,
        "seed": int(seed),
        "seed_namespace": config["namespaces"]["measurement"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity.get("source_tree_sha256", config["source_tree_sha256"]),
        "decoder_binary_sha256": identity.get("decoder_binary_sha256", config["decoder_binary"]["sha256"]),
        "python_version": identity.get("python_version", ""),
        "numpy_version": identity.get("numpy_version", ""),
        "scipy_version": identity.get("scipy_version", ""),
        "ldpc_version": identity.get("ldpc_version", ""),
        "device_name": identity.get("device_name", ""),
        "hostname": identity.get("hostname", ""),
        "conda_environment": identity.get("conda_environment", ""),
        "conda_prefix_matches_python": identity.get("conda_prefix_matches_python", False),
        "n": -1,
        "k": -1,
        "classical_distance": -1,
        "error_stream_sha256": hashlib.sha256().hexdigest(),
        "correction_stream_sha256": hashlib.sha256().hexdigest(),
        "label_stream_sha256": hashlib.sha256().hexdigest(),
        "failure_flags": np.zeros(0, dtype=np.bool_),
        "logical_labels": np.zeros((0, 0), dtype=np.uint8),
        "syndrome_match": np.zeros(0, dtype=np.bool_),
        "bp_converged": np.zeros(0, dtype=np.bool_),
        "bp_iterations": np.zeros(0, dtype=np.int32),
    }


def _validate_correction(correction, n):
    if not isinstance(correction, np.ndarray):
        raise TypeError("decoder correction must be a numpy array")
    if correction.dtype != np.uint8:
        raise TypeError("decoder correction dtype must be uint8")
    if correction.shape != (n,):
        raise ValueError("decoder correction shape is invalid")
    if np.any(correction > 1):
        raise ValueError("decoder correction is not binary")
    return correction.copy()


def run_decoder_shard(code_id, p, shard_index, config):
    """Run one immutable 2500-trial measurement shard.

    Infrastructure failures are returned as INVALID raw payloads so callers can
    save the evidence without redrawing the failed trial.
    """
    config = ensure_config(config)
    token = normalize_p_token(p)
    if isinstance(shard_index, bool) or not isinstance(shard_index, Integral):
        raise ValueError("shard index must be an integer")
    shard_index = int(shard_index)
    if not 0 <= shard_index < config["shards_per_code_p"]:
        raise ValueError("shard index is outside the frozen plan")
    seed = derive_seed(config, "measurement", code_id, token, shard_index)
    identity = {}
    raw = _base_raw(code_id, token, shard_index, config, identity, seed)
    try:
        identity = runtime_identity(config)
        raw.update(_base_raw(code_id, token, shard_index, config, identity, seed))
        model = load_model(config, code_id)
        raw.update({
            "m": model.m, "n": model.n, "k": model.k,
            "classical_distance": model.classical_distance,
        })
        decoder = make_decoder(model, float(token), config)
        trials = config["trials_per_shard"]
        failures = np.zeros(trials, dtype=np.bool_)
        logical_labels = np.zeros((trials, model.k), dtype=np.uint8)
        syndrome_match = np.zeros(trials, dtype=np.bool_)
        bp_converged = np.zeros(trials, dtype=np.bool_)
        bp_iterations = np.zeros(trials, dtype=np.int32)
        error_digest = hashlib.sha256()
        correction_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        rng = np.random.Generator(np.random.PCG64(seed))
        completed = 0
        try:
            for trial in range(trials):
                error = (rng.random(model.n) < float(token)).astype(np.uint8)
                syndrome = parity_product(model.H_Z, error)
                correction = _validate_correction(decoder.decode(syndrome), model.n)
                failed, matched, labels = score_residual_pairing(model, error, correction)
                failures[trial] = failed
                logical_labels[trial] = labels
                syndrome_match[trial] = matched
                bp_converged[trial] = bool(decoder.converge)
                bp_iterations[trial] = int(decoder.iter)
                error_digest.update(error.tobytes())
                correction_digest.update(correction.tobytes())
                label_digest.update(labels.tobytes())
                completed = trial + 1
        except Exception as error:
            raw.update({
                "invalid_reason": "trial_infrastructure_error",
                "exception_type": type(error).__name__,
                "exception_message": "".join(traceback.format_exception_only(type(error), error)).strip()[:1000],
            })
        raw.update({
            "completed_trials": completed,
            "failure_flags": failures[:completed].copy(),
            "logical_labels": logical_labels[:completed].copy(),
            "syndrome_match": syndrome_match[:completed].copy(),
            "bp_converged": bp_converged[:completed].copy(),
            "bp_iterations": bp_iterations[:completed].copy(),
            "error_stream_sha256": error_digest.hexdigest(),
            "correction_stream_sha256": correction_digest.hexdigest(),
            "label_stream_sha256": label_digest.hexdigest(),
        })
        if completed == trials:
            raw.update({"status": "VALID", "invalid_reason": ""})
    except Exception as error:
        raw.update({
            "invalid_reason": "setup_infrastructure_error",
            "exception_type": type(error).__name__,
            "exception_message": "".join(traceback.format_exception_only(type(error), error)).strip()[:1000],
        })
    if set(raw) != RAW_FIELDS:
        raise AssertionError("internal raw schema field mismatch")
    return raw
