import hashlib
import traceback
from numbers import Integral

import numpy as np
from ldpc import BpOsdDecoder

from . import EXPERIMENT_ID, RAW_SCHEMA
from .config import (
    M_VALUES,
    TASKS_PER_M,
    block_code_indices,
    code_id as make_code_id,
    ensure_config,
)
from .identity import runtime_identity
from .model import clear_model_cache, load_model, parity_product
from .seeds import derive_seed


RAW_FIELDS = {
    "schema_version", "status", "invalid_reason", "exception_type",
    "exception_message", "experiment_id", "m", "block_index", "planned_codes",
    "completed_codes", "p_tokens", "trials_per_code_p", "seed_namespace",
    "config_sha256", "registry_sha256", "source_commit", "source_tree_sha256",
    "decoder_binary_sha256", "python_version", "numpy_version",
    "scipy_version", "ldpc_version", "device_name", "hostname",
    "conda_environment", "conda_prefix_matches_python", "n", "k",
    "code_index", "graph_seed", "classical_distance", "classical_H_sha256",
    "logical_frame_sha256", "trial_seed", "failure_flags", "logical_labels",
    "syndrome_match", "bp_converged", "bp_iterations",
    "error_stream_sha256", "correction_stream_sha256", "label_stream_sha256",
}

_ARRAY_FIELDS = {
    "code_index", "graph_seed", "classical_distance", "classical_H_sha256",
    "logical_frame_sha256", "trial_seed", "failure_flags", "logical_labels",
    "syndrome_match", "bp_converged", "bp_iterations",
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


def _base_raw(m, block_index, config, identity):
    codes = len(block_code_indices(m, block_index))
    p_count = len(config["p_tokens"])
    trials = int(config["trials_per_code_p"])
    k = int(m) ** 2
    return {
        "schema_version": RAW_SCHEMA,
        "status": "INVALID",
        "invalid_reason": "not_completed",
        "exception_type": "",
        "exception_message": "",
        "experiment_id": EXPERIMENT_ID,
        "m": int(m),
        "block_index": int(block_index),
        "planned_codes": codes,
        "completed_codes": 0,
        "p_tokens": ",".join(config["p_tokens"]),
        "trials_per_code_p": trials,
        "seed_namespace": config["namespaces"]["measurement"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity.get("source_tree_sha256", config["source_tree_sha256"]),
        "decoder_binary_sha256": identity.get(
            "decoder_binary_sha256", config["decoder_binary"]["sha256"],
        ),
        "python_version": identity.get("python_version", ""),
        "numpy_version": identity.get("numpy_version", ""),
        "scipy_version": identity.get("scipy_version", ""),
        "ldpc_version": identity.get("ldpc_version", ""),
        "device_name": identity.get("device_name", ""),
        "hostname": identity.get("hostname", ""),
        "conda_environment": identity.get("conda_environment", ""),
        "conda_prefix_matches_python": identity.get("conda_prefix_matches_python", False),
        "n": 25 * int(m) ** 2,
        "k": k,
        "code_index": np.zeros(codes, dtype=np.int32),
        "graph_seed": np.zeros(codes, dtype=np.int64),
        "classical_distance": np.zeros(codes, dtype=np.int16),
        "classical_H_sha256": np.full(codes, "", dtype="<U64"),
        "logical_frame_sha256": np.full(codes, "", dtype="<U64"),
        "trial_seed": np.zeros((codes, p_count), dtype=np.int64),
        "failure_flags": np.zeros((codes, p_count, trials), dtype=np.bool_),
        "logical_labels": np.zeros((codes, p_count, trials, k), dtype=np.uint8),
        "syndrome_match": np.zeros((codes, p_count, trials), dtype=np.bool_),
        "bp_converged": np.zeros((codes, p_count, trials), dtype=np.bool_),
        "bp_iterations": np.zeros((codes, p_count, trials), dtype=np.int32),
        "error_stream_sha256": hashlib.sha256().hexdigest(),
        "correction_stream_sha256": hashlib.sha256().hexdigest(),
        "label_stream_sha256": hashlib.sha256().hexdigest(),
    }


def run_code_block(m, block_index, config, registry_rows):
    """Run one immutable task: a contiguous code block over the whole p grid.

    The model is built once per code and reused across every p, which is what
    makes a small trial count per code affordable: at m=8 the logical frame
    costs about nine trials, so amortising it over the nine grid points leaves
    the fixed cost near one percent of the task.

    Infrastructure failures return an INVALID payload rather than raising, so
    the evidence is saved without redrawing any trial.
    """
    config = ensure_config(config)
    if m not in M_VALUES:
        raise ValueError(f"m is outside the frozen panel: {m!r}")
    if isinstance(block_index, bool) or not isinstance(block_index, Integral):
        raise ValueError("block index must be an integer")
    block_index = int(block_index)
    if not 0 <= block_index < TASKS_PER_M[m]:
        raise ValueError("block index is outside the frozen plan")

    indices = block_code_indices(m, block_index)
    tokens = list(config["p_tokens"])
    trials = int(config["trials_per_code_p"])
    identity = {}
    raw = _base_raw(m, block_index, config, identity)
    try:
        identity = runtime_identity(config)
        raw = _base_raw(m, block_index, config, identity)
        expected_ids = [make_code_id(m, index) for index in indices]
        rows = [registry_rows[code] for code in expected_ids]
        error_digest = hashlib.sha256()
        correction_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        completed = 0
        try:
            for slot, row in enumerate(rows):
                clear_model_cache()
                model = load_model(row)
                raw["code_index"][slot] = int(row["code_index"])
                raw["graph_seed"][slot] = int(row["graph_seed"])
                raw["classical_distance"][slot] = int(row["classical_distance"])
                raw["classical_H_sha256"][slot] = model.classical_H_sha256
                raw["logical_frame_sha256"][slot] = model.logical_frame_sha256
                for p_slot, token in enumerate(tokens):
                    seed = derive_seed(config, "measurement", model.code_id, token, 0)
                    raw["trial_seed"][slot, p_slot] = seed
                    decoder = make_decoder(model, float(token), config)
                    rng = np.random.Generator(np.random.PCG64(seed))
                    for trial in range(trials):
                        error = (rng.random(model.n) < float(token)).astype(np.uint8)
                        syndrome = parity_product(model.H_Z, error)
                        correction = _validate_correction(
                            decoder.decode(syndrome), model.n,
                        )
                        failed, matched, labels = score_residual_pairing(
                            model, error, correction,
                        )
                        raw["failure_flags"][slot, p_slot, trial] = failed
                        raw["logical_labels"][slot, p_slot, trial] = labels
                        raw["syndrome_match"][slot, p_slot, trial] = matched
                        raw["bp_converged"][slot, p_slot, trial] = bool(decoder.converge)
                        raw["bp_iterations"][slot, p_slot, trial] = int(decoder.iter)
                        error_digest.update(error.tobytes())
                        correction_digest.update(correction.tobytes())
                        label_digest.update(labels.tobytes())
                completed = slot + 1
        except Exception as error:
            raw.update({
                "invalid_reason": "trial_infrastructure_error",
                "exception_type": type(error).__name__,
                "exception_message": "".join(
                    traceback.format_exception_only(type(error), error)
                ).strip()[:1000],
            })
        raw["completed_codes"] = completed
        raw["error_stream_sha256"] = error_digest.hexdigest()
        raw["correction_stream_sha256"] = correction_digest.hexdigest()
        raw["label_stream_sha256"] = label_digest.hexdigest()
        if completed == len(indices):
            raw.update({"status": "VALID", "invalid_reason": ""})
    except Exception as error:
        raw.update({
            "invalid_reason": "setup_infrastructure_error",
            "exception_type": type(error).__name__,
            "exception_message": "".join(
                traceback.format_exception_only(type(error), error)
            ).strip()[:1000],
        })
    finally:
        clear_model_cache()
    if set(raw) != RAW_FIELDS:
        raise AssertionError("internal raw schema field mismatch")
    return raw
