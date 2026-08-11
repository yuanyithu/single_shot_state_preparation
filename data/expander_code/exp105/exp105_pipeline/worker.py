import hashlib
import traceback
from numbers import Integral

import numpy as np
from ldpc import BpOsdDecoder

from . import EXPERIMENT_ID, RAW_SCHEMA
from .config import (
    block_code_indices,
    code_id as make_code_id,
    ensure_config,
    tasks_per_m,
)
from .identity import runtime_identity
from .model import clear_model_cache, load_model, logical_label, parity_product
from .seeds import derive_seed


RAW_FIELDS = {
    "schema_version", "status", "invalid_reason", "exception_type",
    "exception_message", "experiment_id", "m", "block_index", "planned_codes",
    "completed_codes", "p_tokens", "q_token", "trials_per_code_p",
    "seed_namespace", "config_sha256", "registry_sha256", "source_commit",
    "source_tree_sha256", "decoder_binary_sha256", "python_version",
    "numpy_version", "scipy_version", "ldpc_version", "device_name",
    "hostname", "conda_environment", "conda_prefix_matches_python", "n",
    "n_checks", "k", "code_index", "graph_seed", "classical_distance",
    "classical_H_sha256", "logical_frame_sha256",
    "observable_frame_fingerprint", "trial_seed", "failure_flags",
    "logical_labels", "readout_match", "bp_converged", "bp_iterations",
    "error_stream_sha256", "readout_stream_sha256",
    "correction_stream_sha256", "label_stream_sha256",
}

_ARRAY_FIELDS = {
    "code_index", "graph_seed", "classical_distance", "classical_H_sha256",
    "logical_frame_sha256", "observable_frame_fingerprint", "trial_seed",
    "failure_flags", "logical_labels", "readout_match", "bp_converged",
    "bp_iterations",
}


def trials_for(config, m):
    return int(config["trials_per_code_p"][str(int(m))])


def make_decoder(model, p, config):
    """The frozen BP+OSD-0 identity, on the augmented matrix.

    At q > 0 the readout error is part of what has to be inferred, so the
    decoding problem is [H_Z | I] (eps, mu)^T = y_eff with a mixed channel.
    max_iter follows the block length actually being decoded, which is the same
    rule exp104 applied to its own block length.
    """
    decoder = config["decoder"]
    q = float(config["q_token"])
    width = model.n + model.n_checks
    return BpOsdDecoder(
        model.H_augmented_sparse,
        error_channel=[float(p)] * model.n + [q] * model.n_checks,
        bp_method=decoder["bp_method"],
        max_iter=width,
        schedule=decoder["schedule"],
        serial_schedule_order=list(range(width)),
        osd_method=decoder["osd_method"],
        osd_order=decoder["osd_order"],
        omp_thread_count=decoder["omp_thread_count"],
    )


def score_logical_class(model, error, readout, correction):
    """Score one trial through the exp101 absolute logical label phi_r.

    The trial fails iff the decoder's data estimate lands in a different
    logical class from the truth. `readout_match` records whether the readout
    error itself was recovered; it is a diagnostic split of the failures, not
    part of the failure criterion.

    exp104 additionally required the residual to have zero syndrome. Carrying
    that requirement into q > 0 would be wrong: the protocol ends in a perfect
    final round that measures the residual syndrome exactly and removes it, so
    a residual with nonzero syndrome but trivial class is a success, and
    counting it as a failure would report the readout channel twice.
    """
    residual = np.bitwise_xor(error, correction[:model.n])
    labels = logical_label(model, residual)
    readout_match = bool(
        np.array_equal(correction[model.n:], np.asarray(readout, dtype=np.uint8))
    )
    failed = bool(labels.any())
    return failed, readout_match, labels


def _validate_correction(correction, width):
    if not isinstance(correction, np.ndarray):
        raise TypeError("decoder correction must be a numpy array")
    if correction.dtype != np.uint8:
        raise TypeError("decoder correction dtype must be uint8")
    if correction.shape != (width,):
        raise ValueError("decoder correction shape is invalid")
    if np.any(correction > 1):
        raise ValueError("decoder correction is not binary")
    return correction.copy()


def _base_raw(m, block_index, config, identity):
    codes = len(block_code_indices(config, m, block_index))
    p_count = len(config["p_tokens"])
    trials = trials_for(config, m)
    k = int(m) ** 2
    n_checks = 12 * int(m) ** 2
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
        "q_token": config["q_token"],
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
        "n_checks": n_checks,
        "k": k,
        "code_index": np.zeros(codes, dtype=np.int32),
        "graph_seed": np.zeros(codes, dtype=np.int64),
        "classical_distance": np.zeros(codes, dtype=np.int16),
        "classical_H_sha256": np.full(codes, "", dtype="<U64"),
        "logical_frame_sha256": np.full(codes, "", dtype="<U64"),
        "observable_frame_fingerprint": np.full(codes, "", dtype="<U64"),
        "trial_seed": np.zeros((codes, p_count), dtype=np.int64),
        "failure_flags": np.zeros((codes, p_count, trials), dtype=np.bool_),
        "logical_labels": np.zeros((codes, p_count, trials, k), dtype=np.uint8),
        "readout_match": np.zeros((codes, p_count, trials), dtype=np.bool_),
        "bp_converged": np.zeros((codes, p_count, trials), dtype=np.bool_),
        "bp_iterations": np.zeros((codes, p_count, trials), dtype=np.int32),
        "error_stream_sha256": hashlib.sha256().hexdigest(),
        "readout_stream_sha256": hashlib.sha256().hexdigest(),
        "correction_stream_sha256": hashlib.sha256().hexdigest(),
        "label_stream_sha256": hashlib.sha256().hexdigest(),
    }


def run_code_block(m, block_index, config, registry_rows):
    """Run one immutable task: a contiguous code block over the whole p grid.

    The model is built once per code and reused across every p, which is what
    makes a small trial count per code affordable: the logical frame and the
    label basis cost several trials at m=8, so amortising them over the grid
    leaves the fixed cost near one percent of the task.

    Infrastructure failures return an INVALID payload rather than raising, so
    the evidence is saved without redrawing any trial.
    """
    config = ensure_config(config)
    m = int(m)
    if m not in config["m_values"]:
        raise ValueError(f"m is outside the frozen panel: {m!r}")
    if isinstance(block_index, bool) or not isinstance(block_index, Integral):
        raise ValueError("block index must be an integer")
    block_index = int(block_index)
    counts = {int(key): int(value) for key, value in config["codes_per_m"].items()}
    sizes = {int(key): int(value) for key, value in config["codes_per_task"].items()}
    if not 0 <= block_index < tasks_per_m(counts, sizes)[m]:
        raise ValueError("block index is outside the frozen plan")

    indices = block_code_indices(config, m, block_index)
    tokens = list(config["p_tokens"])
    trials = trials_for(config, m)
    q = float(config["q_token"])
    identity = {}
    raw = _base_raw(m, block_index, config, identity)
    try:
        identity = runtime_identity(config)
        raw = _base_raw(m, block_index, config, identity)
        expected_ids = [make_code_id(m, index) for index in indices]
        rows = [registry_rows[code] for code in expected_ids]
        error_digest = hashlib.sha256()
        readout_digest = hashlib.sha256()
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
                raw["observable_frame_fingerprint"][slot] = (
                    model.observable_frame_fingerprint
                )
                for p_slot, token in enumerate(tokens):
                    seed = derive_seed(config, "measurement", model.code_id, token, 0)
                    raw["trial_seed"][slot, p_slot] = seed
                    decoder = make_decoder(model, float(token), config)
                    rng = np.random.Generator(np.random.PCG64(seed))
                    for trial in range(trials):
                        error = (rng.random(model.n) < float(token)).astype(np.uint8)
                        readout = (rng.random(model.n_checks) < q).astype(np.uint8)
                        effective = np.bitwise_xor(
                            parity_product(model.H_Z, error), readout,
                        )
                        correction = _validate_correction(
                            decoder.decode(effective), model.n + model.n_checks,
                        )
                        failed, readout_match, labels = score_logical_class(
                            model, error, readout, correction,
                        )
                        raw["failure_flags"][slot, p_slot, trial] = failed
                        raw["logical_labels"][slot, p_slot, trial] = labels
                        raw["readout_match"][slot, p_slot, trial] = readout_match
                        raw["bp_converged"][slot, p_slot, trial] = bool(decoder.converge)
                        raw["bp_iterations"][slot, p_slot, trial] = int(decoder.iter)
                        error_digest.update(error.tobytes())
                        readout_digest.update(readout.tobytes())
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
        raw["readout_stream_sha256"] = readout_digest.hexdigest()
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
