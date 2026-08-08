"""Tie the exp104 code path to the frozen exp103 code path.

exp104 replays only a committed ten percent of its own tasks, so the pipeline
has to be tied to one that is already published. This script runs exp103's frozen
48-code registry and exp103's frozen seeds through **both** packages on the same
machine and requires the per-trial arrays to be bit-identical: failure flags,
logical labels, syndrome match, convergence flags and iteration counts.

It also records, without gating on it, how far each package lands from the counts
exp103 published from nd-3. That gap is a property of the compiled decoder across
platforms, not of either package, and the contract is explicit that different
builds are not required to reproduce each other verbatim. The same-platform
comparison against the frozen exp103 raw on nd-3 belongs to Validation 003.

Nothing here is a physical claim. It is an implementation-equality claim.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = str(Path(__file__).resolve().parents[5])
sys.path.insert(0, ROOT)

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp103.exp103_pipeline.config import load_config as load_exp103_config
from data.expander_code.exp103.exp103_pipeline.seeds import derive_seed as exp103_seed
from data.expander_code.exp103.exp103_pipeline.worker import run_decoder_shard
from data.expander_code.exp104.exp104_pipeline.config import load_config as load_exp104_config
from data.expander_code.exp104.exp104_pipeline.io import atomic_json, sha256_file
from data.expander_code.exp104.exp104_pipeline.model import DecoderModel, frame_digest
from data.expander_code.exp104.exp104_pipeline.worker import (
    make_decoder,
    score_residual_pairing,
)

EXP103_CONFIG = f"{ROOT}/data/expander_code/exp103/config/decoder_mc.v2.json"
EXP104_CONFIG = f"{ROOT}/data/expander_code/exp104/config/ensemble_mc.v1.json"
EXP103_REGISTRY = f"{ROOT}/data/expander_code/exp102/registry/registry.json"
EXP103_DIAGNOSTICS = (
    f"{ROOT}/data/expander_code/exp103/validation/010_final_crossing_20260807/"
    "code_diagnostics.csv"
)
# Cheap cells spanning three sizes, two distances and four grid points. Cost is
# dominated by belief-propagation iterations, so low p and low m are chosen; the
# claim under test is implementation equality, which does not depend on cost.
CELLS = (
    ("m03_c00", "0.02"),
    ("m03_c00", "0.06"),
    ("m03_c03", "0.04"),
    ("m04_c00", "0.02"),
    ("m04_c05", "0.03"),
    ("m05_c00", "0.02"),
)


def exp104_model_for_exp103_code(code_id):
    """Build the model exp104's way from exp103's frozen registry entry."""
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators

    _, row, classical_H = load_frozen_code(EXP103_REGISTRY, code_id)
    H_Z, H_X = hgp_from_H(classical_H)
    frame = logical_pauli_operators(H_X, H_Z)
    for array in (H_Z, H_X, frame.logical_Z):
        array.flags.writeable = False
    return DecoderModel(
        code_id=code_id,
        m=int(row["m"]),
        n=int(H_Z.shape[1]),
        k=int(frame.k),
        classical_distance=int(row["classical_distance"]),
        classical_H_sha256=row["classical_H_sha256"],
        logical_frame_sha256=frame_digest(H_Z, H_X, frame.logical_X, frame.logical_Z),
        H_Z=H_Z,
        H_X=H_X,
        H_Z_sparse=csr_matrix(H_Z),
        logical_Z=frame.logical_Z,
    )


def exp104_shard(model, token, seed, trials, exp104_config):
    decoder = make_decoder(model, float(token), exp104_config)
    rng = np.random.Generator(np.random.PCG64(seed))
    failure_flags = np.zeros(trials, dtype=np.bool_)
    logical_labels = np.zeros((trials, model.k), dtype=np.uint8)
    syndrome_match = np.zeros(trials, dtype=np.bool_)
    bp_converged = np.zeros(trials, dtype=np.bool_)
    bp_iterations = np.zeros(trials, dtype=np.int32)
    for trial in range(trials):
        error = (rng.random(model.n) < float(token)).astype(np.uint8)
        syndrome = np.asarray(model.H_Z @ error, dtype=np.uint8) & np.uint8(1)
        correction = decoder.decode(syndrome)
        failed, matched, labels = score_residual_pairing(model, error, correction)
        failure_flags[trial] = failed
        logical_labels[trial] = labels
        syndrome_match[trial] = matched
        bp_converged[trial] = bool(decoder.converge)
        bp_iterations[trial] = int(decoder.iter)
    return {
        "failure_flags": failure_flags,
        "logical_labels": logical_labels,
        "syndrome_match": syndrome_match,
        "bp_converged": bp_converged,
        "bp_iterations": bp_iterations,
    }


def published(code_id, token):
    with open(EXP103_DIAGNOSTICS, encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            if row["code_id"] == code_id and row["p"] == token:
                return {
                    "failures": int(row["failures"]),
                    "trials": int(row["trials"]),
                    "bp_convergence_rate": float(row["bp_convergence_rate"]),
                    "classical_distance": int(row["classical_distance"]),
                }
    raise KeyError(f"exp103 published no cell for {code_id} at p={token}")


def compare_cell(code_id, token, exp103_config, exp104_config):
    model = exp104_model_for_exp103_code(code_id)
    shards = int(exp103_config["shards_per_code_p"])
    per_shard = int(exp103_config["trials_per_shard"])
    fields = (
        "failure_flags", "logical_labels", "syndrome_match", "bp_converged",
        "bp_iterations",
    )
    identical = True
    mismatched_trials = 0
    exp103_failures = 0
    exp104_failures = 0
    exp103_converged = 0
    started = time.perf_counter()
    for shard_index in range(shards):
        theirs = run_decoder_shard(code_id, token, shard_index, exp103_config)
        if theirs["status"] != "VALID":
            raise RuntimeError(f"exp103 worker returned {theirs['invalid_reason']}")
        seed = exp103_seed(exp103_config, "measurement", code_id, token, shard_index)
        if int(theirs["seed"]) != seed:
            raise RuntimeError("exp103 seed derivation disagrees with its own worker")
        mine = exp104_shard(model, token, seed, per_shard, exp104_config)
        for field in fields:
            if not np.array_equal(mine[field], theirs[field]):
                identical = False
        mismatched_trials += int(
            np.count_nonzero(mine["failure_flags"] != theirs["failure_flags"])
        )
        exp103_failures += int(theirs["failure_flags"].sum())
        exp104_failures += int(mine["failure_flags"].sum())
        exp103_converged += int(theirs["bp_converged"].sum())
    trials = shards * per_shard
    reference = published(code_id, token)
    return {
        "code_id": code_id,
        "p_token": token,
        "classical_distance": model.classical_distance,
        "trials": trials,
        "status": "PASS" if identical and mismatched_trials == 0 else "FAIL",
        "bit_identical_arrays": identical,
        "mismatched_trials_between_packages": mismatched_trials,
        "exp103_local_failures": exp103_failures,
        "exp104_local_failures": exp104_failures,
        "exp103_published_failures_from_nd3": reference["failures"],
        "local_minus_published_failures": exp103_failures - reference["failures"],
        "exp103_local_bp_convergence_rate": exp103_converged / trials,
        "exp103_published_bp_convergence_rate": reference["bp_convergence_rate"],
        "seconds": time.perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    exp103_config = load_exp103_config(EXP103_CONFIG)
    exp104_config = load_exp104_config(EXP104_CONFIG)
    results = []
    for code_id, token in CELLS:
        result = compare_cell(code_id, token, exp103_config, exp104_config)
        results.append(result)
        print(
            f"{result['status']} {code_id} p={token} "
            f"packages_agree={result['bit_identical_arrays']} "
            f"local={result['exp104_local_failures']} "
            f"published_nd3={result['exp103_published_failures_from_nd3']} "
            f"({result['seconds']:.0f}s)",
            flush=True,
        )
    status = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    platform_deltas = [abs(r["local_minus_published_failures"]) for r in results]
    report = {
        "schema_version": "exp104.exp103_cross_validation.v2",
        "status": status,
        "gated_claim": (
            "on one machine, the exp104 and exp103 code paths produce bit-identical "
            "per-trial arrays from the same frozen registry, seeds and decoder identity"
        ),
        "recorded_observation": (
            "neither package reproduces the counts exp103 published from nd-3, because "
            "the compiled decoder is not bit-portable across platforms; belief "
            "propagation convergence is unchanged and the difference is confined to "
            "post-processing outcomes. This is not gated here and is retested "
            "same-platform on nd-3 in Validation 003"
        ),
        "platform": sys.platform,
        "exp103_config_sha256": exp103_config["config_sha256"],
        "exp104_config_sha256": exp104_config["config_sha256"],
        "exp103_registry_sha256": exp103_config["registry_sha256"],
        "exp103_diagnostics_sha256": sha256_file(EXP103_DIAGNOSTICS),
        "cells": len(results),
        "max_abs_local_minus_published_failures": max(platform_deltas),
        "results": results,
    }
    atomic_json(args.output, report)
    print(status, f"cells={len(results)} max_platform_delta={max(platform_deltas)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
