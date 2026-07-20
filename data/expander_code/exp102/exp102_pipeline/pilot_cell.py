"""Execute one independently resumable exp102 pilot cell."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .diagnostics import evaluate_gate
from .io import atomic_npz, sha256_json
from .labels import initial_labels
from .q0_pt import Q0PtConfig, run_q0_pt_instance
from .registry import load_frozen_code
from .seeds import derive_seed
from .worker import build_model


def _gate(config, stage):
    gate = dict(config["production_gate"])
    if stage in {"ladder", "gamma"}:
        gate.update(min_swap_rate=0.15, min_round_trips=0,
                    min_sector_changing_round_trips=0, max_rhat=np.inf, min_ess=0,
                    max_instance_mean_spread=np.inf)
    elif stage in {"rounds", "held_out"}:
        gate.update(min_round_trips=8, min_sector_changing_round_trips=4,
                    max_rhat=1.02, min_ess=400)
    else:
        raise ValueError("stage must be ladder, gamma, rounds, or held_out")
    return gate


def run_cell(registry_path, config_path, code_id, p, disorder_index, candidate,
             attempt, stage, source_commit, output_path):
    registry, code, H = load_frozen_code(registry_path, code_id)
    config = load_config(config_path)
    candidate = dict(candidate)
    pt_config = Q0PtConfig(**candidate)
    namespace = (f"pilot_held_out_m{code['m']}_attempt{attempt}" if stage == "held_out"
                 else f"pilot_{stage}_m{code['m']}_attempt{attempt}")
    identity = {
        "namespace": namespace, "stage": stage, "code_id": code_id, "p": float(p),
        "disorder_index": int(disorder_index), "candidate": candidate,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"], "source_commit": source_commit,
        "engine": "numba",
    }
    fingerprint = sha256_json(identity)
    output_path = Path(output_path)
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as old:
            if str(old["task_fingerprint"].item()) == fingerprint:
                return "reused"
        raise ValueError("existing pilot output has a conflicting fingerprint")
    model, frame = build_model(H)
    uniform_seed = derive_seed(namespace, registry["registry_sha256"], code_id,
                               disorder_index, "uniforms")
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
    epsilon = (uniforms < float(p)).astype(np.uint8)
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    results = []
    wall_start, core_start = time.monotonic(), time.process_time()
    for instance, label in enumerate(initial_labels(model.k)):
        seed = derive_seed(namespace, registry["registry_sha256"], code_id,
                           disorder_index, f"p={float(p):.8f}", instance)
        result = run_q0_pt_instance(model, frame, syndrome, float(p), pt_config,
                                    seed, label, engine="numba")
        result["seed"] = seed
        results.append(result)
    core_seconds, wall_seconds = time.process_time() - core_start, time.monotonic() - wall_start
    valid, failures, rhats, esses, statuses = evaluate_gate(
        results, _gate(config, stage), model.k,
        require_trace_gate=stage not in {"ladder", "gamma"},
    )
    swap_attempts = np.asarray([r["swap_attempts"] for r in results])
    swap_accepts = np.asarray([r["swap_accepts"] for r in results])
    logical_attempts = np.asarray([r["logical_attempts"] for r in results])
    logical_accepts = np.asarray([r["logical_accepts"] for r in results])
    atomic_npz(
        output_path, task_fingerprint=np.array(fingerprint), namespace=np.array(namespace),
        stage=np.array(stage), code_id=np.array(code_id), m=np.array(code["m"], dtype=np.int8),
        p=np.array(float(p)), disorder_index=np.array(disorder_index, dtype=np.int16),
        candidate_json=np.array(json.dumps(candidate, sort_keys=True, separators=(",", ":"))),
        attempt=np.array(attempt, dtype=np.int16), valid=np.array(valid),
        failure_reason=np.array(";".join(failures), dtype="U4096"),
        labels=np.asarray([r["labels"] for r in results], dtype=np.uint64),
        swap_attempts=swap_attempts, swap_accepts=swap_accepts,
        swap_rates=swap_accepts / np.maximum(swap_attempts, 1),
        logical_attempts=logical_attempts, logical_accepts=logical_accepts,
        logical_rates=logical_accepts / np.maximum(logical_attempts, 1),
        round_trips=np.asarray([r["round_trips"] for r in results]),
        sector_changing_round_trips=np.asarray([r["sector_changing_round_trips"] for r in results]),
        residual=np.asarray([r["max_hard_coset_residual"] for r in results]),
        rhat=rhats, ess=esses, constant_status=statuses,
        core_seconds=np.array(core_seconds), wall_seconds=np.array(wall_seconds),
        engine=np.array("numba"), source_commit=np.array(source_commit),
        model_fingerprint=np.array(sha256_json({"n": model.num_qubits, "k": model.k})),
        registry_sha256=np.array(registry["registry_sha256"]),
        config_sha256=np.array(config["config_sha256"]),
        section_fingerprint=np.array(code["section_fingerprint"]),
        logical_frame_fingerprint=np.array(code["logical_frame_fingerprint"]),
    )
    return "computed"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry"); parser.add_argument("config"); parser.add_argument("code_id")
    parser.add_argument("p", type=float); parser.add_argument("disorder_index", type=int)
    parser.add_argument("candidate_json"); parser.add_argument("attempt", type=int)
    parser.add_argument("stage", choices=("ladder", "gamma", "rounds", "held_out"))
    parser.add_argument("source_commit"); parser.add_argument("output")
    args = parser.parse_args(argv)
    candidate = json.loads(args.candidate_json)
    print(run_cell(args.registry, args.config, args.code_id, args.p, args.disorder_index,
                   candidate, args.attempt, args.stage, args.source_commit, args.output))


if __name__ == "__main__":
    main()
