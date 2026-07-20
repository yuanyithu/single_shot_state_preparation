import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .diagnostics import basis_character_traces, evaluate_gate
from .exp101_bridge import load_exp101
from .io import atomic_json, atomic_npz, sha256_json
from .labels import initial_labels, pairwise_collision
from .q0_pt import Q0PtConfig, run_q0_pt_instance
from .registry import load_frozen_code
from .seeds import derive_seed


def build_model(H):
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.model import assemble_sector_model
    from exp101_certified_src.observables import build_observable_frame
    H_Z, H_X = hgp_from_H(H)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    return model, build_observable_frame(model)


def run_task(registry_path, config_path, frozen_path, code_id, disorder_index, output_path, namespace="production"):
    registry, code, H = load_frozen_code(registry_path, code_id)
    config = load_config(config_path)
    frozen = json.loads(Path(frozen_path).read_text(encoding="ascii"))
    if frozen.get("status") != "FROZEN_HELD_OUT_PASS":
        raise ValueError("PT configuration is not held-out certified and frozen")
    if namespace != "production":
        raise ValueError("production worker accepts only the production namespace")
    if frozen.get("registry_sha256") != registry["registry_sha256"] or frozen.get("config_sha256") != config["config_sha256"]:
        raise ValueError("frozen configuration identity mismatch")
    if frozen.get("engine") != "numba" or not frozen.get("source_commit"):
        raise ValueError("frozen configuration lacks numba/source identity")
    disorder_index = int(disorder_index)
    if not 0 <= disorder_index < config["num_disorders"]:
        raise ValueError("disorder index out of range")
    m_config = frozen["by_m"][str(code["m"])]
    pt_config = Q0PtConfig(**m_config)
    identity = {"code_id": code_id, "disorder_index": disorder_index,
                "registry_sha256": registry["registry_sha256"],
                "config_sha256": config["config_sha256"],
                "frozen_config_sha256": sha256_json(frozen), "namespace": namespace}
    task_fingerprint = sha256_json(identity)
    output_path = Path(output_path)
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as old:
            if str(old["task_fingerprint"].item()) == task_fingerprint:
                return "reused"
        raise ValueError("existing task output has a conflicting fingerprint")
    wall_start, core_start = time.monotonic(), time.process_time()
    model, frame = build_model(H)
    if model.k != code["k"] or model.k > 64:
        raise ValueError("registry/model logical dimension mismatch")
    u_seed = derive_seed(namespace, registry["registry_sha256"], code_id, disorder_index, "uniforms")
    uniforms = np.random.Generator(np.random.PCG64(u_seed)).random(model.num_qubits)
    qtop, collision, planted, char_u = [], [], [], []
    valid, reasons, rhats, esses = [], [], [], []
    all_labels = []
    diagnostics = []
    constant_statuses = []
    seeds = np.zeros((len(config["p_values"]), 4), dtype=np.int64)
    for p_index, p in enumerate(config["p_values"]):
        epsilon = (uniforms < p).astype(np.uint8)
        syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
        planted_label = frame.label_of(epsilon)
        planted_uint = int(sum(int(value) << bit for bit, value in enumerate(planted_label)))
        instance_results = []
        for instance, label in enumerate(initial_labels(model.k)):
            seed = derive_seed(namespace, registry["registry_sha256"], code_id, disorder_index, f"p={p:.8f}", instance)
            seeds[p_index, instance] = seed
            result = run_q0_pt_instance(model, frame, syndrome, p, pt_config, seed, label,
                                        engine=config["engine"])
            result["seed"] = seed
            instance_results.append(result)
        traces = [result["labels"] for result in instance_results]
        mass, estimate = pairwise_collision(traces, model.k)
        chars = basis_character_traces(np.stack(traces), model.k)
        char_means = chars.mean(axis=2)
        char_estimate = float(np.mean([
            np.mean(char_means[a] * char_means[b])
            for a in range(4) for b in range(a + 1, 4)
        ]))
        is_valid, failure, rhat, ess, statuses = evaluate_gate(instance_results, config["production_gate"], model.k)
        qtop.append(estimate); collision.append(mass)
        planted.append(float(np.mean([np.mean(trace == np.uint64(planted_uint)) for trace in traces])))
        char_u.append(char_estimate); valid.append(is_valid); reasons.append(";".join(failure))
        rhats.append(rhat); esses.append(ess); all_labels.append(np.stack(traces))
        constant_statuses.append(statuses)
        diagnostics.append(instance_results)
    swap_attempts = np.asarray([[r["swap_attempts"] for r in group] for group in diagnostics])
    swap_accepts = np.asarray([[r["swap_accepts"] for r in group] for group in diagnostics])
    logical_attempts = np.asarray([[r["logical_attempts"] for r in group] for group in diagnostics])
    logical_accepts = np.asarray([[r["logical_accepts"] for r in group] for group in diagnostics])
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    atomic_npz(output_path,
        task_fingerprint=np.array(task_fingerprint), code_id=np.array(code_id),
        disorder_index=np.array(disorder_index, dtype=np.int16), m=np.array(code["m"], dtype=np.int8),
        k=np.array(model.k, dtype=np.int8), p_values=np.asarray(config["p_values"], dtype=np.float64),
        qtop=np.asarray(qtop), collision_mass=np.asarray(collision), planted_hit=np.asarray(planted),
        character_u_statistic=np.asarray(char_u), valid=np.asarray(valid, dtype=np.bool_),
        failure_reason=np.asarray(reasons, dtype="U4096"), labels=np.asarray(all_labels, dtype=np.uint64),
        rhat=np.asarray(rhats), ess=np.asarray(esses), instance_seeds=seeds,
        constant_status=np.asarray(constant_statuses, dtype="U40"),
        swap_attempts=swap_attempts, swap_accepts=swap_accepts,
        swap_rates=swap_accepts / np.maximum(swap_attempts, 1),
        logical_attempts=logical_attempts, logical_accepts=logical_accepts,
        logical_rates=logical_accepts / np.maximum(logical_attempts, 1),
        round_trips=np.asarray([[r["round_trips"] for r in group] for group in diagnostics]),
        sector_changing_round_trips=np.asarray([[r["sector_changing_round_trips"] for r in group] for group in diagnostics]),
        max_hard_coset_residual=np.asarray([[r["max_hard_coset_residual"] for r in group] for group in diagnostics]),
        engine=np.array(config["engine"]), source_commit=np.array(frozen["source_commit"]),
        namespace=np.array(namespace), core_seconds=np.array(core_seconds),
        wall_seconds=np.array(wall_seconds), model_fingerprint=np.array(sha256_json({
            "code_id": code_id, "n": model.num_qubits, "k": model.k,
            "section": code["section_fingerprint"],
            "logical_frame": code["logical_frame_fingerprint"],
        })),
        section_fingerprint=np.array(code["section_fingerprint"]),
        logical_frame_fingerprint=np.array(code["logical_frame_fingerprint"]),
        registry_sha256=np.array(registry["registry_sha256"]), config_sha256=np.array(config["config_sha256"]),
        frozen_config_sha256=np.array(sha256_json(frozen)), physics_contract_version=np.array(config["physics_contract_version"]),
        pt_contract_version=np.array(config["pt_contract_version"]), scan_contract_version=np.array(config["scan_contract_version"]))
    return "computed"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry"); parser.add_argument("config"); parser.add_argument("frozen")
    parser.add_argument("code_id"); parser.add_argument("disorder_index", type=int); parser.add_argument("output")
    parser.add_argument("--namespace", default="production")
    args = parser.parse_args(argv)
    print(run_task(args.registry, args.config, args.frozen, args.code_id, args.disorder_index, args.output, args.namespace))


if __name__ == "__main__":
    main()
