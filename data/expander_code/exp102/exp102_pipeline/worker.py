import argparse
import json
import os
import re
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


PRODUCTION_RAW_FIELDS = {
    "task_fingerprint", "code_id", "disorder_index", "m", "k", "p_values",
    "qtop", "collision_mass", "planted_hit", "character_u_statistic", "valid",
    "failure_reason", "labels", "rhat", "ess", "instance_seeds",
    "constant_status", "swap_attempts", "swap_accepts", "swap_rates",
    "logical_attempts", "logical_accepts", "logical_rates", "round_trips",
    "sector_changing_round_trips", "max_hard_coset_residual", "engine",
    "source_commit", "namespace", "core_seconds", "wall_seconds",
    "model_fingerprint", "section_fingerprint", "logical_frame_fingerprint",
    "registry_sha256", "config_sha256", "frozen_config_sha256",
    "physics_contract_version", "pt_contract_version", "scan_contract_version",
}


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


def _raw_scalar(data, field):
    if field not in data or data[field].shape != ():
        raise ValueError(f"production raw field {field!r} must be a scalar")
    return data[field].item()


def _raw_array(data, field, shape, dtype=None):
    if field not in data or data[field].shape != shape:
        raise ValueError(f"production raw field {field!r} has the wrong shape")
    value = data[field].copy()
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"production raw field {field!r} has the wrong dtype")
    return value


def validate_production_raw(path, registry, code, config, frozen, model, frame):
    """Validate a complete production NPZ before it is reused or aggregated."""
    path = Path(path)
    if model.num_qubits != code["n"] or model.k != code["k"]:
        raise ValueError("registry/model identity mismatch during production validation")
    frozen_hash = sha256_json(frozen)
    identity = {
        "code_id": code["code_id"], "disorder_index": None,
        "registry_sha256": registry["registry_sha256"],
        "config_sha256": config["config_sha256"],
        "frozen_config_sha256": frozen_hash, "namespace": "production",
    }
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot read production raw {path}: {exc}") from exc
    with context as data:
        if set(data.files) != PRODUCTION_RAW_FIELDS:
            raise ValueError(f"production raw schema mismatch in {path}")
        disorder = int(_raw_scalar(data, "disorder_index"))
        identity["disorder_index"] = disorder
        fingerprint = sha256_json(identity)
        scalar_expected = {
            "task_fingerprint": fingerprint,
            "code_id": code["code_id"],
            "m": code["m"],
            "k": code["k"],
            "engine": "numba",
            "source_commit": frozen["source_commit"],
            "namespace": "production",
            "model_fingerprint": sha256_json({
                "code_id": code["code_id"], "n": code["n"], "k": code["k"],
                "section": code["section_fingerprint"],
                "logical_frame": code["logical_frame_fingerprint"],
            }),
            "section_fingerprint": code["section_fingerprint"],
            "logical_frame_fingerprint": code["logical_frame_fingerprint"],
            "registry_sha256": registry["registry_sha256"],
            "config_sha256": config["config_sha256"],
            "frozen_config_sha256": frozen_hash,
            "physics_contract_version": config["physics_contract_version"],
            "pt_contract_version": config["pt_contract_version"],
            "scan_contract_version": config["scan_contract_version"],
        }
        for field, expected in scalar_expected.items():
            if str(_raw_scalar(data, field)) != str(expected):
                raise ValueError(f"production raw identity mismatch in {path}: {field}")
        if not 0 <= disorder < config["num_disorders"]:
            raise ValueError(f"production disorder index is out of range in {path}")

        p_count, instances, k = len(config["p_values"]), config["num_instances"], int(code["k"])
        candidate = frozen["by_m"][str(code["m"])]
        temperatures = int(candidate["num_temperatures"])
        measurements = int(candidate["measurement_rounds"])
        p_values = _raw_array(data, "p_values", (p_count,), np.dtype(np.float64))
        if not np.array_equal(p_values, np.asarray(config["p_values"], dtype=np.float64)):
            raise ValueError(f"production p grid mismatch in {path}")
        labels = _raw_array(
            data, "labels", (p_count, instances, measurements), np.dtype(np.uint64),
        )
        if k < 64 and np.any(labels >> np.uint64(k)):
            raise ValueError(f"production labels exceed the logical dimension in {path}")
        qtop = _raw_array(data, "qtop", (p_count,), np.dtype(np.float64))
        collision = _raw_array(data, "collision_mass", (p_count,), np.dtype(np.float64))
        planted = _raw_array(data, "planted_hit", (p_count,), np.dtype(np.float64))
        character = _raw_array(
            data, "character_u_statistic", (p_count,), np.dtype(np.float64),
        )
        valid = _raw_array(data, "valid", (p_count,), np.dtype(np.bool_))
        failures = _raw_array(data, "failure_reason", (p_count,))
        if failures.dtype.kind != "U":
            raise ValueError(f"production failure reasons have the wrong dtype in {path}")
        rhats = _raw_array(data, "rhat", (p_count, k), np.dtype(np.float64))
        esses = _raw_array(data, "ess", (p_count, k), np.dtype(np.float64))
        seeds = _raw_array(data, "instance_seeds", (p_count, instances), np.dtype(np.int64))
        statuses = _raw_array(data, "constant_status", (p_count, k))
        if statuses.dtype.kind != "U":
            raise ValueError(f"production constant statuses have the wrong dtype in {path}")

        swap_attempts = _raw_array(
            data, "swap_attempts", (p_count, instances, temperatures - 1),
        )
        swap_accepts = _raw_array(
            data, "swap_accepts", (p_count, instances, temperatures - 1),
        )
        swap_rates = _raw_array(
            data, "swap_rates", (p_count, instances, temperatures - 1), np.dtype(np.float64),
        )
        logical_attempts = _raw_array(
            data, "logical_attempts", (p_count, instances, temperatures, k),
        )
        logical_accepts = _raw_array(
            data, "logical_accepts", (p_count, instances, temperatures, k),
        )
        logical_rates = _raw_array(
            data, "logical_rates", (p_count, instances, temperatures, k), np.dtype(np.float64),
        )
        round_trips = _raw_array(data, "round_trips", (p_count, instances))
        changing_round_trips = _raw_array(
            data, "sector_changing_round_trips", (p_count, instances),
        )
        residual = _raw_array(data, "max_hard_coset_residual", (p_count, instances))
        for name, attempts, accepts in (
                ("swap", swap_attempts, swap_accepts),
                ("logical", logical_attempts, logical_accepts)):
            if attempts.dtype.kind not in "iu" or accepts.dtype.kind not in "iu":
                raise ValueError(f"production {name} counters have the wrong dtype in {path}")
            if (np.any(attempts < 0) or np.any(accepts < 0)
                    or np.any(accepts > attempts)):
                raise ValueError(f"production {name} counters are inconsistent in {path}")
        for name, values in (
                ("round trips", round_trips),
                ("sector-changing round trips", changing_round_trips),
                ("residual", residual)):
            if values.dtype.kind not in "iu" or np.any(values < 0):
                raise ValueError(f"production {name} counters are invalid in {path}")
        if not np.array_equal(swap_rates, swap_accepts / np.maximum(swap_attempts, 1)):
            raise ValueError(f"production swap rates disagree with counters in {path}")
        if not np.array_equal(logical_rates, logical_accepts / np.maximum(logical_attempts, 1)):
            raise ValueError(f"production logical rates disagree with counters in {path}")
        recomputed_collision = np.empty(p_count)
        recomputed_qtop = np.empty(p_count)
        recomputed_planted = np.empty(p_count)
        recomputed_character = np.empty(p_count)
        recomputed_valid = np.empty(p_count, dtype=np.bool_)
        recomputed_failures = np.empty(p_count, dtype="U4096")
        recomputed_rhat = np.empty((p_count, k))
        recomputed_ess = np.empty((p_count, k))
        recomputed_status = np.empty((p_count, k), dtype="U40")
        uniform_seed = derive_seed(
            "production", registry["registry_sha256"], code["code_id"], disorder, "uniforms",
        )
        uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        for p_index, p in enumerate(config["p_values"]):
            expected_seeds = np.asarray([
                derive_seed(
                    "production", registry["registry_sha256"], code["code_id"], disorder,
                    f"p={float(p):.8f}", instance,
                )
                for instance in range(instances)
            ], dtype=np.int64)
            if not np.array_equal(seeds[p_index], expected_seeds):
                raise ValueError(f"production instance seeds mismatch in {path}")
            recomputed_collision[p_index], recomputed_qtop[p_index] = pairwise_collision(
                labels[p_index], k,
            )
            epsilon = (uniforms < float(p)).astype(np.uint8)
            planted_bits = frame.label_of(epsilon)
            planted_label = np.uint64(sum(
                int(value) << bit for bit, value in enumerate(planted_bits)
            ))
            recomputed_planted[p_index] = float(np.mean([
                np.mean(trace == planted_label) for trace in labels[p_index]
            ]))
            chars = basis_character_traces(labels[p_index], k)
            char_means = chars.mean(axis=2)
            recomputed_character[p_index] = np.mean([
                np.mean(char_means[a] * char_means[b])
                for a in range(instances) for b in range(a + 1, instances)
            ])
            results = [{
                "labels": labels[p_index, instance],
                "swap_attempts": swap_attempts[p_index, instance],
                "swap_accepts": swap_accepts[p_index, instance],
                "logical_attempts": logical_attempts[p_index, instance],
                "logical_accepts": logical_accepts[p_index, instance],
                "round_trips": int(round_trips[p_index, instance]),
                "sector_changing_round_trips": int(changing_round_trips[p_index, instance]),
                "max_hard_coset_residual": int(residual[p_index, instance]),
                "seed": int(seeds[p_index, instance]),
            } for instance in range(instances)]
            gate_result = evaluate_gate(results, config["production_gate"], k)
            recomputed_valid[p_index] = gate_result[0]
            recomputed_failures[p_index] = ";".join(gate_result[1])
            recomputed_rhat[p_index] = gate_result[2]
            recomputed_ess[p_index] = gate_result[3]
            recomputed_status[p_index] = gate_result[4]
        comparisons = (
            ("collision", collision, recomputed_collision, False),
            ("qtop", qtop, recomputed_qtop, False),
            ("planted hit", planted, recomputed_planted, False),
            ("character", character, recomputed_character, False),
            ("validity", valid, recomputed_valid, False),
            ("failures", failures, recomputed_failures, False),
            ("R-hat", rhats, recomputed_rhat, True),
            ("ESS", esses, recomputed_ess, True),
            ("constant status", statuses, recomputed_status, False),
        )
        for name, stored, recomputed, equal_nan in comparisons:
            if not np.array_equal(stored, recomputed, equal_nan=equal_nan):
                raise ValueError(f"production stored {name} disagrees with recomputation in {path}")
        for field in ("core_seconds", "wall_seconds"):
            value = float(_raw_scalar(data, field))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"production timing is invalid in {path}: {field}")
    return {
        "task_fingerprint": fingerprint, "disorder_index": disorder,
        "qtop": qtop, "collision_mass": collision, "planted_hit": planted,
        "valid": valid,
    }


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
    if (frozen.get("engine") != "numba"
            or re.fullmatch(r"[0-9a-f]{40}", str(frozen.get("source_commit", ""))) is None):
        raise ValueError("frozen configuration lacks numba/source identity")
    frozen_hash = sha256_json(frozen)
    if os.environ.get("EXP102_FROZEN_VERIFIED_SHA256") != frozen_hash:
        raise ValueError("production worker requires a report-verified freezer")
    if os.environ.get("EXP102_SOURCE_VERIFIED_COMMIT") != frozen["source_commit"]:
        raise ValueError("production worker source identity was not verified")
    disorder_index = int(disorder_index)
    if not 0 <= disorder_index < config["num_disorders"]:
        raise ValueError("disorder index out of range")
    m_config = frozen["by_m"][str(code["m"])]
    pt_config = Q0PtConfig(**m_config)
    identity = {"code_id": code_id, "disorder_index": disorder_index,
                "registry_sha256": registry["registry_sha256"],
                "config_sha256": config["config_sha256"],
                "frozen_config_sha256": frozen_hash, "namespace": namespace}
    task_fingerprint = sha256_json(identity)
    output_path = Path(output_path)
    if output_path.exists():
        model, frame = build_model(H)
        record = validate_production_raw(output_path, registry, code, config, frozen, model, frame)
        if record["task_fingerprint"] != task_fingerprint or record["disorder_index"] != disorder_index:
            raise ValueError("existing task output has a conflicting fingerprint")
        return "reused"
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
        frozen_config_sha256=np.array(frozen_hash), physics_contract_version=np.array(config["physics_contract_version"]),
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
