"""Independent raw analyzer for the fresh direct-block m8 T1 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import math
from pathlib import Path

import numpy as np


_workflow = import_module(
    "data.expander_code.exp102.validation."
    "055_q0_random_full_column_direct_block_t1_m8_20260724.workflow"
)
_legacy = import_module(
    "data.expander_code.exp102.validation."
    "052_q0_random_full_column_t1_m8_20260724.analyze_t1"
)
_direct = import_module(
    "data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column"
)
_global = import_module("data.expander_code.exp102.exp102_pipeline.q0_global")

CONTRACT_VERSION = _workflow.CONTRACT_VERSION
FAMILIES = _workflow.FAMILIES
NODE_REPORT_VERSION = _workflow.NODE_REPORT_VERSION
RAW_VERSION = _workflow.RAW_VERSION
REPORT_VERSION = _workflow.REPORT_VERSION


class AnalysisConflictError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise AnalysisConflictError(message)


def _exact_b_log_likelihood(b_states_packed, H, syndrome, log_mass):
    """Rebuild the factor indices, then use the kernel's fixed 1-D sum order."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    r, n = H.shape
    bits = np.unpackbits(
        np.asarray(b_states_packed, dtype=np.uint8), axis=1,
        count=r * r, bitorder="little",
    )
    B = bits.reshape(-1, r, r).astype(np.int64, copy=False)
    bh = np.einsum(
        "tij,jk->tik", B, H.astype(np.int64), optimize=False,
    ) & 1
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    a_syndromes = bh ^ Y[None, :, :]
    powers = np.left_shift(np.int64(1), np.arange(r, dtype=np.int64))
    indices = np.einsum(
        "trn,r->tn", a_syndromes.astype(np.int64, copy=False), powers,
        optimize=False,
    )
    values = np.asarray(log_mass, dtype=np.float64)
    return np.asarray(
        [float(values[row].sum()) for row in indices], dtype=np.float64,
    )


def _load_and_verify_raw(path, task, context, schedule, log_mass):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required_metadata = {
        "archive_sha256", "config_sha256", "contract_version",
        "control_content_sha256", "model_fingerprint", "raw_version",
        "replay_seconds", "sampling_seconds", "schedule_sha256", "source_commit",
        "source_manifest_sha256", "syndrome_packed", "task_fingerprint",
        "task_json", "version",
    }
    required_kernel = {
        "burn__counters", "burn__final_b_columns", "burn__selected_columns",
        "burn__old_columns", "burn__new_columns", "conditional_engine",
        "final_b_columns", "final_state_packed", "initial_b_columns",
        "initial_state_packed", "measurement__counters",
        "measurement__selected_columns", "measurement__old_columns",
        "measurement__new_columns", "measurement__b_columns",
        "measurement__b_likelihood", "measurement__b_weights",
        "measurement__blocks", "measurement__labels",
        "measurement__states_packed", "measurement__weights",
        "seed_identity_sha256",
    }
    _require(set(data) == required_metadata | required_kernel, "raw schema changed")
    source = schedule["source_identity"]
    scalar_identity = {
        "archive_sha256": source["archive_sha256"],
        "conditional_engine": "numba_direct_positive_fixed_block_12",
        "config_sha256": context["config_sha"],
        "contract_version": CONTRACT_VERSION,
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "model_fingerprint": context["model"].fingerprint(),
        "raw_version": RAW_VERSION,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_commit": source["source_commit"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "task_fingerprint": task["task_fingerprint"],
        "task_json": _workflow.canonical_json(task),
        "version": _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_VERSION,
    }
    for name, expected in scalar_identity.items():
        _require(str(data[name].item()) == expected, f"raw identity mismatch: {name}")
    _require(
        task["method_id"] == _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_METHOD_ID,
        "task method identity changed",
    )
    _require(
        np.array_equal(data["syndrome_packed"], context["arrays"]["syndrome_packed"])
        and math.isfinite(float(data["sampling_seconds"]))
        and float(data["sampling_seconds"]) > 0.0
        and math.isfinite(float(data["replay_seconds"]))
        and float(data["replay_seconds"]) > 0.0,
        "raw syndrome or timing invalid",
    )
    seed_identity = hashlib.sha256(
        _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_VERSION.encode("ascii") + b"\0"
        + np.asarray([
            task["burn_update_seed"], task["measurement_update_seed"],
            task["observation_seed"],
        ], dtype=">u8").tobytes()
        + np.asarray(0.04, dtype=">f8").tobytes()
        + np.asarray([
            context["config"]["resource"]["burn_updates"],
            context["config"]["resource"]["measurement_updates"],
        ], dtype=">u8").tobytes()
    ).hexdigest()
    _require(
        str(data["seed_identity_sha256"].item()) == seed_identity,
        "raw seed identity mismatch",
    )

    initial = _legacy._unpack(
        data["initial_state_packed"], context["model"].num_qubits,
    )
    family = task["family"]
    if family == "P":
        expected = context["fixed_states"][0]
    elif family == "M0":
        expected = context["fixed_states"][1]
    elif family == "M1":
        expected = context["fixed_states"][2]
    elif family == "S":
        expected = context["fixed_states"][3 + int(task["index"])]
    else:
        expected = _global.uniform_hard_coset_state(
            context["model"], context["syndrome"], task["initialization_seed"],
        )
    _require(np.array_equal(initial, expected), "raw initial state mismatch")
    _require(
        np.array_equal(
            _legacy._state_b_columns(initial, context["H"]),
            data["initial_b_columns"],
        ),
        "raw initial B columns mismatch",
    )
    burn_trace, burn_changes, burn_bits = _legacy._replay_b_transcript(
        data["initial_b_columns"], data["burn__selected_columns"],
        data["burn__old_columns"], data["burn__new_columns"],
    )
    _require(
        np.array_equal(burn_trace[-1], data["burn__final_b_columns"]),
        "raw burn endpoint mismatch",
    )
    measurement_trace, measurement_changes, measurement_bits = (
        _legacy._replay_b_transcript(
            data["burn__final_b_columns"], data["measurement__selected_columns"],
            data["measurement__old_columns"], data["measurement__new_columns"],
        )
    )
    _require(
        np.array_equal(measurement_trace, data["measurement__b_columns"])
        and np.array_equal(measurement_trace[-1], data["final_b_columns"]),
        "raw measurement B transcript mismatch",
    )
    burn_counters = data["burn__counters"]
    measurement_counters = data["measurement__counters"]
    _require(
        np.array_equal(
            burn_counters[:4], [burn_trace.shape[0], burn_changes, burn_bits, 0],
        )
        and int(burn_counters[4]) == 0,
        "raw burn counters mismatch",
    )

    states = _legacy._unpack(
        data["measurement__states_packed"], context["model"].num_qubits,
    )
    _legacy._verify_hgp_syndrome(states, context["H"], context["syndrome"])
    n = context["H"].shape[1]
    r = context["H"].shape[0]
    observed_b = np.packbits(
        states[:, n * n:].reshape(-1, r * r), axis=1, bitorder="little",
    )
    stored_b = _legacy._columns_to_b_packed(measurement_trace, r)
    _require(np.array_equal(observed_b, stored_b), "raw state/B mismatch")
    labels = _legacy._labels_from_states(
        states, _legacy._qubit_signatures(context["frame"]),
    )
    _require(
        np.array_equal(labels, data["measurement__labels"])
        and np.array_equal(states.sum(axis=1), data["measurement__weights"]),
        "raw label or weight mismatch",
    )
    initial_label = int(_global.state_label(context["frame"], initial))
    label_changes = int(labels[0] != initial_label)
    label_changes += int(np.count_nonzero(labels[1:] != labels[:-1]))
    _require(
        np.array_equal(
            measurement_counters[:4], [
                measurement_trace.shape[0], measurement_changes, measurement_bits,
                measurement_trace.shape[0],
            ],
        )
        and int(measurement_counters[4]) == label_changes,
        "raw measurement counters mismatch",
    )
    b_weights = np.bitwise_count(stored_b).sum(axis=1).astype(np.int16)
    b_likelihood = _exact_b_log_likelihood(
        stored_b, context["H"], context["syndrome"], log_mass,
    )
    _require(
        np.array_equal(b_weights, data["measurement__b_weights"])
        and np.array_equal(b_likelihood, data["measurement__b_likelihood"]),
        "raw B weight or likelihood mismatch",
    )
    _require(
        np.array_equal(
            data["measurement__blocks"],
            np.minimum(7, 8 * np.arange(labels.size) // labels.size).astype(np.int8),
        ),
        "raw block clock mismatch",
    )
    _require(
        np.array_equal(
            _legacy._unpack(
                data["final_state_packed"], context["model"].num_qubits,
            ),
            states[-1],
        ),
        "raw final state mismatch",
    )
    return {
        "b_likelihood": b_likelihood,
        "b_packed": stored_b,
        "b_weights": b_weights.astype(np.float64),
        "burn_b_packed": _legacy._columns_to_b_packed(burn_trace, r),
        "burn_changes": burn_changes,
        "family": family,
        "index": int(task["index"]),
        "initial_b_packed": _legacy._columns_to_b_packed(
            data["initial_b_columns"][None, :], r,
        )[0],
        "initial_label": initial_label,
        "label_changes": label_changes,
        "labels": labels,
        "measurement_changes": measurement_changes,
        "path": path,
        "sampling_seconds": float(data["sampling_seconds"]),
        "weights": data["measurement__weights"].astype(np.float64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    _require(not output.exists(), "analysis output already exists")
    config, config_sha = _workflow._load_config()
    context = _workflow._load_control(run_root / "control", config, config_sha)
    schedule = _workflow._load_canonical_json(run_root / "control/schedule.json")
    _workflow._verify_self_hash(schedule, "schedule_sha256")
    preflight = _workflow._load_canonical_json(run_root / "preflight/aggregate.json")
    _workflow._verify_self_hash(preflight, "preflight_sha256")
    _require(
        preflight["status"] == "PASS"
        and preflight["exact_consensus"] is True
        and preflight["schedule_sha256"] == schedule["schedule_sha256"],
        "measurement lacks PASS aggregate preflight",
    )
    task_by_fingerprint = {
        task["task_fingerprint"]: task for task in schedule["tasks"]
    }
    raw_by_fingerprint = {}
    node_reports = {}
    for node in config["resource"]["allowed_nodes"]:
        report_path = run_root / f"measurement/{node}/node_report.json"
        report = _workflow._load_canonical_json(report_path)
        _workflow._verify_self_hash(report, "node_report_sha256")
        _require(
            report["node_report_version"] == NODE_REPORT_VERSION
            and report["node"] == node
            and report["status"] == "COMPLETE"
            and report["schedule_sha256"] == schedule["schedule_sha256"],
            "node report identity changed",
        )
        node_reports[node] = report["node_report_sha256"]
        for record in report["raw_records"]:
            path = run_root / f"measurement/{node}/raw/{record['file']}"
            _require(
                _workflow.sha256_file(path) == record["raw_sha256"],
                "node raw hash mismatch",
            )
            fingerprint = record["task_fingerprint"]
            _require(fingerprint not in raw_by_fingerprint,
                     "duplicate raw fingerprint")
            raw_by_fingerprint[fingerprint] = path
    _require(
        set(raw_by_fingerprint) == set(task_by_fingerprint),
        "raw task set is incomplete",
    )
    mass = _legacy.build_classical_coset_mass(context["H"], 0.04, engine="numba")
    log_mass = np.log(mass)
    records = [
        _load_and_verify_raw(
            raw_by_fingerprint[task["task_fingerprint"]], task, context,
            schedule, log_mass,
        )
        for task in schedule["tasks"]
    ]
    logical_set = _legacy.CharacterSet(
        masks=context["arrays"]["logical_character_masks"],
        basis_positions=context["arrays"]["logical_basis_positions"],
        tier="sampled", k=context["model"].k,
        random_seed=context["metadata"]["logical_character_seed"],
        character_sha256=context["metadata"]["logical_character_sha256"],
    )
    b_set = _legacy.BCharacterSet(
        masks_packed=context["arrays"]["b_character_masks_packed"],
        r=context["H"].shape[0],
        dense_count=config["statistics"]["b_dense_character_count"],
        random_seed=context["metadata"]["b_character_seed"],
        character_sha256=context["metadata"]["b_character_sha256"],
    )
    family_summaries = {
        family: _legacy._family_summary(
            [row for row in records if row["family"] == family],
            context, logical_set, b_set,
        )
        for family in FAMILIES
    }
    comparisons = []
    for left_index, left in enumerate(FAMILIES):
        for right in FAMILIES[left_index + 1:]:
            comparisons.append(_legacy._pair_comparison(
                left, right, family_summaries[left], family_summaries[right],
                logical_set, b_set, context,
            ))
    constant_failures = _legacy._constant_b_freeze_failures(records, b_set)
    bridge = _legacy._map_bridge_gate(family_summaries, context)
    checks = {
        "all_families": all(
            summary["valid"] for summary in family_summaries.values()
        ),
        "all_pairwise_comparisons": all(row["valid"] for row in comparisons),
        "constant_b_freeze": not constant_failures,
        "map_bridge": all(row["valid"] for row in bridge.values()),
        "raw_identity_and_algebra": True,
    }
    status = (
        "DIAGNOSTIC_DIRECT_BLOCK_T1_M8_VIABLE"
        if all(checks.values())
        else "UNRESOLVED_DIRECT_BLOCK_T1_M8"
    )
    raw_paths = sorted(raw_by_fingerprint.values(), key=lambda path: path.as_posix())
    raw_set_sha = hashlib.sha256("".join(
        f"{path.relative_to(run_root).as_posix()}:{_workflow.sha256_file(path)}\n"
        for path in raw_paths
    ).encode("ascii")).hexdigest()
    core = {
        "checks": checks,
        "comparisons": comparisons,
        "config_sha256": config_sha,
        "constant_b_freeze_failures": constant_failures,
        "contract_version": CONTRACT_VERSION,
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "families": {
            family: _legacy._public_family(summary)
            for family, summary in family_summaries.items()
        },
        "map_bridge": bridge,
        "node_report_sha256": node_reports,
        "preflight_sha256": preflight["preflight_sha256"],
        "raw_count": len(raw_paths),
        "raw_set_sha256": raw_set_sha,
        "report_version": REPORT_VERSION,
        "schedule_sha256": schedule["schedule_sha256"],
        "scope": config["scope"],
        "source_identity": schedule["source_identity"],
        "status": status,
    }
    report = {**core, "report_sha256": _workflow.sha256_json(core)}
    _legacy.atomic_json(output, report)
    print(_workflow.canonical_json({
        "checks": checks,
        "report_sha256": report["report_sha256"],
        "status": status,
    }))


if __name__ == "__main__":
    main()
