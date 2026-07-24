"""Sampler-independent raw audit for validation 059."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.q0_global import (  # noqa: E402
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (  # noqa: E402
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed  # noqa: E402


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "local_run_v0"
CONFIG_PATH = ROOT / "pilot_config.json"
OUTPUT = ROOT / "independent_raw_audit.json"
CONTROL_WORKFLOW = (
    "data.expander_code.exp102.validation."
    "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.workflow"
)
AUDIT_VERSION = "exp102.q0_hybrid_row_column.local_pilot.audit.v0"
RAW_VERSION = "exp102.q0_hybrid_row_column.local_pilot.raw.v0"
SAMPLER_VERSION = "exp102.q0_hgp_hybrid_row_column.v0"
COUNTER_COUNT = 9


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_json(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_self_hash(payload, field):
    expected = payload[field]
    unsigned = dict(payload)
    unsigned.pop(field)
    require(sha_json(unsigned) == expected, f"self hash changed: {field}")


def collapsed_a_syndromes(H, Y, b_columns):
    rows, columns = H.shape
    result = np.zeros(columns, dtype=np.uint32)
    for factor in range(columns):
        value = sum(int(Y[row, factor]) << row for row in range(rows))
        for b_column in np.flatnonzero(H[:, factor]):
            value ^= int(b_columns[int(b_column)])
        result[factor] = np.uint32(value)
    return result


def current_row_mask(b_columns, row_index):
    return sum(
        ((int(b_columns[column]) >> row_index) & 1) << column
        for column in range(len(b_columns))
    )


def set_row_mask(b_columns, row_index, mask):
    row_bit = np.uint32(1) << np.uint32(row_index)
    for column in range(len(b_columns)):
        if (int(mask) >> column) & 1:
            b_columns[column] |= row_bit
        else:
            b_columns[column] &= ~row_bit


def replay_stage(raw, prefix, initial_b, *, record):
    b_columns = initial_b.copy()
    selected_columns = raw[f"{prefix}__selected_columns"]
    selected_rows = raw[f"{prefix}__selected_rows"]
    clocks = selected_columns.size
    column_changes = 0
    column_changed_bits = 0
    row_changes = 0
    row_changed_bits = 0
    for clock in range(clocks):
        column = int(selected_columns[clock])
        old_column = int(raw[f"{prefix}__old_columns"][clock])
        new_column = int(raw[f"{prefix}__new_columns"][clock])
        require(int(b_columns[column]) == old_column,
                "audit column transcript old state changed")
        b_columns[column] = np.uint32(new_column)
        column_changes += old_column != new_column
        column_changed_bits += (old_column ^ new_column).bit_count()
        row = int(selected_rows[clock])
        old_row = int(raw[f"{prefix}__old_rows"][clock])
        new_row = int(raw[f"{prefix}__new_rows"][clock])
        require(current_row_mask(b_columns, row) == old_row,
                "audit row transcript old state changed")
        set_row_mask(b_columns, row, new_row)
        row_changes += old_row != new_row
        row_changed_bits += (old_row ^ new_row).bit_count()
        if record:
            require(np.array_equal(
                b_columns, raw[f"{prefix}__b_columns"][clock],
            ), "audit measured B transcript changed")
    counters = np.asarray([
        clocks, clocks, column_changes, column_changed_bits, clocks,
        row_changes, row_changed_bits, clocks if record else 0, 0,
    ], dtype=np.int64)
    return b_columns, counters


def expected_initial(context, task):
    family = task["family"]
    if family == "P":
        return context["fixed_states"][0]
    if family == "M0":
        return context["fixed_states"][1]
    if family == "S0":
        return context["fixed_states"][3]
    require(family == "U", "audit family changed")
    return uniform_hard_coset_state(
        context["model"], context["syndrome"], task["initialization_seed"],
    )


def task_rows(config, config_sha, control_sha, source_commit):
    rows = []
    namespace = config["seed_namespace"]
    for family in config["initialization"]["families"]:
        for index in range(config["initialization"]["trajectories_per_family"]):
            prefix = source_commit, config_sha, control_sha, family, index
            task = {
                "burn_seed": derive_seed(namespace, *prefix, "burn"),
                "family": family,
                "index": index,
                "initialization_seed": derive_seed(
                    namespace, *prefix, "initialization",
                ),
                "measurement_seed": derive_seed(
                    namespace, *prefix, "measurement",
                ),
                "observation_seed": derive_seed(
                    namespace, *prefix, "observation",
                ),
            }
            task["task_fingerprint"] = sha_json(task)
            rows.append(task)
    return rows


def b_bits(values, rows):
    values = np.asarray(values, dtype=np.uint32)
    result = np.empty((values.shape[0], rows, rows), dtype=np.float64)
    for sample in range(values.shape[0]):
        for row in range(rows):
            for column in range(rows):
                result[sample, row, column] = (
                    int(values[sample, column]) >> row
                ) & 1
    return result


def audit_raw(path, receipt, task, context, config, config_sha, source_commit,
              log_mass):
    require(sha_file(path) == receipt["raw_sha256"], "audit raw SHA changed")
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    require(str(raw["raw_version"].item()) == RAW_VERSION
            and str(raw["version"].item()) == SAMPLER_VERSION
            and str(raw["source_commit"].item()) == source_commit
            and str(raw["config_sha256"].item()) == config_sha
            and str(raw["task_json"].item()) == canonical(task)
            and bool(raw["replay_ok"].item()), "audit raw identity changed")
    initial = expected_initial(context, task)
    require(np.array_equal(
        raw["initial_state_packed"], np.packbits(initial, bitorder="little"),
    ), "audit initial state changed")
    _A, initial_B = split_hgp_state(initial, context["H"])
    initial_b = np.asarray([
        sum(int(initial_B[row, column]) << row
            for row in range(initial_B.shape[0]))
        for column in range(initial_B.shape[1])
    ], dtype=np.uint32)
    require(np.array_equal(initial_b, raw["initial_b_columns"]),
            "audit initial B changed")
    burn_b, burn_counters = replay_stage(raw, "burn", initial_b, record=False)
    require(np.array_equal(burn_b, raw["burn__final_b_columns"]),
            "audit burn final B changed")
    measured_b, measurement_counters = replay_stage(
        raw, "measurement", burn_b, record=True,
    )
    require(np.array_equal(measured_b, raw["final_b_columns"]),
            "audit final B changed")
    Y = context["syndrome"].reshape(context["H"].shape)
    burn_a = collapsed_a_syndromes(context["H"], Y, burn_b)
    final_a = collapsed_a_syndromes(context["H"], Y, measured_b)
    require(np.array_equal(burn_a, raw["burn__final_a_syndromes"])
            and np.array_equal(final_a, raw["final_a_syndromes"]),
            "audit cached A syndrome changed")
    states = np.unpackbits(
        raw["measurement__states_packed"], axis=1,
        count=context["model"].num_qubits, bitorder="little",
    ).astype(np.uint8)
    residuals = (
        context["model"].H_check.astype(np.int64)
        @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    require(np.array_equal(
        residuals,
        np.repeat(context["syndrome"][None, :], states.shape[0], axis=0),
    ), "audit state left hard coset")
    labels = np.asarray([
        state_label(context["frame"], state) for state in states
    ], dtype=np.uint64)
    require(np.array_equal(labels, raw["measurement__labels"])
            and np.array_equal(states.sum(axis=1), raw["measurement__weights"]),
            "audit state statistic changed")
    previous = state_label(context["frame"], initial)
    label_changes = 0
    for label in labels:
        label_changes += int(label != previous)
        previous = int(label)
    measurement_counters[8] = label_changes
    require(np.array_equal(burn_counters, raw["burn__counters"])
            and np.array_equal(measurement_counters, raw["measurement__counters"]),
            "audit counters changed")
    for clock, b_columns in enumerate(raw["measurement__b_columns"]):
        a_syndromes = collapsed_a_syndromes(context["H"], Y, b_columns)
        likelihood = float(log_mass[a_syndromes].sum())
        weight = sum(int(value).bit_count() for value in b_columns)
        require(likelihood == float(raw["measurement__b_likelihood"][clock])
                and weight == int(raw["measurement__b_weights"][clock]),
                "audit B observable changed")
        _state_A, state_B = split_hgp_state(states[clock], context["H"])
        state_b = np.asarray([
            sum(int(state_B[row, column]) << row
                for row in range(state_B.shape[0]))
            for column in range(state_B.shape[1])
        ], dtype=np.uint32)
        require(np.array_equal(state_b, b_columns),
                "audit full-state B block changed")
    identity = hashlib.sha256(
        SAMPLER_VERSION.encode("ascii") + b"\0"
        + np.asarray([
            task["burn_seed"], task["measurement_seed"],
            task["observation_seed"],
        ], dtype=">u8").tobytes()
        + np.asarray(config["cell"]["p"], dtype=">f8").tobytes()
        + np.asarray([
            config["clocks"]["burn"], config["clocks"]["measurement"],
        ], dtype=">u8").tobytes()
    ).hexdigest()
    require(str(raw["seed_identity_sha256"].item()) == identity,
            "audit seed identity changed")
    return {
        "blocks": raw["measurement__blocks"],
        "b_columns": raw["measurement__b_columns"],
        "b_likelihood": raw["measurement__b_likelihood"],
        "b_weights": raw["measurement__b_weights"],
        "burn_b_likelihood": float(log_mass[burn_a].sum()),
        "burn_b_weight": sum(int(value).bit_count() for value in burn_b),
        "column_changes": int(measurement_counters[2]),
        "replay_inclusive_seconds": float(
            raw["sampling_seconds"].item() + raw["replay_seconds"].item()
        ),
        "row_changes": int(measurement_counters[5]),
    }


def summarize(records, rows, columns):
    first_weight = []
    last_weight = []
    first_likelihood = []
    last_likelihood = []
    bits = []
    trajectories = []
    for task, raw in records:
        first = raw["blocks"] < 4
        last = ~first
        first_weight.append(raw["b_weights"][first] / (rows * rows))
        last_weight.append(raw["b_weights"][last] / (rows * rows))
        first_likelihood.append(raw["b_likelihood"][first] / columns)
        last_likelihood.append(raw["b_likelihood"][last] / columns)
        bits.append(b_bits(raw["b_columns"][last], rows))
        trajectories.append({
            "burn_b_likelihood_per_factor": raw["burn_b_likelihood"] / columns,
            "burn_b_weight_normalized": raw["burn_b_weight"] / (rows * rows),
            "column_changes": raw["column_changes"],
            "index": task["index"],
            "replay_inclusive_seconds": raw["replay_inclusive_seconds"],
            "row_changes": raw["row_changes"],
        })
    return {
        "bit_means": np.concatenate(bits).mean(axis=0),
        "first_b_likelihood_per_factor": float(np.concatenate(first_likelihood).mean()),
        "first_b_weight_normalized": float(np.concatenate(first_weight).mean()),
        "last_b_likelihood_per_factor": float(np.concatenate(last_likelihood).mean()),
        "last_b_weight_normalized": float(np.concatenate(last_weight).mean()),
        "trajectories": trajectories,
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    config_sha = sha_file(CONFIG_PATH)
    schedule = json.loads((RUN_ROOT / "schedule.json").read_text(encoding="ascii"))
    raw_manifest = json.loads((RUN_ROOT / "raw_manifest.json").read_text(encoding="ascii"))
    report = json.loads((RUN_ROOT / "pilot_report.json").read_text(encoding="ascii"))
    verify_self_hash(schedule, "schedule_sha256")
    verify_self_hash(raw_manifest, "raw_manifest_sha256")
    verify_self_hash(report, "report_sha256")
    source_commit = schedule["source_commit"]
    workflow = import_module(CONTROL_WORKFLOW)
    predecessor_config, predecessor_sha = workflow._load_config()
    context = workflow._load_control(
        workflow.SOURCE_CONTROL_DIR, predecessor_config, predecessor_sha,
    )
    control_sha = context["metadata"]["control_content_sha256"]
    tasks = task_rows(config, config_sha, control_sha, source_commit)
    require(tasks == schedule["tasks"], "audit task schedule changed")
    receipts = raw_manifest["receipts"]
    receipt_map = {row["task_fingerprint"]: row for row in receipts}
    mass = build_classical_coset_mass(context["H"], .04, engine="numba")
    log_mass = np.log(mass)
    by_family = {name: [] for name in config["initialization"]["families"]}
    for task in tasks:
        receipt = receipt_map[task["task_fingerprint"]]
        audited = audit_raw(
            RUN_ROOT / receipt["raw_relpath"], receipt, task, context, config,
            config_sha, source_commit, log_mass,
        )
        by_family[task["family"]].append((task, audited))
    rows, columns = context["H"].shape
    summaries = {
        name: summarize(records, rows, columns)
        for name, records in by_family.items()
    }
    gates_config = config["gates"]
    comparisons = []
    pair_pass = True
    names = list(summaries)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            weight_delta = abs(
                summaries[left]["last_b_weight_normalized"]
                - summaries[right]["last_b_weight_normalized"]
            )
            likelihood_delta = abs(
                summaries[left]["last_b_likelihood_per_factor"]
                - summaries[right]["last_b_likelihood_per_factor"]
            )
            bit_delta = float(np.mean(
                (summaries[left]["bit_means"] - summaries[right]["bit_means"]) ** 2
            ))
            valid = bool(
                weight_delta <= gates_config["max_b_weight_delta"]
                and likelihood_delta <= gates_config["max_b_likelihood_delta"]
                and bit_delta <= gates_config["max_b_bit_mean_squared_delta"]
            )
            pair_pass &= valid
            comparisons.append({
                "b_bit_mean_squared_delta": bit_delta,
                "b_likelihood_delta": likelihood_delta,
                "b_weight_delta": weight_delta,
                "left": left, "right": right, "valid": valid,
            })
    drift = {}
    drift_pass = True
    for family, summary in summaries.items():
        weight_delta = abs(
            summary["first_b_weight_normalized"]
            - summary["last_b_weight_normalized"]
        )
        likelihood_delta = abs(
            summary["first_b_likelihood_per_factor"]
            - summary["last_b_likelihood_per_factor"]
        )
        valid = bool(
            weight_delta <= gates_config["max_b_weight_delta"]
            and likelihood_delta <= gates_config["max_b_likelihood_delta"]
        )
        drift_pass &= valid
        drift[family] = {
            "b_likelihood_delta": likelihood_delta,
            "b_weight_delta": weight_delta,
            "valid": valid,
        }
    u_count = sum(
        row["burn_b_weight_normalized"] <= gates_config["max_u_burn_b_weight"]
        and row["burn_b_likelihood_per_factor"]
        >= gates_config["min_u_burn_b_likelihood"]
        for row in summaries["U"]["trajectories"]
    )
    low_motion = all(
        row["column_changes"] + row["row_changes"] > 0
        for family in ("P", "M0", "S0")
        for row in summaries[family]["trajectories"]
    )
    max_seconds = max(
        row["replay_inclusive_seconds"]
        for summary in summaries.values() for row in summary["trajectories"]
    )
    gates = {
        "all": False,
        "between_family_b_distribution": pair_pass,
        "low_energy_real_motion": low_motion,
        "replay_and_identity": True,
        "runtime": max_seconds <= gates_config["max_replay_inclusive_seconds"],
        "u_burn_collapse": u_count >= gates_config["min_u_collapse_count"],
        "within_family_drift": drift_pass,
    }
    gates["all"] = all(value for key, value in gates.items() if key != "all")
    serializable = {
        family: {key: value for key, value in summary.items() if key != "bit_means"}
        for family, summary in summaries.items()
    }
    recomputed = {
        "comparisons": comparisons,
        "drift": drift,
        "family_summaries": serializable,
        "gates": gates,
        "raw_count": len(receipts),
        "replay_inclusive_max_seconds": max_seconds,
        "status": (
            "LOCAL_HYBRID_B_NECESSARY_GATES_PASS"
            if gates["all"] else "LOCAL_HYBRID_B_NECESSARY_GATES_FAIL"
        ),
        "u_collapse_count": u_count,
    }
    for key, value in recomputed.items():
        require(value == report[key], f"audit report field changed: {key}")
    core = {
        "audit_version": AUDIT_VERSION,
        "pilot_report_sha256": report["report_sha256"],
        "raw_set_sha256": report["raw_set_sha256"],
        "status": "INDEPENDENT_RAW_AUDIT_PASS_LOCAL_HYBRID_B_NECESSARY_GATES_FAIL",
        "verified_raw_count": len(receipts),
    }
    core["audit_sha256"] = sha_json(core)
    with OUTPUT.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
