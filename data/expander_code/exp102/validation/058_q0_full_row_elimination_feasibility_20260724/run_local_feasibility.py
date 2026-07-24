"""Run the frozen local m8 full-row conditional feasibility panel."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _initial_collapsed_masks,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs import (
    FULL_ROW_GIBBS_METHOD_ID,
    FULL_ROW_GIBBS_VERSION,
    build_full_row_elimination_plan,
    full_row_conditional_statistics,
    full_row_elimination_gibbs_update,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
OUTPUT = ROOT / "local_feasibility_report.json"
VERSION = "exp102.q0_full_row_elimination.feasibility.v0"
CONTROL_WORKFLOW = (
    "data.expander_code.exp102.validation."
    "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.workflow"
)
ORACLE_TEST = EXP102_ROOT / "tests/test_q0_hgp_full_row_gibbs.py"
WIDTH_REPORT = ROOT / "row_elimination_width.json"
T1_UPDATES = 2048 + 8192
SAFETY_FACTOR = 2.0
TRAJECTORY_WALL_CAP_SECONDS = 7200.0
MAX_LOG_MASS_BYTES = 256 * (1 << 20)
MAX_INCREMENTAL_PEAK_BYTES = 1 << 30


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _verify_self_hash(payload, field):
    expected = payload[field]
    unsigned = dict(payload)
    unsigned.pop(field)
    if hashlib.sha256(_canonical(unsigned).encode("ascii")).hexdigest() != expected:
        raise RuntimeError(f"self hash changed: {field}")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "maximum": float(np.max(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
    }


def _run_sweep(H, syndrome, log_mass, log_odds, plan, initial_state, seed):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    b_columns, a_syndromes, _ = _initial_collapsed_masks(
        initial_state, syndrome, H,
    )
    initial_b = b_columns.copy()
    initial_a = a_syndromes.copy()
    syndrome_matrix = np.asarray(syndrome, dtype=np.uint8).reshape(H.shape)
    rng = PortablePrng(int(seed))
    transcript = []
    tracemalloc.start()
    started = time.perf_counter()
    for row_index in range(H.shape[0]):
        changed, changed_bits, old, new = full_row_elimination_gibbs_update(
            b_columns, a_syndromes, H, syndrome_matrix, row_index, log_mass,
            log_odds, rng, plan=plan,
        )
        transcript.append((
            int(row_index), bool(changed), int(changed_bits), int(old), int(new),
        ))
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "a_syndromes": a_syndromes,
        "b_columns": b_columns,
        "changed_bits": int(sum(row[2] for row in transcript)),
        "changed_rows": int(sum(row[1] for row in transcript)),
        "elapsed_seconds": float(elapsed),
        "incremental_peak_bytes": int(peak),
        "initial_a_syndromes": initial_a,
        "initial_b_columns": initial_b,
        "transcript": transcript,
    }


def _same_sweep(left, right):
    return bool(
        left["transcript"] == right["transcript"]
        and np.array_equal(left["b_columns"], right["b_columns"])
        and np.array_equal(left["a_syndromes"], right["a_syndromes"])
        and np.array_equal(left["initial_b_columns"], right["initial_b_columns"])
        and np.array_equal(left["initial_a_syndromes"], right["initial_a_syndromes"])
    )


def main():
    oracle = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ORACLE_TEST)],
        cwd=EXP102_ROOT.parents[2], capture_output=True, text=True, check=False,
    )
    workflow = import_module(CONTROL_WORKFLOW)
    config, config_sha = workflow._load_config()
    context = workflow._load_control(
        workflow.SOURCE_CONTROL_DIR, config, config_sha,
    )
    H = np.ascontiguousarray(context["H"], dtype=np.uint8)
    syndrome = np.ascontiguousarray(context["syndrome"], dtype=np.uint8)
    width_report = json.loads(WIDTH_REPORT.read_text(encoding="utf-8"))
    _verify_self_hash(width_report, "report_sha256")
    plan = build_full_row_elimination_plan(H)
    if (
        list(plan.elimination_order) != width_report["selected_order"]
        or plan.h_sha256 != width_report["h_sha256"]
    ):
        raise RuntimeError("frozen width report and implementation plan diverged")

    mass_started = time.perf_counter()
    mass = np.ascontiguousarray(
        build_classical_coset_mass(H, 0.04, engine="numba"), dtype=np.float64,
    )
    mass_seconds = time.perf_counter() - mass_started
    log_mass = np.log(mass)
    if not np.all(np.isfinite(log_mass)):
        raise RuntimeError("m8 log-mass table is non-finite")
    log_odds = math.log(0.04 / 0.96)

    control_sha = context["metadata"]["control_content_sha256"]
    u_seed = derive_seed(VERSION, control_sha, "initialization", "U", 0)
    initial_states = {
        "P": context["fixed_states"][0].copy(),
        "U": uniform_hard_coset_state(context["model"], syndrome, u_seed),
        "M0": context["fixed_states"][1].copy(),
        "S0": context["fixed_states"][3].copy(),
    }
    families = {}
    all_expected_changes = []
    numerical_valid = True
    replay_valid = True
    maximum_seconds_per_update = 0.0
    maximum_incremental_peak = 0
    initial_b = {}
    for family, initial_state in initial_states.items():
        b_columns, a_syndromes, _ = _initial_collapsed_masks(
            initial_state, syndrome, H,
        )
        initial_b[family] = b_columns.copy()
        rows = []
        statistics_started = time.perf_counter()
        for row_index in range(H.shape[0]):
            row = full_row_conditional_statistics(
                H, b_columns, a_syndromes, row_index, log_mass, log_odds,
                plan=plan,
            )
            row["row_index"] = int(row_index)
            rows.append(row)
            all_expected_changes.append(row["expected_hamming_change"])
            numerical_valid &= bool(
                -1e-11 <= row["entropy_bits"] <= H.shape[0] + 1e-11
                and 0.0 <= row["self_probability"] <= 1.0
                and -1e-11 <= row["expected_hamming_change"] <= H.shape[0] + 1e-11
                and -1e-11 <= row["expected_row_weight"] <= H.shape[0] + 1e-11
                and all(math.isfinite(float(value)) for value in row.values())
            )
        statistics_seconds = time.perf_counter() - statistics_started
        sweep_seed = derive_seed(VERSION, control_sha, "sweep", family, 0)
        first = _run_sweep(
            H, syndrome, log_mass, log_odds, plan, initial_state, sweep_seed,
        )
        replay = _run_sweep(
            H, syndrome, log_mass, log_odds, plan, initial_state, sweep_seed,
        )
        replay_ok = _same_sweep(first, replay)
        replay_valid &= replay_ok
        seconds_per_update = max(
            first["elapsed_seconds"], replay["elapsed_seconds"],
        ) / H.shape[0]
        maximum_seconds_per_update = max(
            maximum_seconds_per_update, seconds_per_update,
        )
        incremental_peak = max(
            first["incremental_peak_bytes"], replay["incremental_peak_bytes"],
        )
        maximum_incremental_peak = max(maximum_incremental_peak, incremental_peak)
        families[family] = {
            "conditional_entropy_bits": _summary([
                row["entropy_bits"] for row in rows
            ]),
            "conditional_statistics_seconds": float(statistics_seconds),
            "expected_hamming_change": _summary([
                row["expected_hamming_change"] for row in rows
            ]),
            "expected_row_weight": _summary([
                row["expected_row_weight"] for row in rows
            ]),
            "initial_b_columns_sha256": _sha256_array(b_columns),
            "initial_b_weight": int(sum(int(value).bit_count() for value in b_columns)),
            "initial_label": int(state_label(context["frame"], initial_state)),
            "initial_state_sha256": _sha256_array(initial_state),
            "initial_weight": int(initial_state.sum()),
            "row_statistics": rows,
            "self_probability": _summary([
                row["self_probability"] for row in rows
            ]),
            "sweep": {
                "changed_bits": first["changed_bits"],
                "changed_rows": first["changed_rows"],
                "elapsed_seconds": first["elapsed_seconds"],
                "final_a_syndromes_sha256": _sha256_array(first["a_syndromes"]),
                "final_b_columns_sha256": _sha256_array(first["b_columns"]),
                "incremental_peak_bytes": first["incremental_peak_bytes"],
                "replay_elapsed_seconds": replay["elapsed_seconds"],
                "replay_incremental_peak_bytes": replay["incremental_peak_bytes"],
                "replay_ok": replay_ok,
                "seconds_per_update_conservative": float(seconds_per_update),
                "seed": int(sweep_seed),
                "transcript": [list(row) for row in first["transcript"]],
            },
        }

    pairwise_b_hamming = []
    names = list(initial_states)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            pairwise_b_hamming.append({
                "distance": int(sum(
                    int(a ^ b).bit_count()
                    for a, b in zip(initial_b[left], initial_b[right])
                )),
                "left": left,
                "right": right,
            })
    projected_t1_seconds_with_safety = (
        maximum_seconds_per_update * T1_UPDATES * SAFETY_FACTOR
    )
    gates = {
        "exact_oracle": oracle.returncode == 0,
        "memory": bool(
            log_mass.nbytes <= MAX_LOG_MASS_BYTES
            and maximum_incremental_peak <= MAX_INCREMENTAL_PEAK_BYTES
        ),
        "nontrivial_conditional": max(all_expected_changes) >= 0.1,
        "numerical_ranges": numerical_valid,
        "portable_replay": replay_valid,
        "runtime": projected_t1_seconds_with_safety <= TRAJECTORY_WALL_CAP_SECONDS,
        "width": bool(
            width_report["gate_pass"]
            and plan.induced_width <= 18
            and plan.largest_factor_entries <= (1 << 19)
        ),
    }
    gates["all"] = all(gates.values())
    core = {
        "authority": {
            "formal_authorization": False,
            "maximum_status": "LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE",
            "posterior_estimation": False,
            "production_authorization": False,
            "remote_authorization": False,
        },
        "cell": context["metadata"]["cell"],
        "control_content_sha256": control_sha,
        "families": families,
        "gates": gates,
        "implementation": {
            "method_id": FULL_ROW_GIBBS_METHOD_ID,
            "module_sha256": _sha256_file(
                EXP102_ROOT / "exp102_pipeline/q0_hgp_full_row_gibbs.py"
            ),
            "oracle_test_sha256": _sha256_file(ORACLE_TEST),
            "plan_sha256": plan.plan_sha256,
            "version": FULL_ROW_GIBBS_VERSION,
        },
        "initialization": {
            "families": names,
            "pairwise_b_hamming": pairwise_b_hamming,
            "physical_zero_is_legal": bool(not np.any(syndrome)),
            "shifted_zero_is_P": True,
            "u_seed": int(u_seed),
        },
        "mass": {
            "bytes": int(mass.nbytes),
            "log_mass_bytes": int(log_mass.nbytes),
            "mass_sha256": _sha256_array(mass),
            "seconds": float(mass_seconds),
        },
        "oracle": {
            "command": [sys.executable, "-m", "pytest", "-q", str(ORACLE_TEST)],
            "returncode": int(oracle.returncode),
            "stderr": oracle.stderr.strip(),
            "stdout": oracle.stdout.strip(),
        },
        "plan": plan.as_dict(),
        "resource": {
            "maximum_incremental_peak_bytes": int(maximum_incremental_peak),
            "maximum_seconds_per_update": float(maximum_seconds_per_update),
            "projected_t1_seconds_with_safety": float(
                projected_t1_seconds_with_safety
            ),
            "safety_factor": SAFETY_FACTOR,
            "t1_updates": T1_UPDATES,
            "trajectory_wall_cap_seconds": TRAJECTORY_WALL_CAP_SECONDS,
        },
        "status": (
            "LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE"
            if gates["all"] else "LOCAL_FULL_ROW_CONDITIONAL_NOT_VIABLE"
        ),
        "syndrome_weight": int(syndrome.sum()),
        "version": VERSION,
        "width_report_sha256": width_report["report_sha256"],
    }
    core["report_sha256"] = hashlib.sha256(
        _canonical(core).encode("ascii")
    ).hexdigest()
    OUTPUT.write_text(_canonical(core) + "\n", encoding="utf-8")
    print(json.dumps(core, sort_keys=True, indent=2))
    if not gates["all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
