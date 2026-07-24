"""Independent target-only audit of the full-row feasibility report."""

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
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (  # noqa: E402
    _initial_collapsed_masks,
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed  # noqa: E402


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "local_feasibility_report.json"
WIDTH_REPORT = ROOT / "row_elimination_width.json"
OUTPUT = ROOT / "independent_target_audit.json"
VERSION = "exp102.q0_full_row_elimination.feasibility.audit.v0"
FEASIBILITY_VERSION = "exp102.q0_full_row_elimination.feasibility.v0"
CONTROL_WORKFLOW = (
    "data.expander_code.exp102.validation."
    "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.workflow"
)
T1_UPDATES = 2048 + 8192


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _verify_self_hash(payload, field):
    expected = payload[field]
    unsigned = dict(payload)
    unsigned.pop(field)
    actual = hashlib.sha256(_canonical(unsigned).encode("ascii")).hexdigest()
    if actual != expected:
        raise RuntimeError(f"self hash changed: {field}")


def _expand(values, scope, union_scope):
    ordered_scope = tuple(item for item in union_scope if item in scope)
    permutation = tuple(scope.index(item) for item in ordered_scope)
    values = np.asarray(values, dtype=np.float64)
    if permutation != tuple(range(len(scope))):
        values = np.transpose(values, permutation)
    shape = tuple(2 if item in scope else 1 for item in union_scope)
    return values.reshape(shape)


def _eliminate_log_partition(factors, elimination_order):
    factors = [(tuple(scope), np.asarray(values, dtype=np.float64))
               for scope, values in factors]
    for variable in elimination_order:
        selected = [
            index for index, (scope, _) in enumerate(factors)
            if variable in scope
        ]
        if not selected:
            raise RuntimeError("independent elimination bucket vanished")
        union_scope = tuple(sorted(set().union(*(
            factors[index][0] for index in selected
        ))))
        joint = np.zeros((2,) * len(union_scope), dtype=np.float64)
        for index in selected:
            scope, values = factors[index]
            joint += _expand(values, scope, union_scope)
        axis = union_scope.index(variable)
        message = np.logaddexp.reduce(joint, axis=axis)
        message_scope = tuple(item for item in union_scope if item != variable)
        factors = [
            factor for index, factor in enumerate(factors)
            if index not in selected
        ]
        factors.append((message_scope, message))
    if any(scope for scope, _ in factors):
        raise RuntimeError("independent elimination left live variables")
    return sum(float(np.asarray(values)) for _, values in factors)


def _row_factors(H, b_columns, a_syndromes, row_index, log_mass,
                 log_odds, clamp=None):
    rows, columns = H.shape
    row_bit = 1 << row_index
    old_row = sum(
        ((int(b_columns[column]) >> row_index) & 1) << column
        for column in range(rows)
    )
    h_masks = [sum(
        1 << int(variable) for variable in np.flatnonzero(H[:, factor])
    ) for factor in range(columns)]
    factors = []
    for variable in range(rows):
        values = np.asarray([0.0, log_odds], dtype=np.float64)
        if clamp is not None and variable == clamp[0]:
            values[1 - int(clamp[1])] = -np.inf
        factors.append(((variable,), values))
    for factor, h_mask in enumerate(h_masks):
        scope = tuple(int(value) for value in np.flatnonzero(H[:, factor]))
        old_parity = int(old_row & h_mask).bit_count() & 1
        base = int(a_syndromes[factor]) ^ (row_bit if old_parity else 0)
        values = np.empty((2,) * len(scope), dtype=np.float64)
        for category in range(1 << len(scope)):
            syndrome = base ^ (
                row_bit if category.bit_count() & 1 else 0
            )
            assignment = tuple(
                (category >> local) & 1 for local in range(len(scope))
            )
            values[assignment] = float(log_mass[syndrome])
        factors.append((scope, values))
    return factors, old_row


def _old_score(factors, old_row):
    total = 0.0
    for scope, values in factors:
        index = tuple((old_row >> variable) & 1 for variable in scope)
        total += float(values[index])
    return total


def _independent_row_statistics(H, b_columns, a_syndromes, row_index,
                                log_mass, log_odds, order):
    factors, old_row = _row_factors(
        H, b_columns, a_syndromes, row_index, log_mass, log_odds,
    )
    log_z = _eliminate_log_partition(factors, order)
    self_log_probability = _old_score(factors, old_row) - log_z
    expected_change = 0.0
    for variable in range(H.shape[0]):
        old_bit = (old_row >> variable) & 1
        clamped, _ = _row_factors(
            H, b_columns, a_syndromes, row_index, log_mass, log_odds,
            clamp=(variable, 1 - old_bit),
        )
        expected_change += math.exp(
            _eliminate_log_partition(clamped, order) - log_z
        )
    return {
        "expected_hamming_change": expected_change,
        "self_log_probability": self_log_probability,
        "self_probability": math.exp(self_log_probability),
    }


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    width = json.loads(WIDTH_REPORT.read_text(encoding="utf-8"))
    _verify_self_hash(report, "report_sha256")
    _verify_self_hash(width, "report_sha256")
    workflow = import_module(CONTROL_WORKFLOW)
    config, config_sha = workflow._load_config()
    context = workflow._load_control(
        workflow.SOURCE_CONTROL_DIR, config, config_sha,
    )
    H = np.asarray(context["H"], dtype=np.uint8)
    syndrome = np.asarray(context["syndrome"], dtype=np.uint8)
    log_mass = np.log(build_classical_coset_mass(H, 0.04, engine="numba"))
    log_odds = math.log(0.04 / 0.96)
    control_sha = context["metadata"]["control_content_sha256"]
    u_seed = derive_seed(
        FEASIBILITY_VERSION, control_sha, "initialization", "U", 0,
    )
    states = {
        "P": context["fixed_states"][0].copy(),
        "U": uniform_hard_coset_state(context["model"], syndrome, u_seed),
        "M0": context["fixed_states"][1].copy(),
        "S0": context["fixed_states"][3].copy(),
    }
    order = tuple(int(value) for value in width["selected_order"])
    maximum_self_log_difference = 0.0
    maximum_expected_change_difference = 0.0
    families = {}
    for family, state in states.items():
        b_columns, a_syndromes, _ = _initial_collapsed_masks(
            state, syndrome, H,
        )
        rows = []
        for row_index in range(H.shape[0]):
            actual = _independent_row_statistics(
                H, b_columns, a_syndromes, row_index, log_mass, log_odds,
                order,
            )
            expected = report["families"][family]["row_statistics"][row_index]
            self_difference = abs(
                actual["self_log_probability"]
                - expected["self_log_probability"]
            )
            change_difference = abs(
                actual["expected_hamming_change"]
                - expected["expected_hamming_change"]
            )
            maximum_self_log_difference = max(
                maximum_self_log_difference, self_difference,
            )
            maximum_expected_change_difference = max(
                maximum_expected_change_difference, change_difference,
            )
            rows.append({
                **actual,
                "expected_change_absolute_difference": change_difference,
                "row_index": row_index,
                "self_log_absolute_difference": self_difference,
            })
        update_counts = np.full(H.shape[0], T1_UPDATES // H.shape[0], dtype=np.int64)
        update_counts[:T1_UPDATES % H.shape[0]] += 1
        first_change_union_bound = min(1.0, sum(
            int(update_counts[row["row_index"]])
            * row["expected_hamming_change"]
            for row in rows
        ))
        families[family] = {
            "first_change_by_t1_union_bound": float(first_change_union_bound),
            "maximum_expected_hamming_change": max(
                row["expected_hamming_change"] for row in rows
            ),
            "minimum_self_probability": min(
                row["self_probability"] for row in rows
            ),
            "rows": rows,
        }
    comparison_pass = bool(
        maximum_self_log_difference <= 1e-10
        and maximum_expected_change_difference <= 1e-9
    )
    low_energy_freezing_confirmed = all(
        families[family]["first_change_by_t1_union_bound"] <= 1e-4
        for family in ("P", "M0", "S0")
    )
    core = {
        "authority": {
            "formal_authorization": False,
            "posterior_estimation": False,
            "production_authorization": False,
            "remote_authorization": False,
        },
        "comparison_pass": comparison_pass,
        "families": families,
        "feasibility_report_sha256": report["report_sha256"],
        "low_energy_freezing_confirmed": low_energy_freezing_confirmed,
        "maximum_expected_change_absolute_difference": (
            maximum_expected_change_difference
        ),
        "maximum_self_log_absolute_difference": maximum_self_log_difference,
        "status": (
            "INDEPENDENT_TARGET_AUDIT_PASS_LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE"
            if comparison_pass and low_energy_freezing_confirmed
            else "INDEPENDENT_TARGET_AUDIT_CONFLICT"
        ),
        "version": VERSION,
        "width_report_sha256": width["report_sha256"],
    }
    core["audit_sha256"] = hashlib.sha256(
        _canonical(core).encode("ascii")
    ).hexdigest()
    OUTPUT.write_text(_canonical(core) + "\n", encoding="utf-8")
    print(json.dumps(core, sort_keys=True, indent=2))
    if core["status"] == "INDEPENDENT_TARGET_AUDIT_CONFLICT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
