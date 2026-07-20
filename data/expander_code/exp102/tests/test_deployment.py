from importlib import import_module

import pytest

from data.expander_code.exp102.exp102_pipeline.io import canonical_json


deployment_module = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.build_production_deployment"
)
selected_held_out_records = deployment_module.selected_held_out_records
lpt_assign = deployment_module.lpt_assign


P_VALUES = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)


def _candidate(rounds):
    return {
        "p_hot": 0.49, "num_temperatures": 96, "gamma": 1.0,
        "burn_rounds": rounds[0], "measurement_rounds": rounds[1],
        "sweeps_per_round": 1, "logical_move_repeat": 1,
    }


def _held_out_rows(m, attempt, candidate, valid, core_seconds):
    return [
        {
            "stage": "held_out", "m": m, "attempt": attempt,
            "candidate": candidate, "candidate_key": canonical_json(candidate),
            "code_id": f"m{m:02d}_c{code_index:02d}", "p": p,
            "disorder_index": disorder, "valid": valid,
            "core_seconds": core_seconds, "task_fingerprint": (
                f"m{m}-a{attempt}-c{code_index}-p{p}-d{disorder}"
            ),
        }
        for code_index in range(8)
        for p in P_VALUES
        for disorder in range(8)
    ]


def test_deployment_uses_only_report_selected_held_out_attempt():
    registry = {
        "codes": [
            {"code_id": f"m{m:02d}_c{code_index:02d}", "m": m}
            for m in range(3, 9) for code_index in range(8)
        ],
    }
    failed = _candidate((500, 2000))
    passed = _candidate((1000, 4000))
    records = []
    by_m = {}
    for m in range(3, 9):
        records.extend(_held_out_rows(m, 0, failed, False, 999.0))
        records.extend(_held_out_rows(m, 1, passed, True, float(m)))
        by_m[str(m)] = {
            "all_tuning_pass": True, "all_held_out_pass": True,
            "selected_config": passed,
            "held_out": {"selected_attempt": 1},
        }

    selected, attempts = selected_held_out_records(records, by_m, registry)
    assert len(selected) == 2688
    assert set(attempts.values()) == {1}
    assert {row["attempt"] for row in selected} == {1}
    assert {row["core_seconds"] for row in selected} == set(range(3, 9))

    with pytest.raises(ValueError, match="448 unique valid cells"):
        selected_held_out_records(records[:-1], by_m, registry)


def test_production_lpt_accounts_for_all_seven_p_values_and_node_capacity():
    owners, core_load, wall_load = lpt_assign({"large": 10.0, "small": 1.0})
    assert owners["large"] == "nd-3"
    assert core_load["nd-3"] == 128 * 7 * 10.0
    assert core_load[owners["small"]] >= 128 * 7 * 1.0
    assert wall_load["nd-3"] == pytest.approx(core_load["nd-3"] / 91)
