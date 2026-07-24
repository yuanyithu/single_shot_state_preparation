from importlib import import_module

import numpy as np


pilot = import_module(
    "data.expander_code.exp102.validation."
    "059_q0_hybrid_row_column_local_pilot_20260724.run_local_pilot"
)


def test_frozen_hybrid_pilot_config_and_seed_panel_are_adversarial():
    config, config_sha = pilot.load_config()
    context = pilot.load_context(config, config_sha)
    tasks = pilot.build_tasks(
        config, config_sha, context["metadata"]["control_content_sha256"],
        "1" * 40,
    )
    assert len(tasks) == 16
    assert {task["family"] for task in tasks} == {"P", "U", "M0", "S0"}
    seeds = [
        task[field]
        for task in tasks
        for field in (
            "burn_seed", "initialization_seed", "measurement_seed",
            "observation_seed",
        )
    ]
    assert len(seeds) == len(set(seeds)) == 64
    states = [pilot.initial_state(context, task) for task in tasks]
    residuals = (
        context["model"].H_check.astype(np.int64)
        @ np.stack(states).T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    assert np.array_equal(
        residuals,
        np.repeat(context["syndrome"][None, :], len(states), axis=0),
    )
    u_states = [
        state for task, state in zip(tasks, states) if task["family"] == "U"
    ]
    assert len({state.tobytes() for state in u_states}) == 4
    assert not np.any(np.all(np.stack(states) == 0, axis=1))
    assert config["authority"]["posterior_estimation"] is False
    assert config["authority"]["remote_authorization"] is False


def test_b_bit_unpacking_preserves_row_column_orientation():
    columns = np.asarray([[1, 2], [3, 0]], dtype=np.uint32)
    actual = pilot.b_bits(columns, 2)
    expected = np.asarray([
        [[1, 0], [0, 1]],
        [[1, 0], [1, 0]],
    ], dtype=np.float64)
    assert np.array_equal(actual, expected)
