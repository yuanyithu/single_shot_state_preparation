"""exp106 and exp105 must be the same function at q = 0.05, bit for bit.

This is the gate that actually covers the port. The exp104 comparison runs at
q = 0, where the augmented matrix, the mixed error channel, the readout draw and
the q > 0 failure criterion are all switched off -- so it cannot see a mistake in
any of them. exp105 ran that whole path for 1,057,020 trials at q = 0.05 and its
result is published, so exp106 is required to reproduce it exactly, on exp105's
own frozen registry and its own q.

Nothing here is meant to agree only in a limit. The two packages are the same
function, so every recorded field must match: correction, iteration count,
logical label, readout-match flag and verdict.
"""

import numpy as np
import pytest

from data.expander_code.exp106.exp106_pipeline.equality import (
    build_paired_exp105_models,
    compare_augmented_trials,
)


CASES = [(3, 0), (3, 1), (4, 0), (5, 0), (8, 0)]
# exp105's own grid, at both ends and in the ordered middle where the verdict is
# not saturated and a scoring difference would actually show.
P_TOKENS = ("0.001", "0.01", "0.04")
EXP105_Q = "0.05"
TRIALS = 12


@pytest.fixture(scope="module")
def exp105_registry_rows():
    from data.expander_code.exp105.exp105_pipeline.ensemble import (
        load_registry as load_exp105_registry,
    )

    # The production registry, not the pilot one: it carries all six sizes,
    # and these are the exact codes exp105 published its result on.
    path = "data/expander_code/exp105/config/ensemble_registry.v1.npz"
    registry = load_exp105_registry(path)
    return {row["code_id"]: row for row in registry["codes"]}


@pytest.mark.parametrize("m,index", CASES)
def test_models_are_byte_identical(exp105_registry_rows, m, index):
    exp105_model, exp106_model = build_paired_exp105_models(
        exp105_registry_rows[f"m{m:02d}_c{index:06d}"]
    )
    for name in ("H_Z", "H_X", "logical_Z", "label_basis"):
        assert np.array_equal(
            getattr(exp105_model, name), getattr(exp106_model, name)
        ), name
    assert exp105_model.n == exp106_model.n == 25 * m ** 2
    assert exp105_model.k == exp106_model.k == m ** 2
    assert exp105_model.n_checks == exp106_model.n_checks == 12 * m ** 2
    assert exp105_model.classical_H_sha256 == exp106_model.classical_H_sha256
    assert exp105_model.logical_frame_sha256 == exp106_model.logical_frame_sha256
    assert (
        exp105_model.observable_frame_fingerprint
        == exp106_model.observable_frame_fingerprint
    )


@pytest.mark.parametrize("m,index", CASES)
def test_the_augmented_path_agrees_bit_for_bit(exp105_registry_rows, m, index):
    exp105_model, exp106_model = build_paired_exp105_models(
        exp105_registry_rows[f"m{m:02d}_c{index:06d}"]
    )
    for p in P_TOKENS:
        result = compare_augmented_trials(
            exp105_model, exp106_model, p, EXP105_Q,
            seed=104729 + index, trials=TRIALS,
        )
        assert result["mismatches"] == [], (
            f"m={m} c{index} p={p}: {result['mismatches']}"
        )
        assert result["compared"] == TRIALS


def test_the_comparison_reaches_real_failures(exp105_registry_rows):
    """A live control: the gate must run where the verdict is not constant.

    Comparing two packages that both always succeed proves nothing about the
    scoring criterion, so this asserts the chosen operating point actually
    produces failures to disagree about.
    """
    exp105_model, exp106_model = build_paired_exp105_models(
        exp105_registry_rows["m03_c000000"]
    )
    result = compare_augmented_trials(
        exp105_model, exp106_model, "0.04", EXP105_Q, seed=31337, trials=24,
    )
    assert result["mismatches"] == []
    # exp105 published P_fail(m=3, p=0.04) = 0.5388 at this q, so an all-success
    # or all-failure run over 24 trials would mean the path under test moved.
    assert 0 < result["failures"] < result["compared"]


def test_the_comparison_would_notice_a_difference(exp105_registry_rows):
    """A negative control: perturb one model and require the gate to fire."""
    exp105_model, exp106_model = build_paired_exp105_models(
        exp105_registry_rows["m03_c000000"]
    )
    tampered = np.array(exp106_model.label_basis, copy=True)
    tampered[0, 0] ^= 1
    object.__setattr__(exp106_model, "label_basis", tampered)
    with pytest.raises(ValueError, match="models disagree on label_basis"):
        compare_augmented_trials(
            exp105_model, exp106_model, "0.01", EXP105_Q, seed=1, trials=2,
        )


def test_exp106_reuses_no_predecessor_code(exp105_registry_rows):
    """The ensembles must be disjoint by construction, not by luck.

    Both equality gates read a predecessor's registry on purpose; the panels
    must still share no code. All three draw from a seed stream keyed on the
    master seed, so a collision would mean the fresh seed is not fresh.
    """
    from data.expander_code.exp104.exp104_pipeline.ensemble import (
        load_registry as load_exp104_registry,
    )
    from data.expander_code.exp106.exp106_pipeline.ensemble import load_registry

    ours = load_registry(
        "data/expander_code/exp106/config/ensemble_registry.pilot.v1.npz"
    )
    mine = {str(row["classical_H_sha256"]) for row in ours["codes"]}
    assert mine, "the pilot registry is empty"

    exp104 = load_exp104_registry(
        "data/expander_code/exp104/config/ensemble_registry.v1.json"
    )
    predecessors = {
        "exp105": {
            str(row["classical_H_sha256"]) for row in exp105_registry_rows.values()
        },
        "exp104": {
            str(row["classical_H_sha256"]) for row in exp104["codes"]
        },
    }
    for name, theirs in predecessors.items():
        assert theirs, f"{name}'s registry is empty, so this proves nothing"
        assert mine & theirs == set(), f"exp106 drew a code {name} already measured"
