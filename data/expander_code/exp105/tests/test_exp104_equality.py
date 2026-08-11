"""exp105 and exp104 must be the same function at q = 0.

exp105 adds a branch to a machine exp104 already certified. The dangerous part
is not the new branch but the shared one: ensemble reconstruction, model
assembly and scoring. This gate runs both packages on the same codes, drawn from
exp104's own frozen registry, with the same seeds, and requires identical
models, corrections, iteration counts, verdicts and labels.

exp104's production raw lives on nd-3 and is not tracked in Git, so this is a
package-to-package comparison rather than a replay of stored files -- the same
form exp104's Validation 002 used against exp103.
"""

import numpy as np
import pytest

from data.expander_code.exp105.exp105_pipeline.equality import (
    build_paired_models,
    compare_q0_trials,
    exp104_row_to_exp105,
)


CASES = [(3, 0), (3, 1), (3, 2), (4, 0), (5, 0)]
P_TOKENS = ("0.02", "0.06", "0.10")
TRIALS = 20


@pytest.mark.parametrize("m,index", CASES)
def test_models_are_byte_identical(exp104_registry_rows, m, index):
    row = exp104_registry_rows[f"m{m:02d}_c{index:05d}"]
    exp104_model, exp105_model = build_paired_models(row)
    assert np.array_equal(exp104_model.H_Z, exp105_model.H_Z)
    assert np.array_equal(exp104_model.H_X, exp105_model.H_X)
    assert np.array_equal(exp104_model.logical_Z, exp105_model.logical_Z)
    assert exp104_model.n == exp105_model.n == 25 * m ** 2
    assert exp104_model.k == exp105_model.k == m ** 2
    assert exp105_model.n_checks == 12 * m ** 2
    assert exp104_model.classical_H_sha256 == exp105_model.classical_H_sha256


@pytest.mark.parametrize("m,index", CASES)
def test_q0_verdicts_and_labels_agree(exp104_registry_rows, m, index):
    row = exp104_registry_rows[f"m{m:02d}_c{index:05d}"]
    exp104_model, exp105_model = build_paired_models(row)
    for p in P_TOKENS:
        result = compare_q0_trials(
            exp104_model, exp105_model, p, seed=7919 + index, trials=TRIALS,
        )
        assert result["mismatches"] == [], (
            f"m={m} c{index} p={p}: {result['mismatches']}"
        )
        assert result["compared"] == TRIALS


def test_the_comparison_would_notice_a_difference(exp104_registry_rows):
    """A negative control: perturb one model and require the gate to fire.

    Without this, a comparison that silently compared nothing would pass.
    """
    row = exp104_registry_rows["m03_c00000"]
    exp104_model, exp105_model = build_paired_models(row)
    tampered = np.array(exp105_model.H_Z, copy=True)
    tampered[0, 0] ^= 1
    object.__setattr__(exp105_model, "H_Z", tampered)
    with pytest.raises(ValueError, match="models disagree on H_Z"):
        compare_q0_trials(exp104_model, exp105_model, "0.06", seed=1, trials=2)


def test_row_rekeying_preserves_every_other_field(exp104_registry_rows):
    row = exp104_registry_rows["m03_c00000"]
    converted = exp104_row_to_exp105(row)
    assert converted["code_id"] == "m03_c000000"
    assert set(converted) == set(row)
    for key in row:
        if key != "code_id":
            assert converted[key] == row[key]
