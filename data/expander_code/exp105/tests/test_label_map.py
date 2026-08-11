"""The scoring map phi_r, from three directions.

At q > 0 the residual data error no longer has zero syndrome, so exp104's
residual pairing against logical_Z is not the logical class any more. exp105
scores through the exp101 absolute label phi_r instead. These tests pin that map
down three ways: against exp101's own frame, against an independent
reconstruction from the section's pivot rule, and against exp104's criterion in
the q = 0 limit where the two must coincide exactly.
"""

import numpy as np
import pytest

from data.expander_code.exp105.exp105_pipeline.audit_scorer import (
    apply_label_map,
    independent_label_map,
    in_rowspace,
    row_echelon_basis,
    trivial_class_generators,
)
from data.expander_code.exp105.exp105_pipeline.config import code_id
from data.expander_code.exp105.exp105_pipeline.model import (
    clear_model_cache,
    load_model,
    logical_label,
    parity_product,
)


@pytest.fixture(scope="module", params=[3, 8])
def model(request):
    rows = request.getfixturevalue("registry_rows")
    clear_model_cache()
    model = load_model(rows[code_id(request.param, 0)])
    yield model
    clear_model_cache()


def test_label_basis_matches_exp101_frame(model):
    """model.label_basis is exp101's W, not a re-derivation of it."""
    from data.expander_code.exp105.exp105_pipeline.exp101_bridge import load_exp101

    load_exp101()
    from exp101_certified_src.model import assemble_sector_model
    from exp101_certified_src.observables import build_observable_frame

    sector = assemble_sector_model(model.H_X, model.H_Z, _frame_of(model), "x_error")
    frame = build_observable_frame(sector)
    assert np.array_equal(frame.W_basis, model.label_basis)
    assert frame.fingerprint() == model.observable_frame_fingerprint

    rng = np.random.Generator(np.random.PCG64(11))
    for _ in range(32):
        vector = rng.integers(0, 2, size=model.n, dtype=np.uint8)
        assert np.array_equal(frame.label_of(vector), logical_label(model, vector))


def _frame_of(model):
    from data.expander_code.exp105.exp105_pipeline.exp101_bridge import load_exp101

    load_exp101()
    from exp101_certified_src.logicals import logical_pauli_operators

    return logical_pauli_operators(model.H_X, model.H_Z)


def test_independent_label_map_agrees(model):
    """A second implementation of the section's pivot rule lands on the same map."""
    audit = independent_label_map(model.H_Z, model.logical_Z)
    assert np.array_equal(audit, model.label_basis)

    rng = np.random.Generator(np.random.PCG64(23))
    for _ in range(32):
        vector = rng.integers(0, 2, size=model.n, dtype=np.uint8)
        assert np.array_equal(
            apply_label_map(audit, vector), logical_label(model, vector),
        )


def test_trivial_class_generators_match_the_label_kernel(model):
    """Triviality of phi_r is membership of rowspace(H_X) + im(r_sec)."""
    generators = trivial_class_generators(model.H_X, model.H_Z)
    rng = np.random.Generator(np.random.PCG64(37))
    seen_trivial = 0
    seen_nontrivial = 0
    for _ in range(64):
        vector = rng.integers(0, 2, size=model.n, dtype=np.uint8)
        trivial = not logical_label(model, vector).any()
        assert trivial == in_rowspace(vector, *generators)
        seen_trivial += int(trivial)
        seen_nontrivial += int(not trivial)
    # Random vectors are almost never trivial, so add the cases that are.
    for row in model.H_X[:8]:
        assert not logical_label(model, row).any()
        assert in_rowspace(row, *generators)
    assert seen_nontrivial > 0


def test_stabilizers_and_logicals_land_where_they_should(model):
    """Stabilizers are trivial; the logical move basis pairs as the identity."""
    frame = _frame_of(model)
    for row in model.H_X:
        assert not logical_label(model, row).any()
    pairing = np.stack([logical_label(model, row) for row in frame.logical_X])
    assert np.array_equal(pairing, np.eye(model.k, dtype=np.uint8))


def test_phi_r_reduces_to_exp104_scoring_at_zero_syndrome(model):
    """With a zero-syndrome residual, phi_r is exp104's residual pairing.

    This is the identity that makes the exp104 equality gate meaningful, so it
    is asserted directly rather than inferred from the gate passing.
    """
    rng = np.random.Generator(np.random.PCG64(53))
    checked = 0
    for _ in range(64):
        combination = rng.integers(0, 2, size=model.H_X.shape[0], dtype=np.uint8)
        stabilizer = (combination @ model.H_X) % 2
        logical_row = model.k and rng.integers(0, model.k)
        residual = np.bitwise_xor(
            stabilizer.astype(np.uint8),
            _frame_of(model).logical_X[int(logical_row)].astype(np.uint8),
        )
        assert not parity_product(model.H_Z, residual).any()
        assert np.array_equal(
            logical_label(model, residual),
            parity_product(model.logical_Z, residual),
        )
        checked += 1
    assert checked == 64


def test_label_basis_equals_logical_Z_for_this_family(model):
    """A measured structural property, with its reason asserted alongside it.

    exp101 builds W = Z (I xor r_sec H). Here r_sec places values only on the
    RREF pivot columns of H_Z, and exp101's logical_Z basis is supported
    entirely off those columns, so Z r_sec = 0 and W collapses to Z.

    This is recorded, not relied on: the pipeline always computes the label
    through the certified frame. It is asserted here because it is the reason
    the exp104 equality gate compares labels at all, and because if a future
    logical basis stopped having this property the two would silently diverge.
    """
    from data.expander_code.exp105.exp105_pipeline.exp101_bridge import load_exp101

    load_exp101()
    from exp101_certified_src.gf2 import gf2_row_echelon

    _, pivots = gf2_row_echelon(model.H_Z)
    assert int(model.logical_Z[:, list(pivots)].sum()) == 0, (
        "logical_Z is supported on a pivot column of H_Z; the section term no "
        "longer vanishes and the label map is not the residual pairing"
    )
    assert np.array_equal(model.label_basis, model.logical_Z)


def test_the_failure_criterion_differs_from_exp104_at_nonzero_syndrome(model):
    """Where exp105 genuinely parts company with exp104.

    A residual that is a stabilizer plus one extra flipped qubit has nonzero
    syndrome and trivial-or-not class independent of that syndrome. exp104
    calls every such residual a failure. exp105 scores it by its class alone,
    because the protocol's final perfect round removes the residual syndrome.
    This asserts the two criteria are not interchangeable, so nobody can
    "simplify" exp105 back to exp104's scoring.
    """
    from data.expander_code.exp105.exp105_pipeline.worker import score_logical_class

    rng = np.random.Generator(np.random.PCG64(71))
    disagreements = 0
    for _ in range(64):
        combination = rng.integers(0, 2, size=model.H_X.shape[0], dtype=np.uint8)
        residual = ((combination @ model.H_X) % 2).astype(np.uint8)
        position = int(rng.integers(0, model.n))
        residual[position] ^= 1
        if not parity_product(model.H_Z, residual).any():
            continue
        exp104_failed = True  # nonzero syndrome fails exp104's criterion outright
        exp105_failed = bool(logical_label(model, residual).any())
        disagreements += int(exp104_failed != exp105_failed)
    assert disagreements > 0, (
        "expected residuals with nonzero syndrome but trivial logical class, "
        "which exp104 counts as failures and exp105 does not"
    )


def test_score_logical_class_ignores_the_readout_residual(model):
    """readout_match is reported but must not enter the failure verdict."""
    from data.expander_code.exp105.exp105_pipeline.worker import score_logical_class

    rng = np.random.Generator(np.random.PCG64(97))
    error = rng.integers(0, 2, size=model.n, dtype=np.uint8)
    readout = rng.integers(0, 2, size=model.n_checks, dtype=np.uint8)
    correction = np.concatenate([
        np.array(error, copy=True), np.array(readout, copy=True),
    ])
    failed, matched, labels = score_logical_class(model, error, readout, correction)
    assert failed is False and matched is True and not labels.any()

    wrong_readout = np.array(correction, copy=True)
    wrong_readout[model.n] ^= 1
    failed, matched, labels = score_logical_class(
        model, error, readout, wrong_readout,
    )
    assert matched is False
    assert failed is False, "a wrong readout estimate alone must not fail the trial"
