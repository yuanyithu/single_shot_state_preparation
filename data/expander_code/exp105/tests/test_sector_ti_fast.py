"""The numba sector chain must be the certified reference, bit for bit.

The fast path exists so Track B can afford `m = 3`, where the pure-Python
reference costs about twenty hours per disorder. It is only allowed to be a
speedup, never a different chain, so this file compares the two directly rather
than comparing summary statistics: identical `mu`, `syndrome_mu`, `block_mu` and
`acceptance`, and identical RNG state on return so that a caller alternating the
two still sees one continuous stream.

Bit-exactness is possible here because `_run_label_integrations` draws from
`PortablePrng`, the portable xorshift128+ twin, not from numpy's `default_rng`.
"""

import numpy as np
import pytest

from data.expander_code.exp105.exp105_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp105.anchor.sector_ti_fast import (
    NUMBA_AVAILABLE,
    block_bounds,
    build_kernel_data,
    fast_chain_installed,
    run_fixed_sector_chain_fast,
)

load_exp101()

from exp101_certified_src import model as exp101_model  # noqa: E402
from exp101_certified_src import observables as exp101_observables  # noqa: E402
from exp101_certified_src import run_scan as exp101_run_scan  # noqa: E402
from exp101_certified_src import sector_ti as ti  # noqa: E402
from exp101_certified_src.prng import PortablePrng  # noqa: E402


pytestmark = pytest.mark.skipif(
    not NUMBA_AVAILABLE, reason="numba is unavailable; the reference path runs",
)


@pytest.fixture(scope="module")
def wired():
    """A real m=2 code at the production operating point."""
    H_Z, H_X, logicals, _ = exp101_run_scan.build_code(
        "expander34", 2, "full_rank", None,
    )
    model = exp101_model.assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = exp101_observables.build_observable_frame(model)
    rng = np.random.default_rng(5)
    disorder = exp101_model.disorder_from_uniforms(
        model, 0.05, 0.05,
        data_uniforms=rng.random(model.num_qubits),
        syndrome_uniforms=rng.random(model.num_checks),
    )
    wiring = exp101_model.wire_ensemble(model, disorder, "true_posterior", frame)
    return model, frame, wiring


@pytest.fixture(scope="module")
def short_config():
    return ti.SectorTiConfig(
        num_kp_grid_points=5, num_burn_in_sweeps=20, num_measurements=40,
        num_bootstrap=10,
    )


@pytest.mark.parametrize("label", [0, 1, 3])
def test_fast_chain_is_bit_exact_with_the_reference(wired, short_config, label):
    model, frame, wiring = wired
    proposals = ti.build_sector_preserving_proposals(model, frame)
    kp_grid = np.linspace(0.0, wiring.K_p, short_config.num_kp_grid_points)
    v0 = ti.sector_representative(model, wiring, label)

    reference_rng = PortablePrng(1234)
    fast_rng = PortablePrng(1234)
    reference = ti._run_fixed_sector_chain(
        model, wiring, proposals, v0, kp_grid, short_config, reference_rng,
    )
    fast = run_fixed_sector_chain_fast(
        model, wiring, proposals, v0, kp_grid, short_config, fast_rng,
    )

    assert set(fast) == set(reference)
    for key in ("mu", "syndrome_mu", "block_mu", "acceptance"):
        assert np.array_equal(reference[key], fast[key]), key
    assert (reference_rng.s0, reference_rng.s1) == (fast_rng.s0, fast_rng.s1), (
        "the two chains must consume the same number of draws, or a caller "
        "alternating them would silently diverge"
    )


def test_fast_chain_does_not_start_from_a_shared_state(wired, short_config):
    """A negative control: different seeds must give different trajectories."""
    model, frame, wiring = wired
    proposals = ti.build_sector_preserving_proposals(model, frame)
    kp_grid = np.linspace(0.0, wiring.K_p, short_config.num_kp_grid_points)
    v0 = ti.sector_representative(model, wiring, 0)
    first = run_fixed_sector_chain_fast(
        model, wiring, proposals, v0, kp_grid, short_config, PortablePrng(1),
    )
    second = run_fixed_sector_chain_fast(
        model, wiring, proposals, v0, kp_grid, short_config, PortablePrng(2),
    )
    assert not np.array_equal(first["mu"], second["mu"])


def test_kernel_data_flip_sets_are_the_parity_of_the_support(wired):
    """The hoisted check-flip lists must be exactly what the reference computes."""
    model, frame, wiring = wired
    proposals = ti.build_sector_preserving_proposals(model, frame)
    data = build_kernel_data(model, proposals, bool(wiring.q_zero))
    H = np.asarray(model.H_check, dtype=np.uint8) & 1
    for position in range(data["count"]):
        support = data["qubit_index"][
            data["qubit_offsets"][position]:data["qubit_offsets"][position + 1]
        ]
        flips = data["check_index"][
            data["check_offsets"][position]:data["check_offsets"][position + 1]
        ]
        parity = np.zeros(H.shape[0], dtype=np.uint8)
        for qubit in support:
            parity ^= H[:, int(qubit)]
        assert np.array_equal(np.flatnonzero(parity), flips)


def test_block_bounds_match_numpy_array_split():
    for total in (40, 400, 401, 7):
        for blocks in (1, 3, 8):
            bounds = block_bounds(total, blocks)
            expected = np.array_split(np.arange(total), blocks)
            assert len(bounds) == blocks + 1
            assert bounds[0] == 0 and bounds[-1] == total
            for index, chunk in enumerate(expected):
                assert bounds[index + 1] - bounds[index] == len(chunk)


def test_full_sector_ti_is_bit_exact_end_to_end(wired, short_config):
    """The substitution must not perturb anything downstream of the chain."""
    model, frame, wiring = wired
    reference = ti.run_sector_ti(model, frame, wiring, short_config, 4242)
    with fast_chain_installed() as installed:
        assert installed
        fast = ti.run_sector_ti(model, frame, wiring, short_config, 4242)

    assert fast["q_top"] == reference["q_top"]
    assert fast["flags"] == reference["flags"]
    assert fast["valid_for_aggregation"] == reference["valid_for_aggregation"]
    assert np.array_equal(fast["weights_absolute"], reference["weights_absolute"])
    assert fast["grid_tv"] == reference["grid_tv"]


def test_the_substitution_is_restored(wired, short_config):
    original = ti._run_fixed_sector_chain
    with fast_chain_installed():
        assert ti._run_fixed_sector_chain is run_fixed_sector_chain_fast
    assert ti._run_fixed_sector_chain is original


def test_the_substitution_is_restored_after_a_failure():
    original = ti._run_fixed_sector_chain
    with pytest.raises(RuntimeError):
        with fast_chain_installed():
            raise RuntimeError("boom")
    assert ti._run_fixed_sector_chain is original
