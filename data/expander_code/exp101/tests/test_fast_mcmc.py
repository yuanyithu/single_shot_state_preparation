"""G2.4 单元测试：PRNG 双胞胎逐位一致 + 引擎 bit 级一致 + 独立枚举校验 + 提速。"""

import time

import numpy as np
import pytest

from src.graphs import (
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from src.hgp import classical_parity_check_matrix, hgp_from_H
from src.logicals import logical_pauli_operators
from src.model import assemble_sector_model, draw_disorder, wire_ensemble
from src.observables import build_observable_frame, build_observable_set
from src.prng import NUMBA_AVAILABLE, PortablePrng
from src.reference_mcmc import ReferenceMcmcConfig, run_reference_mcmc
from src.fast_mcmc import build_fast_chain_data, run_fast_mcmc
from tests.util_enum import enum_m_u


def build_setup(classical, sector="x_error", **obs_kwargs):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector=sector)
    frame = build_observable_frame(model)
    obs_set = build_observable_set(frame, **obs_kwargs)
    return model, frame, obs_set


def make_wiring(model, frame, p, q, seed, ensemble="true_posterior"):
    disorder = draw_disorder(model, p, q, np.random.default_rng(seed))
    return wire_ensemble(model, disorder, ensemble, frame)


class TestPortablePrngTwins:
    def test_numba_available(self):
        assert NUMBA_AVAILABLE

    @pytest.mark.parametrize("seed", [0, 1, 12345, 2**60 + 7])
    def test_python_numba_sequences_identical(self, seed):
        from src.prng import nb_fill_permutation, nb_random, nb_randbelow

        py = PortablePrng(seed)
        state = PortablePrng(seed).state_array()
        for _ in range(200):
            assert py.random() == nb_random(state)
        for n in (2, 3, 7, 100):
            assert py.randbelow(n) == nb_randbelow(state, n)
        buffer = np.empty(17, dtype=np.int64)
        nb_fill_permutation(state, buffer)
        assert np.array_equal(py.permutation(17), buffer)

    def test_stream_continuity_across_calls(self):
        py = PortablePrng(9)
        a = [py.random() for _ in range(3)]
        py2 = PortablePrng(9)
        b = [py2.random() for _ in range(3)]
        assert a == b


CASES = [
    # (classical, p, q, ensemble)
    (repetition_parity_check_matrix(2), 0.12, 0.08, "true_posterior"),
    (repetition_parity_check_matrix(2), 0.12, 0.08, "legacy_delta_only"),
    (cycle_parity_check_matrix(2), 0.15, 0.12, "true_posterior"),
    (cycle_parity_check_matrix(2), 0.18, 0.0, "true_posterior"),   # q=0
    (cycle_parity_check_matrix(3), 0.10, 0.05, "true_posterior"),
]


class TestBitLevelEngineIdentity:
    @pytest.mark.parametrize("classical,p,q,ensemble", CASES)
    def test_reference_and_fast_identical(self, classical, p, q, ensemble):
        model, frame, obs_set = build_setup(classical)
        wiring = make_wiring(model, frame, p, q, seed=31, ensemble=ensemble)
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=40, num_measurements=300,
            num_sweeps_between_measurements=2, logical_move_repeat=2,
            record_observable_trajectory=True,
            record_state_trajectory=True,
            debug_invariants=True,
        )
        ref = run_reference_mcmc(model, frame, obs_set, wiring, config, seed=33)
        fast = run_fast_mcmc(model, frame, obs_set, wiring, config, seed=33)
        assert fast["engine"] == "numba"
        assert np.array_equal(ref["observable_sums"], fast["observable_sums"])
        assert np.array_equal(ref["final_state"].v, fast["final_state"].v)
        assert np.array_equal(
            ref["final_state"].syndrome_term, fast["final_state"].syndrome_term
        )
        assert ref["final_state"].data_weight == fast["final_state"].data_weight
        rc, fc = ref["counters"], fast["counters"]
        assert rc.single_bit_attempts == fc.single_bit_attempts
        assert rc.single_bit_accepts == fc.single_bit_accepts
        assert rc.stabilizer_accepts == fc.stabilizer_accepts
        assert np.array_equal(rc.logical_accepts_per_u, fc.logical_accepts_per_u)
        assert np.allclose(ref["energy_trace"], fast["energy_trace"],
                           rtol=0, atol=0)
        assert np.array_equal(
            ref["observable_trajectory"], fast["observable_trajectory"]
        )
        assert np.array_equal(
            ref["state_trajectory"], fast["state_trajectory"]
        )

    def test_identity_with_sampled_tier_and_sector_start(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12349)
        # k=4：非零非 basis 的 u 只有 11 个 ⇒ num_random_u 必须 ≤ 11
        model, frame, obs_set = build_setup(
            classical_parity_check_matrix(graph),
            full_max_k=2, num_random_u=8, u_rand_seed=5,
        )
        assert obs_set.tier == "sampled"
        wiring = make_wiring(model, frame, 0.08, 0.04, seed=35)
        config = ReferenceMcmcConfig(num_burn_in_sweeps=20, num_measurements=100)
        ref = run_reference_mcmc(model, frame, obs_set, wiring, config, seed=37,
                                 sector_bitmask=0b101)
        fast = run_fast_mcmc(model, frame, obs_set, wiring, config, seed=37,
                             sector_bitmask=0b101)
        assert np.array_equal(ref["observable_sums"], fast["observable_sums"])
        assert np.array_equal(ref["final_state"].v, fast["final_state"].v)


class TestFastEngineIndependentChecks:
    @pytest.mark.parametrize("q", [0.0, 0.1])
    def test_p_zero_energy_trace_is_finite_and_matches_reference(self, q):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, p=0.0, q=q, seed=38)
        assert np.isposinf(wiring.K_p)
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=5,
            num_measurements=20,
            record_state_trajectory=True,
            debug_invariants=True,
        )
        reference = run_reference_mcmc(
            model, frame, obs_set, wiring, config, seed=40
        )
        fast = run_fast_mcmc(
            model, frame, obs_set, wiring, config, seed=40
        )
        assert np.all(np.isfinite(reference["energy_trace"]))
        assert np.array_equal(reference["energy_trace"], fast["energy_trace"])
        assert np.array_equal(
            reference["state_trajectory"], fast["state_trajectory"]
        )

    def test_fast_vs_enum(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.12, 0.08, seed=39)
        exact_m, _ = enum_m_u(model, obs_set, wiring)
        config = ReferenceMcmcConfig(
            num_burn_in_sweeps=500, num_measurements=20000,
            record_observable_trajectory=True,
        )
        result = run_fast_mcmc(model, frame, obs_set, wiring, config, seed=41)
        traj = result["observable_trajectory"].astype(np.float64)
        block_means = np.array(
            [b.mean(axis=0) for b in np.array_split(traj, 20, axis=0)]
        )
        stderr = block_means.std(axis=0, ddof=1) / np.sqrt(20)
        z = (result["m_u"] - exact_m) / np.maximum(stderr, 1e-3)
        assert np.max(np.abs(z)) < 5.0

    def test_force_reference_fallback(self):
        model, frame, obs_set = build_setup(repetition_parity_check_matrix(2))
        wiring = make_wiring(model, frame, 0.1, 0.05, seed=43)
        config = ReferenceMcmcConfig(num_burn_in_sweeps=10, num_measurements=50)
        fallback = run_fast_mcmc(model, frame, obs_set, wiring, config, seed=45,
                                 force_reference=True)
        assert fallback["engine"] == "reference_fallback"
        native = run_fast_mcmc(model, frame, obs_set, wiring, config, seed=45)
        assert np.array_equal(
            fallback["observable_sums"], native["observable_sums"]
        )

    def test_speedup_on_m2(self):
        graph = random_biregular_graph_from_m(2, 3, 4, seed=12349)
        model, frame, obs_set = build_setup(classical_parity_check_matrix(graph))
        wiring = make_wiring(model, frame, 0.05, 0.03, seed=47)
        config = ReferenceMcmcConfig(num_burn_in_sweeps=0, num_measurements=200)
        chain_data = build_fast_chain_data(model, obs_set)
        run_fast_mcmc(model, frame, obs_set, wiring,
                      ReferenceMcmcConfig(num_burn_in_sweeps=0,
                                          num_measurements=1),
                      seed=1, chain_data=chain_data)  # JIT 预热
        t0 = time.perf_counter()
        run_fast_mcmc(model, frame, obs_set, wiring, config, seed=49,
                      chain_data=chain_data)
        fast_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        run_reference_mcmc(model, frame, obs_set, wiring, config, seed=49)
        ref_time = time.perf_counter() - t0
        assert fast_time < ref_time / 5.0, (fast_time, ref_time)
