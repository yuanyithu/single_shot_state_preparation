import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.diagnostics import evaluate_gate
from data.expander_code.exp102.exp102_pipeline.labels import (
    bits_to_uint64, initial_labels, pairwise_collision, uint64_to_bits,
)
from data.expander_code.exp102.exp102_pipeline.q0_pt import (
    Q0PtConfig, _run_q0_pt_numba_core, coupling_ladder,
    run_q0_pt_instance, swap_log_acceptance,
)
from data.expander_code.exp102.exp102_pipeline.registry import candidate_seed
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def test_k64_high_bit_and_all_ones():
    bits = np.zeros(64, dtype=np.uint8); bits[63] = 1
    assert int(bits_to_uint64(bits)) == 1 << 63
    assert np.array_equal(uint64_to_bits(bits_to_uint64(bits), 64), bits)
    assert int(initial_labels(64)[1]) == (1 << 64) - 1
    with pytest.raises(ValueError, match=">64"):
        initial_labels(65)


def test_collision_uses_six_independent_pairs_without_clipping():
    traces = [np.array([0, 0, 1, 1], dtype=np.uint64) for _ in range(4)]
    collision, qtop = pairwise_collision(traces, 1)
    assert collision == pytest.approx(0.5)
    assert qtop == pytest.approx(0.0)


def test_ladder_gate_does_not_require_character_trace_convergence():
    results = []
    for instance, label in enumerate((0, 1, 0, 1)):
        results.append({
            "labels": np.full(20, label, dtype=np.uint64), "seed": instance,
            "max_hard_coset_residual": 0,
            "swap_attempts": np.full(2, 100), "swap_accepts": np.full(2, 50),
            "logical_attempts": np.full((3, 1), 100),
            "logical_accepts": np.full((3, 1), 50),
            "round_trips": 0, "sector_changing_round_trips": 0,
        })
    gate = {
        "min_swap_rate": 0.15, "min_swap_accepts": 20,
        "min_round_trips": 0, "min_sector_changing_round_trips": 0,
        "min_hot_logical_rate": 0.01, "min_hot_logical_accepts_per_basis": 20,
        "max_rhat": np.inf, "min_ess": 0, "max_instance_mean_spread": np.inf,
    }
    strict, strict_failures, *_ = evaluate_gate(results, gate, 1)
    ladder, ladder_failures, *_ = evaluate_gate(results, gate, 1, require_trace_gate=False)
    assert not strict and strict_failures == ["basis_0:constant_untrusted"]
    assert ladder and ladder_failures == []


def test_ladder_and_swap_formula():
    K, p = coupling_ladder(0.1, 0.45, 8, 1.5)
    assert p[0] == pytest.approx(0.1) and p[-1] == pytest.approx(0.45)
    assert swap_log_acceptance(K[0], K[1], 7, 3) == pytest.approx((K[0] - K[1]) * 4)


def test_sha_seed_namespaces_are_deterministic_and_distinct():
    master = "00" * 32
    assert candidate_seed(master, 3, 2) == candidate_seed(master, 3, 2)
    assert candidate_seed(master, 3, 2) != candidate_seed(master, 3, 3)
    assert derive_seed("pilot", "x") != derive_seed("production", "x")


def test_q0_pt_nonzero_syndrome_never_leaves_hard_coset():
    from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
    load_exp101()
    from exp101_certified_src.graphs import cycle_parity_check_matrix
    model, frame = build_model(cycle_parity_check_matrix(2))
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8); epsilon[0] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    assert syndrome.any()
    result = run_q0_pt_instance(model, frame, syndrome, 0.1,
        Q0PtConfig(p_hot=0.45, num_temperatures=4, gamma=1.0,
                   burn_rounds=4, measurement_rounds=12), 123, np.uint64(0))
    assert result["max_hard_coset_residual"] == 0
    assert result["labels"].dtype == np.uint64


def test_reference_and_numba_engines_are_bit_identical():
    from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
    load_exp101()
    from exp101_certified_src.graphs import cycle_parity_check_matrix
    model, frame = build_model(cycle_parity_check_matrix(2))
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8); epsilon[1] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    config = Q0PtConfig(0.45, 4, 1.0, 5, 20)
    reference = run_q0_pt_instance(model, frame, syndrome, 0.1, config, 987, 0,
                                   engine="reference")
    accelerated = run_q0_pt_instance(model, frame, syndrome, 0.1, config, 987, 0,
                                     engine="numba")
    for field in ("labels", "swap_attempts", "swap_accepts", "logical_attempts",
                  "logical_accepts", "hot_arrival_labels", "hot_departure_labels"):
        assert np.array_equal(reference[field], accelerated[field]), field
    for field in ("round_trips", "sector_changing_round_trips", "max_hard_coset_residual"):
        assert reference[field] == accelerated[field], field
    assert accelerated["numba_enabled"]
    assert _run_q0_pt_numba_core.nopython_signatures


@pytest.mark.parametrize(
    "burn_rounds,sweeps_per_round,logical_move_repeat,seed",
    [(0, 1, 1, 0), (1, 2, 1, 1), (5, 2, 2, 2**63 - 1)],
)
def test_numba_full_round_kernel_matches_oracle_across_rng_protocol(
    burn_rounds, sweeps_per_round, logical_move_repeat, seed,
):
    from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
    load_exp101()
    from exp101_certified_src.graphs import cycle_parity_check_matrix

    model, frame = build_model(cycle_parity_check_matrix(3))
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, 2]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon % 2).astype(np.uint8)
    config = Q0PtConfig(
        0.475, 5, 1.5, burn_rounds, 17,
        sweeps_per_round, logical_move_repeat,
    )
    results = [
        run_q0_pt_instance(
            model, frame, syndrome, 0.1, config, seed, np.uint64(1), engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    array_fields = (
        "labels", "ladder_K", "ladder_p", "swap_attempts", "swap_accepts",
        "logical_attempts", "logical_accepts", "hot_arrival_labels",
        "hot_departure_labels",
    )
    scalar_fields = (
        "round_trips", "sector_changing_round_trips", "max_hard_coset_residual",
    )
    for field in array_fields:
        assert np.array_equal(results[0][field], results[1][field]), field
    for field in scalar_fields:
        assert results[0][field] == results[1][field], field


def test_common_uniforms_are_nested_across_p():
    uniforms = np.random.Generator(np.random.PCG64(derive_seed("production", "code", 1))).random(1000)
    errors = [uniforms < p for p in (0.04, 0.07, 0.10)]
    assert np.all(errors[0] <= errors[1]) and np.all(errors[1] <= errors[2])
