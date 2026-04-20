import argparse

import numpy as np

from build_toric_code_examples import build_2d_toric_code
from exact_enumeration import compute_exact_logical_observable_means
from linear_section import apply_linear_section, build_linear_section
from main import (
    _build_kernel_basis_from_linear_section,
    _compute_log_odds,
    _run_one_kernel_sweep_zero_syndrome,
)
from mcmc import accumulate_logical_observables, draw_disorder_sample
from preprocessing import build_logical_observable_masks


def _compute_syndrome_bits(parity_check_matrix, chain_bits):
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    chain_bits_uint8 = chain_bits.astype(np.uint8)
    return ((parity_check_matrix_uint8 @ chain_bits_uint8) % 2).astype(bool)


def _build_reference_kernel_state(
        parity_check_matrix,
        linear_section_data,
        disorder_data_error_bits):
    disorder_syndrome_bits = _compute_syndrome_bits(
        parity_check_matrix=parity_check_matrix,
        chain_bits=disorder_data_error_bits,
    )
    section_representative_bits = apply_linear_section(
        disorder_syndrome_bits,
        linear_section_data,
    )
    reference_kernel_state = (
        disorder_data_error_bits ^ section_representative_bits
    )
    return reference_kernel_state


def _run_q0_chain(
        logical_observable_masks,
        disorder_data_error_bits,
        initial_chain_bits,
        kernel_basis,
        data_error_probability,
        num_burn_in_sweeps,
        num_sweeps_between_measurements,
        num_measurements,
        seed):
    rng = np.random.default_rng(seed)
    current_chain_bits = initial_chain_bits.copy()
    current_data_term_bits = current_chain_bits ^ disorder_data_error_bits
    log_odds_data = _compute_log_odds(data_error_probability)

    for _ in range(num_burn_in_sweeps):
        _run_one_kernel_sweep_zero_syndrome(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            kernel_basis=kernel_basis,
            log_odds_data=log_odds_data,
            rng=rng,
        )

    logical_observable_sum_values = np.zeros(
        logical_observable_masks.shape[0],
        dtype=np.int64,
    )
    accepted_count = 0
    attempted_count = 0

    for _ in range(num_measurements):
        for _ in range(num_sweeps_between_measurements):
            accepted_count += _run_one_kernel_sweep_zero_syndrome(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                kernel_basis=kernel_basis,
                log_odds_data=log_odds_data,
                rng=rng,
            )
            attempted_count += kernel_basis.shape[0]

        accumulate_logical_observables(
            current_chain_bits=current_chain_bits,
            logical_observable_masks=logical_observable_masks,
            logical_observable_sum_values=logical_observable_sum_values,
        )

    m_u_values = (
        logical_observable_sum_values / num_measurements
    ).astype(np.float64, copy=False)
    q_top_value = float(np.mean(m_u_values ** 2))
    acceptance_rate = accepted_count / attempted_count
    return m_u_values, q_top_value, acceptance_rate


def _format_array(array):
    return np.array2string(
        np.asarray(array, dtype=np.float64),
        precision=6,
        floatmode="fixed",
        suppress_small=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose q=0 kernel-space mixing by comparing multiple "
            "initial sectors on one disorder sample."
        )
    )
    parser.add_argument("--lattice-size", type=int, default=3)
    parser.add_argument("--data-error-probability", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20240901)
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument("--between", type=int, default=5)
    parser.add_argument("--measurements", type=int, default=2000)
    args = parser.parse_args()

    parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
        lattice_size=args.lattice_size
    )
    linear_section_data = build_linear_section(parity_check_matrix)
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        linear_section_data=linear_section_data,
    )
    kernel_basis = _build_kernel_basis_from_linear_section(
        parity_check_matrix=parity_check_matrix,
        linear_section_data=linear_section_data,
    )

    num_checks, num_qubits = parity_check_matrix.shape
    disorder_rng = np.random.default_rng(args.seed)
    observed_syndrome_bits, disorder_data_error_bits = draw_disorder_sample(
        num_checks=num_checks,
        num_qubits=num_qubits,
        syndrome_error_probability=0.0,
        data_error_probability=args.data_error_probability,
        rng=disorder_rng,
    )

    if np.any(observed_syndrome_bits):
        raise AssertionError("q=0 diagnostic expects the observed syndrome to be 0")

    reference_kernel_state = _build_reference_kernel_state(
        parity_check_matrix=parity_check_matrix,
        linear_section_data=linear_section_data,
        disorder_data_error_bits=disorder_data_error_bits,
    )

    initial_state_map = {
        "zero": np.zeros(num_qubits, dtype=bool),
        "reference": reference_kernel_state,
    }

    num_logical_qubits = dual_logical_z_basis.shape[0]
    for logical_sector in range(1, 1 << num_logical_qubits):
        shifted_state = reference_kernel_state.copy()
        sector_label_terms = []
        for logical_qubit_index in range(num_logical_qubits):
            if ((logical_sector >> logical_qubit_index) & 1) == 0:
                continue
            shifted_state ^= dual_logical_z_basis[logical_qubit_index]
            sector_label_terms.append(f"L{logical_qubit_index}")
        initial_state_map["reference+" + "+".join(sector_label_terms)] = (
            shifted_state
        )

    exact_m_u_values = None
    exact_q_top_value = None
    if num_qubits <= 22:
        exact_m_u_values = compute_exact_logical_observable_means(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=observed_syndrome_bits,
            disorder_data_error_bits=disorder_data_error_bits,
            syndrome_error_probability=0.0,
            data_error_probability=args.data_error_probability,
            logical_observable_masks=logical_observable_masks,
        )
        exact_q_top_value = float(np.mean(exact_m_u_values ** 2))

    print(
        f"L={args.lattice_size}, n={num_qubits}, p={args.data_error_probability:.6f}"
    )
    print(
        "disorder weight |eta| =",
        int(np.count_nonzero(disorder_data_error_bits)),
    )
    if exact_m_u_values is not None:
        print("exact m_u =", _format_array(exact_m_u_values))
        print(f"exact q_top = {exact_q_top_value:.6f}")

    q_top_values = []
    print()
    print(
        f"{'initial_state':<20} {'q_top':>10} {'acceptance':>12} {'max_abs_err':>12}"
    )

    for init_index, (name, initial_chain_bits) in enumerate(initial_state_map.items()):
        m_u_values, q_top_value, acceptance_rate = _run_q0_chain(
            logical_observable_masks=logical_observable_masks,
            disorder_data_error_bits=disorder_data_error_bits,
            initial_chain_bits=initial_chain_bits,
            kernel_basis=kernel_basis,
            data_error_probability=args.data_error_probability,
            num_burn_in_sweeps=args.burn_in,
            num_sweeps_between_measurements=args.between,
            num_measurements=args.measurements,
            seed=args.seed + 1000 + init_index,
        )
        q_top_values.append(q_top_value)

        if exact_m_u_values is None:
            max_abs_error_text = "n/a"
        else:
            max_abs_error_value = float(
                np.max(np.abs(m_u_values - exact_m_u_values))
            )
            max_abs_error_text = f"{max_abs_error_value:.6f}"

        print(
            f"{name:<20} {q_top_value:10.6f} {acceptance_rate:12.6f} "
            f"{max_abs_error_text:>12}"
        )
        print("m_u =", _format_array(m_u_values))

    print()
    print(
        "q_top spread across initial states = "
        f"{(max(q_top_values) - min(q_top_values)):.6f}"
    )


if __name__ == "__main__":
    main()
