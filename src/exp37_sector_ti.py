"""Sector-resolved thermodynamic integration for exp37.

This runner samples x = c xor eta directly.  Logical sectors are labelled by
the corrected observable chain x xor r(Hx xor Heta) xor r(Heta), matching the
main MCMC observable c xor eta xor r(Hc) xor r(Heta).  For each sector it can
estimate <|x|> on a Kp grid, integrate from Kp=0 to the target Kp, and
reconstruct the eight sector weights.
"""

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional remote acceleration
    njit = None

from build_toric_code_examples import (
    build_toric_code_by_family,
    build_zero_syndrome_move_data_by_family,
)
from exact_enumeration import _iter_chain_bit_chunks, _logsumexp
from linear_section import (
    apply_linear_section,
    apply_section,
    build_linear_section,
    build_syndrome_representative_section,
)
from preprocessing import build_checks_touching_each_qubit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP37_ROOT = (
    PROJECT_ROOT / "data" / "3d_toric_code" / "with_measurement_noise"
    / "exp37"
)


if njit is not None:
    @njit(cache=True)
    def _numba_popcount(value):
        count = 0
        while value:
            value &= value - 1
            count += 1
        return count


    @njit(cache=True)
    def _numba_attempt_proposal(
            current_x_bits,
            current_syndrome_term_bits,
            current_data_weight,
            current_syndrome_weight,
            proposal_index,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            kp_value,
            kq_value):
        support_length = data_support_lengths[proposal_index]
        ones = 0
        for support_position in range(support_length):
            qubit_index = data_supports[proposal_index, support_position]
            if current_x_bits[qubit_index]:
                ones += 1
        delta_data_weight = support_length - 2 * ones

        syndrome_support_length = syndrome_support_lengths[proposal_index]
        syndrome_ones = 0
        for support_position in range(syndrome_support_length):
            check_index = syndrome_supports[proposal_index, support_position]
            if current_syndrome_term_bits[check_index]:
                syndrome_ones += 1
        delta_syndrome_weight = syndrome_support_length - 2 * syndrome_ones

        log_acceptance = (
            -kp_value * delta_data_weight
            -kq_value * delta_syndrome_weight
        )
        if log_acceptance >= 0.0 or np.random.random() < math.exp(log_acceptance):
            for support_position in range(support_length):
                qubit_index = data_supports[proposal_index, support_position]
                current_x_bits[qubit_index] = not current_x_bits[qubit_index]
            for support_position in range(syndrome_support_length):
                check_index = syndrome_supports[proposal_index, support_position]
                current_syndrome_term_bits[check_index] = (
                    not current_syndrome_term_bits[check_index]
                )
            return (
                1,
                current_data_weight + delta_data_weight,
                current_syndrome_weight + delta_syndrome_weight,
            )
        return 0, current_data_weight, current_syndrome_weight


    @njit(cache=True)
    def _numba_run_even_winding_heatbath(
            current_x_bits,
            current_data_weight,
            winding_groups,
            kp_value):
        num_groups = winding_groups.shape[0]
        num_moves = winding_groups.shape[1]
        support_size = winding_groups.shape[2]
        if num_groups == 0 or num_moves == 0 or support_size == 0:
            return 0, 0, current_data_weight

        changed_count = 0
        attempted_count = 0
        num_configurations = 1 << num_moves
        delta_by_move = np.empty(num_moves, dtype=np.int64)
        log_weights = np.empty(num_configurations, dtype=np.float64)

        for group_index in range(num_groups):
            attempted_count += num_moves
            for move_index in range(num_moves):
                ones = 0
                for support_position in range(support_size):
                    qubit_index = winding_groups[
                        group_index,
                        move_index,
                        support_position,
                    ]
                    if current_x_bits[qubit_index]:
                        ones += 1
                delta_by_move[move_index] = support_size - 2 * ones

            max_log_weight = -1.0e300
            for configuration in range(num_configurations):
                if _numba_popcount(configuration) % 2 != 0:
                    log_weights[configuration] = -1.0e300
                    continue
                log_weight = 0.0
                for move_index in range(num_moves):
                    if (configuration >> move_index) & 1:
                        log_weight += -kp_value * delta_by_move[move_index]
                log_weights[configuration] = log_weight
                if log_weight > max_log_weight:
                    max_log_weight = log_weight

            total_weight = 0.0
            for configuration in range(num_configurations):
                if log_weights[configuration] <= -0.5e300:
                    continue
                total_weight += math.exp(log_weights[configuration] - max_log_weight)
            if total_weight <= 0.0:
                continue
            threshold = np.random.random() * total_weight
            cumulative = 0.0
            selected_configuration = 0
            for configuration in range(num_configurations):
                if log_weights[configuration] <= -0.5e300:
                    continue
                cumulative += math.exp(log_weights[configuration] - max_log_weight)
                if threshold <= cumulative:
                    selected_configuration = configuration
                    break

            if selected_configuration == 0:
                continue
            for move_index in range(num_moves):
                if (selected_configuration >> move_index) & 1:
                    for support_position in range(support_size):
                        qubit_index = winding_groups[
                            group_index,
                            move_index,
                            support_position,
                        ]
                        current_x_bits[qubit_index] = not current_x_bits[qubit_index]
                    current_data_weight += delta_by_move[move_index]
                    changed_count += 1
        return changed_count, attempted_count, current_data_weight


    @njit(cache=True)
    def _numba_run_sector_sweep(
            current_x_bits,
            current_syndrome_term_bits,
            current_data_weight,
            current_syndrome_weight,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            winding_groups,
            kp_value,
            kq_value,
            winding_heatbath_sweeps):
        accepted_count = 0
        attempted_count = data_support_lengths.shape[0]
        for _ in range(attempted_count):
            proposal_index = np.random.randint(0, attempted_count)
            accepted, current_data_weight, current_syndrome_weight = (
                _numba_attempt_proposal(
                    current_x_bits,
                    current_syndrome_term_bits,
                    current_data_weight,
                    current_syndrome_weight,
                    proposal_index,
                    data_supports,
                    data_support_lengths,
                    syndrome_supports,
                    syndrome_support_lengths,
                    kp_value,
                    kq_value,
                )
            )
            accepted_count += accepted

        heatbath_changed_count = 0
        heatbath_attempted_count = 0
        for _ in range(winding_heatbath_sweeps):
            changed, attempted, current_data_weight = (
                _numba_run_even_winding_heatbath(
                    current_x_bits,
                    current_data_weight,
                    winding_groups,
                    kp_value,
                )
            )
            heatbath_changed_count += changed
            heatbath_attempted_count += attempted
        return (
            accepted_count,
            attempted_count,
            heatbath_changed_count,
            heatbath_attempted_count,
            current_data_weight,
            current_syndrome_weight,
        )


    @njit(cache=True)
    def _numba_run_logical_sector_heatbath(
            current_x_bits,
            current_data_weight,
            logical_supports,
            logical_support_lengths,
            kp_value):
        num_representatives = logical_support_lengths.shape[0]
        if num_representatives <= 1:
            return 0, current_data_weight

        delta_by_representative = np.empty(num_representatives, dtype=np.int64)
        log_weights = np.empty(num_representatives, dtype=np.float64)
        max_log_weight = -1.0e300
        for representative_index in range(num_representatives):
            support_length = logical_support_lengths[representative_index]
            ones = 0
            for support_position in range(support_length):
                qubit_index = logical_supports[
                    representative_index,
                    support_position,
                ]
                if current_x_bits[qubit_index]:
                    ones += 1
            delta_data_weight = support_length - 2 * ones
            delta_by_representative[representative_index] = delta_data_weight
            log_weight = -kp_value * delta_data_weight
            log_weights[representative_index] = log_weight
            if log_weight > max_log_weight:
                max_log_weight = log_weight

        total_weight = 0.0
        for representative_index in range(num_representatives):
            total_weight += math.exp(
                log_weights[representative_index] - max_log_weight
            )
        if total_weight <= 0.0:
            return 0, current_data_weight

        threshold = np.random.random() * total_weight
        cumulative = 0.0
        selected_representative = 0
        for representative_index in range(num_representatives):
            cumulative += math.exp(
                log_weights[representative_index] - max_log_weight
            )
            if threshold <= cumulative:
                selected_representative = representative_index
                break

        if selected_representative == 0:
            return 0, current_data_weight
        support_length = logical_support_lengths[selected_representative]
        for support_position in range(support_length):
            qubit_index = logical_supports[
                selected_representative,
                support_position,
            ]
            current_x_bits[qubit_index] = not current_x_bits[qubit_index]
        return 1, current_data_weight + delta_by_representative[
            selected_representative
        ]


    @njit(cache=True)
    def _numba_run_ais_sweep(
            current_x_bits,
            current_syndrome_term_bits,
            current_data_weight,
            current_syndrome_weight,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            logical_supports,
            logical_support_lengths,
            empty_winding_groups,
            kp_value,
            kq_value,
            logical_heatbath_sweeps):
        (
            accepted_count,
            attempted_count,
            heatbath_changed_count,
            heatbath_attempted_count,
            current_data_weight,
            current_syndrome_weight,
        ) = _numba_run_sector_sweep(
            current_x_bits,
            current_syndrome_term_bits,
            current_data_weight,
            current_syndrome_weight,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            empty_winding_groups,
            kp_value,
            kq_value,
            0,
        )
        for _ in range(logical_heatbath_sweeps):
            changed, current_data_weight = _numba_run_logical_sector_heatbath(
                current_x_bits,
                current_data_weight,
                logical_supports,
                logical_support_lengths,
                kp_value,
            )
            heatbath_changed_count += changed
            heatbath_attempted_count += 1
        return (
            accepted_count,
            attempted_count,
            heatbath_changed_count,
            heatbath_attempted_count,
            current_data_weight,
            current_syndrome_weight,
        )


    @njit(cache=True)
    def _numba_run_fixed_sector_chain(
            initial_x_bits,
            initial_syndrome_term_bits,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            winding_groups,
            kp_grid,
            kq_value,
            num_burn_in_sweeps,
            num_measurements,
            num_sweeps_between_measurements,
            block_count,
            winding_heatbath_sweeps,
            seed):
        np.random.seed(seed)
        current_x_bits = initial_x_bits.copy()
        current_syndrome_term_bits = initial_syndrome_term_bits.copy()
        current_data_weight = 0
        for index in range(current_x_bits.shape[0]):
            if current_x_bits[index]:
                current_data_weight += 1
        current_syndrome_weight = 0
        for index in range(current_syndrome_term_bits.shape[0]):
            if current_syndrome_term_bits[index]:
                current_syndrome_weight += 1

        num_grid = kp_grid.shape[0]
        mu = np.empty(num_grid, dtype=np.float64)
        syndrome_mu = np.empty(num_grid, dtype=np.float64)
        energy_mu = np.empty(num_grid, dtype=np.float64)
        block_mu = np.zeros((num_grid, block_count), dtype=np.float64)
        block_counts = np.zeros((num_grid, block_count), dtype=np.int64)
        acceptance_rate = np.empty(num_grid, dtype=np.float64)
        heatbath_change_rate = np.empty(num_grid, dtype=np.float64)

        for grid_index in range(num_grid):
            kp_value = kp_grid[grid_index]
            accepted_total = 0
            attempted_total = 0
            heatbath_changed_total = 0
            heatbath_attempted_total = 0
            for _ in range(num_burn_in_sweeps):
                (
                    accepted,
                    attempted,
                    heatbath_changed,
                    heatbath_attempted,
                    current_data_weight,
                    current_syndrome_weight,
                ) = _numba_run_sector_sweep(
                    current_x_bits,
                    current_syndrome_term_bits,
                    current_data_weight,
                    current_syndrome_weight,
                    data_supports,
                    data_support_lengths,
                    syndrome_supports,
                    syndrome_support_lengths,
                    winding_groups,
                    kp_value,
                    kq_value,
                    winding_heatbath_sweeps,
                )
                accepted_total += accepted
                attempted_total += attempted
                heatbath_changed_total += heatbath_changed
                heatbath_attempted_total += heatbath_attempted

            data_sum = 0.0
            syndrome_sum = 0.0
            for measurement_index in range(num_measurements):
                for _ in range(num_sweeps_between_measurements):
                    (
                        accepted,
                        attempted,
                        heatbath_changed,
                        heatbath_attempted,
                        current_data_weight,
                        current_syndrome_weight,
                    ) = _numba_run_sector_sweep(
                        current_x_bits,
                        current_syndrome_term_bits,
                        current_data_weight,
                        current_syndrome_weight,
                        data_supports,
                        data_support_lengths,
                        syndrome_supports,
                        syndrome_support_lengths,
                        winding_groups,
                        kp_value,
                        kq_value,
                        winding_heatbath_sweeps,
                    )
                    accepted_total += accepted
                    attempted_total += attempted
                    heatbath_changed_total += heatbath_changed
                    heatbath_attempted_total += heatbath_attempted
                data_sum += current_data_weight
                syndrome_sum += current_syndrome_weight
                block_index = (measurement_index * block_count) // num_measurements
                block_mu[grid_index, block_index] += current_data_weight
                block_counts[grid_index, block_index] += 1

            mu[grid_index] = data_sum / num_measurements
            syndrome_mu[grid_index] = syndrome_sum / num_measurements
            energy_mu[grid_index] = (
                kp_value * mu[grid_index]
                + kq_value * syndrome_mu[grid_index]
            )
            for block_index in range(block_count):
                if block_counts[grid_index, block_index] > 0:
                    block_mu[grid_index, block_index] = (
                        block_mu[grid_index, block_index]
                        / block_counts[grid_index, block_index]
                    )
            if attempted_total == 0:
                acceptance_rate[grid_index] = 0.0
            else:
                acceptance_rate[grid_index] = accepted_total / attempted_total
            if heatbath_attempted_total == 0:
                heatbath_change_rate[grid_index] = 0.0
            else:
                heatbath_change_rate[grid_index] = (
                    heatbath_changed_total / heatbath_attempted_total
                )
        return mu, syndrome_mu, energy_mu, block_mu, acceptance_rate, heatbath_change_rate


    @njit(cache=True)
    def _numba_run_ais_particle(
            initial_x_bits,
            initial_syndrome_term_bits,
            data_supports,
            data_support_lengths,
            syndrome_supports,
            syndrome_support_lengths,
            logical_supports,
            logical_support_lengths,
            kp_grid,
            kq_value,
            num_initial_burn_in_sweeps,
            num_transition_sweeps,
            logical_heatbath_sweeps,
            seed):
        np.random.seed(seed)
        current_x_bits = initial_x_bits.copy()
        current_syndrome_term_bits = initial_syndrome_term_bits.copy()
        current_data_weight = 0
        for index in range(current_x_bits.shape[0]):
            if current_x_bits[index]:
                current_data_weight += 1
        current_syndrome_weight = 0
        for index in range(current_syndrome_term_bits.shape[0]):
            if current_syndrome_term_bits[index]:
                current_syndrome_weight += 1

        empty_winding_groups = np.empty((0, 0, 0), dtype=np.int32)
        for _ in range(num_initial_burn_in_sweeps):
            (
                _,
                _,
                _,
                _,
                current_data_weight,
                current_syndrome_weight,
            ) = _numba_run_ais_sweep(
                current_x_bits,
                current_syndrome_term_bits,
                current_data_weight,
                current_syndrome_weight,
                data_supports,
                data_support_lengths,
                syndrome_supports,
                syndrome_support_lengths,
                logical_supports,
                logical_support_lengths,
                empty_winding_groups,
                kp_grid[0],
                kq_value,
                logical_heatbath_sweeps,
            )

        log_weight = 0.0
        for grid_index in range(1, kp_grid.shape[0]):
            delta_kp = kp_grid[grid_index] - kp_grid[grid_index - 1]
            log_weight += -delta_kp * current_data_weight
            for _ in range(num_transition_sweeps):
                (
                    _,
                    _,
                    _,
                    _,
                    current_data_weight,
                    current_syndrome_weight,
                ) = _numba_run_ais_sweep(
                    current_x_bits,
                    current_syndrome_term_bits,
                    current_data_weight,
                    current_syndrome_weight,
                    data_supports,
                    data_support_lengths,
                    syndrome_supports,
                    syndrome_support_lengths,
                    logical_supports,
                    logical_support_lengths,
                    empty_winding_groups,
                    kp_grid[grid_index],
                    kq_value,
                    logical_heatbath_sweeps,
                )
        return current_x_bits, current_syndrome_term_bits, log_weight
else:
    _numba_run_fixed_sector_chain = None
    _numba_run_ais_particle = None


def _parse_float_list(text):
    text = str(text)
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError("range syntax must be start:stop:step")
        start, stop, step = parts
        if step <= 0.0:
            raise ValueError("range step must be positive")
        count = int(round((stop - start) / step)) + 1
        values = start + step * np.arange(count, dtype=np.float64)
        values = values[values <= stop + 0.5 * step]
        return [float(round(value, 12)) for value in values]
    return [float(part) for part in text.split(",") if part]


def _parse_int_list(text):
    return [int(part) for part in str(text).split(",") if part]


def _compute_k(probability):
    probability = float(probability)
    if not (0.0 < probability < 0.5):
        raise ValueError("probability must be in (0, 0.5)")
    return float(math.log((1.0 - probability) / probability))


def _compute_signature(chain_bits, logical_projection_masks):
    parity_bits = (
        logical_projection_masks.astype(np.uint8)
        @ np.asarray(chain_bits, dtype=np.uint8)
    ) % 2
    signature = 0
    for bit_index, bit in enumerate(parity_bits):
        signature |= int(bit) << bit_index
    return int(signature)


def _compute_decoder_signature(chain_bits,
                               chain_syndrome_bits,
                               primitive_logical_masks,
                               section_data):
    representative_bits = apply_section(
        np.asarray(chain_syndrome_bits, dtype=bool),
        section_data,
    )
    logical_chain_bits = np.asarray(chain_bits, dtype=bool) ^ representative_bits
    return _compute_signature(logical_chain_bits, primitive_logical_masks)


def _compute_corrected_x_decoder_signature(chain_bits,
                                           chain_syndrome_bits,
                                           disorder_syndrome_bits,
                                           disorder_syndrome_representative_bits,
                                           primitive_logical_masks,
                                           section_data):
    """Logical label for x = c xor eta using the corrected observable.

    The project observable is
    c xor eta xor r(Hc) xor r(Heta).  Since Hc = Hx xor Heta, an x-space
    sample must be labelled with x xor r(Hx xor Heta) xor r(Heta).
    """
    chain_syndrome_bits = np.asarray(chain_syndrome_bits, dtype=bool)
    disorder_syndrome_bits = np.asarray(disorder_syndrome_bits, dtype=bool)
    chain_syndrome_representative_bits = apply_section(
        chain_syndrome_bits ^ disorder_syndrome_bits,
        section_data,
    )
    logical_chain_bits = (
        np.asarray(chain_bits, dtype=bool)
        ^ chain_syndrome_representative_bits
        ^ np.asarray(disorder_syndrome_representative_bits, dtype=bool)
    )
    return _compute_signature(logical_chain_bits, primitive_logical_masks)


def _build_logical_projection_masks(parity_check_matrix,
                                    primitive_logical_masks):
    """Build linear P_L(x)=<z, x xor r(Hx)> masks.

    The section is the deterministic GF(2) elimination section.  This keeps
    the exp37 free-energy sectors linear while still removing the syndrome/T
    component before assigning a logical label.
    """
    num_checks, num_qubits = parity_check_matrix.shape
    num_logical = primitive_logical_masks.shape[0]
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    primitive_logical_masks_uint8 = primitive_logical_masks.astype(np.uint8)
    section_data = build_linear_section(parity_check_matrix)
    projection_masks = np.zeros((num_logical, num_qubits), dtype=bool)
    for qubit_index in range(num_qubits):
        basis_vector = np.zeros(num_qubits, dtype=bool)
        basis_vector[qubit_index] = True
        syndrome_bits = parity_check_matrix_uint8[:, qubit_index].astype(bool)
        representative = apply_linear_section(syndrome_bits, section_data)
        logical_chain = basis_vector ^ representative
        projection_masks[:, qubit_index] = (
            (primitive_logical_masks_uint8 @ logical_chain.astype(np.uint8)) % 2
        ).astype(bool)
    return projection_masks


def _signature_bits(signature, num_bits):
    return "".join(
        "1" if (int(signature) >> bit_index) & 1 else "0"
        for bit_index in range(num_bits)
    )


def _build_sector_representatives(zero_syndrome_move_data,
                                  logical_projection_masks,
                                  parity_check_matrix):
    generators = np.asarray(
        zero_syndrome_move_data["start_sector_generators"],
        dtype=bool,
    )
    num_generators, num_qubits = generators.shape
    num_sectors = 1 << num_generators
    representatives = np.zeros((num_sectors, num_qubits), dtype=bool)
    found = np.zeros(num_sectors, dtype=bool)
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)

    for combination in range(num_sectors):
        representative = np.zeros(num_qubits, dtype=bool)
        for generator_index in range(num_generators):
            if (combination >> generator_index) & 1:
                representative ^= generators[generator_index]
        signature = _compute_signature(
            representative,
            logical_projection_masks,
        )
        if found[signature]:
            raise ValueError("sector representative signatures are not unique")
        representatives[signature] = representative
        found[signature] = True

    if not np.all(found):
        raise ValueError("sector generators did not enumerate all sectors")
    syndromes = (parity_check_matrix_uint8 @ representatives.T.astype(np.uint8)) % 2
    if np.any(syndromes):
        raise ValueError("sector representatives must have zero syndrome")
    return representatives


def _pack_supports_from_bit_rows(bit_rows):
    bit_rows = np.asarray(bit_rows, dtype=bool)
    supports = [
        np.flatnonzero(bit_rows[row_index]).astype(np.int32)
        for row_index in range(bit_rows.shape[0])
    ]
    max_support_length = max(1, max(int(support.shape[0]) for support in supports))
    support_array = np.full(
        (len(supports), max_support_length),
        -1,
        dtype=np.int32,
    )
    support_lengths = np.empty(len(supports), dtype=np.int32)
    for row_index, support in enumerate(supports):
        support_lengths[row_index] = int(support.shape[0])
        if support.shape[0] > 0:
            support_array[row_index, :support.shape[0]] = support
    return support_array, support_lengths


def _logical_signature_per_qubit(logical_projection_masks):
    logical_projection_masks = np.asarray(logical_projection_masks, dtype=bool)
    num_logical, num_qubits = logical_projection_masks.shape
    signatures = np.zeros(num_qubits, dtype=np.int64)
    for bit_index in range(num_logical):
        signatures |= logical_projection_masks[bit_index].astype(np.int64) << bit_index
    return signatures


def _xor_syndrome_support_for_qubits(support, checks_touching_each_qubit):
    counts = {}
    for qubit_index in support:
        for check_index in checks_touching_each_qubit[int(qubit_index)]:
            check_index = int(check_index)
            counts[check_index] = counts.get(check_index, 0) ^ 1
    return np.array(
        [check_index for check_index, parity in counts.items() if parity],
        dtype=np.int32,
    )


def _build_projection_kernel_supports(logical_projection_masks):
    projection_matrix = np.asarray(logical_projection_masks, dtype=bool).copy()
    num_rows, num_qubits = projection_matrix.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(num_qubits):
        if pivot_row >= num_rows:
            break
        pivot_candidates = np.flatnonzero(projection_matrix[pivot_row:, column])
        if pivot_candidates.size == 0:
            continue
        selected_row = pivot_row + int(pivot_candidates[0])
        if selected_row != pivot_row:
            projection_matrix[[pivot_row, selected_row]] = projection_matrix[
                [selected_row, pivot_row]
            ]
        for row in range(num_rows):
            if row != pivot_row and projection_matrix[row, column]:
                projection_matrix[row] ^= projection_matrix[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1

    rank = len(pivot_columns)
    pivot_set = set(pivot_columns)
    supports = []
    for free_column in range(num_qubits):
        if free_column in pivot_set:
            continue
        support = [free_column]
        for row, pivot_column in enumerate(pivot_columns):
            if projection_matrix[row, free_column]:
                support.append(int(pivot_column))
        supports.append(np.asarray(sorted(support), dtype=np.int32))

    if len(supports) != num_qubits - rank:
        raise ValueError("failed to build a full projection-kernel basis")
    projection_uint8 = np.asarray(logical_projection_masks, dtype=np.uint8)
    for support in supports:
        test_vector = np.zeros(num_qubits, dtype=np.uint8)
        test_vector[support] = 1
        if np.any((projection_uint8 @ test_vector) % 2):
            raise ValueError("projection-kernel proposal does not preserve sector")
    return supports, rank


def _build_sector_preserving_proposals(parity_check_matrix,
                                       logical_projection_masks,
                                       zero_syndrome_move_data):
    num_qubits = parity_check_matrix.shape[1]
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix,
    )
    qubit_signatures = _logical_signature_per_qubit(logical_projection_masks)

    proposal_supports = []
    proposal_syndrome_supports = []
    proposal_kinds = []

    kernel_supports, projection_rank = _build_projection_kernel_supports(
        logical_projection_masks,
    )
    for support in kernel_supports:
        proposal_supports.append(support)
        proposal_syndrome_supports.append(
            _xor_syndrome_support_for_qubits(
                support,
                checks_touching_each_qubit,
            )
        )
        proposal_kinds.append(0)

    if zero_syndrome_move_data is not None:
        contractible_supports = np.asarray(
            zero_syndrome_move_data["contractible_move_supports"],
            dtype=np.int32,
        )
        for support in contractible_supports:
            proposal_supports.append(np.asarray(support, dtype=np.int32))
            proposal_syndrome_supports.append(np.empty(0, dtype=np.int32))
            proposal_kinds.append(2)

    max_data_support = max(int(support.shape[0]) for support in proposal_supports)
    max_syndrome_support = max(
        int(support.shape[0]) for support in proposal_syndrome_supports
    )
    data_support_array = np.full(
        (len(proposal_supports), max_data_support),
        -1,
        dtype=np.int32,
    )
    syndrome_support_array = np.full(
        (len(proposal_supports), max(max_syndrome_support, 1)),
        -1,
        dtype=np.int32,
    )
    data_support_lengths = np.empty(len(proposal_supports), dtype=np.int32)
    syndrome_support_lengths = np.empty(len(proposal_supports), dtype=np.int32)
    for proposal_index, support in enumerate(proposal_supports):
        data_support_lengths[proposal_index] = int(support.shape[0])
        data_support_array[proposal_index, :support.shape[0]] = support
        syndrome_support = proposal_syndrome_supports[proposal_index]
        syndrome_support_lengths[proposal_index] = int(syndrome_support.shape[0])
        if syndrome_support.shape[0] > 0:
            syndrome_support_array[
                proposal_index,
                :syndrome_support.shape[0],
            ] = syndrome_support

    proposal_kinds = np.asarray(proposal_kinds, dtype=np.int8)
    return {
        "data_supports": data_support_array,
        "data_support_lengths": data_support_lengths,
        "syndrome_supports": syndrome_support_array,
        "syndrome_support_lengths": syndrome_support_lengths,
        "proposal_kinds": proposal_kinds,
        "projection_rank": int(projection_rank),
        "qubit_signature_histogram": np.bincount(
            qubit_signatures,
            minlength=1 << logical_projection_masks.shape[0],
        ).astype(np.int64),
        "num_projection_kernel_proposals": int(
            np.count_nonzero(proposal_kinds == 0)
        ),
        "num_single_zero_qubit_proposals": int(
            np.count_nonzero(
                (proposal_kinds == 0) & (data_support_lengths == 1)
            )
        ),
        "num_pair_proposals": int(
            np.count_nonzero(
                (proposal_kinds == 0) & (data_support_lengths == 2)
            )
        ),
        "num_contractible_proposals": int(np.count_nonzero(proposal_kinds == 2)),
    }


def _build_decoder_reject_proposals(parity_check_matrix,
                                    zero_syndrome_move_data):
    num_qubits = parity_check_matrix.shape[1]
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix,
    )
    proposal_supports = []
    proposal_syndrome_supports = []
    proposal_kinds = []

    for qubit_index in range(num_qubits):
        support = np.array([qubit_index], dtype=np.int32)
        proposal_supports.append(support)
        proposal_syndrome_supports.append(
            np.asarray(
                checks_touching_each_qubit[qubit_index],
                dtype=np.int32,
            )
        )
        proposal_kinds.append(0)

    if zero_syndrome_move_data is not None:
        contractible_supports = np.asarray(
            zero_syndrome_move_data["contractible_move_supports"],
            dtype=np.int32,
        )
        for support in contractible_supports:
            proposal_supports.append(np.asarray(support, dtype=np.int32))
            proposal_syndrome_supports.append(np.empty(0, dtype=np.int32))
            proposal_kinds.append(2)

    max_data_support = max(int(support.shape[0]) for support in proposal_supports)
    max_syndrome_support = max(
        int(support.shape[0]) for support in proposal_syndrome_supports
    )
    data_support_array = np.full(
        (len(proposal_supports), max_data_support),
        -1,
        dtype=np.int32,
    )
    syndrome_support_array = np.full(
        (len(proposal_supports), max(max_syndrome_support, 1)),
        -1,
        dtype=np.int32,
    )
    data_support_lengths = np.empty(len(proposal_supports), dtype=np.int32)
    syndrome_support_lengths = np.empty(len(proposal_supports), dtype=np.int32)
    for proposal_index, support in enumerate(proposal_supports):
        data_support_lengths[proposal_index] = int(support.shape[0])
        data_support_array[proposal_index, :support.shape[0]] = support
        syndrome_support = proposal_syndrome_supports[proposal_index]
        syndrome_support_lengths[proposal_index] = int(syndrome_support.shape[0])
        if syndrome_support.shape[0] > 0:
            syndrome_support_array[
                proposal_index,
                :syndrome_support.shape[0],
            ] = syndrome_support

    proposal_kinds = np.asarray(proposal_kinds, dtype=np.int8)
    return {
        "data_supports": data_support_array,
        "data_support_lengths": data_support_lengths,
        "syndrome_supports": syndrome_support_array,
        "syndrome_support_lengths": syndrome_support_lengths,
        "proposal_kinds": proposal_kinds,
        "num_projection_kernel_proposals": 0,
        "num_single_zero_qubit_proposals": 0,
        "num_pair_proposals": 0,
        "num_single_bit_proposals": int(np.count_nonzero(proposal_kinds == 0)),
        "num_contractible_proposals": int(np.count_nonzero(proposal_kinds == 2)),
    }


def _build_unrestricted_proposals(parity_check_matrix,
                                  zero_syndrome_move_data):
    proposal_data = _build_decoder_reject_proposals(
        parity_check_matrix=parity_check_matrix,
        zero_syndrome_move_data=zero_syndrome_move_data,
    )
    if zero_syndrome_move_data is None:
        return proposal_data

    data_supports = [
        proposal_data["data_supports"][index, :proposal_data["data_support_lengths"][index]].copy()
        for index in range(proposal_data["data_support_lengths"].shape[0])
    ]
    syndrome_supports = [
        proposal_data["syndrome_supports"][index, :proposal_data["syndrome_support_lengths"][index]].copy()
        for index in range(proposal_data["syndrome_support_lengths"].shape[0])
    ]
    proposal_kinds = proposal_data["proposal_kinds"].tolist()
    for support in np.asarray(zero_syndrome_move_data["winding_move_supports"], dtype=np.int32):
        data_supports.append(np.asarray(support, dtype=np.int32))
        syndrome_supports.append(np.empty(0, dtype=np.int32))
        proposal_kinds.append(3)

    max_data_support = max(int(support.shape[0]) for support in data_supports)
    max_syndrome_support = max(int(support.shape[0]) for support in syndrome_supports)
    data_support_array = np.full(
        (len(data_supports), max_data_support),
        -1,
        dtype=np.int32,
    )
    syndrome_support_array = np.full(
        (len(data_supports), max(max_syndrome_support, 1)),
        -1,
        dtype=np.int32,
    )
    data_support_lengths = np.empty(len(data_supports), dtype=np.int32)
    syndrome_support_lengths = np.empty(len(data_supports), dtype=np.int32)
    for index, support in enumerate(data_supports):
        data_support_lengths[index] = int(support.shape[0])
        data_support_array[index, :support.shape[0]] = support
        syndrome_support = syndrome_supports[index]
        syndrome_support_lengths[index] = int(syndrome_support.shape[0])
        if syndrome_support.shape[0] > 0:
            syndrome_support_array[index, :syndrome_support.shape[0]] = syndrome_support
    proposal_kinds = np.asarray(proposal_kinds, dtype=np.int8)
    return {
        "data_supports": data_support_array,
        "data_support_lengths": data_support_lengths,
        "syndrome_supports": syndrome_support_array,
        "syndrome_support_lengths": syndrome_support_lengths,
        "proposal_kinds": proposal_kinds,
        "num_single_bit_proposals": int(np.count_nonzero(proposal_kinds == 0)),
        "num_contractible_proposals": int(np.count_nonzero(proposal_kinds == 2)),
        "num_winding_proposals": int(np.count_nonzero(proposal_kinds == 3)),
    }


def _build_even_winding_groups(zero_syndrome_move_data, lattice_size):
    supports = np.asarray(
        zero_syndrome_move_data["winding_move_supports"],
        dtype=np.int32,
    )
    if supports.shape[0] != 3 * int(lattice_size):
        return []
    groups = []
    for direction in range(3):
        start = direction * int(lattice_size)
        stop = start + int(lattice_size)
        groups.append(supports[start:stop])
    return groups


def _run_even_winding_heatbath(current_x_bits,
                               current_syndrome_term_bits,
                               winding_groups,
                               kp_value,
                               rng):
    del current_syndrome_term_bits
    if kp_value == 0.0:
        minus_kp = 0.0
    else:
        minus_kp = -float(kp_value)
    changed_count = 0
    attempted_count = 0
    for group in winding_groups:
        num_moves = int(group.shape[0])
        if num_moves == 0:
            continue
        attempted_count += num_moves
        delta_by_move = np.empty(num_moves, dtype=np.int64)
        for move_index in range(num_moves):
            support = group[move_index]
            ones = int(np.count_nonzero(current_x_bits[support]))
            delta_by_move[move_index] = int(support.shape[0]) - 2 * ones

        num_configurations = 1 << num_moves
        log_weights = np.full(num_configurations, -np.inf, dtype=np.float64)
        log_weights[0] = 0.0
        for configuration in range(1, num_configurations):
            if configuration.bit_count() % 2:
                continue
            least_bit = configuration & -configuration
            bit_index = least_bit.bit_length() - 1
            previous = configuration ^ least_bit
            if previous.bit_count() % 2:
                previous_least_bit = previous & -previous
                previous_bit_index = previous_least_bit.bit_length() - 1
                previous_even = previous ^ previous_least_bit
                log_weights[configuration] = (
                    log_weights[previous_even]
                    + minus_kp * float(delta_by_move[bit_index])
                    + minus_kp * float(delta_by_move[previous_bit_index])
                )
            else:
                log_weights[configuration] = (
                    log_weights[previous]
                    + minus_kp * float(delta_by_move[bit_index])
                )
        max_log_weight = float(np.max(log_weights))
        if not np.isfinite(max_log_weight):
            continue
        weights = np.exp(log_weights - max_log_weight)
        threshold = float(rng.random() * np.sum(weights))
        cumulative = 0.0
        selected = 0
        for configuration, weight in enumerate(weights):
            cumulative += float(weight)
            if threshold <= cumulative:
                selected = configuration
                break
        if selected == 0:
            continue
        for move_index in range(num_moves):
            if (selected >> move_index) & 1:
                current_x_bits[group[move_index]] ^= True
                changed_count += 1
    return int(changed_count), int(attempted_count)


def _attempt_proposal(current_x_bits,
                      current_syndrome_term_bits,
                      proposal_index,
                      proposals,
                      kp_value,
                      kq_value,
                      rng):
    support_length = int(proposals["data_support_lengths"][proposal_index])
    support = proposals["data_supports"][proposal_index, :support_length]
    ones = int(np.count_nonzero(current_x_bits[support]))
    delta_data_weight = support_length - 2 * ones

    syndrome_support_length = int(
        proposals["syndrome_support_lengths"][proposal_index]
    )
    if syndrome_support_length == 0:
        delta_syndrome_weight = 0
        syndrome_support = None
    else:
        syndrome_support = proposals[
            "syndrome_supports"
        ][proposal_index, :syndrome_support_length]
        syndrome_ones = int(np.count_nonzero(
            current_syndrome_term_bits[syndrome_support]
        ))
        delta_syndrome_weight = syndrome_support_length - 2 * syndrome_ones

    log_acceptance = (
        -float(kp_value) * float(delta_data_weight)
        -float(kq_value) * float(delta_syndrome_weight)
    )
    if log_acceptance >= 0.0 or rng.random() < math.exp(log_acceptance):
        current_x_bits[support] ^= True
        if syndrome_support_length > 0:
            current_syndrome_term_bits[syndrome_support] ^= True
        return True
    return False


def _run_sector_sweep(current_x_bits,
                      current_syndrome_term_bits,
                      proposals,
                      winding_groups,
                      kp_value,
                      kq_value,
                      rng,
                      winding_heatbath_sweeps):
    accepted = 0
    num_proposals = int(proposals["data_support_lengths"].shape[0])
    for proposal_index in rng.permutation(num_proposals):
        accepted += int(
            _attempt_proposal(
                current_x_bits=current_x_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                proposal_index=int(proposal_index),
                proposals=proposals,
                kp_value=kp_value,
                kq_value=kq_value,
                rng=rng,
            )
        )
    heatbath_changed = 0
    heatbath_attempted = 0
    for _ in range(int(winding_heatbath_sweeps)):
        changed, attempted = _run_even_winding_heatbath(
            current_x_bits=current_x_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            winding_groups=winding_groups,
            kp_value=kp_value,
            rng=rng,
        )
        heatbath_changed += changed
        heatbath_attempted += attempted
    return int(accepted), int(num_proposals), heatbath_changed, heatbath_attempted


def _attempt_decoder_reject_proposal(current_x_bits,
                                     current_syndrome_term_bits,
                                     measurement_error_bits,
                                     disorder_syndrome_bits,
                                     disorder_syndrome_representative_bits,
                                     proposal_index,
                                     proposals,
                                     primitive_logical_masks,
                                     section_data,
                                     target_sector,
                                     kp_value,
                                     kq_value,
                                     rng):
    support_length = int(proposals["data_support_lengths"][proposal_index])
    support = proposals["data_supports"][proposal_index, :support_length]
    ones = int(np.count_nonzero(current_x_bits[support]))
    delta_data_weight = support_length - 2 * ones

    syndrome_support_length = int(
        proposals["syndrome_support_lengths"][proposal_index]
    )
    if syndrome_support_length == 0:
        syndrome_support = None
        delta_syndrome_weight = 0
    else:
        syndrome_support = proposals[
            "syndrome_supports"
        ][proposal_index, :syndrome_support_length]
        syndrome_ones = int(np.count_nonzero(
            current_syndrome_term_bits[syndrome_support]
        ))
        delta_syndrome_weight = syndrome_support_length - 2 * syndrome_ones

    log_acceptance = (
        -float(kp_value) * float(delta_data_weight)
        -float(kq_value) * float(delta_syndrome_weight)
    )
    if log_acceptance < 0.0 and rng.random() >= math.exp(log_acceptance):
        return False, False

    current_x_bits[support] ^= True
    if syndrome_support_length > 0:
        current_syndrome_term_bits[syndrome_support] ^= True
    chain_syndrome_bits = current_syndrome_term_bits ^ measurement_error_bits
    signature = _compute_corrected_x_decoder_signature(
        chain_bits=current_x_bits,
        chain_syndrome_bits=chain_syndrome_bits,
        disorder_syndrome_bits=disorder_syndrome_bits,
        disorder_syndrome_representative_bits=(
            disorder_syndrome_representative_bits
        ),
        primitive_logical_masks=primitive_logical_masks,
        section_data=section_data,
    )
    if signature == int(target_sector):
        return True, False

    current_x_bits[support] ^= True
    if syndrome_support_length > 0:
        current_syndrome_term_bits[syndrome_support] ^= True
    return False, True


def _run_decoder_reject_sector_sweep(current_x_bits,
                                     current_syndrome_term_bits,
                                     measurement_error_bits,
                                     disorder_syndrome_bits,
                                     disorder_syndrome_representative_bits,
                                     proposals,
                                     primitive_logical_masks,
                                     section_data,
                                     target_sector,
                                     kp_value,
                                     kq_value,
                                     rng):
    accepted = 0
    sector_rejected = 0
    num_proposals = int(proposals["data_support_lengths"].shape[0])
    for proposal_index in rng.permutation(num_proposals):
        proposal_accepted, proposal_sector_rejected = (
            _attempt_decoder_reject_proposal(
                current_x_bits=current_x_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                measurement_error_bits=measurement_error_bits,
                disorder_syndrome_bits=disorder_syndrome_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                proposal_index=int(proposal_index),
                proposals=proposals,
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
                target_sector=target_sector,
                kp_value=kp_value,
                kq_value=kq_value,
                rng=rng,
            )
        )
        accepted += int(proposal_accepted)
        sector_rejected += int(proposal_sector_rejected)
    return int(accepted), int(num_proposals), int(sector_rejected)


def _run_fixed_sector_chain_decoder_reject(parity_check_matrix,
                                           primitive_logical_masks,
                                           section_data,
                                           sector_representative_bits,
                                           target_sector,
                                           measurement_error_bits,
                                           disorder_syndrome_bits,
                                           disorder_syndrome_representative_bits,
                                           proposals,
                                           kp_grid,
                                           kq_value,
                                           num_burn_in_sweeps,
                                           num_measurements,
                                           num_sweeps_between_measurements,
                                           block_count,
                                           rng,
                                           debug_checks=False):
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    num_grid = int(len(kp_grid))
    block_count = max(1, min(int(block_count), int(num_measurements)))
    block_indices = np.array_split(np.arange(num_measurements), block_count)

    current_x_bits = np.asarray(sector_representative_bits, dtype=bool).copy()
    current_chain_syndrome_bits = (
        (parity_check_matrix_uint8 @ current_x_bits.astype(np.uint8)) % 2
    ).astype(bool)
    current_syndrome_term_bits = current_chain_syndrome_bits ^ measurement_error_bits
    initial_signature = _compute_corrected_x_decoder_signature(
        chain_bits=current_x_bits,
        chain_syndrome_bits=current_chain_syndrome_bits,
        disorder_syndrome_bits=disorder_syndrome_bits,
        disorder_syndrome_representative_bits=(
            disorder_syndrome_representative_bits
        ),
        primitive_logical_masks=primitive_logical_masks,
        section_data=section_data,
    )
    if initial_signature != int(target_sector):
        raise AssertionError(
            f"decoder sector representative mismatch: expected {target_sector}, "
            f"got {initial_signature}"
        )

    mu = np.empty(num_grid, dtype=np.float64)
    syndrome_mu = np.empty(num_grid, dtype=np.float64)
    energy_mu = np.empty(num_grid, dtype=np.float64)
    block_mu = np.empty((num_grid, block_count), dtype=np.float64)
    acceptance_rate = np.empty(num_grid, dtype=np.float64)
    sector_reject_rate = np.empty(num_grid, dtype=np.float64)

    for grid_index, kp_value in enumerate(kp_grid):
        accepted_total = 0
        attempted_total = 0
        sector_rejected_total = 0
        for _ in range(int(num_burn_in_sweeps)):
            accepted, attempted, sector_rejected = _run_decoder_reject_sector_sweep(
                current_x_bits=current_x_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                measurement_error_bits=measurement_error_bits,
                disorder_syndrome_bits=disorder_syndrome_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                proposals=proposals,
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
                target_sector=target_sector,
                kp_value=float(kp_value),
                kq_value=float(kq_value),
                rng=rng,
            )
            accepted_total += accepted
            attempted_total += attempted
            sector_rejected_total += sector_rejected

        data_weight_samples = np.empty(num_measurements, dtype=np.float64)
        syndrome_weight_samples = np.empty(num_measurements, dtype=np.float64)
        for measurement_index in range(int(num_measurements)):
            for _ in range(int(num_sweeps_between_measurements)):
                accepted, attempted, sector_rejected = _run_decoder_reject_sector_sweep(
                    current_x_bits=current_x_bits,
                    current_syndrome_term_bits=current_syndrome_term_bits,
                    measurement_error_bits=measurement_error_bits,
                    disorder_syndrome_bits=disorder_syndrome_bits,
                    disorder_syndrome_representative_bits=(
                        disorder_syndrome_representative_bits
                    ),
                    proposals=proposals,
                    primitive_logical_masks=primitive_logical_masks,
                    section_data=section_data,
                    target_sector=target_sector,
                    kp_value=float(kp_value),
                    kq_value=float(kq_value),
                    rng=rng,
                )
                accepted_total += accepted
                attempted_total += attempted
                sector_rejected_total += sector_rejected
            data_weight_samples[measurement_index] = float(
                np.count_nonzero(current_x_bits)
            )
            syndrome_weight_samples[measurement_index] = float(
                np.count_nonzero(current_syndrome_term_bits)
            )

        if debug_checks:
            recomputed_chain_syndrome_bits = (
                (parity_check_matrix_uint8 @ current_x_bits.astype(np.uint8)) % 2
            ).astype(bool)
            if not np.array_equal(
                    recomputed_chain_syndrome_bits ^ measurement_error_bits,
                    current_syndrome_term_bits):
                raise AssertionError("decoder-reject syndrome cache mismatch")
            signature = _compute_corrected_x_decoder_signature(
                chain_bits=current_x_bits,
                chain_syndrome_bits=recomputed_chain_syndrome_bits,
                disorder_syndrome_bits=disorder_syndrome_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
            )
            if signature != int(target_sector):
                raise AssertionError(
                    f"decoder-reject sector invariant failed: "
                    f"expected {target_sector}, got {signature}"
                )

        mu[grid_index] = float(np.mean(data_weight_samples))
        syndrome_mu[grid_index] = float(np.mean(syndrome_weight_samples))
        energy_mu[grid_index] = float(
            float(kp_value) * mu[grid_index]
            + float(kq_value) * syndrome_mu[grid_index]
        )
        for block_index, indices in enumerate(block_indices):
            block_mu[grid_index, block_index] = float(
                np.mean(data_weight_samples[indices])
            )
        acceptance_rate[grid_index] = (
            0.0 if attempted_total == 0 else accepted_total / attempted_total
        )
        sector_reject_rate[grid_index] = (
            0.0 if attempted_total == 0 else sector_rejected_total / attempted_total
        )

    return {
        "mu": mu,
        "syndrome_mu": syndrome_mu,
        "energy_mu": energy_mu,
        "block_mu": block_mu,
        "acceptance_rate": acceptance_rate,
        "winding_heatbath_change_rate": sector_reject_rate,
    }


def _run_fixed_sector_chain(parity_check_matrix,
                            logical_projection_masks,
                            sector_representative_bits,
                            target_sector,
                            measurement_error_bits,
                            proposals,
                            winding_groups,
                            kp_grid,
                            kq_value,
                            num_burn_in_sweeps,
                            num_measurements,
                            num_sweeps_between_measurements,
                            block_count,
                            winding_heatbath_sweeps,
                            rng,
                            use_numba=False,
                            debug_checks=False):
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    num_grid = int(len(kp_grid))
    block_count = max(1, min(int(block_count), int(num_measurements)))
    block_indices = np.array_split(np.arange(num_measurements), block_count)

    current_x_bits = np.asarray(sector_representative_bits, dtype=bool).copy()
    current_syndrome_term_bits = (
        (parity_check_matrix_uint8 @ current_x_bits.astype(np.uint8)) % 2
    ).astype(bool) ^ measurement_error_bits

    if bool(use_numba) and _numba_run_fixed_sector_chain is not None:
        if debug_checks:
            signature = _compute_signature(
                current_x_bits,
                logical_projection_masks,
            )
            if signature != int(target_sector):
                raise AssertionError(
                    f"sector invariant failed before numba run: "
                    f"expected {target_sector}, got {signature}"
                )
        if winding_groups:
            winding_groups_array = np.asarray(winding_groups, dtype=np.int32)
        else:
            winding_groups_array = np.empty((0, 0, 0), dtype=np.int32)
        numba_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        (
            mu,
            syndrome_mu,
            energy_mu,
            block_mu,
            acceptance_rate,
            heatbath_change_rate,
        ) = _numba_run_fixed_sector_chain(
            current_x_bits,
            current_syndrome_term_bits,
            proposals["data_supports"],
            proposals["data_support_lengths"],
            proposals["syndrome_supports"],
            proposals["syndrome_support_lengths"],
            winding_groups_array,
            np.asarray(kp_grid, dtype=np.float64),
            float(kq_value),
            int(num_burn_in_sweeps),
            int(num_measurements),
            int(num_sweeps_between_measurements),
            int(block_count),
            int(winding_heatbath_sweeps),
            numba_seed,
        )
        return {
            "mu": mu,
            "syndrome_mu": syndrome_mu,
            "energy_mu": energy_mu,
            "block_mu": block_mu,
            "acceptance_rate": acceptance_rate,
            "winding_heatbath_change_rate": heatbath_change_rate,
        }

    mu = np.empty(num_grid, dtype=np.float64)
    syndrome_mu = np.empty(num_grid, dtype=np.float64)
    energy_mu = np.empty(num_grid, dtype=np.float64)
    block_mu = np.empty((num_grid, block_count), dtype=np.float64)
    acceptance_rate = np.empty(num_grid, dtype=np.float64)
    heatbath_change_rate = np.empty(num_grid, dtype=np.float64)

    for grid_index, kp_value in enumerate(kp_grid):
        accepted_total = 0
        attempted_total = 0
        heatbath_changed_total = 0
        heatbath_attempted_total = 0
        for _ in range(int(num_burn_in_sweeps)):
            accepted, attempted, hb_changed, hb_attempted = _run_sector_sweep(
                current_x_bits=current_x_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                proposals=proposals,
                winding_groups=winding_groups,
                kp_value=float(kp_value),
                kq_value=float(kq_value),
                rng=rng,
                winding_heatbath_sweeps=winding_heatbath_sweeps,
            )
            accepted_total += accepted
            attempted_total += attempted
            heatbath_changed_total += hb_changed
            heatbath_attempted_total += hb_attempted

        data_weight_samples = np.empty(num_measurements, dtype=np.float64)
        syndrome_weight_samples = np.empty(num_measurements, dtype=np.float64)
        for measurement_index in range(int(num_measurements)):
            for _ in range(int(num_sweeps_between_measurements)):
                accepted, attempted, hb_changed, hb_attempted = _run_sector_sweep(
                    current_x_bits=current_x_bits,
                    current_syndrome_term_bits=current_syndrome_term_bits,
                    proposals=proposals,
                    winding_groups=winding_groups,
                    kp_value=float(kp_value),
                    kq_value=float(kq_value),
                    rng=rng,
                    winding_heatbath_sweeps=winding_heatbath_sweeps,
                )
                accepted_total += accepted
                attempted_total += attempted
                heatbath_changed_total += hb_changed
                heatbath_attempted_total += hb_attempted
            data_weight_samples[measurement_index] = float(
                np.count_nonzero(current_x_bits)
            )
            syndrome_weight_samples[measurement_index] = float(
                np.count_nonzero(current_syndrome_term_bits)
            )

        if debug_checks:
            signature = _compute_signature(
                current_x_bits,
                logical_projection_masks,
            )
            if signature != int(target_sector):
                raise AssertionError(
                    f"sector invariant failed: expected {target_sector}, got {signature}"
                )
            recomputed = (
                (parity_check_matrix_uint8 @ current_x_bits.astype(np.uint8)) % 2
            ).astype(bool) ^ measurement_error_bits
            if not np.array_equal(recomputed, current_syndrome_term_bits):
                raise AssertionError("syndrome cache mismatch")

        mu[grid_index] = float(np.mean(data_weight_samples))
        syndrome_mu[grid_index] = float(np.mean(syndrome_weight_samples))
        energy_mu[grid_index] = float(
            float(kp_value) * mu[grid_index]
            + float(kq_value) * syndrome_mu[grid_index]
        )
        for block_index, indices in enumerate(block_indices):
            block_mu[grid_index, block_index] = float(
                np.mean(data_weight_samples[indices])
            )
        acceptance_rate[grid_index] = (
            0.0 if attempted_total == 0 else accepted_total / attempted_total
        )
        heatbath_change_rate[grid_index] = (
            0.0
            if heatbath_attempted_total == 0
            else heatbath_changed_total / heatbath_attempted_total
        )

    return {
        "mu": mu,
        "syndrome_mu": syndrome_mu,
        "energy_mu": energy_mu,
        "block_mu": block_mu,
        "acceptance_rate": acceptance_rate,
        "winding_heatbath_change_rate": heatbath_change_rate,
    }


def _integrate_mu(kp_grid, mu_by_sector):
    return np.trapezoid(mu_by_sector, x=kp_grid, axis=-1)


def _weights_from_delta_f(delta_f):
    shifted = -np.asarray(delta_f, dtype=np.float64)
    shifted -= float(np.max(shifted))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("invalid sector weights")
    return weights / total


def _q_top_from_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float((8.0 * np.sum(weights ** 2) - 1.0) / 7.0)


def _bootstrap_ti(kp_grid, block_mu_by_sector, num_bootstrap, rng):
    block_mu_by_sector = np.asarray(block_mu_by_sector, dtype=np.float64)
    num_sectors, num_grid, block_count = block_mu_by_sector.shape
    q_top_samples = np.empty(int(num_bootstrap), dtype=np.float64)
    weights_samples = np.empty((int(num_bootstrap), num_sectors), dtype=np.float64)
    delta_f_samples = np.empty((int(num_bootstrap), num_sectors), dtype=np.float64)
    for sample_index in range(int(num_bootstrap)):
        sampled_mu = np.empty((num_sectors, num_grid), dtype=np.float64)
        for sector in range(num_sectors):
            for grid_index in range(num_grid):
                indices = rng.integers(0, block_count, size=block_count)
                sampled_mu[sector, grid_index] = float(
                    np.mean(block_mu_by_sector[sector, grid_index, indices])
                )
        integrals = _integrate_mu(kp_grid, sampled_mu)
        delta_f = integrals - integrals[0]
        weights = _weights_from_delta_f(delta_f)
        delta_f_samples[sample_index] = delta_f
        weights_samples[sample_index] = weights
        q_top_samples[sample_index] = _q_top_from_weights(weights)
    return {
        "q_top_samples": q_top_samples,
        "q_top_stderr": float(np.std(q_top_samples, ddof=1)),
        "q_top_ci95": np.quantile(q_top_samples, [0.025, 0.975]),
        "weights_stderr": np.std(weights_samples, axis=0, ddof=1),
        "delta_f_stderr": np.std(delta_f_samples, axis=0, ddof=1),
    }


def _logsumexp_by_sector(log_weights, sector_indices, num_sectors):
    sector_log_sums = np.full(num_sectors, -np.inf, dtype=np.float64)
    for sector in range(num_sectors):
        sector_values = log_weights[sector_indices == sector]
        if sector_values.size:
            sector_log_sums[sector] = _logsumexp(sector_values)
    return sector_log_sums


def _ais_weights_from_log_samples(log_weights, sector_indices, num_sectors):
    sector_log_sums = _logsumexp_by_sector(
        log_weights=log_weights,
        sector_indices=sector_indices,
        num_sectors=num_sectors,
    )
    total_log_sum = _logsumexp(sector_log_sums)
    weights = np.exp(sector_log_sums - total_log_sum)
    return weights, sector_log_sums, total_log_sum


def _ais_weights_from_log_matrix(log_weight_matrix):
    sector_log_sums = np.asarray([
        _logsumexp(log_weight_matrix[:, sector])
        for sector in range(log_weight_matrix.shape[1])
    ], dtype=np.float64)
    total_log_sum = _logsumexp(sector_log_sums)
    weights = np.exp(sector_log_sums - total_log_sum)
    return weights, sector_log_sums, total_log_sum


def _bootstrap_ais(log_weights, sector_indices, num_sectors, num_bootstrap, rng):
    num_samples = int(log_weights.shape[0])
    q_top_samples = np.empty(int(num_bootstrap), dtype=np.float64)
    weights_samples = np.empty((int(num_bootstrap), num_sectors), dtype=np.float64)
    delta_f_samples = np.empty((int(num_bootstrap), num_sectors), dtype=np.float64)
    for sample_index in range(int(num_bootstrap)):
        sample_indices = rng.integers(0, num_samples, size=num_samples)
        weights, _, _ = _ais_weights_from_log_samples(
            log_weights=log_weights[sample_indices],
            sector_indices=sector_indices[sample_indices],
            num_sectors=num_sectors,
        )
        delta_f = np.full(num_sectors, np.inf, dtype=np.float64)
        if weights[0] > 0.0:
            positive_weight_mask = weights > 0.0
            delta_f[positive_weight_mask] = -np.log(
                weights[positive_weight_mask] / weights[0]
            )
        weights_samples[sample_index] = weights
        delta_f_samples[sample_index] = delta_f
        q_top_samples[sample_index] = _q_top_from_weights(weights)
    return {
        "q_top_stderr": float(np.std(q_top_samples, ddof=1)),
        "q_top_ci95": np.quantile(q_top_samples, [0.025, 0.975]),
        "weights_stderr": np.std(weights_samples, axis=0, ddof=1),
        "delta_f_stderr": np.nanstd(delta_f_samples, axis=0, ddof=1),
    }


def _bootstrap_ais_matrix(log_weight_matrix, num_bootstrap, rng):
    num_samples = int(log_weight_matrix.shape[0])
    num_sectors = int(log_weight_matrix.shape[1])
    q_top_samples = np.empty(int(num_bootstrap), dtype=np.float64)
    weights_samples = np.empty((int(num_bootstrap), num_sectors), dtype=np.float64)
    delta_f_samples = np.empty((int(num_bootstrap), num_sectors), dtype=np.float64)
    for sample_index in range(int(num_bootstrap)):
        sample_indices = rng.integers(0, num_samples, size=num_samples)
        weights, _, _ = _ais_weights_from_log_matrix(
            log_weight_matrix=log_weight_matrix[sample_indices],
        )
        delta_f = np.full(num_sectors, np.inf, dtype=np.float64)
        if weights[0] > 0.0:
            positive_weight_mask = weights > 0.0
            delta_f[positive_weight_mask] = -np.log(
                weights[positive_weight_mask] / weights[0]
            )
        weights_samples[sample_index] = weights
        delta_f_samples[sample_index] = delta_f
        q_top_samples[sample_index] = _q_top_from_weights(weights)
    return {
        "q_top_stderr": float(np.std(q_top_samples, ddof=1)),
        "q_top_ci95": np.quantile(q_top_samples, [0.025, 0.975]),
        "weights_stderr": np.std(weights_samples, axis=0, ddof=1),
        "delta_f_stderr": np.nanstd(delta_f_samples, axis=0, ddof=1),
    }


def _run_single_ais_task(task):
    if _numba_run_ais_particle is None:
        raise RuntimeError("AIS production requires numba in the active environment")
    started_at = time.perf_counter()
    lattice_size = int(task["lattice_size"])
    p_value = float(task["p_value"])
    q_value = float(task["q_value"])
    disorder_index = int(task["disorder_index"])
    seed = int(task["seed"])
    code_family = str(task.get("code_family", "3d_toric"))
    ais_estimator = str(task.get("ais_estimator", "direct"))
    if ais_estimator not in {"direct", "flip_reweight"}:
        raise ValueError("ais_estimator must be direct or flip_reweight")
    disorder_seed = int(task.get("disorder_seed", seed))
    sample_seed = int(task.get("sample_seed", seed))
    rng_disorder = np.random.default_rng(disorder_seed)
    rng_sample = np.random.default_rng(sample_seed)

    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        code_family,
        lattice_size,
    )
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        code_family,
        lattice_size,
    )
    section_data = build_syndrome_representative_section(parity_check_matrix)
    proposals = _build_unrestricted_proposals(
        parity_check_matrix=parity_check_matrix,
        zero_syndrome_move_data=zero_syndrome_move_data,
    )
    logical_heatbath_sweeps = int(task.get("logical_heatbath_sweeps", 0))
    sector_representatives = _build_sector_representatives(
        zero_syndrome_move_data=zero_syndrome_move_data,
        logical_projection_masks=primitive_logical_masks,
        parity_check_matrix=parity_check_matrix,
    )
    logical_supports, logical_support_lengths = _pack_supports_from_bit_rows(
        sector_representatives,
    )
    num_checks, num_qubits = parity_check_matrix.shape
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)

    eta_bits = rng_disorder.random(num_qubits) < p_value
    measurement_error_bits = rng_disorder.random(num_checks) < q_value
    eta_syndrome_bits = (
        parity_check_matrix_uint8 @ eta_bits.astype(np.uint8)
    ) % 2
    eta_syndrome_bits = eta_syndrome_bits.astype(bool)
    observed_syndrome_bits = eta_syndrome_bits ^ measurement_error_bits
    del observed_syndrome_bits
    disorder_syndrome_representative_bits = apply_section(
        eta_syndrome_bits,
        section_data,
    )

    kp_target = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    kp_grid = np.linspace(
        0.0,
        kp_target,
        int(task["num_kp_grid_points"]),
        dtype=np.float64,
    )
    num_particles = int(task["num_ais_particles"])
    num_sectors = 1 << primitive_logical_masks.shape[0]
    log_weights = np.empty(num_particles, dtype=np.float64)
    sector_indices = np.empty(num_particles, dtype=np.int64)
    flip_log_weight_matrix = np.empty((num_particles, num_sectors), dtype=np.float64)

    for particle_index in range(num_particles):
        initial_x_bits = rng_sample.integers(0, 2, size=num_qubits).astype(bool)
        initial_syndrome_term_bits = (
            (parity_check_matrix_uint8 @ initial_x_bits.astype(np.uint8)) % 2
        ).astype(bool) ^ measurement_error_bits
        particle_seed = int(rng_sample.integers(0, np.iinfo(np.int32).max))
        final_x_bits, final_syndrome_term_bits, log_weight = _numba_run_ais_particle(
            initial_x_bits,
            initial_syndrome_term_bits,
            proposals["data_supports"],
            proposals["data_support_lengths"],
            proposals["syndrome_supports"],
            proposals["syndrome_support_lengths"],
            logical_supports,
            logical_support_lengths,
            kp_grid,
            float(kq_value),
            int(task["num_initial_burn_in_sweeps"]),
            int(task["num_transition_sweeps"]),
            int(logical_heatbath_sweeps),
            particle_seed,
        )
        final_chain_syndrome_bits = final_syndrome_term_bits ^ measurement_error_bits
        sector_index = _compute_corrected_x_decoder_signature(
            chain_bits=final_x_bits,
            chain_syndrome_bits=final_chain_syndrome_bits,
            disorder_syndrome_bits=eta_syndrome_bits,
            disorder_syndrome_representative_bits=(
                disorder_syndrome_representative_bits
            ),
            primitive_logical_masks=primitive_logical_masks,
            section_data=section_data,
        )
        sector_indices[particle_index] = sector_index
        log_weights[particle_index] = float(log_weight)
        for representative_index in range(num_sectors):
            support_length = int(logical_support_lengths[representative_index])
            support = logical_supports[representative_index, :support_length]
            overlap_weight = int(np.count_nonzero(final_x_bits[support]))
            delta_data_weight = int(support_length) - 2 * overlap_weight
            target_sector = int(sector_index) ^ int(representative_index)
            flip_log_weight_matrix[particle_index, target_sector] = (
                float(log_weight) - float(kp_target) * float(delta_data_weight)
            )

    if ais_estimator == "flip_reweight":
        log_weight_matrix = flip_log_weight_matrix
    else:
        log_weight_matrix = np.full(
            (num_particles, num_sectors),
            -np.inf,
            dtype=np.float64,
        )
        log_weight_matrix[np.arange(num_particles), sector_indices] = log_weights

    weights, sector_log_sums, total_log_sum = _ais_weights_from_log_matrix(
        log_weight_matrix=log_weight_matrix,
    )
    q_top = _q_top_from_weights(weights)
    per_particle_log_sums = np.asarray([
        _logsumexp(log_weight_matrix[particle_index])
        for particle_index in range(num_particles)
    ], dtype=np.float64)
    log_sum_w2 = _logsumexp(2.0 * per_particle_log_sums)
    ess = float(math.exp(2.0 * total_log_sum - log_sum_w2))
    bootstrap = _bootstrap_ais_matrix(
        log_weight_matrix=log_weight_matrix,
        num_bootstrap=int(task["num_bootstrap"]),
        rng=np.random.default_rng(
            int(rng_sample.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        ),
    )
    flags = []
    if ess < float(task["min_ais_ess"]):
        flags.append("AIS_LOW_ESS_WARN")
    if ess / float(num_particles) < float(task["min_ais_ess_fraction"]):
        flags.append("AIS_LOW_ESS_FRACTION_WARN")
    if not flags:
        flags.append("PASS")
    delta_f = np.full_like(weights, np.inf, dtype=np.float64)
    if weights[0] > 0.0:
        positive_weight_mask = weights > 0.0
        delta_f[positive_weight_mask] = -np.log(
            weights[positive_weight_mask] / weights[0]
        )
    return {
        "lattice_size": lattice_size,
        "p_value": p_value,
        "q_value": q_value,
        "disorder_index": disorder_index,
        "seed": seed,
        "disorder_seed": disorder_seed,
        "sample_seed": sample_seed,
        "replica_index": int(task.get("replica_index", 0)),
        "ais_estimator": ais_estimator,
        "num_qubits": int(num_qubits),
        "num_checks": int(num_checks),
        "num_ais_particles": int(num_particles),
        "num_bootstrap": int(task["num_bootstrap"]),
        "kp_target": kp_target,
        "kq_value": kq_value,
        "kp_grid": kp_grid,
        "weights": weights,
        "delta_f": delta_f,
        "q_top": q_top,
        "q_top_stderr": bootstrap["q_top_stderr"],
        "q_top_ci95": bootstrap["q_top_ci95"],
        "weights_stderr": bootstrap["weights_stderr"],
        "ais_ess": ess,
        "ais_ess_fraction": ess / float(num_particles),
        "min_ais_ess": float(task["min_ais_ess"]),
        "min_ais_ess_fraction": float(task["min_ais_ess_fraction"]),
        "logical_heatbath_sweeps": int(logical_heatbath_sweeps),
        "log_weights": log_weights,
        "sector_indices": sector_indices,
        "log_weight_matrix": log_weight_matrix,
        "sector_sample_counts": np.bincount(sector_indices, minlength=num_sectors),
        "sector_log_weight_sums": sector_log_sums,
        "flags": ";".join(flags),
        "wall_time_seconds": float(time.perf_counter() - started_at),
    }


def _coarse_indices_from_fine_grid(num_grid):
    indices = np.arange(0, int(num_grid), 2, dtype=np.int64)
    if indices[-1] != int(num_grid) - 1:
        indices = np.concatenate([indices, np.array([int(num_grid) - 1])])
    return indices


def _run_single_ti_task(task):
    started_at = time.perf_counter()
    lattice_size = int(task["lattice_size"])
    p_value = float(task["p_value"])
    q_value = float(task["q_value"])
    disorder_index = int(task["disorder_index"])
    seed = int(task["seed"])
    code_family = str(task.get("code_family", "3d_toric"))
    projection_mode = str(task.get("projection_mode", "linear"))
    rng = np.random.default_rng(seed)

    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        code_family,
        lattice_size,
    )
    if projection_mode == "linear":
        logical_projection_masks = _build_logical_projection_masks(
            parity_check_matrix=parity_check_matrix,
            primitive_logical_masks=primitive_logical_masks,
        )
        section_data = None
    elif projection_mode == "decoder_reject":
        logical_projection_masks = primitive_logical_masks
        section_data = build_syndrome_representative_section(
            parity_check_matrix,
        )
    else:
        raise ValueError("projection_mode must be linear or decoder_reject")
    zero_syndrome_move_data = build_zero_syndrome_move_data_by_family(
        code_family,
        lattice_size,
    )
    num_checks, num_qubits = parity_check_matrix.shape
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)

    data_uniform = rng.random(num_qubits)
    syndrome_uniform = rng.random(num_checks)
    eta_bits = data_uniform < p_value
    measurement_error_bits = syndrome_uniform < q_value
    eta_syndrome_bits = (
        parity_check_matrix_uint8 @ eta_bits.astype(np.uint8)
    ) % 2
    eta_syndrome_bits = eta_syndrome_bits.astype(bool)
    observed_syndrome_bits = eta_syndrome_bits.astype(bool) ^ measurement_error_bits
    if projection_mode == "decoder_reject":
        disorder_syndrome_representative_bits = apply_section(
            eta_syndrome_bits,
            section_data,
        )
    else:
        disorder_syndrome_representative_bits = None

    if projection_mode == "decoder_reject":
        proposals = _build_decoder_reject_proposals(
            parity_check_matrix=parity_check_matrix,
            zero_syndrome_move_data=zero_syndrome_move_data,
        )
    else:
        proposals = _build_sector_preserving_proposals(
            parity_check_matrix=parity_check_matrix,
            logical_projection_masks=logical_projection_masks,
            zero_syndrome_move_data=zero_syndrome_move_data,
        )
    sector_representatives = _build_sector_representatives(
        zero_syndrome_move_data=zero_syndrome_move_data,
        logical_projection_masks=logical_projection_masks,
        parity_check_matrix=parity_check_matrix,
    )
    winding_groups = _build_even_winding_groups(
        zero_syndrome_move_data=zero_syndrome_move_data,
        lattice_size=lattice_size,
    )
    kp_target = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    kp_grid = np.linspace(
        0.0,
        kp_target,
        int(task["num_kp_grid_points"]),
        dtype=np.float64,
    )
    num_sectors = sector_representatives.shape[0]
    num_grid = kp_grid.shape[0]
    block_count = int(task["block_count"])

    mu_by_sector = np.empty((num_sectors, num_grid), dtype=np.float64)
    syndrome_mu_by_sector = np.empty((num_sectors, num_grid), dtype=np.float64)
    energy_mu_by_sector = np.empty((num_sectors, num_grid), dtype=np.float64)
    block_mu_by_sector = np.empty(
        (num_sectors, num_grid, min(block_count, int(task["num_measurements"]))),
        dtype=np.float64,
    )
    acceptance_by_sector = np.empty((num_sectors, num_grid), dtype=np.float64)
    heatbath_by_sector = np.empty((num_sectors, num_grid), dtype=np.float64)

    for sector in range(num_sectors):
        sector_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        sector_rng = np.random.default_rng(sector_seed)
        if projection_mode == "decoder_reject":
            chain_result = _run_fixed_sector_chain_decoder_reject(
                parity_check_matrix=parity_check_matrix,
                primitive_logical_masks=primitive_logical_masks,
                section_data=section_data,
                sector_representative_bits=sector_representatives[sector],
                target_sector=sector,
                measurement_error_bits=measurement_error_bits,
                disorder_syndrome_bits=eta_syndrome_bits,
                disorder_syndrome_representative_bits=(
                    disorder_syndrome_representative_bits
                ),
                proposals=proposals,
                kp_grid=kp_grid,
                kq_value=kq_value,
                num_burn_in_sweeps=int(task["num_burn_in_sweeps"]),
                num_measurements=int(task["num_measurements"]),
                num_sweeps_between_measurements=int(
                    task["num_sweeps_between_measurements"]
                ),
                block_count=block_count,
                rng=sector_rng,
                debug_checks=bool(task.get("debug_checks", False)),
            )
        else:
            chain_result = _run_fixed_sector_chain(
                parity_check_matrix=parity_check_matrix,
                logical_projection_masks=logical_projection_masks,
                sector_representative_bits=sector_representatives[sector],
                target_sector=sector,
                measurement_error_bits=measurement_error_bits,
                proposals=proposals,
                winding_groups=winding_groups,
                kp_grid=kp_grid,
                kq_value=kq_value,
                num_burn_in_sweeps=int(task["num_burn_in_sweeps"]),
                num_measurements=int(task["num_measurements"]),
                num_sweeps_between_measurements=int(
                    task["num_sweeps_between_measurements"]
                ),
                block_count=block_count,
                winding_heatbath_sweeps=int(task["winding_heatbath_sweeps"]),
                rng=sector_rng,
                use_numba=bool(task.get("use_numba", False)),
                debug_checks=bool(task.get("debug_checks", False)),
            )
        mu_by_sector[sector] = chain_result["mu"]
        syndrome_mu_by_sector[sector] = chain_result["syndrome_mu"]
        energy_mu_by_sector[sector] = chain_result["energy_mu"]
        block_mu_by_sector[sector] = chain_result["block_mu"]
        acceptance_by_sector[sector] = chain_result["acceptance_rate"]
        heatbath_by_sector[sector] = chain_result["winding_heatbath_change_rate"]

    integrals = _integrate_mu(kp_grid, mu_by_sector)
    delta_f = integrals - integrals[0]
    weights = _weights_from_delta_f(delta_f)
    q_top = _q_top_from_weights(weights)
    coarse_indices = _coarse_indices_from_fine_grid(num_grid)
    coarse_integrals = _integrate_mu(
        kp_grid[coarse_indices],
        mu_by_sector[:, coarse_indices],
    )
    coarse_delta_f = coarse_integrals - coarse_integrals[0]
    coarse_weights = _weights_from_delta_f(coarse_delta_f)
    coarse_q_top = _q_top_from_weights(coarse_weights)
    grid_tv = float(0.5 * np.sum(np.abs(weights - coarse_weights)))
    grid_q_top_abs_diff = float(abs(q_top - coarse_q_top))

    bootstrap_rng = np.random.default_rng(
        int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    )
    bootstrap = _bootstrap_ti(
        kp_grid=kp_grid,
        block_mu_by_sector=block_mu_by_sector,
        num_bootstrap=int(task["num_bootstrap"]),
        rng=bootstrap_rng,
    )
    flags = []
    if grid_tv > float(task["grid_tv_warning"]):
        flags.append("TI_GRID_TV_WARN")
    if grid_q_top_abs_diff > float(task["grid_q_top_warning"]):
        flags.append("TI_GRID_QTOP_WARN")
    if not flags:
        flags.append("PASS")

    return {
        "lattice_size": lattice_size,
        "p_value": p_value,
        "q_value": q_value,
        "projection_mode": projection_mode,
        "disorder_index": disorder_index,
        "seed": seed,
        "num_qubits": int(num_qubits),
        "num_checks": int(num_checks),
        "kp_target": kp_target,
        "kq_value": kq_value,
        "kp_grid": kp_grid,
        "measurement_error_weight": int(np.count_nonzero(measurement_error_bits)),
        "eta_weight": int(np.count_nonzero(eta_bits)),
        "observed_syndrome_weight": int(np.count_nonzero(observed_syndrome_bits)),
        "mu_by_sector": mu_by_sector,
        "syndrome_mu_by_sector": syndrome_mu_by_sector,
        "energy_mu_by_sector": energy_mu_by_sector,
        "block_mu_by_sector": block_mu_by_sector,
        "acceptance_by_sector": acceptance_by_sector,
        "winding_heatbath_change_by_sector": heatbath_by_sector,
        "integrals": integrals,
        "delta_f": delta_f,
        "weights": weights,
        "q_top": q_top,
        "coarse_delta_f": coarse_delta_f,
        "coarse_weights": coarse_weights,
        "coarse_q_top": coarse_q_top,
        "grid_tv": grid_tv,
        "grid_q_top_abs_diff": grid_q_top_abs_diff,
        "q_top_stderr": bootstrap["q_top_stderr"],
        "q_top_ci95": bootstrap["q_top_ci95"],
        "weights_stderr": bootstrap["weights_stderr"],
        "delta_f_stderr": bootstrap["delta_f_stderr"],
        "flags": ";".join(flags),
        "proposal_summary": {
            "projection_rank": proposals.get("projection_rank", 0),
            "qubit_signature_histogram": proposals[
                "qubit_signature_histogram"
            ].tolist() if "qubit_signature_histogram" in proposals else [],
            "num_projection_kernel_proposals": proposals[
                "num_projection_kernel_proposals"
            ],
            "num_single_bit_proposals": proposals.get(
                "num_single_bit_proposals",
                0,
            ),
            "num_single_zero_qubit_proposals": proposals[
                "num_single_zero_qubit_proposals"
            ],
            "num_pair_proposals": proposals["num_pair_proposals"],
            "num_contractible_proposals": proposals[
                "num_contractible_proposals"
            ],
        },
        "wall_time_seconds": float(time.perf_counter() - started_at),
    }


def _compute_exact_sector_weights_x(parity_check_matrix,
                                    logical_projection_masks,
                                    measurement_error_bits,
                                    p_value,
                                    q_value,
                                    chunk_size):
    kp_value = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    num_checks, num_qubits = parity_check_matrix.shape
    num_sectors = 1 << logical_projection_masks.shape[0]
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    projection_masks_uint8 = logical_projection_masks.astype(np.uint8)
    bit_weights = (1 << np.arange(logical_projection_masks.shape[0], dtype=np.int64))
    log_z = np.full(num_sectors, -np.inf, dtype=np.float64)

    for chain_bits_chunk in _iter_chain_bit_chunks(num_qubits, int(chunk_size)):
        chain_uint8 = chain_bits_chunk.astype(np.uint8)
        data_weights = np.count_nonzero(chain_bits_chunk, axis=1)
        syndrome_bits = (chain_uint8 @ parity_check_matrix_uint8.T) % 2
        syndrome_bits = syndrome_bits.astype(bool) ^ measurement_error_bits[None, :]
        syndrome_weights = np.count_nonzero(syndrome_bits, axis=1)
        logical_bits = (chain_uint8 @ projection_masks_uint8.T) % 2
        sector_indices = logical_bits.astype(np.int64) @ bit_weights
        log_weights = (
            -kp_value * data_weights.astype(np.float64)
            -kq_value * syndrome_weights.astype(np.float64)
        )
        for sector in range(num_sectors):
            sector_log_weights = log_weights[sector_indices == sector]
            if sector_log_weights.size == 0:
                continue
            log_z[sector] = np.logaddexp(
                log_z[sector],
                _logsumexp(sector_log_weights),
            )

    log_z_total = _logsumexp(log_z)
    weights = np.exp(log_z - log_z_total)
    delta_f = -(log_z - log_z[0])
    return {
        "weights": weights,
        "delta_f": delta_f,
        "q_top": _q_top_from_weights(weights),
        "log_z": log_z,
    }


def _compute_exact_sector_weights_decoder(parity_check_matrix,
                                          primitive_logical_masks,
                                          section_data,
                                          disorder_syndrome_bits,
                                          measurement_error_bits,
                                          p_value,
                                          q_value,
                                          chunk_size):
    kp_value = _compute_k(p_value)
    kq_value = _compute_k(q_value)
    num_checks, num_qubits = parity_check_matrix.shape
    num_sectors = 1 << primitive_logical_masks.shape[0]
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    log_z = np.full(num_sectors, -np.inf, dtype=np.float64)
    if num_checks > 26:
        raise ValueError(
            "decoder exact benchmark only supports num_checks <= 26; "
            "use L=2 or add a sparse syndrome cache implementation"
        )
    syndrome_bit_weights = (1 << np.arange(num_checks, dtype=np.int64))
    disorder_syndrome_bits = np.asarray(disorder_syndrome_bits, dtype=bool)
    disorder_syndrome_representative_bits = apply_section(
        disorder_syndrome_bits,
        section_data,
    )
    disorder_syndrome_index = int(
        disorder_syndrome_bits.astype(np.int64) @ syndrome_bit_weights
    )
    disorder_signature = _compute_signature(
        disorder_syndrome_representative_bits,
        primitive_logical_masks,
    )
    signature_lookup = np.full(1 << num_checks, -1, dtype=np.int8)

    for chain_bits_chunk in _iter_chain_bit_chunks(num_qubits, int(chunk_size)):
        chain_uint8 = chain_bits_chunk.astype(np.uint8)
        data_weights = np.count_nonzero(chain_bits_chunk, axis=1)
        chain_syndrome_bits_chunk = (
            chain_uint8 @ parity_check_matrix_uint8.T
        ) % 2
        syndrome_indices = (
            chain_syndrome_bits_chunk.astype(np.int64) @ syndrome_bit_weights
        )
        missing_indices = np.unique(
            syndrome_indices[signature_lookup[
                np.bitwise_xor(
                    syndrome_indices,
                    disorder_syndrome_index,
                )
            ] < 0]
        )
        c_syndrome_indices = np.bitwise_xor(
            missing_indices,
            disorder_syndrome_index,
        )
        for syndrome_index in c_syndrome_indices:
            syndrome_bits = (
                (
                    int(syndrome_index)
                    >> np.arange(num_checks, dtype=np.int64)
                ) & 1
            ).astype(bool)
            representative_bits = apply_section(syndrome_bits, section_data)
            signature_lookup[int(syndrome_index)] = np.int8(
                _compute_signature(representative_bits, primitive_logical_masks)
            )
        syndrome_term_bits = (
            chain_syndrome_bits_chunk.astype(bool)
            ^ measurement_error_bits[None, :]
        )
        syndrome_weights = np.count_nonzero(syndrome_term_bits, axis=1)
        log_weights = (
            -kp_value * data_weights.astype(np.float64)
            -kq_value * syndrome_weights.astype(np.float64)
        )
        raw_logical_bits = (
            chain_uint8 @ primitive_logical_masks.astype(np.uint8).T
        ) % 2
        raw_logical_indices = raw_logical_bits.astype(np.int64) @ (
            1 << np.arange(primitive_logical_masks.shape[0], dtype=np.int64)
        )
        c_syndrome_indices = np.bitwise_xor(
            syndrome_indices,
            disorder_syndrome_index,
        )
        sector_indices = np.bitwise_xor(
            np.bitwise_xor(
                raw_logical_indices,
                signature_lookup[c_syndrome_indices].astype(np.int64),
            ),
            int(disorder_signature),
        )
        for sector in range(num_sectors):
            sector_log_weights = log_weights[sector_indices == sector]
            if sector_log_weights.size == 0:
                continue
            log_z[sector] = np.logaddexp(
                log_z[sector],
                _logsumexp(sector_log_weights),
            )

    log_z_total = _logsumexp(log_z)
    weights = np.exp(log_z - log_z_total)
    delta_f = -(log_z - log_z[0])
    return {
        "weights": weights,
        "delta_f": delta_f,
        "q_top": _q_top_from_weights(weights),
        "log_z": log_z,
    }


def _run_exact_benchmark(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p_value = float(args.p)
    q_values = _parse_float_list(args.q_values)
    lattice_size = int(args.lattice_size)
    parity_check_matrix, primitive_logical_masks = build_toric_code_by_family(
        args.code_family,
        lattice_size,
    )
    logical_projection_masks = _build_logical_projection_masks(
        parity_check_matrix=parity_check_matrix,
        primitive_logical_masks=primitive_logical_masks,
    )
    if str(args.projection_mode) == "decoder_reject":
        section_data = build_syndrome_representative_section(
            parity_check_matrix,
        )
    elif str(args.projection_mode) == "linear":
        section_data = None
    else:
        raise ValueError("projection_mode must be linear or decoder_reject")
    num_checks, num_qubits = parity_check_matrix.shape
    if num_qubits > int(args.max_exact_qubits):
        raise ValueError(
            f"exact benchmark refuses n={num_qubits}; raise --max-exact-qubits if intended"
        )

    records = []
    for q_value in q_values:
        for disorder_index in range(int(args.num_disorder_samples)):
            if bool(args.common_disorder_across_q):
                seed = (
                    int(args.seed_base)
                    + 1000003 * int(lattice_size)
                    + int(disorder_index)
                )
            else:
                seed = (
                    int(args.seed_base)
                    + 1000003 * int(lattice_size)
                    + 1009 * int(round(10000 * float(q_value)))
                    + int(disorder_index)
                )
            rng = np.random.default_rng(seed)
            eta_bits = rng.random(num_qubits) < p_value
            measurement_error_bits = rng.random(num_checks) < q_value
            eta_syndrome_bits = (
                parity_check_matrix.astype(np.uint8) @ eta_bits.astype(np.uint8)
            ) % 2
            observed_syndrome_bits = (
                eta_syndrome_bits.astype(bool) ^ measurement_error_bits
            )
            if str(args.projection_mode) == "decoder_reject":
                exact = _compute_exact_sector_weights_decoder(
                    parity_check_matrix=parity_check_matrix,
                    primitive_logical_masks=primitive_logical_masks,
                    section_data=section_data,
                    disorder_syndrome_bits=eta_syndrome_bits.astype(bool),
                    measurement_error_bits=measurement_error_bits,
                    p_value=p_value,
                    q_value=q_value,
                    chunk_size=int(args.exact_chunk_size),
                )
            else:
                exact = _compute_exact_sector_weights_x(
                    parity_check_matrix=parity_check_matrix,
                    logical_projection_masks=logical_projection_masks,
                    measurement_error_bits=measurement_error_bits,
                    p_value=p_value,
                    q_value=q_value,
                    chunk_size=int(args.exact_chunk_size),
                )
            records.append({
                "q_value": float(q_value),
                "disorder_index": int(disorder_index),
                "seed": int(seed),
                "weights": exact["weights"].tolist(),
                "delta_f": exact["delta_f"].tolist(),
                "q_top": float(exact["q_top"]),
            })
    payload = {
        "mode": "exact_benchmark_x_sector",
        "sector_observable": "corrected_c_eta_section",
        "projection_mode": str(args.projection_mode),
        "code_family": args.code_family,
        "lattice_size": lattice_size,
        "p_value": p_value,
        "q_values": q_values,
        "num_disorder_samples": int(args.num_disorder_samples),
        "seed_base": int(args.seed_base),
        "common_disorder_across_q": bool(args.common_disorder_across_q),
        "records": records,
    }
    json_path = output_dir / "exact_sector_weights.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def _build_tasks(args):
    lattice_sizes = _parse_int_list(args.lattice_sizes)
    q_values = _parse_float_list(args.q_values)
    tasks = []
    for lattice_size in lattice_sizes:
        for q_value in q_values:
            for disorder_index in range(int(args.num_disorder_samples)):
                if bool(args.common_disorder_across_q):
                    seed = (
                        int(args.seed_base)
                        + 1000003 * int(lattice_size)
                        + int(disorder_index)
                    )
                else:
                    seed = (
                        int(args.seed_base)
                        + 1000003 * int(lattice_size)
                        + 1009 * int(round(10000 * float(q_value)))
                        + int(disorder_index)
                    )
                tasks.append({
                    "code_family": args.code_family,
                    "projection_mode": str(getattr(
                        args,
                        "projection_mode",
                        "linear",
                    )),
                    "lattice_size": int(lattice_size),
                    "p_value": float(args.p),
                    "q_value": float(q_value),
                    "disorder_index": int(disorder_index),
                    "seed": int(seed),
                    "num_kp_grid_points": int(args.num_kp_grid_points),
                    "num_burn_in_sweeps": int(getattr(
                        args,
                        "num_burn_in_sweeps",
                        0,
                    )),
                    "num_measurements": int(getattr(
                        args,
                        "num_measurements",
                        0,
                    )),
                    "num_sweeps_between_measurements": int(
                        getattr(args, "num_sweeps_between_measurements", 1)
                    ),
                    "block_count": int(getattr(args, "block_count", 1)),
                    "num_bootstrap": int(args.num_bootstrap),
                    "winding_heatbath_sweeps": int(getattr(
                        args,
                        "winding_heatbath_sweeps",
                        0,
                    )),
                    "use_numba": bool(getattr(args, "use_numba", False)),
                    "grid_tv_warning": float(getattr(args, "grid_tv_warning", 0.02)),
                    "grid_q_top_warning": float(getattr(args, "grid_q_top_warning", 0.02)),
                    "debug_checks": bool(getattr(args, "debug_checks", False)),
                })
    return tasks, lattice_sizes, q_values


def _aggregate_results(results, lattice_sizes, q_values, num_disorder_samples):
    lattice_sizes = list(lattice_sizes)
    q_values = list(q_values)
    shape = (len(lattice_sizes), len(q_values), int(num_disorder_samples))
    q_top = np.full(shape, np.nan, dtype=np.float64)
    q_top_stderr = np.full(shape, np.nan, dtype=np.float64)
    q_top_ci95 = np.full(shape + (2,), np.nan, dtype=np.float64)
    grid_tv = np.full(shape, np.nan, dtype=np.float64)
    grid_q_top_abs_diff = np.full(shape, np.nan, dtype=np.float64)
    weights = np.full(shape + (8,), np.nan, dtype=np.float64)
    weights_stderr = np.full(shape + (8,), np.nan, dtype=np.float64)
    delta_f = np.full(shape + (8,), np.nan, dtype=np.float64)
    delta_f_stderr = np.full(shape + (8,), np.nan, dtype=np.float64)
    flags = np.full(shape, "MISSING", dtype="<U128")
    wall = np.full(shape, np.nan, dtype=np.float64)

    l_index = {int(value): index for index, value in enumerate(lattice_sizes)}
    q_index = {float(value): index for index, value in enumerate(q_values)}
    for result in results:
        li = l_index[int(result["lattice_size"])]
        qi = q_index[float(result["q_value"])]
        di = int(result["disorder_index"])
        q_top[li, qi, di] = float(result["q_top"])
        q_top_stderr[li, qi, di] = float(result["q_top_stderr"])
        q_top_ci95[li, qi, di] = np.asarray(
            result["q_top_ci95"],
            dtype=np.float64,
        )
        grid_tv[li, qi, di] = float(result["grid_tv"])
        grid_q_top_abs_diff[li, qi, di] = float(result["grid_q_top_abs_diff"])
        weights[li, qi, di] = np.asarray(result["weights"], dtype=np.float64)
        weights_stderr[li, qi, di] = np.asarray(
            result["weights_stderr"],
            dtype=np.float64,
        )
        delta_f[li, qi, di] = np.asarray(result["delta_f"], dtype=np.float64)
        delta_f_stderr[li, qi, di] = np.asarray(
            result["delta_f_stderr"],
            dtype=np.float64,
        )
        flags[li, qi, di] = str(result["flags"])
        wall[li, qi, di] = float(result["wall_time_seconds"])

    mean_q_top = np.nanmean(q_top, axis=2)
    if int(num_disorder_samples) > 1:
        disorder_sem = np.nanstd(q_top, axis=2, ddof=1) / math.sqrt(
            float(num_disorder_samples)
        )
    else:
        disorder_sem = np.zeros_like(mean_q_top)
    mcmc_sem = np.sqrt(np.nanmean(q_top_stderr ** 2, axis=2))
    total_sem = np.sqrt(disorder_sem ** 2 + mcmc_sem ** 2)
    pass_fraction = np.empty(mean_q_top.shape, dtype=np.float64)
    for li in range(len(lattice_sizes)):
        for qi in range(len(q_values)):
            pass_fraction[li, qi] = float(
                np.mean(flags[li, qi] == "PASS")
            )
    return {
        "q_top_per_disorder": q_top,
        "q_top_stderr_per_disorder": q_top_stderr,
        "q_top_ci95_per_disorder": q_top_ci95,
        "grid_tv_per_disorder": grid_tv,
        "grid_q_top_abs_diff_per_disorder": grid_q_top_abs_diff,
        "weights_per_disorder": weights,
        "weights_stderr_per_disorder": weights_stderr,
        "delta_f_per_disorder": delta_f,
        "delta_f_stderr_per_disorder": delta_f_stderr,
        "flags_per_disorder": flags,
        "wall_time_seconds_per_disorder": wall,
        "mean_q_top": mean_q_top,
        "disorder_sem_q_top": disorder_sem,
        "mcmc_sem_q_top": mcmc_sem,
        "total_sem_q_top": total_sem,
        "pass_fraction": pass_fraction,
    }


def _aggregate_ais_results(results, lattice_sizes, q_values, num_disorder_samples):
    lattice_sizes = list(lattice_sizes)
    q_values = list(q_values)
    shape = (len(lattice_sizes), len(q_values), int(num_disorder_samples))
    q_top = np.full(shape, np.nan, dtype=np.float64)
    q_top_stderr = np.full(shape, np.nan, dtype=np.float64)
    q_top_ci95 = np.full(shape + (2,), np.nan, dtype=np.float64)
    weights = np.full(shape + (8,), np.nan, dtype=np.float64)
    weights_stderr = np.full(shape + (8,), np.nan, dtype=np.float64)
    delta_f = np.full(shape + (8,), np.nan, dtype=np.float64)
    delta_f_stderr = np.full(shape + (8,), np.nan, dtype=np.float64)
    ais_ess = np.full(shape, np.nan, dtype=np.float64)
    ais_ess_fraction = np.full(shape, np.nan, dtype=np.float64)
    sector_sample_counts = np.full(shape + (8,), -1, dtype=np.int64)
    flags = np.full(shape, "MISSING", dtype="<U128")
    wall = np.full(shape, np.nan, dtype=np.float64)
    num_particles = np.full(shape, 0, dtype=np.int64)

    l_index = {int(value): index for index, value in enumerate(lattice_sizes)}
    q_index = {float(value): index for index, value in enumerate(q_values)}
    grouped_results = {}
    for result in results:
        key = (
            int(result["lattice_size"]),
            float(result["q_value"]),
            int(result["disorder_index"]),
        )
        grouped_results.setdefault(key, []).append(result)

    for key, group in grouped_results.items():
        lattice_size, q_value, disorder_index = key
        li = l_index[lattice_size]
        qi = q_index[q_value]
        di = int(disorder_index)
        group = sorted(group, key=lambda item: int(item.get("replica_index", 0)))
        num_sectors = np.asarray(group[0]["weights"], dtype=np.float64).shape[0]
        if all("log_weight_matrix" in item for item in group):
            group_log_weight_matrix = np.concatenate([
                np.asarray(item["log_weight_matrix"], dtype=np.float64)
                for item in group
            ], axis=0)
            group_sector_indices = np.concatenate([
                np.asarray(item["sector_indices"], dtype=np.int64)
                for item in group
            ])
            group_weights, group_sector_log_sums, group_total_log_sum = (
                _ais_weights_from_log_matrix(
                    log_weight_matrix=group_log_weight_matrix,
                )
            )
            del group_sector_log_sums
            group_row_log_sums = np.asarray([
                _logsumexp(group_log_weight_matrix[row_index])
                for row_index in range(group_log_weight_matrix.shape[0])
            ], dtype=np.float64)
            log_sum_w2 = _logsumexp(2.0 * group_row_log_sums)
            group_ess = float(
                math.exp(2.0 * group_total_log_sum - log_sum_w2)
            )
            bootstrap = _bootstrap_ais_matrix(
                log_weight_matrix=group_log_weight_matrix,
                num_bootstrap=int(group[0].get("num_bootstrap", 200)),
                rng=np.random.default_rng(
                    int(min(item["sample_seed"] for item in group)) + 99173
                ),
            )
            group_delta_f = np.full_like(group_weights, np.inf, dtype=np.float64)
            if group_weights[0] > 0.0:
                positive_weight_mask = group_weights > 0.0
                group_delta_f[positive_weight_mask] = -np.log(
                    group_weights[positive_weight_mask] / group_weights[0]
                )
            group_num_particles = int(group_log_weight_matrix.shape[0])
            group_flags = []
            if group_ess < float(group[0].get("min_ais_ess", 100.0)):
                group_flags.append("AIS_LOW_ESS_WARN")
            if (
                group_ess / float(group_num_particles)
                < float(group[0].get("min_ais_ess_fraction", 0.05))
            ):
                group_flags.append("AIS_LOW_ESS_FRACTION_WARN")
            if not group_flags:
                group_flags.append("PASS")

            q_top[li, qi, di] = _q_top_from_weights(group_weights)
            q_top_stderr[li, qi, di] = float(bootstrap["q_top_stderr"])
            q_top_ci95[li, qi, di] = np.asarray(
                bootstrap["q_top_ci95"],
                dtype=np.float64,
            )
            weights[li, qi, di] = group_weights
            weights_stderr[li, qi, di] = np.asarray(
                bootstrap["weights_stderr"],
                dtype=np.float64,
            )
            delta_f[li, qi, di] = group_delta_f
            delta_f_stderr[li, qi, di] = np.asarray(
                bootstrap["delta_f_stderr"],
                dtype=np.float64,
            )
            ais_ess[li, qi, di] = group_ess
            ais_ess_fraction[li, qi, di] = group_ess / float(group_num_particles)
            sector_sample_counts[li, qi, di] = np.bincount(
                group_sector_indices,
                minlength=num_sectors,
            )
            flags[li, qi, di] = ";".join(group_flags)
            wall[li, qi, di] = float(sum(item["wall_time_seconds"] for item in group))
            num_particles[li, qi, di] = group_num_particles
        else:
            result = group[0]
            q_top[li, qi, di] = float(result["q_top"])
            q_top_stderr[li, qi, di] = float(result["q_top_stderr"])
            q_top_ci95[li, qi, di] = np.asarray(
                result["q_top_ci95"],
                dtype=np.float64,
            )
            weights[li, qi, di] = np.asarray(result["weights"], dtype=np.float64)
            weights_stderr[li, qi, di] = np.asarray(
                result["weights_stderr"],
                dtype=np.float64,
            )
            delta_f[li, qi, di] = np.asarray(result["delta_f"], dtype=np.float64)
            delta_f_stderr[li, qi, di] = np.asarray(
                result.get("delta_f_stderr", np.full(8, np.nan)),
                dtype=np.float64,
            )
            ais_ess[li, qi, di] = float(result["ais_ess"])
            ais_ess_fraction[li, qi, di] = float(result["ais_ess_fraction"])
            sector_sample_counts[li, qi, di] = np.asarray(
                result["sector_sample_counts"],
                dtype=np.int64,
            )
            flags[li, qi, di] = str(result["flags"])
            wall[li, qi, di] = float(result["wall_time_seconds"])
            num_particles[li, qi, di] = int(result.get("num_ais_particles", 0))

    mean_q_top = np.nanmean(q_top, axis=2)
    if int(num_disorder_samples) > 1:
        disorder_sem = np.nanstd(q_top, axis=2, ddof=1) / math.sqrt(
            float(num_disorder_samples)
        )
    else:
        disorder_sem = np.zeros_like(mean_q_top)
    ais_sem = np.sqrt(np.nanmean(q_top_stderr ** 2, axis=2))
    total_sem = np.sqrt(disorder_sem ** 2 + ais_sem ** 2)
    pass_fraction = np.empty(mean_q_top.shape, dtype=np.float64)
    for li in range(len(lattice_sizes)):
        for qi in range(len(q_values)):
            pass_fraction[li, qi] = float(np.mean(flags[li, qi] == "PASS"))
    return {
        "q_top_per_disorder": q_top,
        "q_top_stderr_per_disorder": q_top_stderr,
        "q_top_ci95_per_disorder": q_top_ci95,
        "weights_per_disorder": weights,
        "weights_stderr_per_disorder": weights_stderr,
        "delta_f_per_disorder": delta_f,
        "delta_f_stderr_per_disorder": delta_f_stderr,
        "ais_ess_per_disorder": ais_ess,
        "ais_ess_fraction_per_disorder": ais_ess_fraction,
        "sector_sample_counts_per_disorder": sector_sample_counts,
        "flags_per_disorder": flags,
        "wall_time_seconds_per_disorder": wall,
        "num_ais_particles_per_disorder": num_particles,
        "mean_q_top": mean_q_top,
        "disorder_sem_q_top": disorder_sem,
        "ais_sem_q_top": ais_sem,
        "total_sem_q_top": total_sem,
        "pass_fraction": pass_fraction,
    }


def _write_summary_markdown(output_dir, args, lattice_sizes, q_values, aggregate):
    lines = []
    lines.append("# exp37 sector-TI summary")
    lines.append("")
    lines.append(f"p = {float(args.p):.6g}")
    lines.append(
        "TI config: "
        f"grid={int(args.num_kp_grid_points)}, "
        f"burn={int(args.num_burn_in_sweeps)}, "
        f"measurements={int(args.num_measurements)}, "
        f"stride={int(args.num_sweeps_between_measurements)}, "
        f"blocks={int(args.block_count)}, "
        f"bootstrap={int(args.num_bootstrap)}, "
        f"even_winding_heatbath={int(args.winding_heatbath_sweeps)}"
    )
    lines.append("")
    lines.append("| L | q | mean q_top | total SEM | disorder SEM | MCMC SEM | pass fraction | max grid TV | max |dq_top grid| |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            max_grid_tv = float(np.nanmax(
                aggregate["grid_tv_per_disorder"][li, qi]
            ))
            max_grid_dq = float(np.nanmax(
                aggregate["grid_q_top_abs_diff_per_disorder"][li, qi]
            ))
            lines.append(
                f"| {int(lattice_size)} | {float(q_value):.3f} | "
                f"{aggregate['mean_q_top'][li, qi]:.6f} | "
                f"{aggregate['total_sem_q_top'][li, qi]:.6f} | "
                f"{aggregate['disorder_sem_q_top'][li, qi]:.6f} | "
                f"{aggregate['mcmc_sem_q_top'][li, qi]:.6f} | "
                f"{aggregate['pass_fraction'][li, qi]:.3f} | "
                f"{max_grid_tv:.6f} | {max_grid_dq:.6f} |"
            )
    summary_path = output_dir / "sector_ti_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _write_ais_summary_markdown(output_dir, args, lattice_sizes, q_values, aggregate):
    lines = []
    lines.append("# exp37 unrestricted AIS summary")
    lines.append("")
    lines.append(f"p = {float(args.p):.6g}")
    lines.append(
        "AIS config: "
        f"estimator={str(args.ais_estimator)}, "
        f"grid={int(args.num_kp_grid_points)}, "
        f"particles={int(args.num_ais_particles)}, "
        f"replicates={int(args.num_ais_replicates)}, "
        f"initial_burn={int(args.num_initial_burn_in_sweeps)}, "
        f"transition_sweeps={int(args.num_transition_sweeps)}, "
        f"logical_heatbath={int(args.logical_heatbath_sweeps)}, "
        f"bootstrap={int(args.num_bootstrap)}"
    )
    lines.append("")
    lines.append("| L | q | mean q_top | total SEM | disorder SEM | AIS SEM | pass fraction | min ESS frac | min ESS |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for li, lattice_size in enumerate(lattice_sizes):
        for qi, q_value in enumerate(q_values):
            min_ess_frac = float(np.nanmin(
                aggregate["ais_ess_fraction_per_disorder"][li, qi]
            ))
            min_ess = float(np.nanmin(aggregate["ais_ess_per_disorder"][li, qi]))
            lines.append(
                f"| {int(lattice_size)} | {float(q_value):.3f} | "
                f"{aggregate['mean_q_top'][li, qi]:.6f} | "
                f"{aggregate['total_sem_q_top'][li, qi]:.6f} | "
                f"{aggregate['disorder_sem_q_top'][li, qi]:.6f} | "
                f"{aggregate['ais_sem_q_top'][li, qi]:.6f} | "
                f"{aggregate['pass_fraction'][li, qi]:.3f} | "
                f"{min_ess_frac:.6f} | {min_ess:.2f} |"
            )
    summary_path = output_dir / "ais_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _run_ti(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, lattice_sizes, q_values = _build_tasks(args)
    started_at = time.perf_counter()
    results = []
    if int(args.num_workers) == 1:
        for task_index, task in enumerate(tasks):
            result = _run_single_ti_task(task)
            results.append(result)
            print(
                f"[{task_index + 1}/{len(tasks)}] "
                f"L={result['lattice_size']} q={result['q_value']:.3f} "
                f"d={result['disorder_index']} q_top={result['q_top']:.6f} "
                f"flags={result['flags']} wall={result['wall_time_seconds']:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            future_to_task = {
                executor.submit(_run_single_ti_task, task): task
                for task in tasks
            }
            completed = 0
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] "
                    f"L={result['lattice_size']} q={result['q_value']:.3f} "
                    f"d={result['disorder_index']} q_top={result['q_top']:.6f} "
                    f"flags={result['flags']} wall={result['wall_time_seconds']:.1f}s",
                    flush=True,
                )

    aggregate = _aggregate_results(
        results=results,
        lattice_sizes=lattice_sizes,
        q_values=q_values,
        num_disorder_samples=int(args.num_disorder_samples),
    )
    manifest = {
        "mode": "sector_ti",
        "sector_observable": "corrected_c_eta_section",
        "code_family": args.code_family,
        "projection_mode": str(args.projection_mode),
        "lattice_sizes": lattice_sizes,
        "p_value": float(args.p),
        "q_values": q_values,
        "num_disorder_samples": int(args.num_disorder_samples),
        "seed_base": int(args.seed_base),
        "common_disorder_across_q": bool(args.common_disorder_across_q),
        "num_kp_grid_points": int(args.num_kp_grid_points),
        "num_burn_in_sweeps": int(args.num_burn_in_sweeps),
        "num_measurements": int(args.num_measurements),
        "num_sweeps_between_measurements": int(args.num_sweeps_between_measurements),
        "block_count": int(args.block_count),
        "num_bootstrap": int(args.num_bootstrap),
        "winding_heatbath_sweeps": int(args.winding_heatbath_sweeps),
        "use_numba": bool(args.use_numba),
        "numba_available": bool(_numba_run_fixed_sector_chain is not None),
        "grid_tv_warning": float(args.grid_tv_warning),
        "grid_q_top_warning": float(args.grid_q_top_warning),
        "total_wall_time_seconds": float(time.perf_counter() - started_at),
    }
    npz_path = output_dir / "sector_ti_results.npz"
    np.savez_compressed(
        npz_path,
        manifest_json=np.array(json.dumps(manifest, indent=2)),
        lattice_size_list=np.asarray(lattice_sizes, dtype=np.int64),
        q_values=np.asarray(q_values, dtype=np.float64),
        p_value=np.float64(args.p),
        **aggregate,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path = _write_summary_markdown(
        output_dir=output_dir,
        args=args,
        lattice_sizes=lattice_sizes,
        q_values=q_values,
        aggregate=aggregate,
    )
    return npz_path, summary_path


def _run_ais(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, lattice_sizes, q_values = _build_tasks(args)
    if int(args.num_ais_replicates) > 1:
        replicated_tasks = []
        for task in tasks:
            disorder_seed = int(task["seed"])
            q_seed_offset = 1009 * int(round(10000 * float(task["q_value"])))
            for replica_index in range(int(args.num_ais_replicates)):
                replica_task = dict(task)
                replica_task["replica_index"] = int(replica_index)
                replica_task["disorder_seed"] = disorder_seed
                replica_task["sample_seed"] = int(
                    disorder_seed
                    + q_seed_offset
                    + 1000000007 * int(replica_index)
                )
                replicated_tasks.append(replica_task)
        tasks = replicated_tasks
    for task in tasks:
        task["ais_estimator"] = str(args.ais_estimator)
        task["num_ais_particles"] = int(args.num_ais_particles)
        task["num_initial_burn_in_sweeps"] = int(args.num_initial_burn_in_sweeps)
        task["num_transition_sweeps"] = int(args.num_transition_sweeps)
        task["logical_heatbath_sweeps"] = int(args.logical_heatbath_sweeps)
        task["min_ais_ess"] = float(args.min_ais_ess)
        task["min_ais_ess_fraction"] = float(args.min_ais_ess_fraction)
    started_at = time.perf_counter()
    results = []
    if int(args.num_workers) == 1:
        for task_index, task in enumerate(tasks):
            result = _run_single_ais_task(task)
            results.append(result)
            print(
                f"[{task_index + 1}/{len(tasks)}] "
                f"L={result['lattice_size']} q={result['q_value']:.3f} "
                f"d={result['disorder_index']} q_top={result['q_top']:.6f} "
                f"r={result['replica_index']} "
                f"ESS={result['ais_ess']:.1f} "
                f"flags={result['flags']} wall={result['wall_time_seconds']:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            future_to_task = {
                executor.submit(_run_single_ais_task, task): task
                for task in tasks
            }
            completed = 0
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] "
                    f"L={result['lattice_size']} q={result['q_value']:.3f} "
                    f"d={result['disorder_index']} q_top={result['q_top']:.6f} "
                    f"r={result['replica_index']} "
                    f"ESS={result['ais_ess']:.1f} "
                    f"flags={result['flags']} wall={result['wall_time_seconds']:.1f}s",
                    flush=True,
                )

    aggregate = _aggregate_ais_results(
        results=results,
        lattice_sizes=lattice_sizes,
        q_values=q_values,
        num_disorder_samples=int(args.num_disorder_samples),
    )
    manifest = {
        "mode": "unrestricted_ais_decoder_sector",
        "sector_observable": "corrected_c_eta_section",
        "ais_estimator": str(args.ais_estimator),
        "code_family": args.code_family,
        "projection_mode": "decoder_reject",
        "lattice_sizes": lattice_sizes,
        "p_value": float(args.p),
        "q_values": q_values,
        "num_disorder_samples": int(args.num_disorder_samples),
        "seed_base": int(args.seed_base),
        "common_disorder_across_q": bool(args.common_disorder_across_q),
        "num_kp_grid_points": int(args.num_kp_grid_points),
        "num_ais_particles": int(args.num_ais_particles),
        "num_ais_replicates": int(args.num_ais_replicates),
        "num_initial_burn_in_sweeps": int(args.num_initial_burn_in_sweeps),
        "num_transition_sweeps": int(args.num_transition_sweeps),
        "logical_heatbath_sweeps": int(args.logical_heatbath_sweeps),
        "num_bootstrap": int(args.num_bootstrap),
        "min_ais_ess": float(args.min_ais_ess),
        "min_ais_ess_fraction": float(args.min_ais_ess_fraction),
        "total_wall_time_seconds": float(time.perf_counter() - started_at),
    }
    npz_path = output_dir / "ais_results.npz"
    np.savez_compressed(
        npz_path,
        manifest_json=np.array(json.dumps(manifest, indent=2)),
        lattice_size_list=np.asarray(lattice_sizes, dtype=np.int64),
        q_values=np.asarray(q_values, dtype=np.float64),
        p_value=np.float64(args.p),
        **aggregate,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path = _write_ais_summary_markdown(
        output_dir=output_dir,
        args=args,
        lattice_sizes=lattice_sizes,
        q_values=q_values,
        aggregate=aggregate,
    )
    return npz_path, summary_path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--code-family", default="3d_toric")
    run_parser.add_argument(
        "--projection-mode",
        default="linear",
        choices=("linear", "decoder_reject"),
    )
    run_parser.add_argument("--lattice-sizes", default="3,4,5")
    run_parser.add_argument("--p", type=float, default=0.05)
    run_parser.add_argument("--q-values", default="0.08:0.23:0.01")
    run_parser.add_argument("--num-disorder-samples", type=int, default=4)
    run_parser.add_argument("--seed-base", type=int, default=637000)
    run_parser.add_argument("--common-disorder-across-q", action="store_true")
    run_parser.add_argument("--num-kp-grid-points", type=int, default=33)
    run_parser.add_argument("--num-burn-in-sweeps", type=int, default=64)
    run_parser.add_argument("--num-measurements", type=int, default=128)
    run_parser.add_argument("--num-sweeps-between-measurements", type=int, default=2)
    run_parser.add_argument("--block-count", type=int, default=8)
    run_parser.add_argument("--num-bootstrap", type=int, default=200)
    run_parser.add_argument("--winding-heatbath-sweeps", type=int, default=1)
    run_parser.add_argument("--use-numba", action="store_true")
    run_parser.add_argument("--grid-tv-warning", type=float, default=0.02)
    run_parser.add_argument("--grid-q-top-warning", type=float, default=0.02)
    run_parser.add_argument("--num-workers", type=int, default=1)
    run_parser.add_argument("--debug-checks", action="store_true")
    run_parser.set_defaults(func=_run_ti)

    ais_parser = subparsers.add_parser("ais")
    ais_parser.add_argument("--output-dir", required=True)
    ais_parser.add_argument("--code-family", default="3d_toric")
    ais_parser.add_argument("--lattice-sizes", default="3,4,5")
    ais_parser.add_argument("--p", type=float, default=0.05)
    ais_parser.add_argument("--q-values", default="0.08:0.23:0.01")
    ais_parser.add_argument("--num-disorder-samples", type=int, default=4)
    ais_parser.add_argument("--seed-base", type=int, default=637000)
    ais_parser.add_argument("--common-disorder-across-q", action="store_true")
    ais_parser.add_argument(
        "--ais-estimator",
        choices=["direct", "flip_reweight"],
        default="direct",
    )
    ais_parser.add_argument("--num-kp-grid-points", type=int, default=65)
    ais_parser.add_argument("--num-ais-particles", type=int, default=512)
    ais_parser.add_argument("--num-ais-replicates", type=int, default=1)
    ais_parser.add_argument("--num-initial-burn-in-sweeps", type=int, default=128)
    ais_parser.add_argument("--num-transition-sweeps", type=int, default=4)
    ais_parser.add_argument("--logical-heatbath-sweeps", type=int, default=0)
    ais_parser.add_argument("--num-bootstrap", type=int, default=200)
    ais_parser.add_argument("--min-ais-ess", type=float, default=100.0)
    ais_parser.add_argument("--min-ais-ess-fraction", type=float, default=0.05)
    ais_parser.add_argument("--num-workers", type=int, default=1)
    ais_parser.set_defaults(func=_run_ais)

    exact_parser = subparsers.add_parser("exact-benchmark")
    exact_parser.add_argument("--output-dir", required=True)
    exact_parser.add_argument("--code-family", default="3d_toric")
    exact_parser.add_argument(
        "--projection-mode",
        default="linear",
        choices=("linear", "decoder_reject"),
    )
    exact_parser.add_argument("--lattice-size", type=int, default=2)
    exact_parser.add_argument("--p", type=float, default=0.05)
    exact_parser.add_argument("--q-values", default="0.08,0.15,0.23")
    exact_parser.add_argument("--num-disorder-samples", type=int, default=2)
    exact_parser.add_argument("--seed-base", type=int, default=637000)
    exact_parser.add_argument("--common-disorder-across-q", action="store_true")
    exact_parser.add_argument("--exact-chunk-size", type=int, default=262144)
    exact_parser.add_argument("--max-exact-qubits", type=int, default=24)
    exact_parser.set_defaults(func=_run_exact_benchmark)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    result = args.func(args)
    print(result)


if __name__ == "__main__":
    main()
