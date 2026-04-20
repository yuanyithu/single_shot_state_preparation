import numpy as np

from build_toric_code_examples import build_2d_toric_code
from linear_section import build_linear_section
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)


def draw_disorder_sample(
        num_checks,
        num_qubits,
        syndrome_error_probability,
        data_error_probability,
        rng):
    """
    采样一个 disorder 样本。

    输入：
      num_checks: int，check 总数
      num_qubits: int，qubit 总数
      syndrome_error_probability: float，shape 标量
      data_error_probability: float，shape 标量
      rng: np.random.Generator

    输出：
      observed_syndrome_bits: np.ndarray，shape (num_checks,)，dtype=bool
      disorder_data_error_bits: np.ndarray，shape (num_qubits,)，dtype=bool
    """
    observed_syndrome_bits = (
        rng.random(num_checks) < syndrome_error_probability
    )
    disorder_data_error_bits = (
        rng.random(num_qubits) < data_error_probability
    )
    return observed_syndrome_bits, disorder_data_error_bits


def initialize_mcmc_state(
        num_qubits,
        observed_syndrome_bits,
        disorder_data_error_bits,
        parity_check_matrix,
        rng):
    """
    初始化 MCMC 动态状态。

    输入：
      num_qubits: int
      observed_syndrome_bits: np.ndarray，shape (num_checks,)，dtype=bool
      disorder_data_error_bits: np.ndarray，shape (num_qubits,)，dtype=bool
      parity_check_matrix: np.ndarray，shape (num_checks, num_qubits)，dtype=bool
      rng: np.random.Generator，本实现不使用，但保留接口

    输出：
      current_chain_bits: np.ndarray，shape (num_qubits,)，dtype=bool
      current_data_term_bits: np.ndarray，shape (num_qubits,)，dtype=bool
      current_syndrome_term_bits: np.ndarray，shape (num_checks,)，dtype=bool
    """
    del rng

    current_chain_bits = np.zeros(num_qubits, dtype=bool)
    current_data_term_bits = current_chain_bits ^ disorder_data_error_bits

    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    chain_bits_uint8 = current_chain_bits.astype(np.uint8)
    current_syndrome_bits = (
        parity_check_matrix_uint8 @ chain_bits_uint8
    ) % 2
    current_syndrome_term_bits = (
        current_syndrome_bits.astype(bool) ^ observed_syndrome_bits
    )

    return (
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
    )


def attempt_single_bit_metropolis_update(
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
        qubit_index,
        checks_touching_each_qubit,
        log_odds_data,
        log_odds_syndrome,
        rng):
    """
    尝试一次单比特 Metropolis 更新。

    输入输出：
      current_chain_bits: np.ndarray，shape (num_qubits,)，dtype=bool，原地更新
      current_data_term_bits: np.ndarray，shape (num_qubits,)，dtype=bool，原地更新
      current_syndrome_term_bits: np.ndarray，shape (num_checks,)，dtype=bool，原地更新
      qubit_index: int
      checks_touching_each_qubit: list[np.ndarray[int32]]
      log_odds_data: float
      log_odds_syndrome: float
      rng: np.random.Generator

    输出：
      accepted: bool
    """
    touched_checks = checks_touching_each_qubit[qubit_index]

    if current_data_term_bits[qubit_index]:
        delta_data_weight = -1
    else:
        delta_data_weight = +1

    delta_syndrome_weight = 0
    for check_index in touched_checks:
        if current_syndrome_term_bits[check_index]:
            delta_syndrome_weight -= 1
        else:
            delta_syndrome_weight += 1

    log_acceptance = (
        delta_data_weight * log_odds_data
        + delta_syndrome_weight * log_odds_syndrome
    )

    if log_acceptance >= 0.0:
        accepted = True
    else:
        accepted = bool(rng.random() < np.exp(log_acceptance))

    if not accepted:
        return False

    current_chain_bits[qubit_index] ^= True
    current_data_term_bits[qubit_index] ^= True
    current_syndrome_term_bits[touched_checks] ^= True
    return True


def run_one_sweep(
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
        checks_touching_each_qubit,
        log_odds_data,
        log_odds_syndrome,
        rng):
    """
    执行一次 sweep。

    输入输出：
      current_chain_bits: np.ndarray，shape (num_qubits,)，dtype=bool，原地更新
      current_data_term_bits: np.ndarray，shape (num_qubits,)，dtype=bool，原地更新
      current_syndrome_term_bits: np.ndarray，shape (num_checks,)，dtype=bool，原地更新
      checks_touching_each_qubit: list[np.ndarray[int32]]
      log_odds_data: float
      log_odds_syndrome: float
      rng: np.random.Generator

    输出：
      accepted_count: int
    """
    num_qubits = current_chain_bits.shape[0]
    accepted_count = 0

    for qubit_index in rng.permutation(num_qubits):
        accepted_count += int(
            attempt_single_bit_metropolis_update(
                current_chain_bits=current_chain_bits,
                current_data_term_bits=current_data_term_bits,
                current_syndrome_term_bits=current_syndrome_term_bits,
                qubit_index=int(qubit_index),
                checks_touching_each_qubit=checks_touching_each_qubit,
                log_odds_data=log_odds_data,
                log_odds_syndrome=log_odds_syndrome,
                rng=rng,
            )
        )

    return accepted_count


def accumulate_logical_observables(
        current_chain_bits,
        logical_observable_masks,
        logical_observable_sum_values):
    """
    累加全部逻辑观测量。

    输入输出：
      current_chain_bits: np.ndarray，shape (num_qubits,)，dtype=bool
      logical_observable_masks: np.ndarray，shape (num_masks, num_qubits)，dtype=bool
      logical_observable_sum_values: np.ndarray，shape (num_masks,)，dtype=int64，原地更新
    """
    masked_bits = logical_observable_masks & current_chain_bits
    parity_bits = np.bitwise_xor.reduce(masked_bits, axis=1)
    logical_observable_sum_values += 1 - 2 * parity_bits.astype(np.int64)


def _compute_syndrome_bits_mod2(parity_check_matrix, chain_bits):
    """
    用 full 矩阵乘重算 H_Z · chain_bits mod 2。
    """
    parity_check_matrix_uint8 = parity_check_matrix.astype(np.uint8)
    chain_bits_uint8 = chain_bits.astype(np.uint8)
    syndrome_bits = (parity_check_matrix_uint8 @ chain_bits_uint8) % 2
    return syndrome_bits.astype(bool)


def _run_cache_consistency_test(
        parity_check_matrix,
        checks_touching_each_qubit,
        rng):
    """
    Test A: 增量缓存一致性。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    syndrome_error_probability = 0.17
    data_error_probability = 0.11

    observed_syndrome_bits, disorder_data_error_bits = draw_disorder_sample(
        num_checks=num_checks,
        num_qubits=num_qubits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        rng=rng,
    )

    (
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
    ) = initialize_mcmc_state(
        num_qubits=num_qubits,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        parity_check_matrix=parity_check_matrix,
        rng=rng,
    )

    log_odds_data = np.log(
        data_error_probability / (1.0 - data_error_probability)
    )
    log_odds_syndrome = np.log(
        syndrome_error_probability / (1.0 - syndrome_error_probability)
    )

    for sweep_index in range(200):
        run_one_sweep(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            checks_touching_each_qubit=checks_touching_each_qubit,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
            rng=rng,
        )

        if (sweep_index + 1) % 10 != 0:
            continue

        expected_data_term_bits = (
            current_chain_bits ^ disorder_data_error_bits
        )
        expected_syndrome_term_bits = (
            _compute_syndrome_bits_mod2(parity_check_matrix, current_chain_bits)
            ^ observed_syndrome_bits
        )

        assert np.array_equal(
            current_data_term_bits,
            expected_data_term_bits,
        ), f"Test A failed: data cache mismatch at sweep {sweep_index + 1}"
        assert np.array_equal(
            current_syndrome_term_bits,
            expected_syndrome_term_bits,
        ), f"Test A failed: syndrome cache mismatch at sweep {sweep_index + 1}"


def _run_infinite_temperature_test(
        parity_check_matrix,
        checks_touching_each_qubit,
        rng):
    """
    Test B: p = q = 0.5 时全接受。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    observed_syndrome_bits, disorder_data_error_bits = draw_disorder_sample(
        num_checks=num_checks,
        num_qubits=num_qubits,
        syndrome_error_probability=0.5,
        data_error_probability=0.5,
        rng=rng,
    )

    (
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
    ) = initialize_mcmc_state(
        num_qubits=num_qubits,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        parity_check_matrix=parity_check_matrix,
        rng=rng,
    )

    log_odds_data = np.log(0.5 / 0.5)
    log_odds_syndrome = np.log(0.5 / 0.5)

    for sweep_index in range(50):
        accepted_count = run_one_sweep(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            checks_touching_each_qubit=checks_touching_each_qubit,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
            rng=rng,
        )
        assert accepted_count == num_qubits, (
            "Test B failed: not all proposals accepted at "
            f"sweep {sweep_index + 1}"
        )


def _run_zero_temperature_like_test(
        parity_check_matrix,
        checks_touching_each_qubit,
        rng):
    """
    Test C: p = q = 1e-4 且 disorder = 0 时接受率应很低。
    """
    del rng

    num_checks, num_qubits = parity_check_matrix.shape
    observed_syndrome_bits = np.zeros(num_checks, dtype=bool)
    disorder_data_error_bits = np.zeros(num_qubits, dtype=bool)

    (
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
    ) = initialize_mcmc_state(
        num_qubits=num_qubits,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        parity_check_matrix=parity_check_matrix,
        rng=np.random.default_rng(0),
    )

    data_error_probability = 1e-4
    syndrome_error_probability = 1e-4
    log_odds_data = np.log(
        data_error_probability / (1.0 - data_error_probability)
    )
    log_odds_syndrome = np.log(
        syndrome_error_probability / (1.0 - syndrome_error_probability)
    )

    low_temperature_rng = np.random.default_rng(20260420)
    accepted_counts = np.empty(50, dtype=np.int64)

    for sweep_index in range(50):
        accepted_counts[sweep_index] = run_one_sweep(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            checks_touching_each_qubit=checks_touching_each_qubit,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
            rng=low_temperature_rng,
        )

    mean_acceptance_fraction = accepted_counts.mean() / num_qubits
    assert mean_acceptance_fraction < 0.05, (
        "Test C failed: acceptance fraction too large: "
        f"{mean_acceptance_fraction}"
    )


def _run_realistic_test(
        parity_check_matrix,
        logical_observable_masks,
        checks_touching_each_qubit,
        rng):
    """
    Test D: 真实运行 smoke test。
    """
    num_checks, num_qubits = parity_check_matrix.shape
    syndrome_error_probability = 0.03
    data_error_probability = 0.03

    observed_syndrome_bits, disorder_data_error_bits = draw_disorder_sample(
        num_checks=num_checks,
        num_qubits=num_qubits,
        syndrome_error_probability=syndrome_error_probability,
        data_error_probability=data_error_probability,
        rng=rng,
    )

    (
        current_chain_bits,
        current_data_term_bits,
        current_syndrome_term_bits,
    ) = initialize_mcmc_state(
        num_qubits=num_qubits,
        observed_syndrome_bits=observed_syndrome_bits,
        disorder_data_error_bits=disorder_data_error_bits,
        parity_check_matrix=parity_check_matrix,
        rng=rng,
    )

    log_odds_data = np.log(
        data_error_probability / (1.0 - data_error_probability)
    )
    log_odds_syndrome = np.log(
        syndrome_error_probability / (1.0 - syndrome_error_probability)
    )

    total_accepted_count = 0
    total_attempt_count = 0

    for _ in range(200):
        total_accepted_count += run_one_sweep(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            checks_touching_each_qubit=checks_touching_each_qubit,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
            rng=rng,
        )
        total_attempt_count += num_qubits

    logical_observable_sum_values = np.zeros(
        logical_observable_masks.shape[0],
        dtype=np.int64,
    )

    for _ in range(500):
        total_accepted_count += run_one_sweep(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            current_syndrome_term_bits=current_syndrome_term_bits,
            checks_touching_each_qubit=checks_touching_each_qubit,
            log_odds_data=log_odds_data,
            log_odds_syndrome=log_odds_syndrome,
            rng=rng,
        )
        total_attempt_count += num_qubits
        accumulate_logical_observables(
            current_chain_bits=current_chain_bits,
            logical_observable_masks=logical_observable_masks,
            logical_observable_sum_values=logical_observable_sum_values,
        )

    m_u_values = logical_observable_sum_values / 500.0
    average_acceptance_rate = total_accepted_count / total_attempt_count
    q_top_value = float(np.mean(m_u_values ** 2))

    assert np.all(m_u_values >= -1.0), "Test D failed: m_u < -1"
    assert np.all(m_u_values <= 1.0), "Test D failed: m_u > 1"

    return m_u_values, average_acceptance_rate, q_top_value


if __name__ == "__main__":
    random_number_generator = np.random.default_rng(20260420)
    lattice_size = 3

    parity_check_matrix, dual_logical_z_basis = build_2d_toric_code(
        lattice_size=lattice_size
    )
    linear_section_data = build_linear_section(parity_check_matrix)
    checks_touching_each_qubit = build_checks_touching_each_qubit(
        parity_check_matrix
    )
    logical_observable_masks = build_logical_observable_masks(
        parity_check_matrix=parity_check_matrix,
        dual_logical_z_basis=dual_logical_z_basis,
        linear_section_data=linear_section_data,
    )

    _run_cache_consistency_test(
        parity_check_matrix=parity_check_matrix,
        checks_touching_each_qubit=checks_touching_each_qubit,
        rng=random_number_generator,
    )
    print("Test A passed: cache consistency")

    _run_infinite_temperature_test(
        parity_check_matrix=parity_check_matrix,
        checks_touching_each_qubit=checks_touching_each_qubit,
        rng=random_number_generator,
    )
    print("Test B passed: infinite-temperature limit")

    _run_zero_temperature_like_test(
        parity_check_matrix=parity_check_matrix,
        checks_touching_each_qubit=checks_touching_each_qubit,
        rng=random_number_generator,
    )
    print("Test C passed: zero-temperature-like limit")

    (
        m_u_values,
        average_acceptance_rate,
        q_top_value,
    ) = _run_realistic_test(
        parity_check_matrix=parity_check_matrix,
        logical_observable_masks=logical_observable_masks,
        checks_touching_each_qubit=checks_touching_each_qubit,
        rng=random_number_generator,
    )
    print("Test D passed: realistic run")
    print(f"m_u_values: {m_u_values}")
    print(f"average_acceptance_rate: {average_acceptance_rate}")
    print(f"q_top: {q_top_value}")
