import unittest

import numpy as np

from build_toric_code_examples import (
    build_3d_toric_code,
    build_3d_toric_zero_syndrome_move_data,
)
from main import _build_pt_ladders
from mcmc_parallel_tempering import run_parallel_tempering_measurement
from preprocessing import (
    build_checks_touching_each_qubit,
    build_logical_observable_masks,
)
from linear_section import build_linear_section


class ParallelTemperingSwapSweepTests(unittest.TestCase):
    def test_extra_swap_sweeps_increase_attempt_counts(self):
        parity_check_matrix, dual_logical_z_basis = build_3d_toric_code(3)
        zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(3)
        checks_touching_each_qubit = build_checks_touching_each_qubit(
            parity_check_matrix
        )
        logical_observable_masks = build_logical_observable_masks(
            parity_check_matrix=parity_check_matrix,
            dual_logical_z_basis=dual_logical_z_basis,
            linear_section_data=build_linear_section(parity_check_matrix),
        )
        data_ladder, syndrome_ladder, _ = _build_pt_ladders(
            data_error_probability=0.05,
            syndrome_error_probability=0.08,
            pt_p_hot=None,
            pt_num_temperatures=5,
            pt_ladder_mode="sync_enlarge",
            pt_q_hot=0.32,
        )
        common_kwargs = dict(
            parity_check_matrix=parity_check_matrix,
            observed_syndrome_bits=np.zeros(
                parity_check_matrix.shape[0],
                dtype=bool,
            ),
            disorder_data_error_bits=np.zeros(
                parity_check_matrix.shape[1],
                dtype=bool,
            ),
            syndrome_error_probability=0.08,
            data_error_probability_ladder=data_ladder,
            syndrome_error_probability_ladder=syndrome_ladder,
            logical_observable_masks=logical_observable_masks,
            checks_touching_each_qubit=checks_touching_each_qubit,
            num_burn_in_sweeps=2,
            num_measurements=3,
            num_sweeps_between_measurements=2,
            zero_syndrome_move_data=zero_syndrome_move_data,
            num_zero_syndrome_sweeps_per_cycle=1,
            winding_repeat_factor=1,
            swap_attempt_every_num_sweeps=1,
            cluster_update_enabled=False,
            observable_temperature_mode="cold",
        )

        baseline = run_parallel_tempering_measurement(
            rng=np.random.default_rng(1234),
            swap_sweeps_per_attempt=1,
            **common_kwargs,
        )
        doubled = run_parallel_tempering_measurement(
            rng=np.random.default_rng(1234),
            swap_sweeps_per_attempt=2,
            **common_kwargs,
        )

        np.testing.assert_array_equal(
            doubled["swap_attempt_counts"],
            2 * baseline["swap_attempt_counts"],
        )
        self.assertEqual(int(doubled["pt_swap_sweeps_per_attempt"]), 2)


if __name__ == "__main__":
    unittest.main()
