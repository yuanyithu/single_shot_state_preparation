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


class ClusterQLadderTests(unittest.TestCase):
    def test_cluster_update_runs_with_sync_q_ladder(self):
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

        result = run_parallel_tempering_measurement(
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
            num_measurements=2,
            num_sweeps_between_measurements=1,
            rng=np.random.default_rng(4321),
            zero_syndrome_move_data=zero_syndrome_move_data,
            num_zero_syndrome_sweeps_per_cycle=1,
            winding_repeat_factor=1,
            swap_attempt_every_num_sweeps=1,
            cluster_update_enabled=True,
            cluster_budget_fraction_rho=100.0,
            observable_temperature_mode="cold",
            track_logical_sector_diagnostics=True,
            logical_sector_diagnostic_stride=1,
        )

        self.assertTrue(bool(result["cluster_update_enabled"]))
        self.assertTrue(bool(result["cluster_update_requested_enabled"]))
        self.assertGreater(int(result["cluster_num_attempts"]), 0)
        self.assertEqual(
            result["cluster_by_temperature_attempts"].shape,
            data_ladder.shape,
        )
        self.assertEqual(
            result["pt_cluster_sector_attempted_count_per_temperature"].shape,
            data_ladder.shape,
        )
        self.assertEqual(
            result["pt_cluster_sector_changed_count_per_temperature"].shape,
            data_ladder.shape,
        )
        for key in (
                "pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature",
                "pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature",
                "pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature",
                "pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature",
                "pt_cluster_sector_cold_departure_survived_count_per_origin_temperature",
                "pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature",
                "pt_cluster_sector_cold_departure_other_count_per_origin_temperature",
                "pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature",
                "pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature",
                "pt_cluster_sector_cold_active_remaining_count_per_origin_temperature",
        ):
            self.assertIn(key, result)
            self.assertEqual(result[key].shape, data_ladder.shape)
        self.assertGreaterEqual(
            int(np.sum(result["pt_cluster_sector_attempted_count_per_temperature"])),
            1,
        )

    def test_cluster_sector_diagnostics_can_run_without_full_sector_histogram(self):
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

        result = run_parallel_tempering_measurement(
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
            num_sweeps_between_measurements=1,
            rng=np.random.default_rng(8765),
            zero_syndrome_move_data=zero_syndrome_move_data,
            num_zero_syndrome_sweeps_per_cycle=1,
            winding_repeat_factor=1,
            swap_attempt_every_num_sweeps=1,
            cluster_update_enabled=True,
            cluster_budget_fraction_rho=100.0,
            observable_temperature_mode="cold",
            track_logical_sector_diagnostics=False,
            track_cluster_sector_diagnostics=True,
            logical_sector_diagnostic_stride=1,
        )

        self.assertFalse(bool(result["pt_sector_diagnostics_enabled"]))
        self.assertTrue(bool(result["pt_cluster_sector_diagnostics_enabled"]))
        self.assertIn("pt_cluster_sector_attempted_count_per_temperature", result)
        self.assertNotIn("pt_sector_histogram_per_temperature", result)
        self.assertNotIn("pt_sector_flip_count_per_temperature", result)
        self.assertEqual(
            result["pt_cluster_sector_attempted_count_per_temperature"].shape,
            data_ladder.shape,
        )


if __name__ == "__main__":
    unittest.main()
