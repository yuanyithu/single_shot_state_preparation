import unittest

import numpy as np

from build_toric_code_examples import build_3d_toric_zero_syndrome_move_data
from main import (
    _infer_winding_heatbath_groups,
    _run_winding_plane_heatbath_sweeps,
)


class WindingPlaneHeatbathTests(unittest.TestCase):
    def test_groups_parallel_3d_winding_planes(self):
        zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(3)
        groups = _infer_winding_heatbath_groups(
            zero_syndrome_move_data["winding_move_supports"]
        )

        self.assertEqual(len(groups), 3)
        self.assertTrue(all(group.shape == (3,) for group in groups))
        np.testing.assert_array_equal(groups[0], np.array([0, 1, 2]))
        np.testing.assert_array_equal(groups[1], np.array([3, 4, 5]))
        np.testing.assert_array_equal(groups[2], np.array([6, 7, 8]))

    def test_heatbath_changes_planes_at_infinite_temperature(self):
        zero_syndrome_move_data = build_3d_toric_zero_syndrome_move_data(3)
        num_qubits = zero_syndrome_move_data["winding_moves"].shape[1]
        current_chain_bits = np.zeros(num_qubits, dtype=bool)
        current_data_term_bits = np.zeros(num_qubits, dtype=bool)

        result = _run_winding_plane_heatbath_sweeps(
            current_chain_bits=current_chain_bits,
            current_data_term_bits=current_data_term_bits,
            zero_syndrome_move_data=zero_syndrome_move_data,
            log_odds_data=0.0,
            rng=np.random.default_rng(1234),
            num_sweeps=8,
        )

        self.assertEqual(result["attempted_count"], 8 * 9)
        self.assertGreater(result["changed_count"], 0)
        np.testing.assert_array_equal(current_chain_bits, current_data_term_bits)


if __name__ == "__main__":
    unittest.main()
