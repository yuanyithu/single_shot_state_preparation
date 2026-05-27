import unittest

import numpy as np

from mcmc_diagnostics import (
    DEFAULT_ADAPTIVE_PT_MAX_LOG_GAP_FACTOR,
    adaptive_ladder_from_flow,
    probability_to_coupling,
    sync_pt_enlarge_ladder,
    sync_pt_ladders_from_enlarge,
)


class SyncParallelTemperingLadderTests(unittest.TestCase):
    def test_sync_ladder_uses_common_beta_temperature_path(self):
        p_cold = 0.05
        q_cold = 0.23
        q_hot = 0.44
        heat_scale = sync_pt_enlarge_ladder(
            q_cold=q_cold,
            q_hot=q_hot,
            num_temperatures=9,
        )
        p_ladder, q_ladder = sync_pt_ladders_from_enlarge(
            p_cold=p_cold,
            q_cold=q_cold,
            pt_enlarge=heat_scale,
        )

        self.assertAlmostEqual(float(heat_scale[0]), 1.0)
        self.assertAlmostEqual(float(p_ladder[0]), p_cold)
        self.assertAlmostEqual(float(q_ladder[0]), q_cold)
        self.assertAlmostEqual(float(q_ladder[-1]), q_hot)
        self.assertLess(float(p_ladder[-1]), 0.5)
        self.assertLess(float(q_ladder[-1]), 0.5)
        self.assertGreater(float(p_ladder[-1]), 0.35)

        beta_from_heat = 1.0 / heat_scale
        beta_from_p = np.array([
            probability_to_coupling(value) / probability_to_coupling(p_cold)
            for value in p_ladder
        ])
        beta_from_q = np.array([
            probability_to_coupling(value) / probability_to_coupling(q_cold)
            for value in q_ladder
        ])
        np.testing.assert_allclose(beta_from_p, beta_from_heat, rtol=1e-12)
        np.testing.assert_allclose(beta_from_q, beta_from_heat, rtol=1e-12)

    def test_sync_ladder_rejects_non_hot_q_endpoint(self):
        with self.assertRaisesRegex(ValueError, "q_hot"):
            sync_pt_enlarge_ladder(
                q_cold=0.23,
                q_hot=0.20,
                num_temperatures=9,
            )
        with self.assertRaisesRegex(ValueError, "probability"):
            sync_pt_enlarge_ladder(
                q_cold=0.23,
                q_hot=0.5,
                num_temperatures=9,
            )

    def test_adaptive_ladder_caps_degenerate_large_gap(self):
        initial_heat_scale = sync_pt_enlarge_ladder(
            q_cold=0.08,
            q_hot=0.44,
            num_temperatures=17,
        )
        degenerate_flow = np.zeros(17, dtype=np.float64)
        degenerate_flow[0] = 1.0
        degenerate_flow[1] = 0.5

        adapted_heat_scale, status = adaptive_ladder_from_flow(
            initial_heat_scale,
            degenerate_flow,
        )

        self.assertEqual(status, "ok_capped_gap")
        self.assertAlmostEqual(
            float(adapted_heat_scale[0]),
            float(initial_heat_scale[0]),
        )
        self.assertAlmostEqual(
            float(adapted_heat_scale[-1]),
            float(initial_heat_scale[-1]),
        )
        log_gaps = np.diff(np.log(adapted_heat_scale))
        uniform_gap = (
            np.log(adapted_heat_scale[-1] / adapted_heat_scale[0])
            / float(adapted_heat_scale.size - 1)
        )
        self.assertLessEqual(
            float(np.max(log_gaps)),
            DEFAULT_ADAPTIVE_PT_MAX_LOG_GAP_FACTOR * float(uniform_gap)
            + 1e-12,
        )
        self.assertTrue(np.all(log_gaps >= -1e-12))


if __name__ == "__main__":
    unittest.main()
