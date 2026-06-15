import argparse
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exp37_sector_ti import _build_disorder_uniforms, _build_tasks


def _args(**overrides):
    values = {
        "lattice_sizes": "3,5",
        "q_values": "0.18,0.19",
        "num_disorder_samples": 2,
        "seed_base": 637000,
        "common_disorder_across_q": True,
        "disorder_seed_scope": "auto",
        "disorder_realization_mode": "rng_stream",
        "code_family": "3d_toric",
        "projection_mode": "linear",
        "p": 0.05,
        "num_kp_grid_points": 129,
        "num_burn_in_sweeps": 512,
        "num_measurements": 8192,
        "num_sweeps_between_measurements": 2,
        "block_count": 128,
        "num_bootstrap": 800,
        "winding_heatbath_sweeps": 1,
        "use_numba": True,
        "grid_tv_warning": 0.02,
        "grid_q_top_warning": 0.02,
        "debug_checks": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class Exp37SectorTiSeedScopeTests(unittest.TestCase):
    def test_auto_scope_preserves_legacy_common_q_seeds(self):
        tasks, _, _ = _build_tasks(_args())
        by_key = {
            (
                task["lattice_size"],
                round(task["q_value"], 2),
                task["disorder_index"],
            ): task
            for task in tasks
        }

        seed_l3_q18 = by_key[(3, 0.18, 0)]["disorder_seed"]
        seed_l3_q19 = by_key[(3, 0.19, 0)]["disorder_seed"]
        seed_l5_q18 = by_key[(5, 0.18, 0)]["disorder_seed"]

        self.assertEqual(seed_l3_q18, seed_l3_q19)
        self.assertNotEqual(seed_l3_q18, seed_l5_q18)
        self.assertEqual(by_key[(3, 0.18, 0)]["sample_seed"], seed_l3_q18)

    def test_disorder_index_scope_shares_disorder_seed_across_l_and_q(self):
        tasks, _, _ = _build_tasks(_args(disorder_seed_scope="disorder_index"))
        by_key = {
            (
                task["lattice_size"],
                round(task["q_value"], 2),
                task["disorder_index"],
            ): task
            for task in tasks
        }

        shared_seed = by_key[(3, 0.18, 1)]["disorder_seed"]
        self.assertEqual(shared_seed, by_key[(5, 0.18, 1)]["disorder_seed"])
        self.assertEqual(shared_seed, by_key[(3, 0.19, 1)]["disorder_seed"])
        self.assertEqual(shared_seed, by_key[(5, 0.19, 1)]["disorder_seed"])
        self.assertNotEqual(
            by_key[(3, 0.18, 1)]["sample_seed"],
            by_key[(5, 0.18, 1)]["sample_seed"],
        )

    def test_coordinate_hash_mode_is_explicit_in_tasks(self):
        tasks, _, _ = _build_tasks(_args(
            disorder_seed_scope="disorder_index",
            disorder_realization_mode="coordinate_hash",
        ))

        self.assertTrue(tasks)
        self.assertTrue(
            all(task["disorder_realization_mode"] == "coordinate_hash"
                for task in tasks)
        )

    def test_coordinate_hash_disorder_aligns_same_3d_coordinates_across_l(self):
        seed = 638500
        data3, syndrome3 = _build_disorder_uniforms(
            code_family="3d_toric",
            lattice_size=3,
            num_qubits=3 * 3 ** 3,
            num_checks=3 * 3 ** 3,
            disorder_seed=seed,
            disorder_realization_mode="coordinate_hash",
        )
        data5, syndrome5 = _build_disorder_uniforms(
            code_family="3d_toric",
            lattice_size=5,
            num_qubits=3 * 5 ** 3,
            num_checks=3 * 5 ** 3,
            disorder_seed=seed,
            disorder_realization_mode="coordinate_hash",
        )

        def index(lattice_size, type_index, i, j, k):
            return (
                type_index * lattice_size ** 3
                + (i * lattice_size + j) * lattice_size
                + k
            )

        for type_index in range(3):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        idx3 = index(3, type_index, i, j, k)
                        idx5 = index(5, type_index, i, j, k)
                        self.assertEqual(data3[idx3], data5[idx5])
                        self.assertEqual(syndrome3[idx3], syndrome5[idx5])
                        self.assertNotEqual(data3[idx3], syndrome3[idx3])

    def test_rng_stream_mode_preserves_numpy_generator_order(self):
        seed = 638501
        num_qubits = 3 * 3 ** 3
        num_checks = 3 * 3 ** 3
        data, syndrome = _build_disorder_uniforms(
            code_family="3d_toric",
            lattice_size=3,
            num_qubits=num_qubits,
            num_checks=num_checks,
            disorder_seed=seed,
            disorder_realization_mode="rng_stream",
        )
        rng = __import__("numpy").random.default_rng(seed)

        self.assertTrue((data == rng.random(num_qubits)).all())
        self.assertTrue((syndrome == rng.random(num_checks)).all())


if __name__ == "__main__":
    unittest.main()
