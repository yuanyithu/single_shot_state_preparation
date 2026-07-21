"""G1.7 单元测试：family wrapper + 序列化复现（spec §9）。"""

import json

import numpy as np
import pytest

from src.instance import (
    build_quantum_expander_code_instance,
    rebuild_and_verify,
)


class TestBuildInstance:
    def test_m1_full_pipeline_bruteforce_distance(self):
        """m=1 = K_{4,3}：ker 维数 19 在守卫内 ⇒ 暴力精确距离路径。"""
        instance = build_quantum_expander_code_instance(
            m=1, d_A=3, d_B=4, seed=12345,
            gamma="1/2", delta="1/2", verify_expansion=True,
            compute_logicals=True, compute_distance=True,
        )
        assert instance.css_commutation_ok
        assert (instance.parameters.n, instance.parameters.k) == (25, 13)
        assert instance.distance_method == "bruteforce"
        assert (instance.parameters.d_X, instance.parameters.d_Z,
                instance.parameters.d) == (2, 2, 2)
        assert instance.logicals.k == 13
        assert instance.expansion_result is not None

    def test_m2_theorem_distance_path(self):
        """m=2：ker 维数 52 超守卫 ⇒ 定理路径，provenance 标注。"""
        instance = build_quantum_expander_code_instance(
            m=2, d_A=3, d_B=4, seed=12345,
            compute_logicals=True, compute_distance=True,
        )
        assert (instance.parameters.n, instance.parameters.k) == (100, 4)
        assert instance.distance_method == "hgp_theorem_classical_sides"
        assert instance.parameters.d == instance.classical_side_distances["theorem_min"]
        assert instance.parameters.d_X is None and instance.parameters.d_Z is None
        assert any("theorem" in note for note in instance.notes)
        # 满秩 ⇒ 定理 d = d(ker H)，且 k=m² 注记存在
        assert instance.classical_rank == instance.graph.n_B
        assert any("k = m²" in note for note in instance.notes)

    def test_flags_off_minimal_build(self):
        instance = build_quantum_expander_code_instance(
            m=2, d_A=3, d_B=4, seed=777,
            compute_logicals=False, compute_distance=False,
        )
        assert instance.logicals is None
        assert instance.expansion_result is None
        assert instance.parameters.d is None
        assert instance.parameters.k == instance.H_X.shape[1] - \
            instance.parameters.rank_H_X - instance.parameters.rank_H_Z

    def test_expansion_requires_gamma_delta(self):
        with pytest.raises(ValueError, match="gamma and delta"):
            build_quantum_expander_code_instance(
                m=1, d_A=3, d_B=4, seed=1, verify_expansion=True,
            )

    def test_parameters_match_theory_assertions_run(self):
        """构造内部对 hgp_expected_parameters 的断言在多 seed 下全通过。"""
        for seed in (1, 2, 3, 12345):
            instance = build_quantum_expander_code_instance(
                m=2, d_A=3, d_B=4, seed=seed,
                compute_logicals=True, compute_distance=False,
            )
            assert instance.parameters.n == 100
            assert instance.logicals.k == instance.parameters.k


class TestSerializationAndReproduction:
    def test_roundtrip_and_rebuild(self, tmp_path):
        instance = build_quantum_expander_code_instance(
            m=2, d_A=3, d_B=4, seed=12345,
            compute_logicals=False, compute_distance=False,
        )
        path = instance.save_json(tmp_path / "instance.json")
        with path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        assert saved["schema"] == "exp101.instance.v1"
        assert saved["fingerprint"] == instance.fingerprint()
        rebuilt = rebuild_and_verify(saved)
        assert rebuilt.fingerprint() == instance.fingerprint()
        assert rebuilt.graph.edge_set() == instance.graph.edge_set()

    def test_rebuild_detects_tampered_edges(self, tmp_path):
        instance = build_quantum_expander_code_instance(
            m=1, d_A=3, d_B=4, seed=12345,
            compute_logicals=False, compute_distance=False,
        )
        saved = instance.to_dict()
        saved["edges"][0] = [0, (saved["edges"][0][1] + 1) % 3]
        with pytest.raises(AssertionError):
            rebuild_and_verify(saved)

    def test_fingerprint_sensitive_to_seed(self):
        f1 = build_quantum_expander_code_instance(
            m=2, d_A=3, d_B=4, seed=1, compute_logicals=False,
        ).fingerprint()
        f2 = build_quantum_expander_code_instance(
            m=2, d_A=3, d_B=4, seed=2, compute_logicals=False,
        ).fingerprint()
        assert f1 != f2

    def test_dict_json_serializable(self):
        instance = build_quantum_expander_code_instance(
            m=1, d_A=3, d_B=4, seed=12345,
            gamma="1/10", delta="1/16", verify_expansion=True,
            compute_logicals=True, compute_distance=True,
        )
        text = json.dumps(instance.to_dict())
        assert "fingerprint" in text


class TestSpecExampleScript:
    def test_spec_example_runs_and_reports(self, tmp_path):
        import importlib.util
        from pathlib import Path

        example_path = (
            Path(__file__).resolve().parents[1] / "examples" / "spec_example.py"
        )
        spec = importlib.util.spec_from_file_location("spec_example", example_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        instance = module.main(output_dir=tmp_path)
        assert instance.parameters.n == 100
        assert instance.parameters.k == 4
        assert instance.css_commutation_ok
        assert instance.expansion_result.passed  # γ=1/10 在 n_A=8 下空真通过
        assert instance.expansion_result.vacuous_left
        output = tmp_path / "spec_example_output.txt"
        assert output.exists()
        assert (tmp_path / "spec_example_instance.json").exists()
        content = output.read_text(encoding="utf-8")
        for token in ("seed = 12345", "n = 100", "k = 4", "CSS commutation holds: True"):
            assert token in content
