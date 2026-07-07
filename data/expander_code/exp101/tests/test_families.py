"""G1.8 单元测试：官方家族注册表（快速子集 m≤3；全表由 validation/003 脚本产出）。"""

import pytest

from src.families import (
    FAMILY_RULES,
    build_family_registry,
    find_family_seed,
    validation_member_m1,
)
from src.gf2 import gf2_rank
from src.graphs import random_biregular_graph_from_m
from src.hgp import classical_parity_check_matrix
from src.params import classical_code_distance


class TestFindFamilySeed:
    @pytest.mark.parametrize("m", [2, 3])
    @pytest.mark.parametrize("rule", FAMILY_RULES)
    def test_selected_seed_satisfies_rule(self, m, rule):
        seed, offset, graph, H, rank, d_cl = find_family_seed(m, rule)
        assert rank == graph.n_B  # 两规则都要求满秩
        if rule == "full_rank_d3":
            assert d_cl >= 3
        # 首个满足：offset 之前的 seed 都不满足该规则（逐一复核 = 「首个」的证明）
        for earlier in range(offset):
            candidate = 12345 + earlier
            try:
                g2 = random_biregular_graph_from_m(m, 3, 4, candidate)
            except RuntimeError:
                continue
            H2 = classical_parity_check_matrix(g2)
            r2 = gf2_rank(H2)
            if r2 != g2.n_B:
                continue
            if rule == "full_rank":
                pytest.fail(f"earlier seed {candidate} already satisfies full_rank")
            d2 = classical_code_distance(H2)
            assert d2 is None or d2 < 3, (
                f"earlier seed {candidate} already satisfies full_rank_d3"
            )

    def test_deterministic(self):
        a = find_family_seed(2, "full_rank")[0]
        b = find_family_seed(2, "full_rank")[0]
        assert a == b == 12345  # m=2 首个满秩即 base seed（G1.3 已证满秩）

    def test_m1_impossible_under_rules(self):
        with pytest.raises(RuntimeError):
            find_family_seed(1, "full_rank", max_seed_offset=5)


class TestRegistry:
    def test_registry_m2_m3(self):
        registry = build_family_registry(m_list=(2, 3), build_fingerprint=True)
        vm = registry["validation_members"]["1"]
        assert (vm["n"], vm["k"], vm["quantum_d"]) == (25, 13, 2)
        assert "validation_only" in vm["role"]
        for rule in FAMILY_RULES:
            for m in ("2", "3"):
                entry = registry["members"][rule][m]
                assert entry["seed"] is not None
                assert entry["k"] == int(m) ** 2  # 满秩 ⇒ k=m²
                assert entry["full_rank"]
                assert len(entry["fingerprint"]) == 64
                if rule == "full_rank_d3":
                    assert entry["d_classical"] >= 3
                    assert entry["quantum_d"] >= 3
        # 同 m 两规则若 seed 相同，则 d_cl 必 ≥3；若不同，full_rank 的 d_cl < 3
        for m in ("2", "3"):
            e_a = registry["members"]["full_rank"][m]
            e_b = registry["members"]["full_rank_d3"][m]
            if e_a["seed"] == e_b["seed"]:
                assert e_a["d_classical"] >= 3
            else:
                assert e_a["d_classical"] < 3

    def test_registry_reproducible(self):
        r1 = build_family_registry(m_list=(2,), build_fingerprint=True)
        r2 = build_family_registry(m_list=(2,), build_fingerprint=True)
        assert r1["members"]["full_rank"]["2"] == r2["members"]["full_rank"]["2"]
        assert (
            r1["members"]["full_rank_d3"]["2"]["fingerprint"]
            == r2["members"]["full_rank_d3"]["2"]["fingerprint"]
        )
