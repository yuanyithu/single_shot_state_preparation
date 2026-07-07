"""Family-level wrapper（expander_code.md 规格 §9）+ 序列化与复现校验。

距离来源（provenance，诚实标注）：
  - "bruteforce"：code_parameters 的精确 Gray-code 搜索（ker 维数在守卫内）。
  - "hgp_theorem_classical_sides"：HGP 距离定理。对 HGP(H,H)：
      上界：c ∈ ker H（权重 d_H）嵌入为 c⊗e_j（A×A 块）∈ ker H_Z，且当其配对
        非平凡时为逻辑算符 ⇒ d ≤ d_H；ker Hᵀ 侧同理 ⇒ d ≤ d_Hᵀ。
      下界：Tillich–Zémor 定理 d ≥ min(d_H, d_Hᵀ)。
      满秩 H（k(Hᵀ)=0）时 d = d_H 严格；一般情形 d = min(d_H, d_Hᵀ)。
    该定理已在 exp101 G1.6 于 5 个小码上与暴力精确值全数交叉验证
    （tests/test_params.py::TestClassicalSideTheoremCrossCheck）。
    定理路径只给整体 d（不拆 d_X/d_Z，避免未经验证的分侧断言）。
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .expansion import verify_vertex_expansion
from .graphs import BiregularBipartiteGraph, random_biregular_graph_from_m
from .hgp import (
    classical_parity_check_matrix,
    hgp_from_H,
    hgp_expected_parameters,
    verify_css_commutation,
)
from .gf2 import gf2_rank
from .logicals import logical_pauli_operators, verify_logical_pauli_result
from .params import (
    CodeParameters,
    code_parameters,
    hgp_classical_side_distances,
)


@dataclass
class QuantumExpanderCodeInstance:
    m: int
    d_A: int
    d_B: int
    seed: int
    graph: BiregularBipartiteGraph
    classical_H: np.ndarray
    H_Z: np.ndarray
    H_X: np.ndarray
    css_commutation_ok: bool
    classical_rank: int
    expansion_result: object = None      # ExpansionVerificationResult | None
    logicals: object = None              # LogicalPauliResult | None
    parameters: object = None            # CodeParameters | None
    distance_method: object = None       # str | None
    classical_side_distances: object = None  # dict | None
    notes: list = field(default_factory=list)

    # ---------- 指纹与序列化 ----------

    def fingerprint(self):
        """实例内容指纹（图 + 矩阵），用于 manifest 与复现校验。"""
        payload = {
            "family": [self.m, self.d_A, self.d_B, self.seed],
            "n_A": self.graph.n_A,
            "n_B": self.graph.n_B,
            "edges": sorted(self.graph.edge_set()),
            "H_Z_sha": hashlib.sha256(
                np.ascontiguousarray(self.H_Z).tobytes()
            ).hexdigest(),
            "H_X_sha": hashlib.sha256(
                np.ascontiguousarray(self.H_X).tobytes()
            ).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self):
        """JSON 可序列化 dict（矩阵不入档——由 edges 精确重建并校验指纹）。"""
        data = {
            "schema": "exp101.instance.v1",
            "m": self.m,
            "d_A": self.d_A,
            "d_B": self.d_B,
            "seed": self.seed,
            "n_A": self.graph.n_A,
            "n_B": self.graph.n_B,
            "edges": sorted([list(edge) for edge in self.graph.edge_set()]),
            "rng_description": self.graph.rng_description,
            "construction_attempts": self.graph.construction_attempts,
            "classical_rank": self.classical_rank,
            "css_commutation_ok": self.css_commutation_ok,
            "fingerprint": self.fingerprint(),
            "n": int(self.H_X.shape[1]),
            "distance_method": self.distance_method,
            "notes": list(self.notes),
        }
        if self.parameters is not None:
            data["parameters"] = {
                "n": self.parameters.n,
                "k": self.parameters.k,
                "rank_H_X": self.parameters.rank_H_X,
                "rank_H_Z": self.parameters.rank_H_Z,
                "d_X": self.parameters.d_X,
                "d_Z": self.parameters.d_Z,
                "d": self.parameters.d,
            }
        if self.classical_side_distances is not None:
            data["classical_side_distances"] = self.classical_side_distances
        if self.logicals is not None:
            data["k"] = self.logicals.k
        if self.expansion_result is not None:
            data["expansion"] = {
                "passed": self.expansion_result.passed,
                "gamma": str(self.expansion_result.gamma),
                "delta": str(self.expansion_result.delta),
                "vacuous_left": self.expansion_result.vacuous_left,
                "vacuous_right": self.expansion_result.vacuous_right,
                "worst_left_ratio": (
                    str(self.expansion_result.worst_left_ratio)
                    if self.expansion_result.worst_left_ratio is not None
                    else None
                ),
                "worst_right_ratio": (
                    str(self.expansion_result.worst_right_ratio)
                    if self.expansion_result.worst_right_ratio is not None
                    else None
                ),
            }
        return data

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return path


def build_quantum_expander_code_instance(
        m,
        d_A,
        d_B,
        seed,
        gamma=None,
        delta=None,
        verify_expansion=False,
        compute_logicals=True,
        compute_distance=False,
        max_attempts=10000,
        max_kernel_dim=24,
        expansion_sides="both",
):
    """spec §9 主入口。步骤：图 → H → (H_Z,H_X) → CSS 验证 → [expansion] →
    [logicals] → [[n,k,d]]。"""
    graph = random_biregular_graph_from_m(
        m, d_A, d_B, seed, max_attempts=max_attempts
    )
    classical_H = classical_parity_check_matrix(graph)
    H_Z, H_X = hgp_from_H(classical_H)
    css_ok = verify_css_commutation(H_X, H_Z)
    if not css_ok:
        raise AssertionError("CSS commutation failed on constructed HGP matrices")
    classical_rank = gf2_rank(classical_H)

    instance = QuantumExpanderCodeInstance(
        m=int(m), d_A=int(d_A), d_B=int(d_B), seed=int(seed),
        graph=graph, classical_H=classical_H, H_Z=H_Z, H_X=H_X,
        css_commutation_ok=css_ok, classical_rank=classical_rank,
    )
    expected = hgp_expected_parameters(classical_H, classical_rank)
    if classical_rank == graph.n_B:
        instance.notes.append(f"classical H full rank ⇒ k = m² = {m * m}")
    else:
        instance.notes.append(
            f"classical H rank deficit: rank={classical_rank} < n_B={graph.n_B}; "
            f"k = {expected['k']} ≠ m²"
        )

    if verify_expansion:
        if gamma is None or delta is None:
            raise ValueError("verify_expansion=True requires gamma and delta")
        instance.expansion_result = verify_vertex_expansion(
            graph, gamma, delta, sides=expansion_sides
        )

    if compute_logicals:
        instance.logicals = logical_pauli_operators(H_X, H_Z)
        verify_logical_pauli_result(H_X, H_Z, instance.logicals, strict=True)
        if instance.logicals.k != expected["k"]:
            raise AssertionError("logical count disagrees with HGP k formula")

    if compute_distance:
        try:
            instance.parameters = code_parameters(
                H_X, H_Z, compute_distance=True, max_kernel_dim=max_kernel_dim
            )
            instance.distance_method = "bruteforce"
        except ValueError:
            sides = hgp_classical_side_distances(classical_H)
            instance.classical_side_distances = sides
            instance.parameters = code_parameters(H_X, H_Z, compute_distance=False)
            instance.parameters.d = sides["theorem_min"]
            instance.parameters.distance_computed = sides["theorem_min"] is not None
            instance.distance_method = "hgp_theorem_classical_sides"
            instance.notes.append(
                "distance via HGP theorem (classical sides); per-side d_X/d_Z "
                "not claimed; theorem cross-verified on small codes in G1.6"
            )
    else:
        instance.parameters = code_parameters(H_X, H_Z, compute_distance=False)

    if instance.parameters is not None:
        assert instance.parameters.n == expected["n"]
        assert instance.parameters.k == expected["k"]
        assert instance.parameters.rank_H_X == expected["rank_HX"]
        assert instance.parameters.rank_H_Z == expected["rank_HZ"]

    return instance


def rebuild_and_verify(saved_dict):
    """从序列化 dict 重建实例并校验指纹一致；返回重建的实例。

    重建路径 = 同 (m,d_A,d_B,seed) 重跑构造器（RNG 完全确定）⇒ 边集必须
    逐边一致，指纹必须一致；否则 AssertionError。
    """
    rebuilt = build_quantum_expander_code_instance(
        m=saved_dict["m"],
        d_A=saved_dict["d_A"],
        d_B=saved_dict["d_B"],
        seed=saved_dict["seed"],
        compute_logicals=False,
        compute_distance=False,
    )
    saved_edges = {tuple(edge) for edge in saved_dict["edges"]}
    if rebuilt.graph.edge_set() != frozenset(saved_edges):
        raise AssertionError("rebuilt graph edge set differs from saved instance")
    if rebuilt.fingerprint() != saved_dict["fingerprint"]:
        raise AssertionError("rebuilt instance fingerprint mismatch")
    return rebuilt
