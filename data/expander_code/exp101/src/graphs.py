"""二部图构造（expander_code.md 规格 §1–§2）与退化码用确定性 H 构造器。

约定（与 spec 一致，权威）：
  - G = (A ∪ B, E)，|A| = n_A，|B| = n_B；A 侧度 d_A，B 侧度 d_B。
  - 简单图：无重边（二部图天然无自环）。
  - 随机构造 = configuration model：左 stubs 固定顺序（顶点 0..n_A-1 各重复
    d_A 次），右 stubs 用 random.Random(seed) 洗牌后依次配对；出现重边则整体
    拒绝重来。**单一 RNG 流跨 attempt 连续使用**，保证同 seed 完全可复现。
  - graph 对象存 seed / rng_description / construction_attempts 以便精确重建。
  - 拒绝率事实：简单图接受率 ≈ exp(−(d_A−1)(d_B−1)/2)（大 n 渐近，与 n 无关）。
    (3,4) ≈ e^{-3} ≈ 5%（默认 max_attempts=10000 绰绰有余）；(5,6) ≈ e^{-10} ≈
    4.5e-5 ⇒ 高度数家族必须显式加大 max_attempts（期望 ~2.2 万次）。
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class BiregularBipartiteGraph:
    n_A: int
    n_B: int
    d_A: int
    d_B: int
    A_to_B: list  # list[set[int]]，长度 n_A
    B_to_A: list  # list[set[int]]，长度 n_B
    seed: int
    rng_description: str
    construction_attempts: int
    extra_metadata: dict = field(default_factory=dict)

    # ---------- 校验（失败即 raise ValueError） ----------

    def check_degrees(self):
        for a, neighbors in enumerate(self.A_to_B):
            if len(neighbors) != self.d_A:
                raise ValueError(
                    f"A vertex {a} has degree {len(neighbors)} != d_A={self.d_A}"
                )
        for b, neighbors in enumerate(self.B_to_A):
            if len(neighbors) != self.d_B:
                raise ValueError(
                    f"B vertex {b} has degree {len(neighbors)} != d_B={self.d_B}"
                )

    def check_simple(self):
        # 邻接以 set 存储 ⇒ 无重边等价于 stub 配对无重复；这里校验集合元素范围
        # 与总边数守恒（重边会导致度数或一致性检查失败，这里再显式确认边数）。
        edge_count_from_A = sum(len(neighbors) for neighbors in self.A_to_B)
        edge_count_from_B = sum(len(neighbors) for neighbors in self.B_to_A)
        if edge_count_from_A != self.n_A * self.d_A:
            raise ValueError("A-side edge count inconsistent with biregularity")
        if edge_count_from_B != self.n_B * self.d_B:
            raise ValueError("B-side edge count inconsistent with biregularity")
        for a, neighbors in enumerate(self.A_to_B):
            for b in neighbors:
                if not (0 <= b < self.n_B):
                    raise ValueError(f"A vertex {a} has out-of-range neighbor {b}")
        for b, neighbors in enumerate(self.B_to_A):
            for a in neighbors:
                if not (0 <= a < self.n_A):
                    raise ValueError(f"B vertex {b} has out-of-range neighbor {a}")

    def check_consistency(self):
        for a, neighbors in enumerate(self.A_to_B):
            for b in neighbors:
                if a not in self.B_to_A[b]:
                    raise ValueError(f"edge ({a},{b}) present in A_to_B but not B_to_A")
        for b, neighbors in enumerate(self.B_to_A):
            for a in neighbors:
                if b not in self.A_to_B[a]:
                    raise ValueError(f"edge ({a},{b}) present in B_to_A but not A_to_B")

    def validate(self):
        self.check_degrees()
        self.check_simple()
        self.check_consistency()

    # ---------- 便捷 ----------

    def edge_set(self):
        return frozenset(
            (a, b) for a, neighbors in enumerate(self.A_to_B) for b in neighbors
        )

    def num_edges(self):
        return self.n_A * self.d_A


def family_sizes_from_m(m, d_A, d_B):
    """由 m 给出 (n_A, n_B)：g=gcd(d_A,d_B), a=d_A/g, b=d_B/g；n_A=b·m, n_B=a·m。"""
    if m < 1 or d_A < 1 or d_B < 1:
        raise ValueError("m, d_A, d_B must all be >= 1")
    g = math.gcd(d_A, d_B)
    a = d_A // g
    b = d_B // g
    n_A = b * m
    n_B = a * m
    assert n_A * d_A == n_B * d_B
    return n_A, n_B


def random_biregular_graph_from_m(m, d_A, d_B, seed, max_attempts=10000):
    """configuration model 随机简单 (d_A,d_B)-biregular 二部图（spec §2）。"""
    n_A, n_B = family_sizes_from_m(m, d_A, d_B)
    if d_A > n_B:
        raise ValueError(f"simple graph impossible: d_A={d_A} > n_B={n_B}")
    if d_B > n_A:
        raise ValueError(f"simple graph impossible: d_B={d_B} > n_A={n_A}")

    rng = random.Random(seed)
    rng_description = (
        f"python-stdlib random.Random(seed={seed}); single stream across attempts; "
        "per attempt: rng.shuffle(right_stubs) paired against fixed-order left stubs; "
        "reject attempt on any parallel edge"
    )
    left_stubs = [a for a in range(n_A) for _ in range(d_A)]

    for attempt in range(1, int(max_attempts) + 1):
        right_stubs = [b for b in range(n_B) for _ in range(d_B)]
        rng.shuffle(right_stubs)
        edges = set()
        simple = True
        for a, b in zip(left_stubs, right_stubs):
            edge = (a, b)
            if edge in edges:
                simple = False
                break
            edges.add(edge)
        if not simple:
            continue
        A_to_B = [set() for _ in range(n_A)]
        B_to_A = [set() for _ in range(n_B)]
        for a, b in edges:
            A_to_B[a].add(b)
            B_to_A[b].add(a)
        graph = BiregularBipartiteGraph(
            n_A=n_A,
            n_B=n_B,
            d_A=d_A,
            d_B=d_B,
            A_to_B=A_to_B,
            B_to_A=B_to_A,
            seed=int(seed),
            rng_description=rng_description,
            construction_attempts=attempt,
            extra_metadata={"m": int(m)},
        )
        graph.validate()
        return graph

    raise RuntimeError(
        f"failed to build simple ({d_A},{d_B})-biregular graph with m={m} "
        f"seed={seed} within {max_attempts} attempts"
    )


def complete_bipartite_graph(n_A, n_B):
    """确定性 K_{n_A,n_B}（(d_A,d_B)=(n_B,n_A)）；(3,4) 家族 m=1 的唯一简单图是 K_{4,3}。"""
    A_to_B = [set(range(n_B)) for _ in range(n_A)]
    B_to_A = [set(range(n_A)) for _ in range(n_B)]
    graph = BiregularBipartiteGraph(
        n_A=n_A,
        n_B=n_B,
        d_A=n_B,
        d_B=n_A,
        A_to_B=A_to_B,
        B_to_A=B_to_A,
        seed=-1,
        rng_description="deterministic complete bipartite graph (no RNG)",
        construction_attempts=1,
    )
    graph.validate()
    return graph


# ---------- 退化码用确定性经典 H 构造器（供 hgp_from_H；非 spec 必需，exp101 验证用） ----------

def cycle_parity_check_matrix(m):
    """单圈 cycle code：m×m circulant，H[i,i]=H[i,(i+1)%m]=1。

    rank = m−1，k(H)=k(Hᵀ)=1；HGP(H) = 2D toric code [[2m²,2,m]]。要求 m>=2。
    """
    import numpy as np

    if m < 2:
        raise ValueError("cycle code requires m >= 2")
    matrix = np.zeros((m, m), dtype=np.uint8)
    for i in range(m):
        matrix[i, i] = 1
        matrix[i, (i + 1) % m] = 1
    return matrix


def repetition_parity_check_matrix(m):
    """路径（repetition）code：(m−1)×m，H[i,i]=H[i,i+1]=1。

    满秩 m−1，k(H)=1，k(Hᵀ)=0；HGP(H) 为 surface-code 型 [[m²+(m−1)²,1,m]]。要求 m>=2。
    """
    import numpy as np

    if m < 2:
        raise ValueError("repetition code requires m >= 2")
    matrix = np.zeros((m - 1, m), dtype=np.uint8)
    for i in range(m - 1):
        matrix[i, i] = 1
        matrix[i, i + 1] = 1
    return matrix
