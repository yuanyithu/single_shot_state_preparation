"""精确 vertex-expansion 验证（expander_code.md 规格 §5）。

定义（精确、有理数比较，禁浮点）：
  左侧：∀ S ⊆ A, 1 ≤ |S| ≤ γ·|A| ⇒ |Γ(S)| ≥ (1−δ)·d_A·|S|
  右侧：∀ T ⊆ B, 1 ≤ |T| ≤ γ·|B| ⇒ |Γ(T)| ≥ (1−δ)·d_B·|T|
诊断比：|Γ(S)| / (d_A·|S|)（右侧同理）。

实现要点：
  - γ、δ 只接受 Fraction / str（"1/10"）/ int；float 一律拒绝（保证精确性）。
  - 全子集枚举（指数复杂度是规格允许的）；超过 max_subsets 显式报错，不静默截断。
  - γ·n < 1 时无可检子集 ⇒ 空真通过（vacuous 字段显式记录，呼应 plan §6 风险 5）。
  - witness 取「诊断比最小的违例子集」（信息量最大），并附精确 required 值。
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import comb


def _as_fraction(value, name):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(
        f"{name} must be Fraction/str/int for exact arithmetic, got {type(value).__name__}"
    )


@dataclass
class ExpansionVerificationResult:
    passed: bool
    checked_left: bool
    checked_right: bool
    worst_left_ratio: object = None       # Fraction | None
    worst_right_ratio: object = None      # Fraction | None
    failing_side: object = None           # "left" | "right" | None
    failing_subset: object = None         # list[int] | None
    failing_neighborhood_size: object = None   # int | None
    required_neighborhood_size: object = None  # Fraction | None
    # exp101 附加诊断（spec 之外，便于留档）
    gamma: object = None
    delta: object = None
    max_subset_size_left: int = 0
    max_subset_size_right: int = 0
    num_subsets_checked_left: int = 0
    num_subsets_checked_right: int = 0
    vacuous_left: bool = False
    vacuous_right: bool = False
    worst_left_subset: object = None
    worst_right_subset: object = None
    notes: list = field(default_factory=list)


def _neighbor_bitmasks(adjacency):
    masks = []
    for neighbors in adjacency:
        mask = 0
        for v in neighbors:
            mask |= 1 << v
        masks.append(mask)
    return masks


def _check_one_side(adjacency, degree, num_vertices, gamma, delta, max_subsets):
    """返回 (worst_ratio, worst_subset, witness dict|None, num_checked, max_size, vacuous)。"""
    max_size = 0
    for size in range(1, num_vertices + 1):
        if Fraction(size) <= gamma * num_vertices:
            max_size = size
        else:
            break
    if max_size == 0:
        return None, None, None, 0, 0, True

    total_subsets = sum(comb(num_vertices, size) for size in range(1, max_size + 1))
    if total_subsets > max_subsets:
        raise ValueError(
            f"subset enumeration too large: {total_subsets} > max_subsets={max_subsets}; "
            "raise max_subsets explicitly if intended"
        )

    masks = _neighbor_bitmasks(adjacency)
    required_factor = (Fraction(1) - delta) * degree  # (1−δ)·d
    worst_ratio = None
    worst_subset = None
    witness = None
    witness_ratio = None
    num_checked = 0
    for size in range(1, max_size + 1):
        for subset in combinations(range(num_vertices), size):
            num_checked += 1
            union = 0
            for vertex in subset:
                union |= masks[vertex]
            neighborhood_size = union.bit_count()
            ratio = Fraction(neighborhood_size, degree * size)
            if worst_ratio is None or ratio < worst_ratio:
                worst_ratio = ratio
                worst_subset = list(subset)
            required = required_factor * size
            if Fraction(neighborhood_size) < required:
                if witness_ratio is None or ratio < witness_ratio:
                    witness_ratio = ratio
                    witness = {
                        "subset": list(subset),
                        "neighborhood_size": neighborhood_size,
                        "required": required,
                    }
    return worst_ratio, worst_subset, witness, num_checked, max_size, False


def verify_vertex_expansion(graph, gamma, delta, sides="both", return_witness=True,
                            max_subsets=5_000_000):
    """精确验证 (γ, δ) vertex expansion（spec §5）。"""
    gamma = _as_fraction(gamma, "gamma")
    delta = _as_fraction(delta, "delta")
    if not (0 <= gamma <= 1):
        raise ValueError("gamma must be in [0, 1]")
    if not (0 <= delta <= 1):
        raise ValueError("delta must be in [0, 1]")
    if sides not in ("both", "left", "right"):
        raise ValueError("sides must be both/left/right")

    result = ExpansionVerificationResult(
        passed=True,
        checked_left=sides in ("both", "left"),
        checked_right=sides in ("both", "right"),
        gamma=gamma,
        delta=delta,
    )

    witnesses = []
    if result.checked_left:
        (worst, worst_subset, witness, checked, max_size, vacuous) = _check_one_side(
            graph.A_to_B, graph.d_A, graph.n_A, gamma, delta, max_subsets
        )
        result.worst_left_ratio = worst
        result.worst_left_subset = worst_subset
        result.num_subsets_checked_left = checked
        result.max_subset_size_left = max_size
        result.vacuous_left = vacuous
        if vacuous:
            result.notes.append(
                f"left check vacuous: gamma*n_A = {gamma * graph.n_A} < 1"
            )
        if witness is not None:
            witnesses.append(("left", witness))
    if result.checked_right:
        (worst, worst_subset, witness, checked, max_size, vacuous) = _check_one_side(
            graph.B_to_A, graph.d_B, graph.n_B, gamma, delta, max_subsets
        )
        result.worst_right_ratio = worst
        result.worst_right_subset = worst_subset
        result.num_subsets_checked_right = checked
        result.max_subset_size_right = max_size
        result.vacuous_right = vacuous
        if vacuous:
            result.notes.append(
                f"right check vacuous: gamma*n_B = {gamma * graph.n_B} < 1"
            )
        if witness is not None:
            witnesses.append(("right", witness))

    if witnesses:
        result.passed = False
        if return_witness:
            side, witness = witnesses[0]
            result.failing_side = side
            result.failing_subset = witness["subset"]
            result.failing_neighborhood_size = witness["neighborhood_size"]
            result.required_neighborhood_size = witness["required"]
    return result
