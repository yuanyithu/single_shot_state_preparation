"""官方码族注册（G1.8）。

官方家族：(d_A, d_B) = (3, 4)，seed 自 base_seed=12345 递增扫描。
两种候选规则（D3 待用户定，注册表两列都算）：
  - rule "full_rank"：简单图构造成功 且 rank(H) = n_B（⇒ k = m²）。
  - rule "full_rank_d3"：再加 d_classical(ker H) ≥ 3（⇔ H 列互异；
    排除权重 ≤2 的经典码字 ⇒ 量子 d ≥ 3（满秩时 d = d_classical，定理路径））。

特殊成员 m=1：唯一简单图 K_{4,3}，rank=1 ≠ n_B=3，不满足任何规则——
作为「验证成员」单独登记（[[25,13,2]]，大 k 测例），scaling 家族从 m=2 起。

注册条目字段：m, seed, seed_offset, construction_attempts, n_A, n_B, n, k,
classical_rank, d_classical, quantum_d(来源标注), fingerprint。
注册函数纯确定性：同参数必出同表。
"""

from .gf2 import gf2_rank
from .graphs import random_biregular_graph_from_m
from .hgp import classical_parity_check_matrix, hgp_expected_parameters
from .instance import build_quantum_expander_code_instance
from .params import classical_code_distance

OFFICIAL_D_A = 3
OFFICIAL_D_B = 4
OFFICIAL_BASE_SEED = 12345

FAMILY_RULES = ("full_rank", "full_rank_d3")


def find_family_seed(
        m,
        rule,
        d_A=OFFICIAL_D_A,
        d_B=OFFICIAL_D_B,
        base_seed=OFFICIAL_BASE_SEED,
        max_seed_offset=100000,
        max_attempts=10000,
):
    """自 base_seed 递增取首个满足 rule 的 seed；返回 (seed, offset, graph, H, rank, d_cl)。

    找不到（如 m=1 满秩不可能）时抛 RuntimeError，调用方决定如何登记。
    """
    if rule not in FAMILY_RULES:
        raise ValueError(f"unknown rule {rule}")
    for offset in range(int(max_seed_offset)):
        seed = base_seed + offset
        try:
            graph = random_biregular_graph_from_m(
                m, d_A, d_B, seed, max_attempts=max_attempts
            )
        except RuntimeError:
            continue
        classical_H = classical_parity_check_matrix(graph)
        rank = gf2_rank(classical_H)
        if rank != graph.n_B:
            continue
        d_classical = classical_code_distance(classical_H)
        if rule == "full_rank_d3" and (d_classical is None or d_classical < 3):
            continue
        return seed, offset, graph, classical_H, rank, d_classical
    raise RuntimeError(
        f"no seed satisfying rule={rule} for m={m} within offset {max_seed_offset}"
    )


def _member_entry(m, seed, offset, graph, classical_H, rank, d_classical,
                  build_fingerprint=True):
    expected = hgp_expected_parameters(classical_H, rank)
    entry = {
        "m": int(m),
        "seed": int(seed),
        "seed_offset": int(offset),
        "construction_attempts": int(graph.construction_attempts),
        "n_A": graph.n_A,
        "n_B": graph.n_B,
        "n": expected["n"],
        "k": expected["k"],
        "classical_rank": int(rank),
        "full_rank": bool(rank == graph.n_B),
        "d_classical": d_classical,
    }
    if rank == graph.n_B:
        entry["quantum_d"] = d_classical
        entry["quantum_d_method"] = "hgp_theorem_classical_sides(full-rank ⇒ d=d_classical)"
    else:
        entry["quantum_d"] = None
        entry["quantum_d_method"] = None
    if build_fingerprint:
        instance = build_quantum_expander_code_instance(
            m=m, d_A=graph.d_A, d_B=graph.d_B, seed=seed,
            compute_logicals=False, compute_distance=False,
        )
        entry["fingerprint"] = instance.fingerprint()
    return entry


def validation_member_m1(build_fingerprint=True):
    """m=1 特殊验证成员 K_{4,3}（任何 seed 都给同一张图；用 base seed 登记）。"""
    graph = random_biregular_graph_from_m(1, OFFICIAL_D_A, OFFICIAL_D_B,
                                          OFFICIAL_BASE_SEED)
    classical_H = classical_parity_check_matrix(graph)
    rank = gf2_rank(classical_H)
    entry = _member_entry(
        1, OFFICIAL_BASE_SEED, 0, graph, classical_H, rank,
        classical_code_distance(classical_H), build_fingerprint,
    )
    entry["role"] = "validation_only（K_{4,3}：rank=1，[[25,13,2]]，大 k 测例；不属 scaling 家族）"
    # K_{4,3} 两侧均非满秩，d 有已验证的暴力值 2（G1.6）
    entry["quantum_d"] = 2
    entry["quantum_d_method"] = "bruteforce(G1.6 验证)"
    return entry


def build_family_registry(m_list=(2, 3, 4, 5, 6), rules=FAMILY_RULES,
                          build_fingerprint=True):
    """构建官方注册表（确定性）。返回 dict，含验证成员 m=1 与各规则成员表。"""
    registry = {
        "family": {"d_A": OFFICIAL_D_A, "d_B": OFFICIAL_D_B,
                   "base_seed": OFFICIAL_BASE_SEED},
        "rules_definition": {
            "full_rank": "simple graph & rank(H)=n_B (k=m²)",
            "full_rank_d3": "full_rank & d_classical>=3 (H 列互异 ⇒ 量子 d>=3)",
        },
        "validation_members": {"1": validation_member_m1(build_fingerprint)},
        "members": {rule: {} for rule in rules},
    }
    for rule in rules:
        for m in m_list:
            try:
                seed, offset, graph, classical_H, rank, d_cl = find_family_seed(
                    m, rule
                )
                registry["members"][rule][str(m)] = _member_entry(
                    m, seed, offset, graph, classical_H, rank, d_cl,
                    build_fingerprint,
                )
            except RuntimeError as error:
                registry["members"][rule][str(m)] = {
                    "m": int(m), "seed": None, "note": str(error),
                }
    return registry


def registry_markdown(registry):
    """人类可读 md 表（可入 git；json 按仓库策略留本地）。"""
    lines = [
        "# 官方 (3,4) 家族注册表",
        "",
        f"- d_A={registry['family']['d_A']}, d_B={registry['family']['d_B']}, "
        f"base_seed={registry['family']['base_seed']}",
        "- 规则 full_rank：简单图 + 满秩（k=m²）；full_rank_d3：再加 H 列互异（d≥3）",
        "",
        "## 验证成员",
        "",
        "| m | seed | [[n,k,d]] | 备注 |",
        "|---|---|---|---|",
    ]
    vm = registry["validation_members"]["1"]
    lines.append(
        f"| 1 | {vm['seed']} | [[{vm['n']},{vm['k']},{vm['quantum_d']}]] | {vm['role']} |"
    )
    for rule, members in registry["members"].items():
        lines += ["", f"## 规则 {rule}", "",
                  "| m | seed | offset | attempts | n | k | rank | d_cl | 量子 d(来源) | fingerprint |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for m_key in sorted(members, key=int):
            e = members[m_key]
            if e.get("seed") is None:
                lines.append(f"| {e['m']} | — | — | — | — | — | — | — | — | {e.get('note','')} |")
                continue
            fingerprint = e.get("fingerprint", "")[:16]
            lines.append(
                f"| {e['m']} | {e['seed']} | {e['seed_offset']} | "
                f"{e['construction_attempts']} | {e['n']} | {e['k']} | "
                f"{e['classical_rank']} | {e['d_classical']} | "
                f"{e['quantum_d']} ({e['quantum_d_method']}) | {fingerprint}… |"
            )
    return "\n".join(lines) + "\n"
