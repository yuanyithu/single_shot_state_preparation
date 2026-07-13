"""tests 共享的微型全枚举工具（独立于 src/enumerate_exact，供采样器对拍）。"""

import numpy as np

from src.gf2 import gf2_matmul
from src.observables import observable_values


def enum_m_u(model, observable_set, wiring):
    """全空间（或 q=0 coset）精确 m_u；n ≤ 16 用。返回 (m_u, state_probs dict)。"""
    n = model.num_qubits
    weights = {}
    total_m = np.zeros(observable_set.num_u, dtype=np.float64)
    norm = 0.0
    for value in range(1 << n):
        v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
        syndrome_term = (
            gf2_matmul(model.H_check, v[:, None])[:, 0]
            ^ wiring.gibbs_syndrome_argument
        )
        weight_s = int(syndrome_term.sum())
        if wiring.q_zero:
            if weight_s:
                continue
            log_weight = -wiring.K_p * float(v.sum())
        else:
            log_weight = (
                -wiring.K_p * float(v.sum()) - wiring.K_q * float(weight_s)
            )
        weight = float(np.exp(log_weight))
        weights[value] = weight
        total_m += weight * observable_values(
            observable_set, wiring, v
        ).astype(np.float64)
        norm += weight
    probabilities = {state: w / norm for state, w in weights.items()}
    return total_m / norm, probabilities


def enum_class_weights(model, frame, wiring):
    """绝对 label 的精确类权重 P(ℓ)，shape (2^k,)；n ≤ 16 用。"""
    from src.gf2 import gf2_matmul as _mm

    k = model.k
    n = model.num_qubits
    class_weights = np.zeros(1 << k, dtype=np.float64)
    for value in range(1 << n):
        v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
        syndrome_term = (
            _mm(model.H_check, v[:, None])[:, 0]
            ^ wiring.gibbs_syndrome_argument
        )
        weight_s = int(syndrome_term.sum())
        if wiring.q_zero:
            if weight_s:
                continue
            log_weight = -wiring.K_p * float(v.sum())
        else:
            log_weight = (
                -wiring.K_p * float(v.sum()) - wiring.K_q * float(weight_s)
            )
        label = frame.label_of(v)
        bits = 0
        for b, val in enumerate(label):
            if val:
                bits |= 1 << b
        class_weights[bits] += float(np.exp(log_weight))
    return class_weights / class_weights.sum()
