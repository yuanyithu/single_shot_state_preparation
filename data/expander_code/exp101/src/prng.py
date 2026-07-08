"""可移植 PRNG（G2.4）：python 与 numba 双胞胎，逐位一致。

目的：reference 与 fast 两引擎共用同一随机流 ⇒ bit 级轨迹一致成为可测 gate。
算法：splitmix64 播种 → xorshift128+；random() = (next >> 11) * 2^-53；
randbelow(n) = next % n（n ≪ 2^64，取模偏差 ~n/2^64 物理可忽略；且提议分布与
状态无关 ⇒ Metropolis 详细平衡严格成立，正确性不依赖均匀性完美）。
permutation(n) = Fisher–Yates（i 从 n−1 到 1，j=randbelow(i+1)）。

python 侧全程 64 位掩码；numba 侧 uint64 自然回绕——tests 验证两侧序列逐位一致。
"""

import numpy as np

_MASK = (1 << 64) - 1
_INV_2_53 = 1.0 / 9007199254740992.0  # 2^-53

try:  # numba 可选
    from numba import njit, uint64

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False


def splitmix64_stream(seed, count):
    """播种流（python 权威实现）。"""
    z = seed & _MASK
    out = []
    for _ in range(count):
        z = (z + 0x9E3779B97F4A7C15) & _MASK
        x = z
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
        x = x ^ (x >> 31)
        out.append(x)
    return out


class PortablePrng:
    """python 侧权威实现。"""

    def __init__(self, seed):
        s = splitmix64_stream(int(seed), 2)
        self.s0, self.s1 = s[0], s[1]
        if self.s0 == 0 and self.s1 == 0:
            self.s1 = 1  # xorshift 全零态禁止

    def next_uint64(self):
        x = self.s0
        y = self.s1
        self.s0 = y
        x = (x ^ (x << 23)) & _MASK
        x = x ^ (x >> 17)
        x = x ^ y ^ (y >> 26)
        self.s1 = x
        return (x + y) & _MASK

    def random(self):
        return (self.next_uint64() >> 11) * _INV_2_53

    def randbelow(self, n):
        return int(self.next_uint64() % n)

    def permutation(self, n):
        arr = np.arange(n, dtype=np.int64)
        for i in range(n - 1, 0, -1):
            j = self.randbelow(i + 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def state_array(self):
        return np.array([self.s0, self.s1], dtype=np.uint64)


def prng_state_from_seed(seed):
    return PortablePrng(seed).state_array()


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def nb_next_uint64(state):
        x = state[0]
        y = state[1]
        state[0] = y
        x = x ^ (x << uint64(23))
        x = x ^ (x >> uint64(17))
        x = x ^ y ^ (y >> uint64(26))
        state[1] = x
        return x + y

    @njit(cache=True)
    def nb_random(state):
        return float(nb_next_uint64(state) >> uint64(11)) * _INV_2_53

    @njit(cache=True)
    def nb_randbelow(state, n):
        return int(nb_next_uint64(state) % uint64(n))

    @njit(cache=True)
    def nb_fill_permutation(state, buffer):
        n = buffer.shape[0]
        for i in range(n):
            buffer[i] = i
        for i in range(n - 1, 0, -1):
            j = nb_randbelow(state, i + 1)
            tmp = buffer[i]
            buffer[i] = buffer[j]
            buffer[j] = tmp
else:  # pragma: no cover
    nb_next_uint64 = None
    nb_random = None
    nb_randbelow = None
    nb_fill_permutation = None
