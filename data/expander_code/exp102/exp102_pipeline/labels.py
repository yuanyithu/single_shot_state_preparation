import numpy as np


def bits_to_uint64(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError("label bits must be one-dimensional")
    if bits.size > 64:
        raise ValueError("logical labels with k>64 are unsupported")
    value = np.uint64(0)
    for bit in np.flatnonzero(bits):
        value |= np.uint64(1) << np.uint64(bit)
    return value


def uint64_to_bits(label, k):
    if not 0 <= int(k) <= 64:
        raise ValueError("logical labels require 0<=k<=64")
    value = np.uint64(label)
    return np.array([(value >> np.uint64(i)) & np.uint64(1) for i in range(k)], dtype=np.uint8)


def initial_labels(k):
    if not 0 <= int(k) <= 64:
        raise ValueError("logical labels with k>64 are unsupported")
    all_ones = np.uint64((1 << int(k)) - 1) if k < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
    even = np.uint64(0)
    odd = np.uint64(0)
    for i in range(k):
        if i % 2:
            odd |= np.uint64(1) << np.uint64(i)
        else:
            even |= np.uint64(1) << np.uint64(i)
    return np.array([0, all_ones, even, odd], dtype=np.uint64)


def pairwise_collision(label_traces, k):
    traces = [np.asarray(trace, dtype=np.uint64) for trace in label_traces]
    if len(traces) != 4 or any(trace.size == 0 for trace in traces):
        raise ValueError("collision estimator requires four nonempty traces")
    collisions = []
    for a in range(4):
        values_a, counts_a = np.unique(traces[a], return_counts=True)
        probs_a = dict(zip(values_a.tolist(), counts_a / traces[a].size))
        for b in range(a + 1, 4):
            values_b, counts_b = np.unique(traces[b], return_counts=True)
            probs_b = dict(zip(values_b.tolist(), counts_b / traces[b].size))
            collisions.append(sum(prob * probs_b.get(label, 0.0) for label, prob in probs_a.items()))
    collision = float(np.mean(collisions))
    uniform = 2.0 ** (-int(k))
    return collision, (collision - uniform) / (1.0 - uniform)
