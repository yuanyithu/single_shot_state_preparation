import hashlib

import numpy as np


def wilson_interval(failures, trials, confidence=0.95):
    if trials <= 0:
        return np.nan, np.nan
    if confidence != 0.95:
        raise ValueError("exp103 freezes 95% Wilson intervals")
    z = 1.959963984540054
    rate = failures / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    return center - radius, center + radius


def _bootstrap_seed(config, label):
    payload = ":".join([
        config["master_seed_hex"], config["namespaces"]["bootstrap"],
        config["registry_sha256"], label,
    ])
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)


def simultaneous_bootstrap(code_failures, code_trials, m_indices, config, label):
    """Grouped-code plus parametric-binomial bootstrap with one max-deviation band."""
    m_indices = tuple(int(value) for value in m_indices)
    indices = np.asarray(m_indices, dtype=np.int64)
    failures = np.asarray(code_failures, dtype=np.int64)[indices, :, :]
    trials = np.asarray(code_trials, dtype=np.int64)[indices, :, :]
    if failures.shape[1:] != (8, 13) or np.any(trials <= 0):
        raise ValueError("bootstrap input must be a complete 8-code by 13-point panel")
    rates = failures / trials
    point = rates.mean(axis=1)
    adjacent_point = np.diff(point, axis=0)
    endpoint_point = point[-1] - point[0]
    coordinates = np.concatenate([point.ravel(), endpoint_point, adjacent_point.ravel()])
    replicates = int(config["bootstrap"]["replicates"])
    rng = np.random.Generator(np.random.PCG64(_bootstrap_seed(config, label)))
    max_deviation = np.empty(replicates, dtype=np.float64)
    for bootstrap_index in range(replicates):
        means = np.empty_like(point)
        for local_m in range(len(m_indices)):
            selected = rng.integers(0, 8, size=8)
            selected_trials = trials[local_m, selected]
            selected_rates = rates[local_m, selected]
            shots = rng.binomial(selected_trials, selected_rates) / selected_trials
            means[local_m] = shots.mean(axis=0)
        adjacent = np.diff(means, axis=0)
        endpoint = means[-1] - means[0]
        draw = np.concatenate([means.ravel(), endpoint, adjacent.ravel()])
        max_deviation[bootstrap_index] = np.max(np.abs(draw - coordinates))
    half_width = float(np.quantile(max_deviation, 0.95, method="higher"))
    return {
        "point": point,
        "point_low": np.clip(point - half_width, 0.0, 1.0),
        "point_high": np.clip(point + half_width, 0.0, 1.0),
        "endpoint": endpoint_point,
        "endpoint_low": np.clip(endpoint_point - half_width, -1.0, 1.0),
        "endpoint_high": np.clip(endpoint_point + half_width, -1.0, 1.0),
        "adjacent": adjacent_point,
        "adjacent_low": np.clip(adjacent_point - half_width, -1.0, 1.0),
        "adjacent_high": np.clip(adjacent_point + half_width, -1.0, 1.0),
        "half_width": half_width,
    }


def _crossing_transitions(delta):
    delta = np.asarray(delta, dtype=np.float64)
    nonzero = [(index, np.sign(value)) for index, value in enumerate(delta) if np.isfinite(value) and value != 0.0]
    correct, reverse = [], []
    for (left_index, left_sign), (right_index, right_sign) in zip(nonzero, nonzero[1:]):
        if left_sign < 0 < right_sign:
            correct.append((left_index, right_index))
        elif left_sign > 0 > right_sign:
            reverse.append((left_index, right_index))
    return correct, reverse


def crossing_brackets(delta):
    correct, reverse = _crossing_transitions(delta)
    return [left for left, _ in correct], [left for left, _ in reverse]


def _certified_bracket(delta, low, high, p_values):
    transitions, reverse_transitions = _crossing_transitions(delta)
    correct = [left for left, _ in transitions]
    reverse = [left for left, _ in reverse_transitions]
    if len(correct) != 1 or reverse:
        return None, correct, reverse
    left, right = transitions[0]
    if right == left + 1 and high[left] < 0.0 and low[right] > 0.0:
        return (float(p_values[left]), float(p_values[right])), correct, reverse
    return None, correct, reverse


def classify_final_crossing(
    p_values, delta38, delta38_low, delta38_high,
    adjacent_delta, adjacent_low, adjacent_high,
):
    p_values = np.asarray(p_values, dtype=np.float64)
    certified, correct, reverse = _certified_bracket(
        delta38, delta38_low, delta38_high, p_values,
    )
    if not correct:
        return {
            "status": "EXP103_NO_CORRECT_CROSSING_IN_WINDOW",
            "bracket": None,
            "compatible_triple": None,
            "correct_reversals": 0,
            "reverse_reversals": len(reverse),
        }
    if len(correct) != 1 or reverse:
        return {
            "status": "EXP103_DECODER_CROSSING_INCONCLUSIVE",
            "bracket": None,
            "compatible_triple": None,
            "correct_reversals": len(correct),
            "reverse_reversals": len(reverse),
        }
    compatible_triple = None
    triple_intersections = []
    adjacent_conflict = False
    for triple_index in range(4):
        first, first_correct, first_reverse = _certified_bracket(
            adjacent_delta[triple_index], adjacent_low[triple_index],
            adjacent_high[triple_index], p_values,
        )
        second, second_correct, second_reverse = _certified_bracket(
            adjacent_delta[triple_index + 1], adjacent_low[triple_index + 1],
            adjacent_high[triple_index + 1], p_values,
        )
        adjacent_conflict = adjacent_conflict or (
            len(first_correct) > 1 or bool(first_reverse)
            or len(second_correct) > 1 or bool(second_reverse)
        )
        if first is not None and second is not None:
            if max(first[0], second[0]) <= min(first[1], second[1]):
                triple_intersections.append((
                    max(first[0], second[0]), min(first[1], second[1]),
                    [3 + triple_index, 4 + triple_index, 5 + triple_index],
                ))
    if certified is None:
        status = "EXP103_DECODER_CROSSING_INCONCLUSIVE"
        bracket = None
    elif adjacent_conflict:
        status = "EXP103_DECODER_CROSSING_INCONCLUSIVE"
        bracket = None
    elif not triple_intersections:
        status = "EXP103_PAIRWISE_BRACKET_ONLY"
        bracket = certified
    else:
        common_low = max([certified[0], *(item[0] for item in triple_intersections)])
        common_high = min([certified[1], *(item[1] for item in triple_intersections)])
        if common_low <= common_high:
            compatible_triple = triple_intersections[0][2]
            status = "EXP103_DECODER_CROSSING_RESOLVED"
            bracket = certified
        else:
            status = "EXP103_DECODER_CROSSING_INCONCLUSIVE"
            bracket = None
    return {
        "status": status,
        "bracket": bracket,
        "compatible_triple": compatible_triple,
        "correct_reversals": len(correct),
        "reverse_reversals": len(reverse),
    }
