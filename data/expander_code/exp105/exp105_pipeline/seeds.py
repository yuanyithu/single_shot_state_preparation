import hashlib
from numbers import Integral

from .config import ensure_config, normalize_p_token, normalize_q_token


def _digest_seed(payload):
    return int.from_bytes(
        hashlib.sha256(payload.encode("ascii")).digest()[:8], "big",
    ) & ((1 << 63) - 1)


def candidate_seed(config, m, candidate_index):
    """Seed for one ensemble candidate, before the acceptance rule is applied.

    The stream is indexed by candidate rather than by accepted code so that the
    rejected candidates stay reconstructible and the acceptance rate is itself
    auditable evidence.
    """
    config = ensure_config(config)
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, Integral):
        raise ValueError("candidate index must be an integer")
    payload = ":".join([
        config["master_seed_hex"], config["namespaces"]["ensemble"],
        str(int(m)), str(int(candidate_index)),
    ])
    return _digest_seed(payload)


def derive_seed(config, namespace_key, code_id, p, trial_index=0):
    """Seed for one (code, p) trial stream at the frozen q.

    q enters the payload even though it is fixed, so that an exp105 stream can
    never collide with an exp104 stream at the same (code, p) by construction,
    and so that the q = 0 equality path is a visibly different stream.
    """
    config = ensure_config(config)
    if namespace_key not in config["namespaces"]:
        raise ValueError(f"unknown seed namespace {namespace_key!r}")
    token = normalize_p_token(p, config["p_tokens"])
    q_token = normalize_q_token(config["q_token"])
    if isinstance(trial_index, bool) or not isinstance(trial_index, Integral):
        raise ValueError("trial index must be an integer")
    payload = ":".join([
        config["master_seed_hex"], config["namespaces"][namespace_key],
        config["registry_sha256"], str(code_id), token, q_token,
        str(int(trial_index)),
    ])
    return _digest_seed(payload)


def replay_selection_seed(config):
    """Seed that fixes the committed replay subsample before production runs."""
    config = ensure_config(config)
    payload = ":".join([
        config["master_seed_hex"], config["namespaces"]["replay"],
        config["registry_sha256"], config["q_token"],
        "replay-block-selection",
    ])
    return _digest_seed(payload)
