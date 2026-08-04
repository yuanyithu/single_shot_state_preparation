import hashlib
from numbers import Integral

from .config import ensure_config, normalize_p_token


def derive_seed(config, namespace_key, code_id, p, shard_index):
    config = ensure_config(config)
    if namespace_key not in config["namespaces"]:
        raise ValueError(f"unknown seed namespace {namespace_key!r}")
    token = normalize_p_token(p)
    if isinstance(shard_index, bool) or not isinstance(shard_index, Integral):
        raise ValueError("shard index must be an integer")
    payload = ":".join([
        config["master_seed_hex"], config["namespaces"][namespace_key],
        config["registry_sha256"], str(code_id), token, str(int(shard_index)),
    ])
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)
