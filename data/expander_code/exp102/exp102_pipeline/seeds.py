import hashlib


def derive_seed(namespace, *parts):
    payload = ":".join(["exp102", str(namespace), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)
