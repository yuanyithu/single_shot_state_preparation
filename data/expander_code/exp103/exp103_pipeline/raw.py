from pathlib import Path
from numbers import Integral

import numpy as np

from .config import normalize_p_token
from .io import atomic_npz, sha256_file
from .worker import RAW_FIELDS


ARRAY_FIELDS = {
    "failure_flags", "logical_labels", "syndrome_match", "bp_converged", "bp_iterations",
}


def raw_filename(code_id, p, shard_index):
    token = normalize_p_token(p).replace(".", "p")
    if isinstance(shard_index, bool) or not isinstance(shard_index, Integral):
        raise ValueError("shard index must be an integer")
    return f"{code_id}__p{token}__s{int(shard_index):02d}.npz"


def save_raw(path, raw, refuse_overwrite=True):
    path = Path(path)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"raw shard already exists: {path}")
    if set(raw) != RAW_FIELDS:
        raise ValueError("raw payload fields do not match exp103.raw.v1")
    values = {}
    for key, value in raw.items():
        values[key] = np.asarray(value) if key in ARRAY_FIELDS else np.asarray(value)
    atomic_npz(path, values)
    return sha256_file(path)


def load_raw(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != RAW_FIELDS:
            raise ValueError("raw NPZ fields do not match exp103.raw.v1")
        raw = {}
        for key in data.files:
            value = data[key]
            raw[key] = value.copy() if key in ARRAY_FIELDS else value.item()
    return raw
