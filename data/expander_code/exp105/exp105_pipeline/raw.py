from numbers import Integral
from pathlib import Path

import numpy as np

from .config import ensure_config, tasks_per_m
from .io import atomic_npz, sha256_file
from .worker import RAW_FIELDS, _ARRAY_FIELDS


ARRAY_FIELDS = set(_ARRAY_FIELDS)


def task_counts(config):
    config = ensure_config(config)
    counts = {int(key): int(value) for key, value in config["codes_per_m"].items()}
    sizes = {int(key): int(value) for key, value in config["codes_per_task"].items()}
    return tasks_per_m(counts, sizes)


def raw_filename(config, m, block_index):
    config = ensure_config(config)
    m = int(m)
    if m not in config["m_values"]:
        raise ValueError(f"m is outside the frozen panel: {m!r}")
    if isinstance(block_index, bool) or not isinstance(block_index, Integral):
        raise ValueError("block index must be an integer")
    if not 0 <= int(block_index) < task_counts(config)[m]:
        raise ValueError("block index is outside the frozen plan")
    return f"m{m:02d}__b{int(block_index):04d}.npz"


def save_raw(path, raw, refuse_overwrite=True):
    path = Path(path)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"raw task already exists: {path}")
    if set(raw) != RAW_FIELDS:
        raise ValueError("raw payload fields do not match exp105.raw.v1")
    atomic_npz(path, {key: np.asarray(value) for key, value in raw.items()})
    return sha256_file(path)


def load_raw(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != RAW_FIELDS:
            raise ValueError("raw NPZ fields do not match exp105.raw.v1")
        raw = {}
        for key in data.files:
            value = data[key]
            raw[key] = value.copy() if key in ARRAY_FIELDS else value.item()
    return raw
