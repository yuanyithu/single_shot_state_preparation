"""Track B: the transport-free q_top anchor, kept out of the measured package.

`exp105_pipeline` is identity-bound: its `source_tree_sha256` is recorded in the
frozen configs, in every raw file and in the published aggregate. Track B shares
no code path with Track A's measurement, so living here keeps that identity
intact -- adding a module to `exp105_pipeline` after the measurement would make
the live tree disagree with the artifacts bound to it.

This package also isolates its numba cache, and that is not cosmetic. exp101's
`prng.py` and `fast_mcmc.py` compile cached kernels under whichever module name
imported them: `src.prng` when exp101's own suite runs, `exp101_certified_src.prng`
when the bridge loads it. Numba's on-disk cache cannot serve both, so a shared
cache directory makes whichever suite runs second fail with
`ModuleNotFoundError: No module named 'src.prng'` -- a failure that looks like a
broken package and is really a cache collision. `setdefault` leaves an explicit
override in place, which is what the nd-3 runner relies on.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).resolve().parent / ".numba-cache"),
)
