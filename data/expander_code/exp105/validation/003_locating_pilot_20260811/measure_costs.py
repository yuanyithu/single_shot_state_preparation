"""Outcome-blind cost benchmark for the frozen allocation rule.

The allocation rule (EXPERIMENT_CONTRACT.md section 6) needs a per-code setup
cost `kappa_m` and a per-trial cost `c_m` at every `m` in the production panel,
while the locating pilot only runs `m = 3` and `m = 8`. This script measures the
missing costs directly.

It is outcome blind by construction: it times the real worker path but records
only seconds and peak memory. No failure flag, logical label or convergence flag
is written anywhere, so nothing measured here can influence anything except the
budget arithmetic. Codes 0 and 1 of the production ensemble namespace are used
because they are the codes the production run will actually build.

Per-trial cost is taken as the **maximum** over the benchmarked grid points, so
the number the rule consumes is an upper bound rather than an expectation.
"""

import json
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from data.expander_code.exp105.exp105_pipeline.config import (
    M_VALUES,
    NAMESPACES,
    Q_TOKEN,
    PILOT_P_TOKENS,
)
from data.expander_code.exp105.exp105_pipeline.ensemble import generate_codes
from data.expander_code.exp105.exp105_pipeline.io import atomic_json, sha256_json
from data.expander_code.exp105.exp105_pipeline.model import (
    clear_model_cache,
    load_model,
    logical_label,
    parity_product,
)

from ldpc import BpOsdDecoder


CODES_PER_M = 2
TRIALS = 12
# First, middle and last of the pilot grid: cost varies with p only through
# belief propagation, and these bracket the range the production grid can land
# in whichever way the grid rule resolves.
BENCH_P_TOKENS = [
    PILOT_P_TOKENS[0],
    PILOT_P_TOKENS[len(PILOT_P_TOKENS) // 2],
    PILOT_P_TOKENS[-1],
]


def _rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2)


def _decoder(model, p, q):
    width = model.n + model.n_checks
    return BpOsdDecoder(
        model.H_augmented_sparse,
        error_channel=[float(p)] * model.n + [float(q)] * model.n_checks,
        bp_method="product_sum",
        max_iter=width,
        schedule="serial",
        serial_schedule_order=list(range(width)),
        osd_method="osd_0",
        osd_order=0,
        omp_thread_count=1,
    )


def main():
    q = float(Q_TOKEN)
    rows = []
    for m in M_VALUES:
        panel, _ = generate_codes(
            m, CODES_PER_M, namespace=NAMESPACES["ensemble"],
        )
        setup_seconds = []
        per_trial = {}
        for row in panel:
            clear_model_cache()
            start = time.perf_counter()
            model = load_model(row)
            setup_seconds.append(time.perf_counter() - start)
            for token in BENCH_P_TOKENS:
                decoder = _decoder(model, float(token), q)
                rng = np.random.Generator(np.random.PCG64(0xC057 + m))
                start = time.perf_counter()
                for _ in range(TRIALS):
                    error = (rng.random(model.n) < float(token)).astype(np.uint8)
                    readout = (rng.random(model.n_checks) < q).astype(np.uint8)
                    effective = np.bitwise_xor(
                        parity_product(model.H_Z, error), readout,
                    )
                    correction = decoder.decode(effective)
                    # Exercise the scoring path so its cost is included, then
                    # discard the verdict without recording it anywhere.
                    logical_label(
                        model,
                        np.bitwise_xor(error, np.asarray(correction)[:model.n]),
                    )
                elapsed = (time.perf_counter() - start) / TRIALS
                per_trial.setdefault(token, []).append(elapsed)
        clear_model_cache()
        rows.append({
            "m": int(m),
            "n": 25 * m ** 2,
            "n_checks": 12 * m ** 2,
            "k": m ** 2,
            "codes_benchmarked": CODES_PER_M,
            "trials_per_point": TRIALS,
            "kappa_seconds_upper": float(max(setup_seconds)),
            "per_trial_seconds_by_p": {
                token: float(max(values)) for token, values in per_trial.items()
            },
            "c_seconds_upper": float(
                max(max(values) for values in per_trial.values())
            ),
        })
        print(
            f"m={m} kappa={rows[-1]['kappa_seconds_upper']:.3f}s "
            f"c={rows[-1]['c_seconds_upper']:.4f}s/trial",
            flush=True,
        )

    core = {
        "schema_version": "exp105.cost_benchmark.v1",
        "outcome_blind": True,
        "q": Q_TOKEN,
        "benchmark_p_tokens": BENCH_P_TOKENS,
        "seed_namespace": NAMESPACES["ensemble"],
        "device": sys.platform,
        "peak_rss_gib": _rss_gib(),
        "per_m": rows,
    }
    report = dict(core, report_sha256=sha256_json(core))
    output = Path(__file__).resolve().parent / "cost_benchmark.json"
    atomic_json(output, report)
    print(f"wrote {output.name} sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
