"""Small reproducible oracle-vs-Numba q=0 PT benchmark."""

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.q0_pt import (  # noqa: E402
    Q0PtConfig, _run_q0_pt_numba_core, run_q0_pt_instance,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code  # noqa: E402
from data.expander_code.exp102.exp102_pipeline.worker import build_model  # noqa: E402


ARRAY_FIELDS = (
    "labels", "ladder_K", "ladder_p", "swap_attempts", "swap_accepts",
    "logical_attempts", "logical_accepts", "hot_arrival_labels",
    "hot_departure_labels",
)
SCALAR_FIELDS = (
    "round_trips", "sector_changing_round_trips", "max_hard_coset_residual",
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-id", default="m03_c00")
    parser.add_argument("--num-temperatures", type=int, default=8)
    parser.add_argument("--burn-rounds", type=int, default=20)
    parser.add_argument("--measurement-rounds", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    registry_path = PROJECT_ROOT / "data/expander_code/exp102/registry/registry.json"
    _, code, H = load_frozen_code(registry_path, args.code_id)
    model, frame = build_model(H)
    epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
    epsilon[[0, min(3, model.num_qubits - 1), min(7, model.num_qubits - 1)]] = 1
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    config = Q0PtConfig(
        p_hot=0.45,
        num_temperatures=args.num_temperatures,
        gamma=1.0,
        burn_rounds=args.burn_rounds,
        measurement_rounds=args.measurement_rounds,
    )

    compile_started = perf_counter()
    run_q0_pt_instance(model, frame, syndrome, 0.1, config, 12345, 0, engine="numba")
    compile_and_warm_seconds = perf_counter() - compile_started

    timings = {"reference": [], "numba": []}
    latest = {}
    for engine in ("reference", "numba"):
        for _ in range(args.repeats):
            started = perf_counter()
            latest[engine] = run_q0_pt_instance(
                model, frame, syndrome, 0.1, config, 12345, 0, engine=engine
            )
            timings[engine].append(perf_counter() - started)

    for field in ARRAY_FIELDS:
        if not np.array_equal(latest["reference"][field], latest["numba"][field]):
            raise AssertionError(f"oracle mismatch: {field}")
    for field in SCALAR_FIELDS:
        if latest["reference"][field] != latest["numba"][field]:
            raise AssertionError(f"oracle mismatch: {field}")

    reference_seconds = statistics.median(timings["reference"])
    numba_seconds = statistics.median(timings["numba"])
    result = {
        "benchmark": "exp102.q0_pt.full_round.v1",
        "code_id": args.code_id,
        "m": int(code["m"]),
        "n": int(model.num_qubits),
        "k": int(model.k),
        "num_temperatures": args.num_temperatures,
        "burn_rounds": args.burn_rounds,
        "measurement_rounds": args.measurement_rounds,
        "repeats": args.repeats,
        "compile_and_warm_seconds": compile_and_warm_seconds,
        "reference_median_seconds": reference_seconds,
        "numba_median_seconds": numba_seconds,
        "speedup": reference_seconds / numba_seconds,
        "bit_identical": True,
        "nopython_kernel": bool(_run_q0_pt_numba_core.nopython_signatures),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="ascii")
    print(rendered, end="")


if __name__ == "__main__":
    main()
