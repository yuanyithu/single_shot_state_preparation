"""Launch an ordered list of exp102 ladder candidates from nd-0."""

import argparse
import shlex
import subprocess
import time
from pathlib import Path


WORKERS = {"nd-2": 75, "nd-3": 91}
RELATIVE = Path("data/expander_code/exp102/validation/002_numba_smoke_20260719")


def remote_command(arguments):
    return " ".join(shlex.quote(str(value)) for value in arguments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True); parser.add_argument("--source-commit", required=True)
    parser.add_argument("--p-hot", required=True, type=float); parser.add_argument("--m-values", required=True)
    parser.add_argument("--first-attempt", required=True, type=int)
    parser.add_argument("--temperatures", required=True, help="comma-separated ordered R values")
    args = parser.parse_args()
    source = Path.home() / ".single_shot/repos" / args.run_id / "source"
    run_root = Path.home() / ".single_shot/runs" / args.run_id
    for offset, temperatures in enumerate(int(value) for value in args.temperatures.split(",")):
        attempt = args.first_attempt + offset
        for node, workers in WORKERS.items():
            stage_dir = run_root / "ladder" / f"attempt_{attempt:03d}" / node
            log = Path.home() / ".single_shot/logs" / f"{args.run_id}_ladder_a{attempt:03d}_{node}.log"
            shell = (
                f"cd {shlex.quote(str(source))}; "
                f"export PYTHONPATH=. NUMBA_CACHE_DIR={shlex.quote(str(source.parent / ('numba-cache-' + node)))} "
                "NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; "
                f"bash {RELATIVE / 'run_stage_wrapper.sh'} {shlex.quote(str(stage_dir))} {shlex.quote(str(log))} "
                f"conda run -n 11 --no-capture-output python {RELATIVE / 'run_ladder_stage.py'} "
                f"{node} --num-workers {workers} --run-id {args.run_id} "
                f"--source-commit {args.source_commit} --stage ladder --attempt {attempt} "
                f"--p-hot {args.p_hot} --num-temperatures {temperatures} --gamma 1.0 "
                f"--burn-rounds 500 --measurement-rounds 2000 --m-values {args.m_values}"
            )
            command = remote_command(("screen", "-dmS", f"exp102_ladder_a{attempt:03d}_{node}",
                                      "bash", "-lc", shell))
            subprocess.run(("ssh", node, command), check=True)
        while True:
            if any((run_root / "ladder" / f"attempt_{attempt:03d}" / node / "FAILED").exists()
                   for node in WORKERS):
                raise RuntimeError(f"ladder attempt {attempt} failed")
            if all((run_root / "ladder" / f"attempt_{attempt:03d}" / node / "SUCCESS").exists()
                   for node in WORKERS):
                print(f"attempt={attempt} R={temperatures} SUCCESS", flush=True)
                break
            time.sleep(2)


if __name__ == "__main__":
    main()
