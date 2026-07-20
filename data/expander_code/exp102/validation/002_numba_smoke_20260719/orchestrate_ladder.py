"""Launch an ordered list of exp102 ladder candidates from nd-0."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


WORKERS = {"nd-2": 75, "nd-3": 91}
RELATIVE = Path("data/expander_code/exp102/validation/002_numba_smoke_20260719")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def remote_command(arguments):
    return " ".join(shlex.quote(str(value)) for value in arguments)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_shared_deployment(deployment_root, source_commit, archive_sha256, manifest_sha256):
    deployment_root = Path(deployment_root)
    required = {
        "source": deployment_root / "source",
        "archive": deployment_root / "SOURCE.tar",
        "commit": deployment_root / "SOURCE_COMMIT",
        "archive_marker": deployment_root / "ARCHIVE_SHA256",
        "manifest": deployment_root / "SOURCE_MANIFEST.json",
    }
    if (not required["source"].is_dir()
            or any(not path.is_file() for key, path in required.items() if key != "source")):
        raise ValueError("shared deployment bundle is incomplete")
    if required["commit"].read_text(encoding="ascii").strip() != source_commit:
        raise ValueError("shared deployment commit marker mismatch")
    if required["archive_marker"].read_text(encoding="ascii").strip() != archive_sha256:
        raise ValueError("shared deployment archive marker mismatch")
    if _sha256_file(required["archive"]) != archive_sha256:
        raise ValueError("shared deployment archive SHA256 mismatch")
    if _sha256_file(required["manifest"]) != manifest_sha256:
        raise ValueError("shared deployment manifest SHA256 mismatch")
    manifest = json.loads(required["manifest"].read_text(encoding="ascii"))
    if (manifest.get("source_commit") != source_commit
            or manifest.get("archive_sha256") != archive_sha256):
        raise ValueError("shared deployment manifest identity mismatch")


def verified_bootstrap(deployment_root, source_commit, archive_sha256,
                       manifest_sha256, stage_dir, log_file, command):
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if SHA256_PATTERN.fullmatch(archive_sha256) is None:
        raise ValueError("archive SHA256 must be 64 lowercase hex characters")
    if SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise ValueError("manifest SHA256 must be 64 lowercase hex characters")
    archive = deployment_root / "SOURCE.tar"
    verifier = RELATIVE / "run_verified_source.sh"
    wrapper = RELATIVE / "run_stage_wrapper.sh"
    guarded_command = (
        "set -euo pipefail; "
        f"tar -xOf {shlex.quote(str(archive))} {shlex.quote(verifier.as_posix())} "
        f"| bash -s -- {shlex.quote(str(deployment_root))} "
        f"{shlex.quote(source_commit)} {shlex.quote(archive_sha256)} "
        f"{shlex.quote(manifest_sha256)} {remote_command(command)}"
    )
    return (
        "set -euo pipefail; "
        f"printf '%s  %s\\n' {shlex.quote(archive_sha256)} {shlex.quote(str(archive))} "
        "| sha256sum -c - >/dev/null; "
        f"tar -xOf {shlex.quote(str(archive))} {shlex.quote(wrapper.as_posix())} "
        f"| bash -s -- {shlex.quote(str(stage_dir))} {shlex.quote(str(log_file))} "
        f"bash -c {shlex.quote(guarded_command)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True); parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--p-hot", required=True, type=float); parser.add_argument("--m-values", required=True)
    parser.add_argument("--first-attempt", required=True, type=int)
    parser.add_argument("--temperatures", required=True, help="comma-separated ordered R values")
    parser.add_argument("--attempt-timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run id must contain only letters, digits, dot, underscore, and dash")
    if args.attempt_timeout_seconds <= 0:
        raise ValueError("attempt timeout must be positive")
    deployment_root = Path.home() / ".single_shot/repos" / args.run_id
    verify_shared_deployment(
        deployment_root, args.source_commit, args.archive_sha256, args.manifest_sha256,
    )
    for node in WORKERS:
        subprocess.run(("ssh", node, "true"), check=True)
    source = deployment_root / "source"
    run_root = Path.home() / ".single_shot/runs" / args.run_id
    for offset, temperatures in enumerate(int(value) for value in args.temperatures.split(",")):
        attempt = args.first_attempt + offset
        conflicts = [
            run_root / "ladder" / f"attempt_{attempt:03d}" / node / marker
            for node in WORKERS for marker in ("RUNNING", "SUCCESS", "FAILED")
            if (run_root / "ladder" / f"attempt_{attempt:03d}" / node / marker).exists()
        ]
        if conflicts:
            raise FileExistsError("ladder stage markers already exist: " + ", ".join(map(str, conflicts)))
        for node, workers in WORKERS.items():
            stage_dir = run_root / "ladder" / f"attempt_{attempt:03d}" / node
            log = Path.home() / ".single_shot/logs" / f"{args.run_id}_ladder_a{attempt:03d}_{node}.log"
            stage_command = (
                "env",
                f"NUMBA_CACHE_DIR={source.parent / ('numba-cache-' + node)}",
                "NUMBA_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "conda", "run", "-n", "11", "--no-capture-output", "python",
                RELATIVE / "run_ladder_stage.py", node,
                "--num-workers", workers, "--run-id", args.run_id,
                "--source-commit", args.source_commit, "--stage", "ladder",
                "--attempt", attempt, "--p-hot", args.p_hot,
                "--num-temperatures", temperatures, "--gamma", "1.0",
                "--burn-rounds", "500", "--measurement-rounds", "2000",
                "--m-values", args.m_values,
            )
            shell = verified_bootstrap(
                deployment_root, args.source_commit, args.archive_sha256,
                args.manifest_sha256, stage_dir, log, stage_command,
            )
            command = remote_command(("screen", "-dmS",
                                      f"exp102_{args.run_id}_ladder_a{attempt:03d}_{node}",
                                      "bash", "-lc", shell))
            subprocess.run(("ssh", node, command), check=True)
        deadline = time.monotonic() + args.attempt_timeout_seconds
        while True:
            if any((run_root / "ladder" / f"attempt_{attempt:03d}" / node / "FAILED").exists()
                   for node in WORKERS):
                raise RuntimeError(f"ladder attempt {attempt} failed")
            if all((run_root / "ladder" / f"attempt_{attempt:03d}" / node / "SUCCESS").exists()
                   for node in WORKERS):
                print(f"attempt={attempt} R={temperatures} SUCCESS", flush=True)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ladder attempt {attempt} produced no terminal marker within "
                    f"{args.attempt_timeout_seconds:g} seconds"
                )
            time.sleep(2)


if __name__ == "__main__":
    main()
