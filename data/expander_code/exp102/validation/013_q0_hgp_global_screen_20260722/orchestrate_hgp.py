"""Orchestrate the immutable 24-hour HGP diagnostic on nd-0.

This module never calls the scientific workflow directly.  Every workflow
action is launched in a fresh remote screen through the verified source
archive and ``run_hgp_wrapper.sh``.  The wrapper's immutable SUCCESS markers
form the only authority chain between stages.

The default ``preflight`` phase stops after aggregate runtime/digest consensus
so the frozen artifact can be replayed locally.  A separate explicit
``measurement`` phase revalidates that authority chain before freezing the
control and launching any sampler task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import tarfile
import time


WORKFLOW_MODULE = (
    "data.expander_code.exp102.validation."
    "013_q0_hgp_global_screen_20260722.workflow"
)
VERIFY_RELATIVE = (
    "data/expander_code/exp102/validation/"
    "002_numba_smoke_20260719/run_verified_source.sh"
)
WRAPPER_RELATIVE = (
    "data/expander_code/exp102/validation/"
    "013_q0_hgp_global_screen_20260722/run_hgp_wrapper.sh"
)
CONFIG_RELATIVE = (
    "data/expander_code/exp102/config/q0_hgp_global.screen.v1.json"
)
REGISTRY_RELATIVE = "data/expander_code/exp102/registry/registry.json"

PREFLIGHT_NODES = ("nd-1", "nd-2", "nd-3")
EXECUTION_NODES = ("nd-2", "nd-3")
LOCAL_ATTESTATION_VERSION = "exp102.q0_hgp_global.screen.local_attestation.v1"
LOCAL_SOLVER_POLICY = (
    "stored_generation_identity_exact_artifact_replay_no_local_milp"
)
ND0_PERSISTENCE_TOKEN = "exp102_q0_hgp_nd0_nohup_setsid_v1"
ND0_LAUNCHER_VERSION = "exp102.q0_hgp.nd0_nohup_setsid.v1"
SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def _read_frozen_config(deployment_root, expected_archive_sha256):
    archive = Path(deployment_root) / "SOURCE.tar"
    if _sha256_file(archive) != expected_archive_sha256:
        raise ValueError("HGP orchestrator source archive SHA mismatch")
    with tarfile.open(archive, "r:*") as handle:
        try:
            member = handle.getmember(CONFIG_RELATIVE)
        except KeyError as exc:
            raise ValueError("HGP config is absent from the source archive") from exc
        if not member.isfile():
            raise ValueError("HGP config archive member is not a regular file")
        stream = handle.extractfile(member)
        if stream is None:
            raise ValueError("HGP config archive member is not a regular file")
        payload = stream.read()
    config = json.loads(payload.decode("ascii"))
    execution = config.get("execution", {})
    capacities = execution.get("capacities", {})
    analysis = execution.get("analysis", {})
    if (tuple(execution.get("execution_nodes", ())) != EXECUTION_NODES
            or set(capacities) != set(EXECUTION_NODES)
            or any(isinstance(capacities[node], bool)
                   or int(capacities[node]) <= 0 for node in EXECUTION_NODES)
            or analysis.get("node") != "nd-3"
            or isinstance(analysis.get("capacity"), bool)
            or int(analysis.get("capacity", 0)) <= 0
            or int(analysis.get("num_workers", -1))
            != int(analysis.get("capacity", 0))):
        raise ValueError("HGP archived execution topology is invalid")
    if set(config.get("resource_tiers", {})) != {"T1", "T2", "T3"}:
        raise ValueError("HGP archived resource tiers changed")
    return config, hashlib.sha256(payload).hexdigest()


def _validate_nd0_persistence(args, base):
    if os.environ.get("EXP102_HGP_ORCHESTRATOR_PERSISTENCE") != (
            ND0_PERSISTENCE_TOKEN):
        raise ValueError("HGP orchestrator requires the nd-0 setsid launcher")
    if os.getsid(0) != os.getpid():
        raise ValueError("HGP orchestrator must be a detached session leader")

    token = hashlib.sha256(args.run_id.encode("ascii")).hexdigest()[:8]
    expected_guard = (
        Path(base) / "logs"
        / f".{args.run_id}_hgp_orchestrator_{token}_{args.phase}.launch"
    )
    supplied_guard = os.environ.get("EXP102_HGP_ORCHESTRATOR_GUARD")
    if supplied_guard is None:
        raise ValueError("HGP orchestrator launch guard is absent")
    try:
        guard = Path(supplied_guard).resolve(strict=True)
        canonical_guard = expected_guard.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("HGP orchestrator launch guard is invalid") from exc
    if (guard != canonical_guard or not guard.is_dir()
            or expected_guard.is_symlink()):
        raise ValueError("HGP orchestrator launch guard is not canonical")

    metadata_path = guard / "LAUNCH.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError("HGP orchestrator launch metadata is invalid")
    metadata = _read_json(metadata_path)
    expected_attestation_sha = (
        args.local_attestation_sha256
        if args.phase == "measurement" else None
    )
    if (set(metadata) != {
            "archive_sha256", "command_sha256", "launcher_version",
            "local_attestation_sha256", "manifest_sha256", "phase",
            "run_id", "source_commit"}
            or metadata.get("launcher_version") != ND0_LAUNCHER_VERSION
            or metadata.get("run_id") != args.run_id
            or metadata.get("phase") != args.phase
            or metadata.get("source_commit") != args.source_commit
            or metadata.get("archive_sha256") != args.archive_sha256
            or metadata.get("manifest_sha256")
            != args.source_manifest_sha256
            or metadata.get("local_attestation_sha256")
            != expected_attestation_sha
            or SHA256_RE.fullmatch(str(metadata.get("command_sha256", "")))
            is None):
        raise ValueError("HGP orchestrator launch metadata identity is invalid")

    pid_path = guard / "ORCHESTRATOR_PID"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(pid_path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(f"{os.getpid()}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("HGP orchestrator PID metadata already exists") from exc
    return guard


def _require_verified_launch(args, home):
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        raise ValueError("HGP run ID is invalid")
    if SHA1_RE.fullmatch(args.source_commit) is None:
        raise ValueError("HGP source commit must be a full lowercase SHA")
    if any(SHA256_RE.fullmatch(value) is None for value in (
            args.archive_sha256, args.source_manifest_sha256)):
        raise ValueError("HGP deployment SHA256 is invalid")
    if os.environ.get("EXP102_SOURCE_COMMIT") != args.source_commit:
        raise ValueError("HGP orchestrator itself must run from verified source")
    if platform.node().split(".", 1)[0] != "nd-0":
        raise ValueError("HGP orchestrator is owned by storage node nd-0")

    base = (Path(home).resolve() / ".single_shot")
    deployment_root = base / "repos" / args.run_id
    run_root = base / "runs" / args.run_id
    if not (base / "logs").is_dir():
        raise ValueError("HGP server log root is absent")
    if not deployment_root.is_dir():
        raise FileNotFoundError("HGP launch deployment is absent")
    if args.phase == "preflight" and run_root.exists():
        raise FileExistsError("HGP preflight requires a fresh run root")
    if args.phase == "measurement" and not run_root.is_dir():
        raise FileNotFoundError(
            "HGP measurement requires the completed preflight run root"
        )

    # Publish the detached session-leader PID before any potentially slow
    # archive I/O, so the outer launcher never times out on an untracked
    # orchestrator that can continue in its own session.
    _validate_nd0_persistence(args, base)
    archive = deployment_root / "SOURCE.tar"
    manifest = deployment_root / "SOURCE_MANIFEST.json"
    marker = deployment_root / "SOURCE_COMMIT"
    source = deployment_root / "source"
    if not (archive.is_file() and manifest.is_file() and marker.is_file()
            and source.is_dir()):
        raise ValueError("HGP deployment evidence is incomplete")
    if (_sha256_file(archive) != args.archive_sha256
            or _sha256_file(manifest) != args.source_manifest_sha256
            or marker.read_text(encoding="ascii").strip()
            != args.source_commit):
        raise ValueError("HGP deployment identity does not match launch arguments")
    return deployment_root, run_root


def _verified_stage_shell(deployment_root, source_commit, archive_sha256,
                          source_manifest_sha256, stage, stage_dir, log_file,
                          prerequisites, workflow_argv):
    archive = Path(deployment_root) / "SOURCE.tar"
    wrapper_arguments = [
        "bash", WRAPPER_RELATIVE, stage, str(stage_dir), str(log_file),
    ]
    for marker in prerequisites:
        wrapper_arguments.extend(("--require-success", str(marker)))
    wrapper_arguments.extend(("--", "python", "-m", WORKFLOW_MODULE))
    wrapper_arguments.extend(str(value) for value in workflow_argv)
    verified_arguments = [
        str(deployment_root), source_commit, archive_sha256,
        source_manifest_sha256, "conda", "run", "-n", "11",
        "--no-capture-output", *wrapper_arguments,
    ]
    checksum_line = f"{archive_sha256}  {archive}"
    return "\n".join((
        "set -euo pipefail",
        f"printf '%s\\n' {shlex.quote(checksum_line)} | sha256sum -c - >/dev/null",
        (
            f"tar -xOf {shlex.quote(str(archive))} "
            f"{shlex.quote(VERIFY_RELATIVE)} | bash -s -- "
            + shlex.join(verified_arguments)
        ),
    ))


def _remote_command(arguments):
    return shlex.join(tuple(str(value) for value in arguments))


class _Stage:
    def __init__(self, *, key, node, stage, workflow_argv, stage_dir,
                 log_file, bootstrap_log, prerequisites, session):
        self.key = key
        self.node = node
        self.stage = stage
        self.workflow_argv = tuple(workflow_argv)
        self.stage_dir = Path(stage_dir)
        self.log_file = Path(log_file)
        self.bootstrap_log = Path(bootstrap_log)
        self.prerequisites = tuple(Path(value) for value in prerequisites)
        self.session = session

    @property
    def success(self):
        return self.stage_dir / "SUCCESS"

    @property
    def failed(self):
        return self.stage_dir / "FAILED"


class HgpOrchestrator:
    def __init__(self, *, run_id, source_commit, archive_sha256,
                 source_manifest_sha256, deployment_root, run_root, config,
                 config_file_sha256, local_attestation=None,
                 local_attestation_sha256=None, poll_seconds=5.0):
        self.run_id = run_id
        self.source_commit = source_commit
        self.archive_sha256 = archive_sha256
        self.source_manifest_sha256 = source_manifest_sha256
        self.deployment_root = Path(deployment_root)
        self.run_root = Path(run_root)
        self.config = config
        self.config_file_sha256 = config_file_sha256
        self.local_attestation = (
            None if local_attestation is None else Path(local_attestation)
        )
        self.local_attestation_sha256 = local_attestation_sha256
        self.poll_seconds = float(poll_seconds)
        self.registry = REGISTRY_RELATIVE
        self.config_path = CONFIG_RELATIVE
        self.token = hashlib.sha256(run_id.encode("ascii")).hexdigest()[:8]
        self.control_root = self.run_root / "control"
        self.artifact_root = self.run_root / "hgp_global/artifacts"
        self.artifact_manifest = self.control_root / "hgp_artifacts.json"
        self.schedule = self.control_root / "HGP_GLOBAL_24H_SCHEDULE.json"
        self.preflight_root = self.run_root / "hgp_global/preflight"
        self.preflight = self.control_root / "hgp_preflight.json"
        self.control = self.control_root / "hgp_measurement_control.json"
        self.raw_root = self.run_root / "hgp_global/raw"
        self.node_report_root = self.run_root / "hgp_global/node_reports"
        self.report = self.control_root / "hgp_report.json"
        self.decision = self.control_root / "hgp_decision.json"
        self.package = self.control_root / "hgp_terminal_package.json"
        self.marker_root = self.run_root / "hgp_global/markers"
        self.log_root = Path.home() / ".single_shot/logs"

    def _common(self, *, artifact_manifest=False):
        values = [
            "--source-commit", self.source_commit,
            "--archive-sha256", self.archive_sha256,
            "--source-manifest-sha256", self.source_manifest_sha256,
            "--artifact-root", self.artifact_root,
        ]
        if artifact_manifest:
            values.extend(("--artifact-manifest", self.artifact_manifest))
        values.extend((
            "--schedule", self.schedule,
            "--registry", self.registry,
            "--config", self.config_path,
        ))
        return values

    def _stage(self, key, node, stage, action, arguments, prerequisites=()):
        safe_key = key.replace("_", "-")
        return _Stage(
            key=key, node=node, stage=stage,
            workflow_argv=(action, *arguments),
            stage_dir=self.marker_root / key,
            log_file=self.log_root / f"{self.run_id}_hgp_{safe_key}.log",
            bootstrap_log=(
                self.log_root / f"{self.run_id}_hgp_{safe_key}_bootstrap.log"
            ),
            prerequisites=prerequisites,
            session=f"e102h_{self.token}_{safe_key[:18]}",
        )

    def schedule_stage(self):
        arguments = [
            "--source-commit", self.source_commit,
            "--archive-sha256", self.archive_sha256,
            "--source-manifest-sha256", self.source_manifest_sha256,
            "--registry", self.registry, "--config", self.config_path,
            "--run-id", self.run_id, "--output", self.schedule,
        ]
        return self._stage(
            "00_schedule", "nd-1", "build-schedule", "build-schedule",
            arguments,
        )

    def artifact_stage(self, schedule_success):
        return self._stage(
            "01_artifacts", "nd-1", "build-artifacts", "build-artifacts",
            [*self._common(), "--output", self.artifact_manifest],
            (schedule_success,),
        )

    def preflight_node_stages(self, artifact_success):
        return [
            self._stage(
                f"02_preflight_{node}", node, "preflight", "preflight-node",
                [
                    node, *self._common(artifact_manifest=True),
                    "--output-root", self.preflight_root,
                ],
                (artifact_success,),
            )
            for node in PREFLIGHT_NODES
        ]

    def preflight_combine_stage(self, node_successes):
        arguments = [*self._common(artifact_manifest=True)]
        for node in PREFLIGHT_NODES:
            arguments.extend((
                "--node-report",
                f"{node}={self.preflight_root / 'nodes' / node / 'preflight.json'}",
            ))
        arguments.extend(("--output", self.preflight))
        return self._stage(
            "03_preflight_combine", "nd-1", "preflight",
            "combine-preflight", arguments, node_successes,
        )

    def control_stage(self, preflight_success):
        return self._stage(
            "04_control", "nd-1", "freeze-control", "build-control",
            [
                *self._common(artifact_manifest=True),
                "--preflight", self.preflight,
                "--output", self.control,
            ],
            (preflight_success,),
        )

    def screen_stages(self, control_success):
        capacities = self.config["execution"]["capacities"]
        return [
            self._stage(
                f"05_screen_{node}", node, "screen", "run-node",
                [
                    node, *self._common(artifact_manifest=True),
                    "--control", self.control,
                    "--preflight", self.preflight,
                    "--raw-root", self.raw_root,
                    "--output", self.node_report_root / f"{node}.json",
                    "--num-workers", int(capacities[node]),
                ],
                (control_success,),
            )
            for node in EXECUTION_NODES
        ]

    def analysis_stage(self, screen_successes):
        analysis = self.config["execution"]["analysis"]
        arguments = [
            analysis["node"], *self._common(artifact_manifest=True),
            "--control", self.control,
            "--preflight", self.preflight,
        ]
        for node in EXECUTION_NODES:
            arguments.extend((
                "--node-report", f"{node}={self.node_report_root / (node + '.json')}",
            ))
        arguments.extend((
            "--raw-root", self.raw_root,
            "--output", self.report,
            "--decision-output", self.decision,
            "--package-output", self.package,
            "--num-workers", int(analysis["num_workers"]),
        ))
        return self._stage(
            "06_analyze", analysis["node"], "analyze", "analyze",
            arguments, screen_successes,
        )

    def _launch(self, stage):
        for marker in stage.prerequisites:
            _validate_success_marker(
                marker, source_commit=self.source_commit,
            )
        if any(path.exists() for path in (
                stage.stage_dir / "RUNNING", stage.success, stage.failed,
                stage.log_file, stage.bootstrap_log)):
            raise FileExistsError(f"HGP stage evidence already exists: {stage.key}")
        verified_shell = _verified_stage_shell(
            self.deployment_root, self.source_commit, self.archive_sha256,
            self.source_manifest_sha256, stage.stage, stage.stage_dir,
            stage.log_file, stage.prerequisites, stage.workflow_argv,
        )
        # Redirect the entire bootstrap in the screen's login shell.  The
        # scientific workflow itself still writes its immutable stage log.
        wrapped_shell = "\n".join((
            "set -euo pipefail",
            f"{{\n{verified_shell}\n}} "
            f"> {shlex.quote(str(stage.bootstrap_log))} 2>&1",
        ))
        remote = _remote_command((
            "screen", "-dmS", stage.session, "bash", "-lc", wrapped_shell,
        ))
        subprocess.run((
            "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=4", stage.node, remote,
        ), check=True)
        print(_canonical_json({
            "event": "launched", "key": stage.key, "node": stage.node,
            "screen": stage.session, "stage": stage.stage,
        }), flush=True)

    def _session_alive(self, stage):
        remote = _remote_command((
            "screen", "-S", stage.session, "-Q", "select", ".",
        ))
        completed = subprocess.run((
            "ssh", "-o", "BatchMode=yes", stage.node, remote,
        ), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return completed.returncode == 0

    def _stop(self, stage):
        remote = _remote_command(("screen", "-S", stage.session, "-X", "quit"))
        subprocess.run((
            "ssh", "-o", "BatchMode=yes", stage.node, remote,
        ), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run_batch(self, stages, deadline_unix):
        stages = tuple(stages)
        if not stages:
            raise ValueError("HGP orchestration batch cannot be empty")
        if time.time() >= float(deadline_unix):
            raise TimeoutError("HGP stage deadline expired before launch")
        launched = []
        try:
            for stage in stages:
                self._launch(stage)
                launched.append(stage)
            last_probe = time.time()
            pending = {stage.key: stage for stage in stages}
            while pending:
                now = time.time()
                for key, stage in tuple(pending.items()):
                    if stage.failed.exists():
                        raise RuntimeError(
                            f"HGP stage failed: {key}: "
                            f"{stage.failed.read_text(encoding='ascii').strip()}"
                        )
                    if stage.success.exists():
                        _validate_success_marker(
                            stage.success, expected_stage=stage.stage,
                            source_commit=self.source_commit,
                        )
                        del pending[key]
                        print(_canonical_json({
                            "event": "completed", "key": key,
                            "success": str(stage.success),
                        }), flush=True)
                if not pending:
                    break
                if now >= float(deadline_unix):
                    raise TimeoutError("HGP stage batch exceeded its frozen deadline")
                if now - last_probe >= 60.0:
                    for stage in pending.values():
                        if not self._session_alive(stage):
                            raise RuntimeError(
                                f"HGP screen exited without a terminal marker: {stage.key}"
                            )
                    last_probe = now
                time.sleep(min(self.poll_seconds, max(0.1, deadline_unix - now)))
        except BaseException:
            for stage in launched:
                if not stage.success.exists() and not stage.failed.exists():
                    self._stop(stage)
            raise
        return tuple(stage.success for stage in stages)

    def run_preflight(self):
        schedule_stage = self.schedule_stage()
        self.run_batch((schedule_stage,), time.time() + 900.0)
        schedule = _validate_schedule_output(
            self.schedule, self.run_id, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )

        artifact = self.artifact_stage(schedule_stage.success)
        self.run_batch((artifact,), schedule["preflight_deadline_unix"])
        preflight_nodes = self.preflight_node_stages(artifact.success)
        node_successes = self.run_batch(
            preflight_nodes, schedule["preflight_deadline_unix"],
        )
        combine = self.preflight_combine_stage(node_successes)
        self.run_batch((combine,), schedule["preflight_deadline_unix"])
        preflight = _validate_aggregate_preflight(
            self.preflight, schedule, self.config, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )
        result = {
            "event": "preflight_ready_for_local_audit",
            "run_id": self.run_id,
            "selected_resource_tier": preflight["selected_resource_tier"],
            "schedule": str(self.schedule),
            "schedule_file_sha256": _sha256_file(self.schedule),
            "artifact_manifest": str(self.artifact_manifest),
            "artifact_manifest_file_sha256": _sha256_file(
                self.artifact_manifest,
            ),
            "artifact_root": str(self.artifact_root),
            "preflight": str(self.preflight),
            "preflight_file_sha256": _sha256_file(self.preflight),
            "control_freeze_deadline_unix": schedule[
                "control_freeze_deadline_unix"
            ],
        }
        print(_canonical_json(result), flush=True)
        return result

    def run_measurement(self):
        schedule_stage = self.schedule_stage()
        artifact = self.artifact_stage(schedule_stage.success)
        preflight_nodes = self.preflight_node_stages(artifact.success)
        combine = self.preflight_combine_stage(
            tuple(stage.success for stage in preflight_nodes),
        )
        _validate_success_marker(
            schedule_stage.success, expected_stage="build-schedule",
            source_commit=self.source_commit,
        )
        _validate_success_marker(
            artifact.success, expected_stage="build-artifacts",
            source_commit=self.source_commit,
        )
        for stage in preflight_nodes:
            _validate_success_marker(
                stage.success, expected_stage="preflight",
                source_commit=self.source_commit,
            )
        _validate_success_marker(
            combine.success, expected_stage="preflight",
            source_commit=self.source_commit,
        )
        schedule = _validate_schedule_output(
            self.schedule, self.run_id, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )
        preflight = _validate_aggregate_preflight(
            self.preflight, schedule, self.config, self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
            self.config_file_sha256,
        )
        if self.local_attestation is None:
            raise ValueError("HGP measurement requires a local attestation")
        expected_attestation = (
            self.control_root / "HGP_LOCAL_PREFLIGHT_ATTESTATION.json"
        )
        if self.local_attestation.resolve(strict=True) != expected_attestation.resolve(
                strict=True):
            raise ValueError("HGP local attestation path is not canonical")
        _validate_local_attestation(
            self.local_attestation, self.local_attestation_sha256,
            schedule, preflight, self.artifact_manifest,
            _sha256_file(self.registry), self.config_file_sha256,
            self.source_commit,
            self.archive_sha256, self.source_manifest_sha256,
        )

        control = self.control_stage(combine.success)
        self.run_batch((control,), schedule["control_freeze_deadline_unix"])
        _validate_control_output(
            self.control, preflight, schedule, self.config,
            self.source_commit, self.archive_sha256,
            self.source_manifest_sha256,
        )

        screens = self.screen_stages(control.success)
        screen_successes = self.run_batch(
            screens, schedule["screen_deadline_unix"],
        )
        analysis = self.analysis_stage(screen_successes)
        self.run_batch((analysis,), schedule["analysis_deadline_unix"])
        package = _validate_terminal_output(
            self.package, self.source_commit, schedule,
        )
        result = {
            "event": "terminal", "run_id": self.run_id,
            "status": package["status"],
            "terminal_package": str(self.package),
            "package_sha256": package["package_sha256"],
        }
        print(_canonical_json(result), flush=True)
        return result

    def run(self, phase):
        if phase == "preflight":
            return self.run_preflight()
        if phase == "measurement":
            return self.run_measurement()
        raise ValueError("unknown HGP orchestration phase")


def _validate_success_marker(path, expected_stage=None, source_commit=None):
    marker = _read_json(path)
    if (set(marker) != {
            "stage", "source_commit", "stage_fingerprint",
            "prerequisite_success_sha256", "completed_utc"}
            or (expected_stage is not None
                and marker.get("stage") != expected_stage)
            or (source_commit is not None
                and marker.get("source_commit") != source_commit)
            or SHA256_RE.fullmatch(str(marker.get("stage_fingerprint", "")))
            is None
            or not isinstance(marker.get("prerequisite_success_sha256"), list)
            or any(SHA256_RE.fullmatch(str(value)) is None
                   for value in marker["prerequisite_success_sha256"])):
        raise ValueError(f"HGP SUCCESS marker is invalid: {path}")
    return marker


def _validate_schedule_output(path, run_id, source_commit, archive_sha256,
                              source_manifest_sha256, config_file_sha256):
    schedule = _read_json(path)
    required = (
        "started_unix", "preflight_deadline_unix",
        "control_freeze_deadline_unix", "screen_deadline_unix",
        "analysis_deadline_unix",
    )
    identity = dict(schedule)
    stored_sha = identity.pop("schedule_sha256", None)
    if (schedule.get("run_id") != run_id
            or schedule.get("source_commit") != source_commit
            or schedule.get("archive_sha256") != archive_sha256
            or schedule.get("source_manifest_sha256")
            != source_manifest_sha256
            or schedule.get("config_file_sha256") != config_file_sha256
            or schedule.get("source_identity", {}).get("mode") != "archive"
            or schedule.get("source_identity", {}).get("source_commit")
            != source_commit
            or schedule.get("source_identity", {}).get("archive_sha256")
            != archive_sha256
            or schedule.get("source_identity", {}).get("manifest_sha256")
            != source_manifest_sha256
            or stored_sha != hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest()
            or any(not isinstance(schedule.get(name), (int, float))
                   for name in required)
            or not all(float(schedule[left]) < float(schedule[right])
                       for left, right in zip(required, required[1:]))
            or float(schedule["started_unix"]) > time.time()):
        raise ValueError("HGP frozen schedule output is invalid")
    return schedule


def _validate_aggregate_preflight(path, schedule, config, source_commit,
                                  archive_sha256, source_manifest_sha256,
                                  config_file_sha256):
    report = _read_json(path)
    tier = report.get("selected_resource_tier")
    completed = float(report.get("completed_unix", float("inf")))
    if (report.get("status") != "PASS"
            or tier not in config.get("resource_tiers", {})
            or report.get("source_commit") != source_commit
            or report.get("archive_sha256") != archive_sha256
            or report.get("source_manifest_sha256")
            != source_manifest_sha256
            or report.get("config_file_sha256") != config_file_sha256
            or report.get("schedule_sha256") != schedule["schedule_sha256"]
            or not isinstance(report.get("source_identity"), dict)
            or report["source_identity"].get("mode") != "archive"
            or report["source_identity"].get("source_commit") != source_commit
            or report["source_identity"].get("archive_sha256")
            != archive_sha256
            or report["source_identity"].get("manifest_sha256")
            != source_manifest_sha256
            or not (float(schedule["started_unix"]) <= completed
                    <= float(schedule["preflight_deadline_unix"]))):
        raise RuntimeError(
            "HGP aggregate preflight is not a PASS authority for measurement"
        )
    return report


def _validate_control_output(path, preflight, schedule, config, source_commit,
                             archive_sha256, source_manifest_sha256):
    control = _read_json(path)
    expected_count = int(config["task_counts"]["total_measurement"])
    if (control.get("resource_tier")
            != preflight.get("selected_resource_tier")
            or control.get("source_commit") != source_commit
            or control.get("archive_sha256") != archive_sha256
            or control.get("source_manifest_sha256")
            != source_manifest_sha256
            or int(control.get("task_count", -1)) != expected_count
            or len(control.get("tasks", ())) != expected_count
            or tuple(control.get("execution_nodes", ())) != EXECUTION_NODES):
        raise RuntimeError("HGP frozen measurement control is invalid")
    return control


def _validate_local_attestation(
        path, expected_file_sha256, schedule, preflight,
        artifact_manifest_path, registry_file_sha256, config_file_sha256,
        source_commit,
        archive_sha256, source_manifest_sha256):
    path = Path(path).resolve(strict=True)
    if (SHA256_RE.fullmatch(str(expected_file_sha256)) is None
            or _sha256_file(path) != expected_file_sha256):
        raise ValueError("HGP local attestation file SHA mismatch")
    value = _read_json(path)
    expected_fields = {
        "attestation_version", "status", "source_commit", "archive_sha256",
        "source_manifest_sha256", "registry_file_sha256",
        "config_file_sha256", "schedule_sha256", "schedule_file_sha256",
        "artifact_manifest_sha256", "artifact_manifest_file_sha256",
        "preflight_file_sha256", "remote_canonical_digest_sha256",
        "local_canonical_digest_sha256", "exact_canonical_match",
        "mismatch_paths", "importance_sampling_transcript_sha256",
        "solver_identity_policy", "local_environment",
        "portability_review", "completed_unix", "attestation_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("HGP local attestation schema is invalid")
    identity = dict(value)
    stored_sha = identity.pop("attestation_sha256", None)
    status = value.get("status")
    exact = value.get("exact_canonical_match")
    remote_digest = preflight.get("canonical_digest_sha256")
    expected_is = preflight.get("canonical_digest", {}).get(
        "importance_sampling_transcript_sha256",
    )
    environment = value.get("local_environment")
    common_valid = (
        value.get("attestation_version") == LOCAL_ATTESTATION_VERSION
        and status == "PASS"
        and value.get("source_commit") == source_commit
        and value.get("archive_sha256") == archive_sha256
        and value.get("source_manifest_sha256") == source_manifest_sha256
        and value.get("registry_file_sha256") == registry_file_sha256
        and value.get("config_file_sha256") == config_file_sha256
        and value.get("schedule_sha256") == schedule["schedule_sha256"]
        and value.get("schedule_file_sha256")
        == _sha256_file(Path(path).parent / "HGP_GLOBAL_24H_SCHEDULE.json")
        and value.get("artifact_manifest_file_sha256")
        == _sha256_file(artifact_manifest_path)
        and value.get("artifact_manifest_sha256")
        == _read_json(artifact_manifest_path).get("artifact_manifest_sha256")
        and value.get("preflight_file_sha256")
        == _sha256_file(Path(path).parent / "hgp_preflight.json")
        and value.get("remote_canonical_digest_sha256") == remote_digest
        and isinstance(expected_is, list)
        and value.get("importance_sampling_transcript_sha256") == expected_is
        and value.get("solver_identity_policy") == LOCAL_SOLVER_POLICY
        and isinstance(environment, dict)
        and set(environment) == {
            "system", "machine", "python", "numpy", "scipy",
            "map_solver_identity_current",
        }
        and all(isinstance(item, str) and item for item in environment.values())
        and stored_sha == hashlib.sha256(
            _canonical_json(identity).encode("ascii")
        ).hexdigest()
        and float(schedule["started_unix"])
        <= float(value.get("completed_unix", float("inf")))
        <= float(schedule["control_freeze_deadline_unix"])
    )
    if not common_valid:
        raise ValueError("HGP local attestation identity is invalid")
    if (exact is not True
            or value.get("local_canonical_digest_sha256") != remote_digest
            or value.get("portability_review") is not None
            or value.get("mismatch_paths") != []):
        raise ValueError("HGP exact local attestation is inconsistent")
    return value


def _validate_terminal_output(path, source_commit, schedule):
    package = _read_json(path)
    identity = dict(package)
    stored_sha = identity.pop("package_sha256", None)
    if (package.get("source_identity", {}).get("source_commit")
            != source_commit
            or package.get("schedule_sha256") != schedule["schedule_sha256"]
            or package.get("formal_authorization") is not False
            or package.get("production_authorization") is not False
            or stored_sha != hashlib.sha256(
                _canonical_json(identity).encode("ascii")
            ).hexdigest()):
        raise ValueError("HGP terminal package is invalid")
    return package


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument(
        "--phase", choices=("preflight", "measurement"),
        default="preflight",
    )
    parser.add_argument("--local-attestation")
    parser.add_argument("--local-attestation-sha256")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if not 0.1 <= float(args.poll_seconds) <= 60.0:
        raise ValueError("HGP orchestrator poll interval is invalid")
    if args.phase == "preflight" and (
            args.local_attestation is not None
            or args.local_attestation_sha256 is not None):
        raise ValueError("HGP preflight cannot accept a local attestation")
    if args.phase == "measurement" and (
            args.local_attestation is None
            or args.local_attestation_sha256 is None):
        raise ValueError("HGP measurement requires local attestation path and SHA")
    deployment_root, run_root = _require_verified_launch(
        args, Path.home(),
    )
    config, config_file_sha256 = _read_frozen_config(
        deployment_root, args.archive_sha256,
    )
    orchestrator = HgpOrchestrator(
        run_id=args.run_id, source_commit=args.source_commit,
        archive_sha256=args.archive_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        deployment_root=deployment_root, run_root=run_root, config=config,
        config_file_sha256=config_file_sha256,
        local_attestation=args.local_attestation,
        local_attestation_sha256=args.local_attestation_sha256,
        poll_seconds=args.poll_seconds,
    )
    orchestrator.run(args.phase)


if __name__ == "__main__":
    main()
