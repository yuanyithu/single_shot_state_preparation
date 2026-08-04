import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .aggregate import _registry, _validate_raw, aggregate_decoder_scan, save_aggregate
from .config import load_config
from .identity import require_tracked_clean_evidence, runtime_identity, verify_frozen_repository
from .io import atomic_json, canonical_json, sha256_file, sha256_json
from .preflight import run_resource_preflight, validate_resource_preflight_report
from .raw import load_raw, raw_filename, save_raw
from .replay import (
    build_replay_report, expected_replay_keys, replay_decoder_shard,
    validate_replay_report,
)
from .report import generate_final_report
from .loader import load_exp103_crossing
from .worker import run_decoder_shard


EXP103_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RAW_ROOT = EXP103_ROOT / "raw"
CANONICAL_STAGE_RAW_ROOTS = {
    "stage1": CANONICAL_RAW_ROOT / "stage1",
    "stage2": CANONICAL_RAW_ROOT / "stage2",
}
CANONICAL_STAGE1_REPLAY = CANONICAL_STAGE_RAW_ROOTS["stage1"] / "REPLAY_STAGE1.json"
CANONICAL_STAGE1_AGGREGATE = EXP103_ROOT / "final_results" / "stage1_aggregate.npz"
CANONICAL_FINAL_AGGREGATE = EXP103_ROOT / "final_results" / "decoder_crossing.npz"
CANONICAL_FINAL_RESULTS = EXP103_ROOT / "final_results"
CANONICAL_STAGE1_TECHNICAL = (
    EXP103_ROOT / "validation" / "003_m3_m5_scan_20260804" / "technical_report.json"
)


def _require_canonical_path(path, canonical, purpose):
    if path is None or Path(path).resolve() != Path(canonical).resolve():
        raise ValueError(f"{purpose} must use the canonical path {canonical}")
    return Path(canonical)


def _canonical_replay_scope(raw_root):
    resolved = Path(raw_root).resolve()
    for scope, canonical in CANONICAL_STAGE_RAW_ROOTS.items():
        if resolved == canonical.resolve():
            return scope
    raise ValueError(
        "formal replay raw root must be the canonical exp103 raw/stage1 or raw/stage2 directory"
    )


def _require_canonical_aggregate_output(path, replay_scope):
    canonical_by_scope = {
        "stage1": CANONICAL_STAGE1_AGGREGATE,
        "final_combined": CANONICAL_FINAL_AGGREGATE,
    }
    canonical = canonical_by_scope.get(replay_scope)
    if canonical is not None:
        return _require_canonical_path(path, canonical, f"{replay_scope} aggregate output")
    if Path(path).resolve() in {
        candidate.resolve() for candidate in canonical_by_scope.values()
    }:
        raise ValueError(
            f"cannot write replay scope {replay_scope!r} to a canonical aggregate path"
        )
    return Path(path)


def _save_code_task(task):
    code_id, config_path, stage_root = task
    config = load_config(config_path)
    rows = _registry(config)
    if code_id not in rows:
        raise ValueError(f"code is outside the frozen primary panel: {code_id}")
    results = []
    for p_token in config["p_tokens"]:
        for shard_index in range(config["shards_per_code_p"]):
            output = Path(stage_root) / raw_filename(code_id, p_token, shard_index)
            if output.exists():
                existing = load_raw(output)
                reason = _validate_raw(
                    existing, config, rows[code_id], code_id, p_token, shard_index,
                )
                if reason is not None:
                    raise ValueError(
                        f"existing immutable raw does not match its canonical key "
                        f"({reason}): {output}"
                    )
                results.append((str(output), "RESUMED"))
                continue
            raw = run_decoder_shard(code_id, p_token, shard_index, config)
            save_raw(output, raw)
            results.append((str(output), raw["status"]))
            if raw["status"] != "VALID":
                break
        if results[-1][1] == "INVALID":
            break
    return code_id, results


def _replay_code_task(task):
    paths, config_path = task
    return [replay_decoder_shard(path, config_path) for path in paths]


def _require_stage_preflight(path, config, stage):
    canonical = Path(__file__).resolve().parents[1] / "validation" / "002_local_resource_preflight_20260804" / "resource_preflight.json"
    if Path(path).resolve() != canonical.resolve():
        raise ValueError("formal scan requires the canonical Validation 002 report")
    require_tracked_clean_evidence(path)
    report = json.loads(Path(path).read_text(encoding="ascii"))
    validate_resource_preflight_report(report, config)
    if report["stages"][stage]["status"] != "PASS":
        raise ValueError(f"{stage} is blocked by the local resource preflight")


def _require_validation001(config):
    path = Path(__file__).resolve().parents[1] / "validation" / "001_contract_oracles_20260804" / "report.json"
    require_tracked_clean_evidence(path)
    report = json.loads(path.read_text(encoding="ascii"))
    if set(report) != {
        "schema_version", "status", "config_sha256", "registry_sha256",
        "source_commit", "source_tree_sha256", "bplsd_binary_sha256",
        "environment", "test_commands", "test_counts", "oracle_checks",
        "distance_two_codes", "all_tests_passed", "authority",
        "exp102_status_unchanged",
    }:
        raise ValueError("Validation 001 report fields mismatch")
    if report.get("schema_version") != "exp103.validation001.v1" or report.get("status") != "PASS":
        raise ValueError("Validation 001 has not passed")
    for field in ("config_sha256", "source_commit", "source_tree_sha256", "registry_sha256"):
        expected = config["config_sha256"] if field == "config_sha256" else config[field]
        if report.get(field) != expected:
            raise ValueError(f"Validation 001 identity mismatch for {field}")
    if (
        report["bplsd_binary_sha256"] != config["bplsd_binary"]["sha256"]
        or report["environment"] != config["environment"]
        or report["all_tests_passed"] is not True
        or report["authority"] != "IMPLEMENTATION_AND_LOCAL_RESOURCE_PREFLIGHT_ONLY"
        or report["exp102_status_unchanged"] is not True
    ):
        raise ValueError("Validation 001 frozen assertions mismatch")


def _require_stage1_technical(path, config):
    if path is None:
        raise ValueError("Stage 2 requires the frozen Stage 1 technical report")
    _require_canonical_path(
        path, CANONICAL_STAGE1_TECHNICAL,
        "Stage 2 Validation 003 technical report",
    )
    require_tracked_clean_evidence(CANONICAL_STAGE1_TECHNICAL)
    require_tracked_clean_evidence(CANONICAL_STAGE1_AGGREGATE)
    report = json.loads(CANONICAL_STAGE1_TECHNICAL.read_text(encoding="ascii"))
    expected = _compute_stage1_technical_report(
        CANONICAL_STAGE1_AGGREGATE,
        CANONICAL_STAGE1_REPLAY,
        CANONICAL_STAGE_RAW_ROOTS["stage1"],
        config,
    )
    if report != expected:
        differing = sorted(
            field for field in set(report) | set(expected)
            if report.get(field) != expected.get(field)
        )
        raise ValueError(
            "Stage 1 technical report does not match current canonical evidence: "
            + ",".join(differing)
        )
    return report


def _compute_stage1_technical_report(aggregate_path, replay_path, raw_root, config):
    aggregate = load_exp103_crossing(aggregate_path)
    replay_report = json.loads(Path(replay_path).read_text(encoding="ascii"))
    validate_replay_report(replay_report, raw_root, config, "stage1")
    replay_sha256 = sha256_json(replay_report)
    if (
        aggregate["replay_status"] != "PASS"
        or aggregate["replay_scope"] != "stage1"
        or aggregate["replay_report_sha256"] != replay_sha256
        or aggregate["raw_manifest_sha256"] != replay_report["raw_manifest_sha256"]
        or aggregate["replay_report_json"] != canonical_json(replay_report)
    ):
        raise ValueError("Stage 1 aggregate is not bound to the supplied replay evidence")
    reportable = int(np.sum(aggregate["code_status"][:24] == "REPORTABLE"))
    if (
        reportable != 312
        or not np.all(aggregate["code_status"][:24] == "REPORTABLE")
        or not np.all(aggregate["code_status"][24:] == "INCOMPLETE")
        or not np.all(aggregate["m_status"][:3] == "REPORTABLE")
        or not np.all(aggregate["m_status"][3:] == "INCOMPLETE")
        or aggregate["overall_status"] != "INCOMPLETE"
        or aggregate["terminal_status"] != "EXP103_INCOMPLETE"
        or json.loads(aggregate["unexpected_raw_errors_json"])
    ):
        raise ValueError("Stage 1 aggregate does not contain exactly the frozen 312 code-p cells")
    return {
        "schema_version": "exp103.stage1_technical.v1",
        "config_sha256": config["config_sha256"],
        "status": "TECHNICAL_PASS",
        "reportable_code_p": reportable,
        "measurement_shards": replay_report["shards"],
        "replay_status": replay_report["status"],
        "outcome_blind_extension_decision": True,
        "aggregate_sha256": sha256_file(aggregate_path),
        "replay_report_sha256": replay_sha256,
        "raw_manifest_sha256": replay_report["raw_manifest_sha256"],
    }


def build_stage1_technical_report(aggregate_path, replay_path, raw_root, config_path):
    return _compute_stage1_technical_report(
        aggregate_path, replay_path, raw_root, load_config(config_path),
    )


def run_stage(config_path, stage, raw_root, preflight_report, num_workers, stage1_technical_report=None):
    config = load_config(config_path)
    if num_workers != config["preflight"]["num_workers"]:
        raise ValueError("formal scans require --num-workers 8")
    _require_canonical_path(raw_root, CANONICAL_RAW_ROOT, "formal scan raw root")
    runtime_identity(config, verify_source=True)
    verify_frozen_repository(config_path)
    _require_validation001(config)
    _require_stage_preflight(preflight_report, config, stage)
    if stage == "stage2":
        _require_stage1_technical(stage1_technical_report, config)
    raw_root = Path(raw_root)
    tasks = []
    for m in config["stage_m_values"][stage]:
        for code_index in range(8):
            code_id = f"m{m:02d}_c{code_index:02d}"
            tasks.append((code_id, str(config_path), raw_root / stage))
    statuses = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_save_code_task, task) for task in tasks]
        for future in as_completed(futures):
            _, code_results = future.result()
            statuses.extend(code_results)
    if any(status not in {"VALID", "RESUMED"} for _, status in statuses):
        raise RuntimeError("one or more formal shards saved INVALID evidence")
    return {
        "scheduled_codes": len(tasks), "measurement_shards": len(statuses),
        "fresh_shards": sum(status == "VALID" for _, status in statuses),
        "resumed_shards": sum(status == "RESUMED" for _, status in statuses),
        "stage": stage,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output", required=True)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--config", required=True)
    scan.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    scan.add_argument("--raw-root", required=True)
    scan.add_argument("--preflight-report", required=True)
    scan.add_argument("--num-workers", type=int, required=True)
    scan.add_argument("--stage1-technical-report")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--config", required=True)
    replay.add_argument("--raw-root", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--num-workers", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", required=True)
    aggregate.add_argument("--raw-root", required=True)
    aggregate.add_argument("--output", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--config", required=True)
    report.add_argument("--result", required=True)
    report.add_argument("--output-dir", required=True)
    technical = subparsers.add_parser("stage1-technical")
    technical.add_argument("--config", required=True)
    technical.add_argument("--aggregate", required=True)
    technical.add_argument("--replay-report", required=True)
    technical.add_argument("--raw-root", required=True)
    technical.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        config = load_config(args.config)
        verify_frozen_repository(args.config)
        _require_validation001(config)
        canonical = Path(__file__).resolve().parents[1] / "validation" / "002_local_resource_preflight_20260804" / "resource_preflight.json"
        if Path(args.output).resolve() != canonical.resolve():
            raise ValueError("resource preflight must use the canonical Validation 002 path")
        if canonical.exists():
            raise FileExistsError(f"resource preflight evidence is immutable: {canonical}")
        report = run_resource_preflight(config)
        atomic_json(args.output, report)
        print(report["status"])
    elif args.command == "scan":
        print(json.dumps(run_stage(
            args.config, args.stage, args.raw_root, args.preflight_report, args.num_workers,
            args.stage1_technical_report,
        ), sort_keys=True))
    elif args.command == "replay":
        config = load_config(args.config)
        if args.num_workers != 8:
            raise ValueError("formal replay requires --num-workers 8")
        expected_scope = _canonical_replay_scope(args.raw_root)
        runtime_identity(config, verify_source=True)
        verify_frozen_repository(args.config)
        _require_validation001(config)
        paths = sorted(Path(args.raw_root).rglob("*.npz"))
        by_code = {}
        observed_keys = set()
        for path in paths:
            raw = load_raw(path)
            by_code.setdefault(raw["code_id"], []).append(path)
            observed_keys.add((raw["code_id"], raw["p_token"], int(raw["shard_index"])))
        if observed_keys != expected_replay_keys(config, expected_scope):
            raise ValueError(f"formal replay must contain exactly the frozen {expected_scope} shards")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            nested = list(executor.map(
                _replay_code_task,
                [(sorted(code_paths), args.config) for _, code_paths in sorted(by_code.items())],
            ))
        results = [item for group in nested for item in group]
        report = build_replay_report(args.raw_root, results, config)
        expected_output = Path(args.raw_root) / f"REPLAY_{report['scope'].upper()}.json"
        if report["scope"] == "invalid" or Path(args.output).resolve() != expected_output.resolve():
            raise ValueError(f"replay report must use canonical path {expected_output}")
        if expected_output.exists():
            raise FileExistsError(f"replay evidence is immutable: {expected_output}")
        atomic_json(args.output, report)
        print(report["status"])
    elif args.command == "aggregate":
        config = load_config(args.config)
        _require_canonical_path(args.raw_root, CANONICAL_RAW_ROOT, "formal aggregate raw root")
        runtime_identity(config, verify_source=True)
        verify_frozen_repository(args.config)
        _require_validation001(config)
        result = aggregate_decoder_scan(args.raw_root, args.config)
        _require_canonical_aggregate_output(args.output, result["replay_scope"])
        save_aggregate(args.output, result)
        print(result["terminal_status"])
    elif args.command == "report":
        config = load_config(args.config)
        _require_canonical_path(
            args.result, CANONICAL_FINAL_AGGREGATE,
            "formal final report aggregate",
        )
        _require_canonical_path(
            args.output_dir, CANONICAL_FINAL_RESULTS,
            "formal final report output directory",
        )
        runtime_identity(config, verify_source=True)
        verify_frozen_repository(args.config)
        _require_validation001(config)
        result = generate_final_report(args.result, args.output_dir)
        print(result["terminal_status"])
    else:
        _require_canonical_path(
            args.aggregate, CANONICAL_STAGE1_AGGREGATE,
            "Stage 1 technical aggregate",
        )
        _require_canonical_path(
            args.replay_report, CANONICAL_STAGE1_REPLAY,
            "Stage 1 technical replay report",
        )
        _require_canonical_path(
            args.raw_root, CANONICAL_STAGE_RAW_ROOTS["stage1"],
            "Stage 1 technical raw root",
        )
        result = build_stage1_technical_report(
            args.aggregate, args.replay_report, args.raw_root, args.config,
        )
        canonical = CANONICAL_STAGE1_TECHNICAL
        if Path(args.output).resolve() != canonical.resolve():
            raise ValueError("Stage 1 technical evidence must use the canonical Validation 003 path")
        if canonical.exists():
            raise FileExistsError(f"Stage 1 technical evidence is immutable: {canonical}")
        atomic_json(args.output, result)
        print(result["status"])


if __name__ == "__main__":
    main()
