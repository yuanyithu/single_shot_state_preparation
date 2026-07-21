"""Materialize the frozen 15-bias and 1280-measurement controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_REGISTRY_RELATIVE,
    all_methods,
    build_bias_manifest,
    build_measurement_manifest,
    config_sha256,
    defect_methods,
    load_config,
    load_registry,
    sha256_file,
    validate_control,
    validate_runtime_consensus,
    validate_schedule,
    verify_remote_stage,
)


def _context(args):
    registry_path = Path(args.registry)
    config_path = Path(args.config)
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    schedule = validate_schedule(
        args.schedule, registry, config, args.source_commit,
    )
    runtime = validate_runtime_consensus(
        args.runtime_report, args.source_commit, registry["registry_sha256"],
        config_sha256(config), schedule["archive_sha256"],
        schedule["source_manifest_sha256"],
    )
    return registry_path, config_path, registry, config, schedule, runtime


def _validate_created(path, registry, config, expected_kind, expected_count,
                      expected_methods, tier):
    manifest = json.loads(Path(path).read_text(encoding="ascii"))
    validate_control(manifest, registry, config)
    if (manifest.get("kind") != expected_kind
            or len(manifest.get("tasks", [])) != expected_count
            or len({entry["task_fingerprint"] for entry in manifest["tasks"]})
            != expected_count
            or manifest.get("method_tiers")
            != [[method, tier] for method in expected_methods]):
        raise ValueError("created diagnostic control is not the frozen task set")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("bias", "measurement"))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_RELATIVE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_RELATIVE))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--raw-root")
    parser.add_argument("--deployment-root")
    parser.add_argument("--bias-control")
    parser.add_argument("--bias-ownership")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("diagnostic control output already exists")
    registry_path, config_path, registry, config, schedule, runtime = _context(args)
    tier = runtime["selected_resource_tier"]

    if args.kind == "bias":
        build_bias_manifest(
            registry_path, config_path, args.source_commit, tier, output,
        )
        manifest = _validate_created(
            output, registry, config, "diagnostic_defect_bias", 15,
            defect_methods(), tier,
        )
    else:
        required = {
            "run_root": args.run_root, "raw_root": args.raw_root,
            "deployment_root": args.deployment_root,
            "bias_control": args.bias_control,
            "bias_ownership": args.bias_ownership,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "measurement control requires verified bias evidence: "
                + ",".join(missing)
            )
        bias_evidence = verify_remote_stage(
            args.run_root, args.raw_root, args.bias_control,
            args.bias_ownership, args.deployment_root, args.schedule,
            args.runtime_report, registry_path, config_path,
        )
        if bias_evidence["raw_count"] != 15:
            raise ValueError("diagnostic bias evidence is not exactly 15 tasks")
        build_measurement_manifest(
            registry_path, config_path, args.source_commit, tier, output,
            args.bias_control, args.raw_root,
        )
        manifest = _validate_created(
            output, registry, config, "diagnostic_measurement", 1280,
            all_methods(), tier,
        )
    print(json.dumps({
        "kind": args.kind, "task_count": len(manifest["tasks"]),
        "resource_tier": tier, "output": str(output),
        "sha256": sha256_file(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
