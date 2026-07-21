"""Canonical reference/Numba digest for the isolated diagnostic sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    canonical_json,
    verify_source_identity,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    DefectTraceConfig,
    HardCosetConfig,
    build_joint_blocks,
    build_logical_proposal_catalog,
    canonical_global_trajectory_digest,
    frozen_character_set,
    run_defect_trace_trajectory,
    run_hardcoset_trajectory,
    tune_defect_bias,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
from data.expander_code.exp102.exp102_pipeline.worker import build_model

from .common import (
    CONTRACT_VERSION,
    DIGEST_CONSENSUS_VERSION,
    DIGEST_NODE_VERSION,
    EXPECTED_PREFLIGHT_NODES,
    atomic_json,
    config_sha256,
    load_config,
    load_registry,
    pipeline_attr,
    sha256_file,
    sha256_json,
    uniform_seed_for_cell,
)


def _identity(source_commit, config, registry, cell, method, family,
              trajectory, namespace):
    identity_type = pipeline_attr("ScreenSeedIdentity")
    return identity_type(
        seed_root=config["seed_root"],
        source_commit=source_commit,
        config_sha256=config_sha256(config),
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(cell),
        method_id=method,
        resource_tier="DIGEST",
        init_family=family,
        trajectory_index=trajectory,
        trajectory_namespace=namespace,
    )


def canonical_digest(registry_path, config_path, source_commit):
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    cell = config["panels"]["HARD2"]["cells"][1]
    if cell["code_id"] != "m08_c06":
        raise ValueError("diagnostic digest requires frozen m08_c06 sentinel")
    _, code, H = load_frozen_code(registry_path, cell["code_id"])
    model, frame = build_model(H)
    seed = uniform_seed_for_cell(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(seed)).random(
        model.num_qubits
    )
    epsilon = (uniforms < cell["p"]).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    catalog = build_logical_proposal_catalog(model, frame)
    character_seed = pipeline_attr("character_seed")(
        config, registry["registry_sha256"], code["code_id"],
    )
    characters = frozen_character_set(model.k, character_seed)
    records = [{
        "kind": "catalog_and_characters",
        "catalog_sha256": catalog.catalog_sha256,
        "character_sha256": characters.character_sha256,
        "catalog_size": catalog.size,
        "character_count": int(characters.masks.size),
    }]

    for method in ("RC8-QC1", "RC8-J08"):
        sampler = HardCosetConfig(method, cell["p"], 2, 8)
        identity = _identity(
            source_commit, config, registry, cell, method, "P", 0,
            "q0_global_screen_cross_node_v1",
        )
        joint = (
            build_joint_blocks(
                model, frame, catalog, sampler.joint_block_size,
            ) if sampler.joint_block_size else None
        )
        values = [
            run_hardcoset_trajectory(
                model, frame, syndrome, sampler, identity, epsilon,
                engine=engine, catalog=catalog, joint=joint,
            )
            for engine in ("reference", "numba")
        ]
        digests = [canonical_global_trajectory_digest(value) for value in values]
        if digests[0] != digests[1]:
            raise AssertionError(f"{method} diagnostic digest mismatch")
        records.append({"kind": method, "digest": digests[0]})

    defect = DefectTraceConfig("DT16", cell["p"], 2, 8)
    defect_identity = _identity(
        source_commit, config, registry, cell, "DT16", "P", 0,
        "q0_global_screen_cross_node_v1",
    )
    fixed_bias = np.linspace(-0.25, 0.25, defect.dmax + 1)
    values = [
        run_defect_trace_trajectory(
            model, frame, syndrome, defect, defect_identity, epsilon,
            fixed_bias, "a" * 64, engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    digests = [canonical_global_trajectory_digest(value) for value in values]
    if digests[0] != digests[1]:
        raise AssertionError("DT16 diagnostic digest mismatch")
    records.append({"kind": "DT16_fixed_bias", "digest": digests[0]})

    tiny_model, _ = build_model(np.asarray([[1, 1, 1]], dtype=np.uint8))
    tiny_syndrome = np.zeros(tiny_model.num_checks, dtype=np.uint8)
    tiny_cell = {
        "code_id": "oracle_n10", "p": 0.10, "disorder_index": 0,
        "disorder_source": "diagnostic_oracle",
    }
    tiny_config = DefectTraceConfig("DT16", 0.10, 2, 8)
    identities = [
        _identity(
            source_commit, config, registry, tiny_cell, "DT16", "TUNE",
            index, "q0_global_screen_cross_node_bias_v1",
        )
        for index in range(8)
    ]
    tuned = [
        tune_defect_bias(
            tiny_model, tiny_syndrome, tiny_config, identities, engine=engine,
        )
        for engine in ("reference", "numba")
    ]
    if (tuned[0]["bias_sha256"] != tuned[1]["bias_sha256"]
            or not np.array_equal(
                tuned[0]["tuning_histogram"], tuned[1]["tuning_histogram"],
            )):
        raise AssertionError("diagnostic bias tuning digest mismatch")
    records.append({
        "kind": "DT16_bias_tuning", "digest": tuned[0]["bias_sha256"],
    })
    digest = hashlib.sha256(canonical_json(records).encode("ascii")).hexdigest()
    return {
        "digest_version": DIGEST_NODE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "records": records,
        "canonical_digest": digest,
        "registry_sha256": registry["registry_sha256"],
        "diagnostic_config_sha256": config_sha256(config),
    }


def combine_digest_reports(report_paths, output_path=None):
    if set(report_paths) != set(EXPECTED_PREFLIGHT_NODES):
        raise ValueError("diagnostic digest consensus requires all three nodes")
    reports = {
        node: json.loads(Path(report_paths[node]).read_text(encoding="ascii"))
        for node in EXPECTED_PREFLIGHT_NODES
    }
    first = reports[EXPECTED_PREFLIGHT_NODES[0]]
    identity_fields = (
        "digest_version", "contract_version", "canonical_digest", "records",
        "source_commit", "source_identity", "registry_sha256",
        "diagnostic_config_sha256",
    )
    expected = {key: first.get(key) for key in identity_fields}
    for node, report in reports.items():
        if (report.get("node") != node
                or report.get("environment", {}).get("system") != "Linux"
                or {key: report.get(key) for key in identity_fields} != expected
                or report.get("digest_version") != DIGEST_NODE_VERSION
                or report.get("contract_version") != CONTRACT_VERSION
                or not isinstance(report.get("source_identity"), dict)
                or report["source_identity"].get("mode") != "archive"
                or report["source_identity"].get("source_commit")
                != report.get("source_commit")
                or not math.isfinite(float(
                    report.get("completed_unix", math.nan)
                ))):
            raise ValueError(
                f"diagnostic digest mismatch or unverified source: {node}"
            )
    result = {
        "report_version": DIGEST_CONSENSUS_VERSION,
        **{key: value for key, value in expected.items()
           if key != "digest_version"},
        "nodes": list(EXPECTED_PREFLIGHT_NODES),
        "node_report_sha256": {
            node: sha256_file(report_paths[node])
            for node in EXPECTED_PREFLIGHT_NODES
        },
        "completed_unix_max": max(
            float(report["completed_unix"]) for report in reports.values()
        ),
        "status": "PASS",
    }
    if output_path is not None:
        atomic_json(output_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("config")
    parser.add_argument("source_commit")
    parser.add_argument("--require-verified-source", action="store_true")
    parser.add_argument("--node", choices=EXPECTED_PREFLIGHT_NODES)
    parser.add_argument("--output")
    parser.add_argument("--combine-report", action="append", default=[])
    args = parser.parse_args(argv)
    if args.combine_report:
        mappings = dict(value.split("=", 1) for value in args.combine_report)
        result = combine_digest_reports(mappings, args.output)
        print(json.dumps(result, sort_keys=True))
        return
    source_identity = (
        verify_source_identity(Path.cwd(), args.source_commit)
        if args.require_verified_source else None
    )
    result = canonical_digest(args.registry, args.config, args.source_commit)
    result.update({
        "source_commit": args.source_commit,
        "source_identity": source_identity,
        "node": args.node if args.node is not None else socket.gethostname(),
        "completed_unix": time.time(),
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(), "numpy": np.__version__,
        },
    })
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
