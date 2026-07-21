"""Canonical Linux digest for exp102 q=0 global-sampling kernels."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import socket
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.global_discovery import (
    character_seed,
    load_global_discovery_config,
    uniform_seed_for_cell,
)
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    verify_source_identity,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    DefectTraceConfig,
    GlobalSeedIdentity,
    HardCosetConfig,
    build_joint_blocks,
    build_logical_proposal_catalog,
    canonical_global_trajectory_digest,
    frozen_character_set,
    run_defect_trace_trajectory,
    run_hardcoset_trajectory,
    tune_defect_bias,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _identity(source_commit, config, registry, cell, method, family, trajectory, namespace):
    from data.expander_code.exp102.exp102_pipeline.io import sha256_json
    return GlobalSeedIdentity(
        source_commit=source_commit,
        config_sha256=config["discovery_config_sha256"],
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
    config = load_global_discovery_config(config_path, registry)
    _, code, H = load_frozen_code(registry_path, "m08_c06")
    model, frame = build_model(H)
    cell = config["panels"]["HARD2"]["cells"][1]
    seed = uniform_seed_for_cell(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(seed)).random(model.num_qubits)
    epsilon = (uniforms < cell["p"]).astype(np.uint8)
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)
    catalog = build_logical_proposal_catalog(model, frame)
    characters = frozen_character_set(
        model.k, character_seed(registry["registry_sha256"], code["code_id"]),
    )
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
            "q0_global_cross_node_v1",
        )
        joint = (
            build_joint_blocks(model, frame, catalog, sampler.joint_block_size)
            if sampler.joint_block_size else None
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
            raise AssertionError(f"{method} reference/Numba digest mismatch")
        records.append({"kind": method, "digest": digests[0]})

    defect = DefectTraceConfig("DT16", cell["p"], 2, 8)
    defect_identity = _identity(
        source_commit, config, registry, cell, "DT16", "P", 0,
        "q0_global_cross_node_v1",
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
        raise AssertionError("defect trace reference/Numba digest mismatch")
    records.append({"kind": "DT16_fixed_bias", "digest": digests[0]})

    tiny_model, _ = build_model(np.asarray([[1, 1, 1]], dtype=np.uint8))
    tiny_syndrome = np.zeros(tiny_model.num_checks, dtype=np.uint8)
    tiny_cell = {
        "code_id": "oracle_n10", "p": 0.10, "disorder_index": 0,
        "disorder_source": "oracle",
    }
    tiny_config = DefectTraceConfig("DT16", 0.10, 2, 8)
    identities = [
        _identity(
            source_commit, config, registry, tiny_cell, "DT16", "TUNE", index,
            "q0_global_cross_node_bias_v1",
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
            or not np.array_equal(tuned[0]["tuning_histogram"], tuned[1]["tuning_histogram"])):
        raise AssertionError("bias tuning reference/Numba mismatch")
    records.append({"kind": "DT16_bias_tuning", "digest": tuned[0]["bias_sha256"]})
    payload = canonical_json(records).encode("ascii")
    return {
        "records": records,
        "canonical_digest": hashlib.sha256(payload).hexdigest(),
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
    }


def combine_digest_reports(report_paths, output_path=None):
    expected_nodes = ("nd-1", "nd-2", "nd-3")
    if set(report_paths) != set(expected_nodes):
        raise ValueError("digest consensus requires nd-1, nd-2, and nd-3")
    reports = {
        node: json.loads(Path(report_paths[node]).read_text(encoding="ascii"))
        for node in expected_nodes
    }
    first = reports[expected_nodes[0]]
    identity_fields = (
        "canonical_digest", "records", "source_commit", "source_identity",
        "registry_sha256", "discovery_config_sha256",
    )
    expected = {field: first.get(field) for field in identity_fields}
    for node, report in reports.items():
        if (report.get("node") != node
                or report.get("environment", {}).get("system") != "Linux"
                or {field: report.get(field) for field in identity_fields} != expected
                or not isinstance(report.get("source_identity"), dict)
                or report["source_identity"].get("mode") != "archive"
                or not np.isfinite(float(report.get("completed_unix", float("nan"))))):
            raise ValueError(f"cross-node digest mismatch or unverified source: {node}")
    result = {
        "report_version": "exp102.q0_global.digest_consensus.v1",
        **expected,
        "nodes": list(expected_nodes),
        "node_report_sha256": {
            node: sha256_file(report_paths[node]) for node in expected_nodes
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
    parser.add_argument("--node", choices=("nd-1", "nd-2", "nd-3"))
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
            "hostname": socket.gethostname(), "python": platform.python_version(),
            "numpy": np.__version__,
        },
    })
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
