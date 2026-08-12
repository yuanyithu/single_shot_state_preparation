"""Local command line for exp106: config authoring, tasks, replay, aggregation."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import CONFIG_SCHEMA, EXPERIMENT_ID, PILOT_CONFIG_SCHEMA
from .aggregate import ARRAY_FIELDS, SCALAR_FIELDS, aggregate_scan
from .config import (
    BOOTSTRAP,
    COMPUTE_HOST,
    CROSSING,
    DECODER,
    DECODER_BINARY_SHA256,
    DECODER_BINARY_SUFFIX,
    ENSEMBLE,
    MASTER_SEED_HEX,
    NAMESPACES,
    OBJECTIVE,
    PREFLIGHT,
    Q_TOKEN,
    REGISTRY_PATH_BY_PHASE,
    SCHEMA_BY_PHASE,
    canonical_config_filename,
    REMOTE_CONFIG_SCHEMA,
    REMOTE_CONDA_PREFIX,
    REMOTE_DECODER_BINARY_SHA256,
    REMOTE_DECODER_BINARY_SUFFIX,
    REMOTE_EXECUTION_DEFAULTS,
    REMOTE_LDPC_SOURCE,
    REMOTE_SUPPORT_PACKAGES,
    REPLAY,
    load_config,
    plan_for_phase,
)
from .ensemble import load_registry, registry_index
from .identity import source_tree_sha256
from .io import atomic_json, sha256_json
from .raw import raw_filename, save_raw
from .replay import (
    committed_replay_blocks,
    replay_task,
    validate_replay_against_raw,
)
from .worker import run_code_block


REPO_ROOT = Path(__file__).resolve().parents[4]


def _base_config(registry_sha256, source_commit, tree_sha256, phase):
    m_values, p_tokens, codes_per_m, trials, codes_per_task = plan_for_phase(phase)
    return {
        "schema_version": PILOT_CONFIG_SCHEMA if phase == "pilot" else CONFIG_SCHEMA,
        "phase": phase,
        "experiment_id": EXPERIMENT_ID,
        "objective": OBJECTIVE,
        "registry_path": REGISTRY_PATH_BY_PHASE[phase],
        "registry_sha256": registry_sha256,
        "source_commit": source_commit,
        "source_tree_sha256": tree_sha256,
        "master_seed_hex": MASTER_SEED_HEX,
        "m_values": m_values,
        "codes_per_m": {str(m): codes_per_m[m] for m in m_values},
        "p_tokens": p_tokens,
        "q_token": Q_TOKEN,
        "trials_per_code_p": {str(m): trials[m] for m in m_values},
        "codes_per_task": {str(m): codes_per_task[m] for m in m_values},
        "ensemble": ENSEMBLE,
        "decoder": DECODER,
        "environment": {
            "device_name": "macmini",
            "hostname": "ymini.local",
            "conda_environment": "12",
            "conda_prefix_matches_python": True,
            "python": "3.12.12", "numpy": "2.4.1", "scipy": "1.17.0", "ldpc": "2.4.1",
        },
        "decoder_binary": {
            "module": "ldpc.bposd_decoder._bposd_decoder",
            "filename_suffix": DECODER_BINARY_SUFFIX,
            "sha256": DECODER_BINARY_SHA256,
        },
        "namespaces": NAMESPACES,
        "bootstrap": BOOTSTRAP,
        "replay": REPLAY,
        "preflight": PREFLIGHT,
        "crossing": CROSSING,
    }


def _remote_config(base, phase="production_remote"):
    config = dict(base)
    config.update({
        "schema_version": SCHEMA_BY_PHASE[phase],
        "phase": phase,
        "environment": {
            "device_name": COMPUTE_HOST,
            "hostname": COMPUTE_HOST,
            "conda_environment": REMOTE_CONDA_PREFIX,
            "conda_prefix_matches_python": True,
            "python": "3.12.12", "numpy": "2.4.1", "scipy": "1.17.0", "ldpc": "2.4.1",
        },
        "decoder_binary": {
            "module": "ldpc.bposd_decoder._bposd_decoder",
            "filename_suffix": REMOTE_DECODER_BINARY_SUFFIX,
            "sha256": REMOTE_DECODER_BINARY_SHA256,
        },
        "execution_profile": dict(
            REMOTE_EXECUTION_DEFAULTS, conda_environment=REMOTE_CONDA_PREFIX,
        ),
        "ldpc_source": REMOTE_LDPC_SOURCE,
        "support_packages": REMOTE_SUPPORT_PACKAGES,
    })
    return config


def command_write_config(args):
    registry = load_registry(REPO_ROOT / REGISTRY_PATH_BY_PHASE[args.phase])
    tree = source_tree_sha256()
    base = _base_config(
        registry["registry_sha256"], args.source_commit, tree, args.phase,
    )
    config_dir = Path(__file__).resolve().parents[1] / "config"
    filename = canonical_config_filename(SCHEMA_BY_PHASE[args.phase])
    atomic_json(config_dir / filename, base)
    print("local ", sha256_json(base))
    if args.remote:
        # The pilot has a remote form because the allocation rule is only
        # meaningful on costs measured by the machine that will spend the
        # budget. It runs `cost-benchmark` there and nothing else: no scan,
        # no aggregation, no published statistic.
        phase = "pilot_remote" if args.phase == "pilot" else "production_remote"
        remote = _remote_config(base, phase)
        filename = canonical_config_filename(SCHEMA_BY_PHASE[phase])
        atomic_json(config_dir / filename, remote)
        print("remote", sha256_json(remote))
    return 0


def command_run_task(args):
    config = load_config(args.config)
    rows = registry_index(load_registry(REPO_ROOT / config["registry_path"]))
    raw = run_code_block(args.m, args.block, config, rows)
    path = Path(args.output) / raw_filename(config, args.m, args.block)
    digest = save_raw(path, raw)
    print(f"{raw['status']} {path} {digest}")
    return 0 if raw["status"] == "VALID" else 1


def command_replay_task(args):
    config = load_config(args.config)
    rows = registry_index(load_registry(REPO_ROOT / config["registry_path"]))
    result = replay_task(args.raw, config, rows)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def command_replay_plan(args):
    config = load_config(args.config)
    blocks = committed_replay_blocks(config)
    total = sum(len(value) for value in blocks.values())
    print(json.dumps({"tasks": total, "blocks": blocks}, sort_keys=True))
    return 0


def command_aggregate(args):
    config = load_config(args.config)
    replay_report = None
    if args.replay_report:
        replay_report = json.loads(Path(args.replay_report).read_text(encoding="ascii"))
        validate_replay_against_raw(replay_report, args.raw_root, config)
        replay_report = dict(
            replay_report, report_sha256=sha256_json(replay_report),
        )
    aggregate = aggregate_scan(args.raw_root, config, replay_report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{
        key: np.asarray(aggregate[key]) for key in ARRAY_FIELDS + SCALAR_FIELDS
    })
    print(f"{aggregate['overall_status']} {aggregate['terminal_status']} {output}")
    return 0 if aggregate["overall_status"] == "COMPLETE" else 1


def build_parser():
    parser = argparse.ArgumentParser(description="exp106 local pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write-config")
    write.add_argument("--source-commit", required=True)
    write.add_argument(
        "--phase", choices=("pilot", "production"), required=True,
    )
    write.add_argument("--remote", action="store_true")
    write.set_defaults(handler=command_write_config)

    task = sub.add_parser("run-task")
    task.add_argument("--config", required=True)
    task.add_argument("--m", type=int, required=True)
    task.add_argument("--block", type=int, required=True)
    task.add_argument("--output", required=True)
    task.set_defaults(handler=command_run_task)

    replay = sub.add_parser("replay-task")
    replay.add_argument("--config", required=True)
    replay.add_argument("--raw", required=True)
    replay.set_defaults(handler=command_replay_task)

    plan = sub.add_parser("replay-plan")
    plan.add_argument("--config", required=True)
    plan.set_defaults(handler=command_replay_plan)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--config", required=True)
    aggregate.add_argument("--raw-root", required=True)
    aggregate.add_argument("--replay-report")
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=command_aggregate)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
