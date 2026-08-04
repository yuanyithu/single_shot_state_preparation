import json

import numpy as np
import pytest

from data.expander_code.exp103.exp103_pipeline import cli
from data.expander_code.exp103.exp103_pipeline import report as report_module
from data.expander_code.exp103.exp103_pipeline.io import canonical_json, sha256_json
from data.expander_code.exp103.exp103_pipeline.raw import raw_filename, save_raw


def _technical_report(config, hash_character="a"):
    digest = hash_character * 64
    return {
        "schema_version": "exp103.stage1_technical.v1",
        "config_sha256": config["config_sha256"],
        "status": "TECHNICAL_PASS",
        "reportable_code_p": 312,
        "measurement_shards": 1248,
        "replay_status": "PASS",
        "outcome_blind_extension_decision": True,
        "aggregate_sha256": digest,
        "replay_report_sha256": digest,
        "raw_manifest_sha256": digest,
    }


def _install_canonical_stage1_paths(monkeypatch, tmp_path):
    raw_root = tmp_path / "raw" / "stage1"
    raw_root.mkdir(parents=True)
    replay_path = raw_root / "REPLAY_STAGE1.json"
    aggregate_path = tmp_path / "final_results" / "stage1_aggregate.npz"
    aggregate_path.parent.mkdir()
    technical_path = tmp_path / "validation" / "technical_report.json"
    technical_path.parent.mkdir()
    monkeypatch.setattr(cli, "CANONICAL_STAGE_RAW_ROOTS", {"stage1": raw_root})
    monkeypatch.setattr(cli, "CANONICAL_STAGE1_REPLAY", replay_path)
    monkeypatch.setattr(cli, "CANONICAL_STAGE1_AGGREGATE", aggregate_path)
    monkeypatch.setattr(cli, "CANONICAL_STAGE1_TECHNICAL", technical_path)
    return raw_root, replay_path, aggregate_path, technical_path


def test_stage2_gate_recomputes_report_from_current_canonical_evidence(
    monkeypatch, tmp_path, frozen_config,
):
    raw_root, replay_path, aggregate_path, technical_path = (
        _install_canonical_stage1_paths(monkeypatch, tmp_path)
    )
    report = _technical_report(frozen_config)
    technical_path.write_text(json.dumps(report), encoding="ascii")
    aggregate_path.write_bytes(b"aggregate")
    replay_path.write_text("{}", encoding="ascii")
    checked = []
    monkeypatch.setattr(
        cli, "require_tracked_clean_evidence", lambda path: checked.append(path),
    )
    observed = {}

    def recompute(aggregate, replay, raw, config):
        observed.update(
            aggregate=aggregate, replay=replay, raw=raw, config=config,
        )
        return report

    monkeypatch.setattr(cli, "_compute_stage1_technical_report", recompute)
    assert cli._require_stage1_technical(technical_path, frozen_config) == report
    assert checked == [technical_path, aggregate_path]
    assert observed == {
        "aggregate": aggregate_path,
        "replay": replay_path,
        "raw": raw_root,
        "config": frozen_config,
    }


@pytest.mark.parametrize(
    "changed_field",
    ["aggregate_sha256", "replay_report_sha256", "raw_manifest_sha256"],
)
def test_stage2_gate_rejects_stale_but_well_formed_digest(
    monkeypatch, tmp_path, frozen_config, changed_field,
):
    _, _, aggregate_path, technical_path = _install_canonical_stage1_paths(
        monkeypatch, tmp_path,
    )
    stored = _technical_report(frozen_config)
    current = dict(stored)
    current[changed_field] = "b" * 64
    technical_path.write_text(json.dumps(stored), encoding="ascii")
    aggregate_path.write_bytes(b"aggregate")
    monkeypatch.setattr(cli, "require_tracked_clean_evidence", lambda _path: None)
    monkeypatch.setattr(
        cli, "_compute_stage1_technical_report",
        lambda *_args, **_kwargs: current,
    )
    with pytest.raises(ValueError, match=changed_field):
        cli._require_stage1_technical(technical_path, frozen_config)


def test_stage1_report_computation_validates_live_replay_and_raw_manifest(
    monkeypatch, tmp_path, frozen_config,
):
    replay = {
        "status": "PASS",
        "shards": 1248,
        "raw_manifest_sha256": "c" * 64,
    }
    replay_path = tmp_path / "REPLAY_STAGE1.json"
    replay_path.write_text(json.dumps(replay), encoding="ascii")
    aggregate_path = tmp_path / "stage1_aggregate.npz"
    aggregate_path.write_bytes(b"aggregate payload")
    raw_root = tmp_path / "stage1"
    raw_root.mkdir()
    replay_sha = sha256_json(replay)
    code_status = np.full((48, 13), "INCOMPLETE", dtype="U12")
    code_status[:24] = "REPORTABLE"
    m_status = np.full((6, 13), "INCOMPLETE", dtype="U12")
    m_status[:3] = "REPORTABLE"
    aggregate = {
        "replay_status": "PASS",
        "replay_scope": "stage1",
        "replay_report_sha256": replay_sha,
        "raw_manifest_sha256": replay["raw_manifest_sha256"],
        "replay_report_json": canonical_json(replay),
        "code_status": code_status,
        "m_status": m_status,
        "overall_status": "INCOMPLETE",
        "terminal_status": "EXP103_INCOMPLETE",
        "unexpected_raw_errors_json": "[]",
    }
    monkeypatch.setattr(cli, "load_exp103_crossing", lambda path: aggregate)
    validated = []

    def validate(report, root, config, scope):
        validated.append((report, root, config, scope))

    monkeypatch.setattr(cli, "validate_replay_report", validate)
    result = cli._compute_stage1_technical_report(
        aggregate_path, replay_path, raw_root, frozen_config,
    )
    assert validated == [(replay, raw_root, frozen_config, "stage1")]
    assert result["raw_manifest_sha256"] == replay["raw_manifest_sha256"]
    assert result["aggregate_sha256"] != "c" * 64


def test_formal_scan_and_replay_reject_noncanonical_same_named_raw_paths(
    tmp_path, frozen_config,
):
    impostor_root = tmp_path / "raw"
    impostor_stage = impostor_root / "stage1"
    impostor_stage.mkdir(parents=True)
    with pytest.raises(ValueError, match="formal scan raw root.*canonical"):
        cli.run_stage(
            frozen_config["config_path"], "stage1", impostor_root,
            tmp_path / "preflight.json", 8,
        )
    with pytest.raises(ValueError, match="formal replay raw root.*canonical"):
        cli.main([
            "replay", "--config", frozen_config["config_path"],
            "--raw-root", str(impostor_stage), "--output",
            str(impostor_stage / "REPLAY_STAGE1.json"), "--num-workers", "8",
        ])


def test_canonical_aggregate_paths_are_reserved_by_replay_scope(tmp_path):
    with pytest.raises(ValueError, match="stage1 aggregate output"):
        cli._require_canonical_aggregate_output(tmp_path / "wrong.npz", "stage1")
    assert cli._require_canonical_aggregate_output(
        cli.CANONICAL_STAGE1_AGGREGATE, "stage1",
    ) == cli.CANONICAL_STAGE1_AGGREGATE
    with pytest.raises(ValueError, match="cannot write replay scope"):
        cli._require_canonical_aggregate_output(
            cli.CANONICAL_STAGE1_AGGREGATE, "none",
        )


def test_formal_report_requires_canonical_paths_before_identity_gates(
    monkeypatch, tmp_path, frozen_config,
):
    canonical_result = tmp_path / "final_results" / "decoder_crossing.npz"
    canonical_output = canonical_result.parent
    monkeypatch.setattr(cli, "CANONICAL_FINAL_AGGREGATE", canonical_result)
    monkeypatch.setattr(cli, "CANONICAL_FINAL_RESULTS", canonical_output)
    monkeypatch.setattr(cli, "load_config", lambda _path: frozen_config)
    monkeypatch.setattr(
        cli, "runtime_identity",
        lambda *_args, **_kwargs: pytest.fail("identity gate ran for a noncanonical path"),
    )

    with pytest.raises(ValueError, match="formal final report aggregate.*canonical"):
        cli.main([
            "report", "--config", frozen_config["config_path"],
            "--result", str(tmp_path / "impostor.npz"),
            "--output-dir", str(canonical_output),
        ])
    with pytest.raises(ValueError, match="formal final report output directory.*canonical"):
        cli.main([
            "report", "--config", frozen_config["config_path"],
            "--result", str(canonical_result),
            "--output-dir", str(tmp_path / "elsewhere"),
        ])


def test_formal_report_runs_all_frozen_gates_before_generation(
    monkeypatch, tmp_path, frozen_config,
):
    canonical_result = tmp_path / "final_results" / "decoder_crossing.npz"
    canonical_output = canonical_result.parent
    monkeypatch.setattr(cli, "CANONICAL_FINAL_AGGREGATE", canonical_result)
    monkeypatch.setattr(cli, "CANONICAL_FINAL_RESULTS", canonical_output)
    monkeypatch.setattr(cli, "load_config", lambda _path: frozen_config)
    calls = []
    monkeypatch.setattr(
        cli, "runtime_identity",
        lambda config, verify_source: calls.append(
            ("runtime", config, verify_source),
        ),
    )
    monkeypatch.setattr(
        cli, "verify_frozen_repository",
        lambda path: calls.append(("repository", path)),
    )
    monkeypatch.setattr(
        cli, "_require_validation001",
        lambda config: calls.append(("validation001", config)),
    )

    def generate(result_path, output_dir):
        calls.append(("generate", result_path, output_dir))
        return {"terminal_status": "EXP103_DECODER_CROSSING_INCONCLUSIVE"}

    monkeypatch.setattr(cli, "generate_final_report", generate)
    cli.main([
        "report", "--config", frozen_config["config_path"],
        "--result", str(canonical_result),
        "--output-dir", str(canonical_output),
    ])
    assert calls == [
        ("runtime", frozen_config, True),
        ("repository", frozen_config["config_path"]),
        ("validation001", frozen_config),
        ("generate", str(canonical_result), str(canonical_output)),
    ]


@pytest.mark.parametrize("filename", report_module.FINAL_REPORT_FILENAMES)
def test_final_report_refuses_every_preexisting_target_without_overwrite(
    monkeypatch, tmp_path, filename,
):
    output_dir = tmp_path / "final_results"
    output_dir.mkdir()
    target = output_dir / filename
    target.write_bytes(b"frozen evidence")
    monkeypatch.setattr(
        report_module, "load_exp103_crossing",
        lambda _path: pytest.fail("aggregate loaded after immutable-target conflict"),
    )

    with pytest.raises(FileExistsError, match="final report evidence is immutable"):
        report_module.generate_final_report(tmp_path / "result.npz", output_dir)
    assert target.read_bytes() == b"frozen evidence"
    assert sorted(path.name for path in output_dir.iterdir()) == [filename]


def test_code_task_resumes_only_a_canonical_valid_raw_key(
    monkeypatch, tmp_path, frozen_config, raw_factory,
):
    config = dict(frozen_config)
    config["p_tokens"] = ["0.02"]
    config["shards_per_code_p"] = 1
    output = tmp_path / raw_filename("m03_c00", "0.02", 0)
    save_raw(output, raw_factory("m03_c00", "0.02", 0))
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    validate_raw = cli._validate_raw
    monkeypatch.setattr(
        cli, "_validate_raw",
        lambda raw, _config, row, code_id, p_token, shard_index: validate_raw(
            raw, frozen_config, row, code_id, p_token, shard_index,
        ),
    )

    def forbidden_fresh_run(*_args, **_kwargs):
        raise AssertionError("a canonical existing shard must be resumed")

    monkeypatch.setattr(cli, "run_decoder_shard", forbidden_fresh_run)
    code_id, results = cli._save_code_task(("m03_c00", "config.json", tmp_path))
    assert code_id == "m03_c00"
    assert results == [(str(output), "RESUMED")]


def test_code_task_rejects_valid_status_with_wrong_embedded_raw_key(
    monkeypatch, tmp_path, frozen_config, raw_factory,
):
    config = dict(frozen_config)
    config["p_tokens"] = ["0.02"]
    config["shards_per_code_p"] = 1
    output = tmp_path / raw_filename("m03_c00", "0.02", 0)
    wrong_key = raw_factory("m03_c00", "0.02", 0)
    wrong_key["code_id"] = "m03_c01"
    save_raw(output, wrong_key)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    validate_raw = cli._validate_raw
    monkeypatch.setattr(
        cli, "_validate_raw",
        lambda raw, _config, row, code_id, p_token, shard_index: validate_raw(
            raw, frozen_config, row, code_id, p_token, shard_index,
        ),
    )

    with pytest.raises(ValueError, match="canonical key.*identity_mismatch:code_id"):
        cli._save_code_task(("m03_c00", "config.json", tmp_path))
