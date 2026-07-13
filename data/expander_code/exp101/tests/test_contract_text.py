"""Repository-level guards against reviving pre-alignment semantics."""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXP101_ROOT = Path(__file__).resolve().parents[1]

CURRENT_NARRATIVE_PATHS = [
    EXP101_ROOT / "report.md",
    EXP101_ROOT / "paper_alignment_report.md",
    EXP101_ROOT / "prompt.md",
    EXP101_ROOT / "notes" / "02_env.md",
    EXP101_ROOT / "validation" / "README.md",
]

HISTORICAL_RUNNERS = [
    EXP101_ROOT / "validation" / "003_family_registry_20260707" /
    "build_registry.py",
    EXP101_ROOT / "validation" / "004_v1_main_matrix_20260708" /
    "run_v1.py",
    EXP101_ROOT / "validation" / "004_v1_main_matrix_20260708" /
    "finalize_v1.py",
    EXP101_ROOT / "validation" / "005_v2_analytic_limits_20260708" /
    "run_v2.py",
    EXP101_ROOT / "validation" / "006_v1c_frame_ab_20260709" /
    "run_v1c.py",
    EXP101_ROOT / "validation" /
    "007_pairwise_characterization_20260709" / "run_pairwise_char.py",
    EXP101_ROOT / "validation" / "008_v3_nishimori_20260709" /
    "run_v3.py",
    EXP101_ROOT / "validation" /
    "009_v4_v6_redundancy_torture_20260709" / "run_v4_v6.py",
    EXP101_ROOT / "validation" / "011_g4_profile_20260709" /
    "run_profile.py",
    EXP101_ROOT / "validation" / "012_g4_physics_smoke_20260709" /
    "run_physics.py",
]


def _current_authoritative_text():
    paths = [
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "CLAUDE.md",
        EXP101_ROOT / "AGENTS.md",
        EXP101_ROOT / "PHYSICS_CONTRACT.md",
        EXP101_ROOT / "plan.md",
        EXP101_ROOT / "status.md",
        *CURRENT_NARRATIVE_PATHS,
        EXP101_ROOT / "notes" / "00_interface_recon.md",
        EXP101_ROOT / "notes" / "01_model_spec.md",
    ]
    paths.extend(sorted((EXP101_ROOT / "src").glob("*.py")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_current_contract_never_reverses_plus_zero_sector_mapping():
    text = _current_authoritative_text()
    wrong_x_mapping = re.compile(
        r"(?:x_error|X\s*错误)\s*/\s*H_Z(?:\s*checks)?\s*"
        r"(?:<->|↔|对应)\s*`?\|0(?:>|_L|bar)",
        re.IGNORECASE,
    )
    wrong_z_mapping = re.compile(
        r"(?:z_error|Z\s*错误)\s*/\s*H_X(?:\s*checks)?\s*"
        r"(?:<->|↔|对应)\s*`?\|\+(?:>|_L|bar)",
        re.IGNORECASE,
    )
    assert wrong_x_mapping.search(text) is None
    assert wrong_z_mapping.search(text) is None


def test_current_source_has_no_removed_pairwise_qtop_or_w0_output():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((EXP101_ROOT / "src").glob("*.py"))
    )
    assert "q_top_pairwise" not in source
    assert "m_u_pairwise" not in source
    assert re.search(r"[\"']w0[\"']\s*:", source) is None
    assert "sector_ti_results.npz" not in source
    assert 'PROTOCOL_VERSION = "exp101.scan.v2"' in source


def test_local_contract_does_not_leak_machine_environment_rules():
    local_agents = (EXP101_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert re.search(r"conda\s+(?:activate|run)", local_agents, re.I) is None
    assert re.search(r"(?:环境|environment)\s*`?12`?", local_agents, re.I) is None
    assert "miniforge" not in local_agents.lower()
    assert "macmini" not in local_agents.lower()


def test_current_narratives_keep_v2_done_and_historical_boundary_explicit():
    texts = {
        path.name: path.read_text(encoding="utf-8")
        for path in CURRENT_NARRATIVE_PATHS
    }
    assert "ERRATUM" in texts["report.md"]
    assert "PRE_ALIGNMENT" in texts["report.md"]
    assert "DONE" in texts["report.md"]
    assert "DONE" in texts["paper_alignment_report.md"]
    assert "完整 Clifford" in texts["paper_alignment_report.md"]
    assert texts["prompt.md"].startswith("# PRE_ALIGNMENT")
    assert "已停用" in texts["prompt.md"]
    assert "exp101.scan.v2" in texts["02_env.md"]
    assert "ProcessPoolExecutor" in texts["02_env.md"]
    assert "DONE" in texts["02_env.md"]
    assert "当前 v2 认证：PASS" in texts["README.md"]
    assert "001`–`013" in texts["README.md"]


def test_historical_runners_cannot_overwrite_alignment_warning():
    for path in HISTORICAL_RUNNERS:
        source = path.read_text(encoding="utf-8")
        assert source.startswith("# PRE_ALIGNMENT"), path
        assert "PRE_ALIGNMENT（自动生成保护）" in source, path


def test_tracked_historical_summaries_remain_marked_pre_alignment():
    historical_root = EXP101_ROOT / "validation"
    numbered_summaries = sorted(historical_root.glob("0??_*/summary.md"))
    summaries = [
        historical_root / "003_family_registry_20260707" /
        "family_registry.md",
        *[
            path for path in numbered_summaries
            if int(path.parent.name[:3]) <= 13
        ],
    ]
    assert summaries
    for path in summaries:
        assert "PRE_ALIGNMENT" in path.read_text(encoding="utf-8"), path


def test_validation_014_runner_forces_fresh_auditable_evidence():
    runner = (
        EXP101_ROOT / "validation" / "014_paper_alignment_20260713" /
        "run_alignment_evidence.py"
    ).read_text(encoding="utf-8")
    assert "force_recompute=True" in runner
    assert 'assert pt_report["computed"] == 1' in runner
    assert 'assert pt_report["reused"] == 0' in runner
    assert "implementation_fingerprint" in runner
    assert "git_worktree_dirty" in runner
    assert "pytest_full_output.txt" in runner
    assert "pytest_exit_code.txt" in runner
    assert 'HERE / "summary.md"' in runner
