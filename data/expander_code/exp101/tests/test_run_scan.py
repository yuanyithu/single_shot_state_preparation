"""G2.7 单元测试：扫描入口（chunk 原子性/续采/merge schema/确定性）。"""

import json

import numpy as np
import pytest

from src.run_scan import (
    PROTOCOL_VERSION,
    build_code,
    merge,
    scan,
    task_seed,
)

FAST_TI = dict(num_kp_grid_points=9, num_burn_in_sweeps=40,
               num_measurements=120, block_count=6, num_bootstrap=60)
FAST_DIRECT = dict(num_burn_in_sweeps=150, num_measurements=1200,
                   num_starts=2)


class TestTaskSeeds:
    def test_scope_and_determinism(self):
        a = task_seed("fp", "x_error", "true_posterior", 0.1, 0.05, 3, "s")
        b = task_seed("fp", "x_error", "true_posterior", 0.1, 0.05, 3, "s")
        assert a == b
        assert a != task_seed("fp", "x_error", "true_posterior", 0.1, 0.05, 4, "s")
        assert a != task_seed("fp", "x_error", "true_posterior", 0.1, 0.06, 3, "s")
        assert a != task_seed("fp", "x_error", "repo_compat", 0.1, 0.05, 3, "s")
        assert a != task_seed("fp2", "x_error", "true_posterior", 0.1, 0.05, 3, "s")


class TestBuildCode:
    def test_known_families(self):
        for family, size, expected_nk in [
            ("surface", 2, (5, 1)), ("toric", 2, (8, 2)), ("k43", 1, (25, 13)),
            ("expander34", 2, (100, 4)),
        ]:
            H_Z, H_X, logicals, meta = build_code(family, size)
            assert (H_X.shape[1], logicals.k) == expected_nk
            assert "classical_sha" in meta


class TestScanEndToEnd:
    def test_ti_scan_resume_and_schema(self, tmp_path):
        out = tmp_path / "scan_ti"
        npz_path, report = scan(
            out, "surface", [2], 0.12, [0.08, 0.15], 2,
            engine="ti", engine_config=FAST_TI,
        )
        assert (report["reused"], report["computed"], report["total"]) == (0, 4, 4)
        assert report["failed"] == []
        # 续采：全部 chunk 复用，结果不变
        with np.load(npz_path, allow_pickle=True) as data:
            q_top_first = data["q_top_per_disorder"].copy()
        npz_path2, report2 = scan(
            out, "surface", [2], 0.12, [0.08, 0.15], 2,
            engine="ti", engine_config=FAST_TI,
        )
        assert (report2["reused"], report2["computed"], report2["total"]) == (4, 0, 4)
        with np.load(npz_path2, allow_pickle=True) as data:
            assert np.array_equal(data["q_top_per_disorder"], q_top_first)
        # schema 字段
        with np.load(npz_path, allow_pickle=True) as data:
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["protocol"] == PROTOCOL_VERSION
            assert manifest["ensemble"] == "true_posterior"
            assert manifest["git_commit_sha"]
            assert manifest["per_size_k"]["2"] == 1
            assert data["q_top_per_disorder"].shape == (1, 2, 2)
            assert data["weights_per_disorder"].shape[3] == 2  # 2^k=2 槽
            assert data["m_u_per_disorder"].shape == (1, 2, 2, 1)
            assert data["lattice_size_list"].tolist() == [2]  # 兼容别名
            assert np.all(np.char.find(
                data["flags_per_disorder"].astype(str), "") >= -1)
            assert data["mean_q_top"].shape == (1, 2)
        # 无 .tmp 残留（原子写）
        assert not list((out / "chunks").glob("*.tmp"))

    def test_deterministic_across_fresh_runs(self, tmp_path):
        r1, _ = scan(tmp_path / "a", "surface", [2], 0.12, [0.08], 2,
                     engine="ti", engine_config=FAST_TI)
        r2, _ = scan(tmp_path / "b", "surface", [2], 0.12, [0.08], 2,
                     engine="ti", engine_config=FAST_TI)
        with np.load(r1, allow_pickle=True) as d1, \
                np.load(r2, allow_pickle=True) as d2:
            assert np.array_equal(d1["q_top_per_disorder"],
                                  d2["q_top_per_disorder"])
            assert np.array_equal(d1["disorder_seed_per_disorder"],
                                  d2["disorder_seed_per_disorder"])

    def test_corrupted_chunk_recomputed(self, tmp_path):
        out = tmp_path / "scan_corrupt"
        scan(out, "surface", [2], 0.12, [0.08], 1, engine="ti",
             engine_config=FAST_TI)
        chunk = next((out / "chunks").glob("task_*.json"))
        chunk.write_text("{broken", encoding="utf-8")
        _, report = scan(out, "surface", [2], 0.12, [0.08], 1, engine="ti",
                         engine_config=FAST_TI)
        assert report["computed"] == 1  # 损坏被重算

    def test_direct_engine_scan(self, tmp_path):
        npz_path, report = scan(
            tmp_path / "scan_direct", "surface", [2], 0.12, [0.08], 1,
            engine="direct", engine_config=FAST_DIRECT,
        )
        assert report["computed"] == 1
        with np.load(npz_path, allow_pickle=True) as data:
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["mode"] == "exp101_direct"
            q_top = data["q_top_per_disorder"]
            assert np.isfinite(q_top).all()

    def test_repo_compat_ensemble_recorded_and_distinct_disorder(self, tmp_path):
        r_true, _ = scan(tmp_path / "e_true", "surface", [2], 0.12, [0.08], 1,
                         engine="ti", engine_config=FAST_TI,
                         ensemble="true_posterior")
        r_repo, _ = scan(tmp_path / "e_repo", "surface", [2], 0.12, [0.08], 1,
                         engine="ti", engine_config=FAST_TI,
                         ensemble="repo_compat")
        with np.load(r_true, allow_pickle=True) as dt, \
                np.load(r_repo, allow_pickle=True) as dr:
            assert json.loads(str(dt["manifest_json"]))["ensemble"] \
                == "true_posterior"
            assert json.loads(str(dr["manifest_json"]))["ensemble"] \
                == "repo_compat"
            # 系综影响 seed scope（防误合并）
            assert not np.array_equal(dt["disorder_seed_per_disorder"],
                                      dr["disorder_seed_per_disorder"])

    def test_toric_multi_size_padding(self, tmp_path):
        """不同 m（k=2 恒定）多尺寸合并；槽位 pad 语义。"""
        npz_path, _ = scan(
            tmp_path / "scan_ms", "toric", [2, 3], 0.12, [0.10], 1,
            engine="ti", engine_config=FAST_TI,
        )
        with np.load(npz_path, allow_pickle=True) as data:
            assert data["q_top_per_disorder"].shape == (2, 1, 1)
            assert np.isfinite(data["q_top_per_disorder"]).all()
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["per_size_k"] == {"2": 2, "3": 2}


class TestParallelism:
    def test_parallel_matches_serial_bit_identical(self, tmp_path):
        """确定性：num_workers=4 与 =1 必须逐位一致（seed scope 与 worker 数无关）。"""
        r1, rep1 = scan(tmp_path / "serial", "surface", [2], 0.12,
                        [0.08, 0.15], 2, engine="ti", engine_config=FAST_TI,
                        num_workers=1)
        r4, rep4 = scan(tmp_path / "par", "surface", [2], 0.12,
                        [0.08, 0.15], 2, engine="ti", engine_config=FAST_TI,
                        num_workers=4)
        assert rep4["num_workers"] == 4 and rep4["computed"] == 4
        assert rep4["failed"] == []
        with np.load(r1, allow_pickle=True) as d1, \
                np.load(r4, allow_pickle=True) as d4:
            assert np.array_equal(d1["q_top_per_disorder"],
                                  d4["q_top_per_disorder"])
            assert np.array_equal(d1["disorder_seed_per_disorder"],
                                  d4["disorder_seed_per_disorder"])
            assert np.array_equal(d1["m_u_per_disorder"],
                                  d4["m_u_per_disorder"])

    def test_merge_handles_missing_chunk(self, tmp_path):
        out = tmp_path / "miss"
        scan(out, "surface", [2], 0.12, [0.08, 0.15], 2, engine="ti",
             engine_config=FAST_TI, num_workers=1)
        chunk = sorted((out / "chunks").glob("task_*.json"))[0]
        chunk.unlink()   # 模拟失败/缺失 cell
        npz_path = merge(out, "surface", [2], 0.12, [0.08, 0.15], 2,
                         "x_error", "true_posterior", "ti", FAST_TI,
                         "full_rank")
        with np.load(npz_path, allow_pickle=True) as data:
            manifest = json.loads(str(data["manifest_json"]))
            assert manifest["missing_chunks"] == 1
            flags = data["flags_per_disorder"].astype(str)
            assert (flags == "MISSING").sum() == 1
            assert np.isfinite(data["q_top_per_disorder"]).sum() == 3
