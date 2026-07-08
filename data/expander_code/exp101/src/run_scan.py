"""扫描入口（G2.7）：per-(m,q,disorder) 任务 → 原子 chunk → merge 兼容 NPZ。

结构（对齐主项目 production_chunked_scan 的必要最小子集，notes/00 §2.5）：
  - 固定 p（与 exp37/exp41 的 per-p NPZ 约定一致），扫 (code_size m) × q × disorder。
  - 每任务 chunk JSON 原子写（tmp+rename），重跑自动跳过已完成任务（断点续采）。
  - disorder seed scope = sha256(family_fp | sector | ensemble | p | q |
    disorder_index | proto)——与执行顺序/节点无关。
  - merge → sector_ti_results.npz 兼容字段 + manifest（commit SHA、版本、
    ensemble、frame/family 指纹、u-set 协议、逐 m 的 weights 槽位布局）。
    k 随 m 变化 ⇒ weights/delta_f 槽位按各 m 的槽数取最大并以 NaN pad，
    布局记录在 manifest["weights_layout"]。
引擎：
  - "ti"（默认，生产路径）：run_sector_ti；full 档存 2^k 权重，pairwise 档存
    k+1 槽（[F(ℓ_ref), F(ℓ_ref⊕e_u)...] 的 ΔF 与 m_u^pair）。
  - "direct"：run_multi_start + 收敛 gate + 聚合（q_top_basis 等）。
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .families import find_family_seed
from .gates import GateThresholds, evaluate_convergence_gate, run_multi_start
from .graphs import (
    complete_bipartite_graph,
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from .hgp import classical_parity_check_matrix, hgp_from_H
from .logicals import logical_pauli_operators
from .model import assemble_sector_model, disorder_from_uniforms, wire_ensemble
from .observables import (
    aggregate_observables,
    build_observable_frame,
    build_observable_set,
)
from .reference_mcmc import ReferenceMcmcConfig
from .sector_ti import SectorTiConfig, run_sector_ti

PROTOCOL_VERSION = "exp101.scan.v1"


# ---------- code family registry ----------

def build_code(family, size, family_rule="full_rank", family_seed=None):
    """family ∈ {expander34, toric, surface, k43}；返回 (H_Z, H_X, logicals, meta)。"""
    if family == "expander34":
        if size == 1:
            graph = random_biregular_graph_from_m(1, 3, 4, 12345)
            seed_used = 12345
        elif family_seed is not None:
            graph = random_biregular_graph_from_m(size, 3, 4, family_seed)
            seed_used = family_seed
        else:
            seed_used, _, graph, _, _, _ = find_family_seed(size, family_rule)
        classical = classical_parity_check_matrix(graph)
        meta = {"family": family, "size": int(size), "seed": int(seed_used),
                "rule": family_rule}
    elif family == "toric":
        classical = cycle_parity_check_matrix(size)
        meta = {"family": family, "size": int(size)}
    elif family == "surface":
        classical = repetition_parity_check_matrix(size)
        meta = {"family": family, "size": int(size)}
    elif family == "k43":
        classical = classical_parity_check_matrix(complete_bipartite_graph(4, 3))
        meta = {"family": family, "size": 1}
    else:
        raise ValueError(f"unknown family {family}")
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    meta["classical_sha"] = hashlib.sha256(
        np.ascontiguousarray(classical).tobytes()
    ).hexdigest()
    return H_Z, H_X, logicals, meta


def task_seed(family_fp, sector, ensemble, p, q, disorder_index, stream):
    payload = (
        f"{PROTOCOL_VERSION}|{family_fp}|{sector}|{ensemble}"
        f"|p={p:.10g}|q={q:.10g}|dis={int(disorder_index)}|{stream}"
    )
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


# ---------- scan config ----------

DEFAULT_TI = dict(num_kp_grid_points=33, num_burn_in_sweeps=200,
                  num_measurements=400, block_count=8, num_bootstrap=200)
DEFAULT_DIRECT = dict(num_burn_in_sweeps=500, num_measurements=4000,
                      num_starts=4)


def _atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, ensure_ascii=False)
    tmp.rename(path)


def _git_commit_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[4], timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _versions():
    import numpy

    versions = {"python": platform.python_version(),
                "numpy": numpy.__version__}
    try:
        import numba

        versions["numba"] = numba.__version__
    except ImportError:
        versions["numba"] = None
    return versions


def run_single_task(models_cache, family, size, sector, ensemble, p, q,
                    disorder_index, engine, engine_config, family_rule,
                    family_seed, u_rand_count):
    """单任务：确定性 disorder → 引擎 → 可 JSON 化结果 dict。"""
    key = (family, size)
    if key not in models_cache:
        H_Z, H_X, logicals, meta = build_code(family, size, family_rule,
                                              family_seed)
        model = assemble_sector_model(H_X, H_Z, logicals, sector=sector)
        frame = build_observable_frame(model)
        models_cache[key] = (model, frame, meta)
    model, frame, meta = models_cache[key]
    family_fp = meta["classical_sha"]

    dis_seed = task_seed(family_fp, sector, ensemble, p, q, disorder_index,
                         "disorder")
    rng_dis = np.random.default_rng(dis_seed)
    disorder = disorder_from_uniforms(
        model, p, q,
        data_uniforms=rng_dis.random(model.num_qubits),
        syndrome_uniforms=rng_dis.random(model.num_checks),
    )
    wiring = wire_ensemble(model, disorder, ensemble, frame)
    engine_seed = task_seed(family_fp, sector, ensemble, p, q, disorder_index,
                            f"engine:{engine}")

    started = time.perf_counter()
    result = {
        "family": meta, "sector": sector, "ensemble": ensemble,
        "p": float(p), "q": float(q), "disorder_index": int(disorder_index),
        "k": int(model.k), "n": int(model.num_qubits),
        "eta_weight": int(disorder.eta_weight),
        "delta_weight": int(disorder.delta_weight),
        "disorder_seed": int(dis_seed), "engine_seed": int(engine_seed),
        "engine": engine,
        "ell_ref": int(sum(1 << b for b, v in
                           enumerate(wiring.reference_label) if v)),
    }
    if engine == "ti":
        config = SectorTiConfig(**engine_config)
        ti = run_sector_ti(model, frame, wiring, config, seed=engine_seed)
        result.update({
            "tier": ti["tier"],
            "delta_f": ti["delta_f"].tolist(),
            "delta_f_stderr": ti["delta_f_stderr"].tolist(),
            "flags": ti["flags"],
        })
        if ti["tier"] == "full":
            result.update({
                "q_top": ti["q_top"], "q_top_stderr": ti["q_top_stderr"],
                "w0": ti["w0"],
                "weights": ti["weights_relative"].tolist(),
                "m_u_basis": ti["m_u_basis"].tolist(),
                "grid_tv": ti["grid_tv"],
                "grid_q_top_abs_diff": ti["grid_q_top_abs_diff"],
                "num_weight_slots": len(ti["weights_relative"]),
            })
        else:
            result.update({
                "q_top": ti["q_top_pairwise"],
                "q_top_stderr": float(
                    np.std(ti["m_u_pairwise"] ** 2, ddof=1)
                    / np.sqrt(max(model.k, 2))
                ),
                "w0": None,
                "weights": ti["delta_f"].tolist(),  # pairwise 槽=ΔF 序列
                "m_u_basis": ti["m_u_pairwise"].tolist(),
                "grid_tv": None, "grid_q_top_abs_diff": None,
                "num_weight_slots": len(ti["delta_f"]),
            })
    elif engine == "direct":
        cfg = dict(engine_config)
        num_starts = cfg.pop("num_starts", 4)
        config = ReferenceMcmcConfig(record_observable_trajectory=True, **cfg)
        obs_set = build_observable_set(
            frame,
            u_rand_seed=task_seed(family_fp, sector, ensemble, p, q,
                                  disorder_index, "uset")
            if model.k > 10 else None,
            num_random_u=u_rand_count,
        )
        starts = run_multi_start(model, frame, obs_set, wiring, config,
                                 base_seed=engine_seed,
                                 num_starts=num_starts)
        report = evaluate_convergence_gate(starts, thresholds=GateThresholds())
        pooled_m = np.mean([s["m_u"] for s in starts], axis=0)
        agg = aggregate_observables(obs_set, pooled_m)
        result.update({
            "tier": obs_set.tier,
            "q_top": agg["q_top_all"], "q_top_basis": agg["q_top_basis"],
            "w0": agg["w0"], "purity": agg["purity"],
            "m_u_pooled": pooled_m.tolist(),
            "gate_passed": bool(report.passed),
            "gate_failed_checks": report.failed_checks,
            "flags": "PASS" if report.passed else ";".join(
                report.failed_checks),
            "num_weight_slots": 0,
            "delta_f": [], "delta_f_stderr": [],
        })
    else:
        raise ValueError(f"unknown engine {engine}")
    result["wall_time_seconds"] = time.perf_counter() - started
    return result


def scan(output_dir, family, size_list, p_value, q_values, num_disorders,
         sector="x_error", ensemble="true_posterior", engine="ti",
         engine_config=None, family_rule="full_rank", family_seed=None,
         u_rand_count=64):
    """主扫描：断点续采 + merge。返回 (npz_path, merge_report)。"""
    output_dir = Path(output_dir)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    engine_config = dict(engine_config or (
        DEFAULT_TI if engine == "ti" else DEFAULT_DIRECT
    ))

    models_cache = {}
    reused = 0
    computed = 0
    tasks = []
    for size in size_list:
        for q in q_values:
            for disorder_index in range(num_disorders):
                tasks.append((size, float(q), disorder_index))
    for size, q, disorder_index in tasks:
        tag = f"m{size}_q{q:.6g}_d{disorder_index}"
        chunk_path = chunk_dir / f"task_{tag}.json"
        if chunk_path.exists():
            try:
                with chunk_path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload.get("protocol") == PROTOCOL_VERSION:
                    reused += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass  # 损坏 chunk 重算
        result = run_single_task(
            models_cache, family, size, sector, ensemble, p_value, q,
            disorder_index, engine, engine_config, family_rule, family_seed,
            u_rand_count,
        )
        _atomic_write_json(chunk_path, {
            "protocol": PROTOCOL_VERSION, "result": result,
        })
        computed += 1

    npz_path = merge(output_dir, family, size_list, p_value, q_values,
                     num_disorders, sector, ensemble, engine, engine_config,
                     family_rule)
    return npz_path, {"reused": reused, "computed": computed,
                      "total": len(tasks)}


def merge(output_dir, family, size_list, p_value, q_values, num_disorders,
          sector, ensemble, engine, engine_config, family_rule):
    """chunk → NPZ（sector_ti_results.npz 兼容字段）+ manifest。"""
    output_dir = Path(output_dir)
    chunk_dir = output_dir / "chunks"
    num_m, num_q, num_d = len(size_list), len(q_values), int(num_disorders)

    def load(size, q, disorder_index):
        tag = f"m{size}_q{q:.6g}_d{disorder_index}"
        with (chunk_dir / f"task_{tag}.json").open(encoding="utf-8") as handle:
            return json.load(handle)["result"]

    all_results = [
        [[load(size, float(q), d) for d in range(num_d)]
         for q in q_values]
        for size in size_list
    ]
    max_slots = max(
        (r["num_weight_slots"] for ms in all_results for qs in ms for r in qs),
        default=0,
    )
    max_k = max(r["k"] for ms in all_results for qs in ms for r in qs)

    def tensor(getter, default=np.nan, dtype=np.float64, extra=()):
        arr = np.full((num_m, num_q, num_d, *extra), default, dtype=dtype)
        for i in range(num_m):
            for j in range(num_q):
                for d in range(num_d):
                    value = getter(all_results[i][j][d])
                    if value is None:
                        continue
                    arr[(i, j, d)] = value
        return arr

    q_top = tensor(lambda r: r["q_top"])
    q_top_stderr = tensor(lambda r: r.get("q_top_stderr"))
    weights = np.full((num_m, num_q, num_d, max(max_slots, 1)), np.nan)
    delta_f = np.full_like(weights, np.nan)
    m_u = np.full((num_m, num_q, num_d, max(max_k, 1)), np.nan)
    flags = np.full((num_m, num_q, num_d), "", dtype="U128")
    for i in range(num_m):
        for j in range(num_q):
            for d in range(num_d):
                r = all_results[i][j][d]
                w = r.get("weights") or []
                weights[i, j, d, :len(w)] = w
                df = r.get("delta_f") or []
                delta_f[i, j, d, :len(df)] = df
                mu = r.get("m_u_basis") or r.get("m_u_pooled") or []
                m_u[i, j, d, :len(mu)] = mu[:max_k]
                flags[i, j, d] = str(r.get("flags", ""))[:128]

    manifest = {
        "protocol": PROTOCOL_VERSION,
        "mode": f"exp101_{engine}",
        "family": family, "family_rule": family_rule,
        "sector": sector, "ensemble": ensemble,
        "code_size_list": list(map(int, size_list)),
        "p_value": float(p_value),
        "q_values": list(map(float, q_values)),
        "num_disorder_samples": num_d,
        "engine_config": engine_config,
        "git_commit_sha": _git_commit_sha(),
        "versions": _versions(),
        "hostname": platform.node(),
        "per_size_meta": {
            str(size): all_results[i][0][0]["family"]
            for i, size in enumerate(size_list)
        },
        "per_size_k": {
            str(size): all_results[i][0][0]["k"]
            for i, size in enumerate(size_list)
        },
        "weights_layout": (
            "full: relative class weights (2^k slots, NaN-padded); "
            "pairwise: delta_f per label [ℓ_ref, ℓ_ref^e_u...] (k+1 slots)"
        ),
        "created_unix": time.time(),
    }
    npz_path = output_dir / "sector_ti_results.npz"
    np.savez_compressed(
        npz_path,
        manifest_json=json.dumps(manifest, ensure_ascii=False),
        code_size_list=np.asarray(size_list, dtype=np.int64),
        lattice_size_list=np.asarray(size_list, dtype=np.int64),  # 兼容别名
        q_values=np.asarray(q_values, dtype=np.float64),
        p_value=np.float64(p_value),
        q_top_per_disorder=q_top,
        q_top_stderr_per_disorder=q_top_stderr,
        weights_per_disorder=weights,
        delta_f_per_disorder=delta_f,
        m_u_per_disorder=m_u,
        ell_ref_per_disorder=tensor(lambda r: r["ell_ref"], default=-1,
                                    dtype=np.int64),
        flags_per_disorder=flags,
        wall_time_seconds_per_disorder=tensor(
            lambda r: r["wall_time_seconds"]),
        disorder_seed_per_disorder=tensor(lambda r: r["disorder_seed"],
                                          default=-1, dtype=np.int64),
        sample_seed_per_disorder=tensor(lambda r: r["engine_seed"],
                                        default=-1, dtype=np.int64),
        mean_q_top=np.nanmean(q_top, axis=2),
        disorder_sem_q_top=(
            np.nanstd(q_top, axis=2, ddof=1) / np.sqrt(num_d)
            if num_d > 1 else np.full((num_m, num_q), np.nan)
        ),
        pass_fraction=np.mean(
            np.char.find(flags, "PASS") >= 0, axis=2
        ).astype(np.float64),
    )
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return npz_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="exp101 scan entry")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", default="expander34",
                        choices=["expander34", "toric", "surface", "k43"])
    parser.add_argument("--size-list", type=int, nargs="+", required=True)
    parser.add_argument("--p-value", type=float, required=True)
    parser.add_argument("--q-values", type=float, nargs="+", required=True)
    parser.add_argument("--num-disorders", type=int, required=True)
    parser.add_argument("--sector", default="x_error",
                        choices=["x_error", "z_error"])
    parser.add_argument("--ensemble", default="true_posterior",
                        choices=["true_posterior", "repo_compat"])
    parser.add_argument("--engine", default="ti", choices=["ti", "direct"])
    parser.add_argument("--family-rule", default="full_rank",
                        choices=["full_rank", "full_rank_d3"])
    parser.add_argument("--family-seed", type=int, default=None)
    parser.add_argument("--ti-grid-points", type=int, default=33)
    parser.add_argument("--ti-burn-in", type=int, default=200)
    parser.add_argument("--ti-measurements", type=int, default=400)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    engine_config = None
    if args.engine == "ti":
        engine_config = dict(
            DEFAULT_TI,
            num_kp_grid_points=args.ti_grid_points,
            num_burn_in_sweeps=args.ti_burn_in,
            num_measurements=args.ti_measurements,
        )
    npz_path, report = scan(
        args.output_dir, args.family, args.size_list, args.p_value,
        args.q_values, args.num_disorders, sector=args.sector,
        ensemble=args.ensemble, engine=args.engine,
        engine_config=engine_config, family_rule=args.family_rule,
        family_seed=args.family_seed,
    )
    print(f"npz: {npz_path}\nreport: {report}")


if __name__ == "__main__":
    main()
