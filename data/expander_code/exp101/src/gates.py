"""收敛诊断与 gate（G2.6）。

判据（plan §3 G2.6 + CLAUDE.md 物理图像节）：
  1. per-u split-R̂（跨起点链）≤ max_r_hat
  2. per-u ESS ≥ min_ess
  3. 跨起点 q_top spread ≤ max_q_top_spread
  4. **sector transport（硬判据，「共冻 ≠ 收敛」）**：
       worst-u 冷端 logical 接受率 ≥ min_cold_logical_acceptance
       或 PT round_trips ≥ min_round_trips（且 min swap rate > 0）。
     不满足则 FAIL——即使多起点一致（同 sector 共冻会假性一致）。
     nan 接受率（0 次尝试，如 logical_move_repeat=0）按冻结处理。

诊断实现独立于主项目（公式标准）：
  - integrated_autocorrelation_time：Sokal 自动窗（c=5）
  - split_r_hat：Gelman–Rubin split-R̂（每链对半拆）
  - effective_sample_size：N_total / τ（跨链拼合的保守版）
"""

from dataclasses import dataclass, field, replace

import numpy as np


# ---------- 基础诊断 ----------

def autocovariance(series):
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    centered = series - series.mean()
    # FFT avoids the quadratic cost of running the gate for every sampled
    # character across four long production chains.
    fft_size = 1 << max(1, (2 * n - 1).bit_length())
    transformed = np.fft.rfft(centered, n=fft_size)
    result = np.fft.irfft(
        transformed * np.conjugate(transformed), n=fft_size
    )[:n]
    return result / n


def integrated_autocorrelation_time(series, window_factor=5.0):
    """Sokal 自动窗：τ = 1 + 2 Σ_{t≤M} ρ_t，M = 最小的 M ≥ c·τ(M)。"""
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    if n < 8 or np.allclose(series, series[0]):
        return float("inf") if np.allclose(series, series[0]) else 1.0
    gamma = autocovariance(series)
    if gamma[0] <= 0:
        return 1.0
    rho = gamma / gamma[0]
    tau = 1.0
    for M in range(1, n // 2):
        tau = 1.0 + 2.0 * float(np.sum(rho[1:M + 1]))
        if M >= window_factor * tau:
            break
    return max(tau, 1e-3)


def split_r_hat(chains):
    """chains: (num_chains, num_samples)。每链对半拆后按 Gelman–Rubin 计算。

    退化（组内方差为 0）：均值全等 → 1.0；均值不等 → inf。
    """
    chains = np.asarray(chains, dtype=np.float64)
    num_chains, num_samples = chains.shape
    half = num_samples // 2
    if half < 2:
        return float("nan")
    split = chains[:, : 2 * half].reshape(num_chains * 2, half)
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    W = float(variances.mean())
    B = half * float(means.var(ddof=1))
    if W <= 1e-300:
        return 1.0 if B <= 1e-300 else float("inf")
    var_plus = (half - 1) / half * W + B / half
    return float(np.sqrt(var_plus / W))


def effective_sample_size(chains):
    """保守 ESS：Σ_chains n_i / τ_i（逐链 τ，取和）。全常数链 → 0。"""
    chains = np.asarray(chains, dtype=np.float64)
    total = 0.0
    for chain in chains:
        if np.allclose(chain, chain[0]):
            continue  # 冻结链贡献 0
        total += chain.shape[0] / integrated_autocorrelation_time(chain)
    return float(total)


# ---------- gate ----------

@dataclass
class GateThresholds:
    max_r_hat: float = 1.05
    min_ess: float = 200.0
    max_q_top_spread: float = 0.03
    max_m_u_spread: float = 0.10   # 符号敏感！q_top=mean m_u² 是符号盲的，
                                   # 不同 sector 共冻给出相同 q_top——必须用
                                   # m_u 本身的跨起点极差探测 sector 分歧
    min_cold_logical_acceptance: float = 1e-4
    min_round_trips: int = 1


@dataclass
class GateReport:
    passed: bool
    failed_checks: list
    metrics: dict
    thresholds: GateThresholds
    notes: list = field(default_factory=list)


def run_multi_start(model, frame, observable_set, wiring, config, base_seed,
                    num_starts=8, sector_bitmasks=None, engine="fast"):
    """多起点运行（sector 轮转初始化），产出 evaluate_convergence_gate 的输入。

    要求 config.record_observable_trajectory=True（gate 需要轨迹）。
    q>0 用 zero+sector 平移起点；q=0 用 section+sector 平移（auto 模式内部处理）。
    """
    from .fast_mcmc import run_fast_mcmc
    from .observables import aggregate_observables
    from .reference_mcmc import run_reference_mcmc

    if not config.record_observable_trajectory:
        raise ValueError("multi-start gate requires record_observable_trajectory")
    if sector_bitmasks is None:
        sector_bitmasks = default_sector_bitmasks(model.k, num_starts)
    runner = run_fast_mcmc if engine == "fast" else run_reference_mcmc
    results = []
    for start_index, bitmask in enumerate(sector_bitmasks):
        result = runner(model, frame, observable_set, wiring, config,
                        seed=int(base_seed) + start_index,
                        sector_bitmask=int(bitmask))
        result["aggregates"] = aggregate_observables(
            observable_set, result["m_u"]
        )
        result["sector_bitmask"] = int(bitmask)
        results.append(result)
    return results


def default_sector_bitmasks(k, num_starts=8):
    """起点 sector 轮转：0,1,2,...（截断到 2^k 上限）。k 大时是子集轮转。"""
    if k <= 0:
        return [0] * int(num_starts)
    upper = 1 << min(k, 20)
    return [i % upper for i in range(int(num_starts))]


def evaluate_convergence_gate(start_results, pt_result=None, thresholds=None,
                              q_top_key="q_top_all"):
    """start_results：多起点运行列表，每项 dict 至少含
        "observable_trajectory" (num_meas, num_u) int8
        "aggregates" dict（含 q_top_basis / q_top_all）
        "acceptance" dict（含 logical_per_u）
    pt_result：可选 PtResult（提供 round_trips 与 swap 传输信息）。
    """
    thresholds = thresholds or GateThresholds()
    failed = []
    notes = []
    num_starts = len(start_results)
    if num_starts < 2:
        raise ValueError("gate requires >= 2 starts")

    trajectories = [np.asarray(r["observable_trajectory"], dtype=np.float64)
                    for r in start_results]
    num_u = trajectories[0].shape[1]

    # 1/2. per-u split-R̂ 与 ESS（跨起点链）
    r_hat_per_u = np.zeros(num_u)
    ess_per_u = np.zeros(num_u)
    for u in range(num_u):
        chains = np.stack([t[:, u] for t in trajectories])
        r_hat_per_u[u] = split_r_hat(chains)
        ess_per_u[u] = effective_sample_size(chains)
    finite_r = r_hat_per_u[np.isfinite(r_hat_per_u)]
    max_r_hat = float(np.max(r_hat_per_u)) if num_u else 1.0
    min_ess = float(np.min(ess_per_u)) if num_u else float("inf")
    if not np.isfinite(max_r_hat) or max_r_hat > thresholds.max_r_hat:
        failed.append(f"max_r_hat>{thresholds.max_r_hat}")
    if min_ess < thresholds.min_ess:
        failed.append(f"min_ess<{thresholds.min_ess:.0f}")

    # 3a. q_top spread（幅值一致性）
    q_top_values = np.array(
        [float(r["aggregates"][q_top_key]) for r in start_results]
    )
    q_top_spread = float(q_top_values.max() - q_top_values.min())
    if q_top_spread > thresholds.max_q_top_spread:
        failed.append(f"q_top_spread>{thresholds.max_q_top_spread}")

    # 3b. m_u spread（符号敏感——探测不同 sector 共冻）
    m_u_per_start = np.stack([t.mean(axis=0) for t in trajectories])
    m_u_spread = (
        float((m_u_per_start.max(axis=0) - m_u_per_start.min(axis=0)).max())
        if num_u else 0.0
    )
    if m_u_spread > thresholds.max_m_u_spread:
        failed.append(f"m_u_spread>{thresholds.max_m_u_spread}")

    # 4. sector transport（硬判据；nan 接受率按 0 处理）
    worst_u_acceptance = float("inf")
    for r in start_results:
        per_u = np.asarray(r["acceptance"]["logical_per_u"], dtype=np.float64)
        per_u = np.where(np.isfinite(per_u), per_u, 0.0)
        if per_u.size:
            worst_u_acceptance = min(worst_u_acceptance, float(per_u.min()))
    if not np.isfinite(worst_u_acceptance):
        worst_u_acceptance = 0.0
    local_transport_ok = (
        worst_u_acceptance >= thresholds.min_cold_logical_acceptance
    )
    pt_transport_ok = False
    round_trips = None
    min_swap_rate = None
    if pt_result is not None:
        measured_round_trips = getattr(
            pt_result, "measurement_round_trips", None
        )
        round_trips = int(
            pt_result.round_trips
            if measured_round_trips is None else measured_round_trips
        )
        min_swap_rate = float(pt_result.swap_rates().min())
        pt_transport_ok = (
            round_trips >= thresholds.min_round_trips and min_swap_rate > 0.0
        )
    if not (local_transport_ok or pt_transport_ok):
        failed.append("sector_transport_insufficient")
        notes.append(
            "共冻 ≠ 收敛：多起点一致不足为凭——worst-u 冷端 logical 接受率 "
            f"{worst_u_acceptance:.2e} 低于阈值且无 PT 往返证据"
        )

    metrics = {
        "max_r_hat": max_r_hat,
        "r_hat_per_u": r_hat_per_u,
        "min_ess": min_ess,
        "ess_per_u": ess_per_u,
        "q_top_values": q_top_values,
        "q_top_spread": q_top_spread,
        "m_u_per_start": m_u_per_start,
        "m_u_spread": m_u_spread,
        "worst_u_cold_logical_acceptance": worst_u_acceptance,
        "local_transport_ok": local_transport_ok,
        "pt_round_trips": round_trips,
        "pt_min_swap_rate": min_swap_rate,
        "pt_transport_ok": pt_transport_ok,
        "num_starts": num_starts,
    }
    return GateReport(
        passed=not failed,
        failed_checks=failed,
        metrics=metrics,
        thresholds=thresholds,
        notes=notes,
    )


def evaluate_pt_convergence_gate(
    pt_results, observable_set, thresholds=None, min_instances=4
):
    """Gate independent PT instances using their cold-end trajectories.

    Statistical R-hat/ESS checks use one cold trajectory per PT instance.
    Transport is deliberately stricter than the direct-chain OR rule: every
    instance must complete enough round trips, every adjacent swap edge must
    have nonzero acceptance, and the pooled worst logical-basis acceptance
    must pass the established cold-end threshold.
    """
    thresholds = thresholds or GateThresholds()
    pt_results = list(pt_results)
    if len(pt_results) < 2:
        raise ValueError("PT gate requires at least two independent instances")

    starts = []
    from .observables import aggregate_observables

    for result in pt_results:
        trajectory = result.observable_trajectory_cold
        if trajectory is None:
            raise ValueError(
                "PT gate requires record_observable_trajectory=True"
            )
        aggregates = aggregate_observables(
            observable_set, np.asarray(result.m_u_cold)
        )
        starts.append({
            "observable_trajectory": trajectory,
            "aggregates": aggregates,
            "acceptance": {
                "logical_per_u": result.cold_logical_acceptance_per_u(),
            },
        })

    # Reuse the common statistical diagnostics without allowing local moves
    # to substitute for the PT-specific transport requirements below.
    diagnostic_thresholds = replace(
        thresholds, min_cold_logical_acceptance=0.0
    )
    base = evaluate_convergence_gate(
        starts, thresholds=diagnostic_thresholds
    )
    failed = [
        check for check in base.failed_checks
        if check != "sector_transport_insufficient"
    ]

    measurement_round_trips_per_instance = np.asarray(
        [
            result.round_trips
            if getattr(result, "measurement_round_trips", None) is None
            else result.measurement_round_trips
            for result in pt_results
        ], dtype=np.int64
    )
    burn_in_round_trips_per_instance = np.asarray([
        -1
        if getattr(result, "burn_in_round_trips", None) is None
        else result.burn_in_round_trips
        for result in pt_results
    ], dtype=np.int64)
    min_swap_rate_per_instance = np.asarray([
        float(np.min(result.swap_rates())) for result in pt_results
    ])
    logical_attempts = np.sum([
        result.counters_per_rung[0].logical_attempts_per_u
        for result in pt_results
    ], axis=0)
    logical_accepts = np.sum([
        result.counters_per_rung[0].logical_accepts_per_u
        for result in pt_results
    ], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_acceptance = np.where(
            logical_attempts > 0,
            logical_accepts / np.maximum(logical_attempts, 1),
            0.0,
        )
    pooled_acceptance = np.where(
        np.isfinite(pooled_acceptance), pooled_acceptance, 0.0
    )
    pooled_worst = (
        float(np.min(pooled_acceptance)) if pooled_acceptance.size else 0.0
    )

    if len(pt_results) < int(min_instances):
        failed.append(f"pt_instance_count<{int(min_instances)}")
    if np.any(
        measurement_round_trips_per_instance < thresholds.min_round_trips
    ):
        failed.append("pt_instance_round_trips_insufficient")
    if np.any(min_swap_rate_per_instance <= 0.0):
        failed.append("pt_adjacent_swap_rate_zero")
    if pooled_worst < thresholds.min_cold_logical_acceptance:
        failed.append("pt_pooled_logical_acceptance_insufficient")

    metrics = dict(base.metrics)
    metrics.update({
        "pt_instance_count": len(pt_results),
        "pt_round_trips_per_instance": measurement_round_trips_per_instance,
        "pt_burn_in_round_trips_per_instance": (
            burn_in_round_trips_per_instance
        ),
        "pt_measurement_round_trips_per_instance": (
            measurement_round_trips_per_instance
        ),
        "pt_min_swap_rate_per_instance": min_swap_rate_per_instance,
        "pt_pooled_logical_acceptance_per_u": pooled_acceptance,
        "pt_pooled_worst_basis_logical_acceptance": pooled_worst,
        "pt_all_instances_round_trip_ok": bool(np.all(
            measurement_round_trips_per_instance
            >= thresholds.min_round_trips
        )),
        "pt_all_adjacent_swap_edges_nonzero": bool(np.all(
            min_swap_rate_per_instance > 0.0
        )),
    })
    notes = list(base.notes)
    if failed:
        notes.append(
            "PT production requires four independent cold traces, per-instance "
            "round trips, nonzero adjacent swaps, and pooled logical transport."
        )
    return GateReport(
        passed=not failed,
        failed_checks=failed,
        metrics=metrics,
        thresholds=thresholds,
        notes=notes,
    )
