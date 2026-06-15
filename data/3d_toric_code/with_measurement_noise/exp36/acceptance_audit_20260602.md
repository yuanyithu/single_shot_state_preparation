# exp36 acceptance audit: sector-histogram convergence and q_top sampling

更新时间：2026-06-02

## 判定范围

本审计只判定当前目标窗口：

- `L=3,4,5,6`
- `p=0.05`
- `q=0.08,0.09,...,0.23`
- disorder average: `64` disorder per `(L,q)`

核心物理要求是：固定 disorder 后，从不同 Wilson-loop sector 初态出发，长时间采样
得到同一个 cold-sector 驻留分布；随后才能相信该 disorder 的 Wilson loop 期望和
`q_top`，再做 disorder average。

## 接受的算法/生产版本

当前接受版本是 exp36/018 生产配置，加上 019/020 对边界点的 8-start reference：

- production: `018_production64_full_q_grid_20260531`
- targeted references:
  - `019_targeted_8start_sector_reference_20260531`
  - `020_q0p18_L3_d10_8start_reference_20260601`

018 生产配置：

- `pt_ladder_mode=sync_enlarge`
- `pt_q_hot=0.35`
- `pt_num_temperatures=17`
- `pt_cold_edge_swap_stride=4`
- `cluster_update_enabled=True`
- `cluster_budget_fraction_rho=0.15`
- `observable_temperature_mode=cold`
- `num_burn_in_sweeps=150`
- `effective_num_burn_in_sweeps_list=[675,750,750,750]` for `L=3,4,5,6`
- `num_sweeps_between_measurements=6`
- `num_measurements_per_disorder=1024`
- `num_start_chains=4`
- `num_replicas_per_start=1`
- `q_positive_initial_chain_mode=sector`
- start labels: `000,100,010,110`
- `q_top_block_count=8`
- `seed_base=518000`
- `common_random_disorder_across_p=True`
- `chunk_size=1`

019/020 reference 配置只用于复查 marginal fail：

- `num_start_chains=8`
- start labels: `000,100,010,110,001,101,011,111`
- `num_measurements_per_disorder=8192`
- `q_top_block_count=32`
- 其余热化/更新参数与 018 同一物理版本保持一致。

## 证据 1：全 q-grid sector gate 覆盖完整

018 已对 `q=0.08..0.23` 的全部 16 个 q 生成 final NPZ。每个 q 覆盖
`L=3,4,5,6` 和每个尺寸 `64` disorder。本地已对每个 q 的 final chunks 跑
10000-bootstrap sector-histogram gate。

审计命令结果：

```text
missing_reports []
wrong_reps []
num_summary_rows 64
fail_rows [('0.130', '3', '1'), ('0.180', '3', '1'), ('0.230', '3', '1')]
```

解释：64 个 `(q,L)` 生产点中，61 个点生产 gate 直接通过；剩余 3 个点都是
`L=3` 的单-disorder 边界 flag。

## 证据 2：生产边界 fail 被 8-start reference 清除

| q | L | disorder | 018 start-TV | 018 boot p99 | 018 q_top | reference | ref start-TV | ref boot p99 | ref q_top | 判定 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0.13 | 3 | 52 | 0.0400 | 0.0342 | 0.865445 | 019 | 0.0035 | 0.0128 | 0.880904 | 清除 |
| 0.18 | 3 | 10 | 0.0195 | 0.0176 | 0.964710 | 020 | 0.0033 | 0.0065 | 0.969400 | 清除 |
| 0.23 | 3 | 15 | 0.0332 | 0.0312 | 0.892532 | 019 | 0.0056 | 0.0125 | 0.888930 | 清除 |

这些 reference 的共同特征是：top sector probabilities 与 018 相容，但 8-start
长链下的 start-TV 和 `q_top spread` 明显下降。因此这些生产 flag 更像有限
measurement 的 histogram 波动，不是固定 disorder 下稳定的 sector metastability。

## 证据 3：q_top 方差

最大 disorder SEM：

| L | max SEM | at q | 约 95% CI | q_top range |
|---:|---:|---:|---:|---:|
| 3 | 0.017068 | 0.23 | 0.033453 | 0.923398..0.986364 |
| 4 | 0.003326 | 0.17 | 0.006519 | 0.989885..0.999835 |
| 5 | 0.000561 | 0.20 | 0.001100 | 0.996903..0.999843 |
| 6 | 0.000740 | 0.20 | 0.001450 | 0.994312..0.999686 |

判定：

- 对 `L>=4`，方差已经很小，足以稳定说明大尺寸在该 q 窗口内仍接近 `q_top=1`。
- `L=3` 方差较大，但最大 95% CI 约 `0.033`，足以给出当前窗口下的
  disorder-averaged `q_top` 曲线和误差条。
- 这些误差不支持精确 threshold/crossing 定位；但当前目标不是定 threshold，
  而是确认算法可热化并给出可信 q_top 采样值。

## 证据 4：q_top 曲线物理方向

合并图：

- `018_production64_full_q_grid_20260531/analysis/exp36_018_fixed_p050_q080_230_final_sem95.png`
- `018_production64_full_q_grid_20260531/analysis/exp36_018_fixed_p050_q080_230_final_gap_ci95.png`
- `018_production64_full_q_grid_20260531/analysis/exp36_018_fixed_p050_q080_230_final_summary.json`

gap summary：

```text
L3-L4: min -0.066487, max -0.013470, crossing_windows []
L4-L5: min -0.007320, max -0.000008, crossing_windows []
L5-L6: min  0.000157, max  0.002894, crossing_windows []
```

解释：在 `p=0.05,q<=0.23` 中，大尺寸没有出现 exp35 那种明显反物理的系统性下降。
`L=4,5,6` 都接近 1，`L=3` 随 q 增大下降且 disorder 方差较大。当前窗口不显示干净
q-threshold crossing。

## 判定

当前 exp36 版本满足本轮停止条件：

1. 已引入并使用 cold-sector histogram gate，而不是只看 `q_top`。
2. 已覆盖目标参数窗口 `L=3,4,5,6; p=0.05; q=0.08..0.23`。
3. 固定 disorder 下，不同初态的 sector histogram 没有稳定不一致信号。
4. 生产中的 3 个 marginal fail 已被 `8/8` 初态、`m=8192` reference 清除。
5. `q_top` disorder SEM 对当前物理判断可接受。

因此，当前算法配置可以作为“已通过 sector-histogram 热化检验的 q_top 采样版本”。

本地代码 sanity：

```text
python -m py_compile src/summarize_exp36_sector_gate.py src/plot_fixed_p_q_scan.py src/production_chunked_scan.py src/main.py
```

已通过。

## 明确边界

- 这不是 q-threshold 的最终定位。当前窗口内没有干净 crossing，后续若要定边界，
  需要换参数窗口或更大样本/尺寸。
- 018 生产本身只覆盖 `4/8` 初态；全 8 初态证据来自 targeted reference，而不是每个
  production disorder 都跑 8-start。
- 若后续改变更新参数、温度 ladder、observable 公式或初始化方式，需要重新跑
  sector-histogram gate，不能直接继承本审计结论。
