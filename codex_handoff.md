# 交接给 codex：3D toric code `q>0` sampler 可信化收尾

> 最后更新：2026-04-23。写给下一位接手的 agent。读完这一份文档就能直接继续干活，不需要再回溯前面的对话。

---

## 1. 一句话现状

**3D toric code 在带 measurement noise (`q>0`) 下的 MCMC sampler，原本在 `L≥5, low p, low q` 被 topological sector trap 卡死。现在已经写好 parallel tempering 作为修复手段，在 L=2 exact 和 L=5 multi-start 两个层面都被证明正确，但还没有接入生产扫描管线，physics 图还不能重做。**

预计再 1–2 天可以把可信版本固定下来。

---

## 2. 项目整体背景

本项目是做 3D toric code 作为 single-shot 量子纠错码的数值仿真，目标是画 `q_top(p, q, L)` 相图，定位 threshold。核心采样问题是 fixed-disorder 的 posterior `π(c | syndrome, data_error)` 采样，其中 chain `c` 同时要服从 syndrome constraint。

权重结构（对 chain 的非规范化 log posterior）：

```
log π(c) = log_odds_data × W_data(c) + log_odds_syndrome × W_syndrome(c)
log_odds(x) = log(x / (1 - x))        # 注意是 p/(1-p)，不是 (1-p)/p
W_data = |c XOR disorder_data_error_bits|
W_syndrome = |H_Z @ c XOR observed_syndrome|
```

关键物理约束：
- `q=0` 模式：`W_syndrome` 强制为 0，只能走 zero-syndrome move（vertex-star contractible + winding sheet）
- `q>0` 模式：可以同时用 single-bit flip + zero-syndrome move 的 hybrid sampler

关键 code 路径：
- 构码 / zero-syndrome move generators：[src/build_toric_code_examples.py:227](src/build_toric_code_examples.py#L227), [src/build_toric_code_examples.py:290](src/build_toric_code_examples.py#L290)
- 主采样循环：[src/main.py:714](src/main.py#L714) `_run_single_disorder_measurement`
- 生产扫描入口：[src/main.py:1002](src/main.py#L1002) `run_disorder_average_simulation`
- 诊断工具：[src/diagnose_3d_q_positive_mixing.py](src/diagnose_3d_q_positive_mixing.py)
- exact 对照（L=2 上限）：[src/exact_enumeration.py](src/exact_enumeration.py)
- **新增** parallel tempering：[src/mcmc_parallel_tempering.py](src/mcmc_parallel_tempering.py)

---

## 3. 病灶（必须理解的物理）

**3D toric code 的 Z-check 是 plaquette (不是 cube)，zero-syndrome move 里跨 sector 的唯一通道是 weight = L² 的 winding sheet**。在 ordered phase（`p < p_c ≈ 0.3`）：

```
winding-sheet Metropolis acceptance ≈ exp(-L² × |log_odds|)
```

`L=5, p=0.22` 下这个数大约是 `exp(-31) ≈ 3e-14`。**实际上永远 reject。** 所以 sector 间完全 frozen，但 sector 内部靠 vertex-star contractible move 依然能 mix — 这会让单链在 observable 上**看起来收敛**，但数值其实不是真实 posterior。

这是 topological 相变模型里的标准失败模式。标准解是 parallel tempering in `p`：加几个热温度（`p > p_c`），热链里 `log_odds` 小，winding sheet acceptance 变高（甚至 ~50%），sector 可以自由流动；再通过 replica swap 把热链的 state 传回冷链。

---

## 4. 已经完成并钉死的工作

### A. 诊断能力升级
把原来只输出全局 aggregate 的诊断 harness 升级成能看到每个 (disorder, start, replica) 链的 breakdown：

- [src/main.py](src/main.py) 的 `_run_measurement_update_cycle` / `_run_single_disorder_measurement` 现在返回拆开的 `contractible_acceptance_rate` 和 `winding_acceptance_rate`
- [src/diagnose_3d_q_positive_mixing.py:418](src/diagnose_3d_q_positive_mixing.py#L418) `_summarize_multi_chain_batch` 的输出 JSON 里多了：
  - `num_chains_that_never_flipped_sector`（最直接的 trap smoking gun）
  - `first_signature_change_index_per_chain`
  - `winding_acceptance_rate_per_chain`
  - `final_q_top_per_chain` / `final_m_u_per_chain`
  - per-disorder `q_top_spread` / `m_u_spread_linf` / `max_r_hat`

### B. PT 实现
[src/mcmc_parallel_tempering.py](src/mcmc_parallel_tempering.py) 的 `run_parallel_tempering_measurement`：

- Ladder 由调用方给 `data_error_probability_ladder`（1D array, K 个温度, index 0 是 cold）
- 每 sweep 所有温度各跑一次 `_run_measurement_update_cycle`，然后 alternating even/odd pair 做一次 swap
- Swap 概率：`min(1, exp((L_j - L_i)(W_i - W_j)))`，`W = data_term_weight`, `L = log_odds`
- Swap 交换 `(chain_bits, data_term_bits, syndrome_term_bits)` 三个数组，用 python list 引用互换（O(1)）
- 返回 `m_u_values_per_temperature` / `q_top_value_per_temperature` / swap stats

### C. PT 接入诊断 harness
[src/diagnose_3d_q_positive_mixing.py:408](src/diagnose_3d_q_positive_mixing.py#L408) 的 `_run_pt_single_chain` 把 PT 冷链输出 **shape 成和非 PT 单链兼容的 dict**，于是可以复用 `_summarize_multi_chain_batch`。新增 suite `c2`：

```
python src/diagnose_3d_q_positive_mixing.py --suite c2 \
  --c2-lattice-sizes 5 --c2-p-cold-values 0.22,0.26 \
  --c2-p-hot 0.44 --c2-num-temperatures 9 --c2-q-value 0.005
```

默认 ladder 用 `_equal_log_odds_ladder(p_cold, p_hot, K)` 构造（均匀分 log-odds，在 3D toric code 上是合理起点）。

### D. 回归测试永久化
[src/exact_enumeration.py](src/exact_enumeration.py) 的 `__main__` 现在跑 4 个测试：

1. `_run_3d_q_positive_exact_vs_mcmc_regression`：扩展到 14 grid points（原来 9 点 + 5 个 physics ballpark 点 `q=5e-3 × p∈{0.20..0.28}`），所有点 MCMC 与 exact L=2 agree
2. `_run_3d_q_positive_exact_vs_pt_regression`：3 grid points × 9 温度 = 27 组 PT vs exact 对照，误差最大 `|Δq_top|=0.006, |Δm_u|=0.024`（tolerance 0.056）
3. q→0 continuity diagnostic（原有）
4. q>0 single-bit brute-force（原有）

全部 PASS。未来任何改动都会被这个 regression 套抓到。

---

## 5. 一项关键数据：PT 有没有治好

L=5, p=0.22, q=0.005, 1 disorder, 8 multi-start, 1 replica, 512 measurements：

| 指标 | non-PT (A1 smoke) | PT (c2 smoke) |
|---|---|---|
| `num_chains_that_never_flipped_sector` | 5/8 | **0/8** |
| `mean_q_top_spread_per_disorder` | 0.228 | **0.048** |
| `max_r_hat_across_disorders` | 1.077 | **1.006** |
| `min_effective_sample_size` | 13 | **73** |
| `mean_q_top` | 0.956 | 0.956 |

`q_top` 数值几乎不变（这个 disorder 下 posterior 本就 sector-0 concentrated），但 PT 下**所有收敛指标都达标**，non-PT 不达标。这正是 PT 应有的行为：不改变答案，只让我们有资格相信答案。

---

## 6. 还没干的事（按优先级）

### [P0] 把 PT 接到生产扫描管线
**这是唯一还挡着 physics 图的事。**

- 改 [src/main.py:1088-1114](src/main.py#L1088-L1114) 的 `run_disorder_average_simulation`：当前只在 `syndrome_error_probability == 0.0` 分支启用多起点 + spread 诊断，`q>0` 分支是单链 all-zero 起点。
- 推荐实现：
  1. 把 `q>0` 分支也改成多起点（复用 [src/main.py:658](src/main.py#L658) `_build_q0_initial_chain_bits_per_start`，它对 q>0 的 syndrome 也适用）
  2. 在 `q>0` 分支额外加一个参数 `use_parallel_tempering=False` / `pt_ladder=None`，当 PT 启用时每个 start × replica 走 `run_parallel_tempering_measurement`，否则走原 `_run_single_disorder_measurement`
  3. 返回字段里加 per-disorder / per-start 的 spread flag（不丢弃 disorder 样本，只标红）
- 估工 **半天到一天**（low risk；q=0 路径不能被动）

### [P1] Convergence gate 正式化
写一个 `src/mcmc_convergence_gate.py`，输入多起点链数据，输出布尔 + 详细摘要。门的初版阈值：

- `max_r_hat_across_disorders < 1.05`
- `min_effective_sample_size > 200`
- `mean_q_top_spread_per_disorder < 0.03`
- PT 冷链 `winding_acceptance_rate > 1e-4`（或 swap 带入的 sector 轮换 > 0）

estimate 半天。放在生产 scan 之后做 post-process。

### [P2] 正式跑一次 physics 扫描
在 P0+P1 落地后：

- 默认 budget：4 disorders × 8 starts × 2 replicas × 4096 measurements × 6 sweeps_between，L ∈ {3, 4, 5}, q=0.005, p ∈ [0.20, 0.30] 细扫
- 大概率要上服务器（AGENTS.md 里写了 nd-1/nd-2/nd-3 节点 + conda env `11`）
- 生产图更新完之后更新 [实验报告_3D_toric_code.md](实验报告_3D_toric_code.md)

### [P3] 其它 clean-up（低优先级，视情况）
- 加密 ladder 的精细调参（当前 9-T equal-log-odds 在 L=5 上 min swap acc 0.26，已经够用，但如果 L=6 以上需要再密）
- `_run_b1` / `_run_d1` 之前规划的 q→0 continuity 升级（叠 exact 曲线）可以在 physics 扫描后做
- 老的 `winding_repeat_factor` / `num_zero_syndrome_sweeps_per_cycle` 旋钮在 PT 下已经不再是主要 mixing 手段，可以考虑把它们从 production default 里下掉

---

## 7. 立即下一步（具体到第一条命令）

P0 的入口：

```bash
# 1. 读懂 q>0 分支当前在干什么
grep -n "syndrome_error_probability == 0.0" src/main.py    # ~1088
# 2. 把 q0_num_start_chains multi-start block 的条件从 "q=0 only" 解耦出来
#    同时加一个 pt_ladder 参数的路径分支
```

实现策略：在 `run_disorder_average_simulation` 签名加 `pt_ladder=None`。如果 `pt_ladder is not None`，所有 start × replica 都跑 PT；否则走 non-PT 老路径。`pt_ladder=None` + q>0 时，就退化成"q>0 多起点 + spread flag"，不需要 PT 的成本。

验证：
- q=0 path smoke 不变（跑 `python src/diagnose_3d_q_positive_mixing.py --suite first_batch` 应该不变）
- q>0 non-PT path：新 multi-start vs 旧单链 all-zero，在 L=3 agree within SEM
- q>0 PT path：L=2 上用 `compute_exact_logical_observable_means` 做 anchor，cold-chain q_top 在 tolerance 内

---

## 8. 常见坑（不是 bug）

1. **`single_bit_acceptance_rate = 0.0` 在 q 很小时是正常的**。Single-bit flip 必触发 ΔW_syndrome ≈ ±4（3D toric code vertex 邻接 6 edges，但每 check 是 plaquette-4），在 `q=0.005` 下 `exp(-4 × log_odds_syndrome) ≈ 1e-8`。已经证实这不是我们的 bug，在直接非 PT 跑同 config 时也是 0。mixing 全靠 zero-syndrome move。

2. **`log_odds = log(p/(1-p))`**（代码约定），不是 `log((1-p)/p)`。在写 PT swap 公式时容易搞反。正确公式：
   ```
   log_swap_ratio = (L_j - L_i) × (W_i - W_j)
   ```
   其中 `W` 是 `data_term_bits.sum()`。syndrome 项对称消去（因为所有温度 `q` 相同）。

3. **PT swap 必须同时交换 `chain_bits, data_term_bits, syndrome_term_bits`**，因为 data_term / syndrome_term 是 chain 状态的缓存。disorder_data_error_bits 和 observed_syndrome_bits 是 "环境"，不交换。

4. **`first_signature_change_index == -1`** 是 "从未换过 sector" 的标记（`chain_analysis["first_signature_change_index"]`），`_summarize_multi_chain_batch` 里会统计。

5. **c2 smoke 里 `cold q_top ≈ 0.96` 不是 mixing 没修好**。8 个 sector 起点都 agree 在 [0.93, 0.98]，说明 posterior 本就 sector-concentrated。c2 之前 non-PT 的 0.956 "碰巧正确"，但没有多起点 agree 支撑，所以不能信。PT 之后有了支撑，所以可以信。

---

## 9. 验证命令速查

```bash
# 环境：conda env 12（AGENTS.md 要求）
conda activate 12

# 短 smoke：c2 suite，1 disorder，几秒出结果
python src/diagnose_3d_q_positive_mixing.py --suite c2 \
  --c2-lattice-sizes 5 --c2-p-cold-values 0.22 --c2-q-value 0.005 \
  --c2-num-temperatures 9 --c2-p-hot 0.44 \
  --disorder-seeds 1 --num-measurements-per-disorder 512 \
  --base-num-burn-in-sweeps 100 --burn-in-multiplier 1 \
  --num-replicas-per-start 1 --num-sweeps-between-measurements 2 \
  --run-root /tmp/c2_smoke

# 完整 regression（~6 min）：L=2 exact vs MCMC + exact vs PT
python src/exact_enumeration.py

# 对照用 A1 smoke（non-PT 基线）
python src/diagnose_3d_q_positive_mixing.py --suite a1 \
  --a1-lattice-sizes 5 --a1-p-values 0.22 --a1-q-value 0.005 \
  --disorder-seeds 1 --num-measurements-per-disorder 512 \
  --base-num-burn-in-sweeps 100 --burn-in-multiplier 1 \
  --num-replicas-per-start 1 --num-sweeps-between-measurements 2 \
  --run-root /tmp/a1_smoke
```

诊断输出 JSON 里最要看的字段：`num_chains_that_never_flipped_sector`（应该是 0/8）、`mean_q_top_spread_per_disorder`（应该 < 0.05）、`max_r_hat_across_disorders`（应该 < 1.05）。

---

## 10. 时间节点预估

| 节点 | 内容 | 预估 |
|---|---|---|
| **Day 0** (即刻) | 读完本交接 + 跑一次完整 regression 确认基线 | 0.5 hr |
| **Day 1** | P0：生产管线接 PT + 多起点 + spread flag；在 L=3 回归验证 non-PT path 不变 | 半天到一天 |
| **Day 2** | P1：convergence gate；P2：启动正式 physics 扫描（在服务器 nd-1/2/3） | 一天 |
| **Day 3** | 扫描结果出来、做 threshold collapse 图、更新 [实验报告_3D_toric_code.md](实验报告_3D_toric_code.md)；commit + push | 半天到一天 |

遇到的最可能意外是 P2 物理扫描里发现 PT 在某些 `(p, q, L)` 组合仍然 R̂ 不达标。备选方案（按采纳优先级）：
1. 加密 ladder（`--c2-num-temperatures 13` / `--c2-p-hot 0.50`）
2. 加 longer burn-in × 系统尺寸
3. 加 cluster move（2-star 或 Wolff-like，不在现有代码里，需要新实现）

---

## 11. 设计上的关键取舍（不要回退）

- **PT swap 只在 ladder 相邻温度之间做**（不是全部两两），避免 O(K²) 成本
- **PT 内部不交换 disorder / observed_syndrome**，只交换 chain state。这是正确做法；如果交换 disorder，posterior 目标都变了。
- **Multi-start 用的是 8 个 topological sector（`_build_q0_initial_chain_bits_per_start` 生成）而不是随机扰动**。随机扰动没法探测 sector trap，topological 多起点 + PT 才是正解组合。
- **disorder 样本不丢弃**，不通过 convergence gate 的点打 flag 但保留数据，避免"条件抽样"偏差。

---

## 12. Reference 文档（按重要性）

- [codex_handoff.md](codex_handoff.md) — 本文档
- [/Users/yuan/.claude/plans/3d-tingly-snowglobe.md](/Users/yuan/.claude/plans/3d-tingly-snowglobe.md) — 完整 debug 方案（A/B/C 三条 route）
- [实验报告_3D_toric_code.md](实验报告_3D_toric_code.md) — 实验日志（最新条目在 line 830 前后）
- [AGENTS.md](AGENTS.md) — 项目 onboarding 规则（conda env, 服务器, commit 规则）
- [架构设计.md](架构设计.md) — 原始数学推导与接口设计

有问题问用户（他懂物理、懂代码，但不需要你解释底层细节）。祝顺利。
