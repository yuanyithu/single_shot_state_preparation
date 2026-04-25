# 实验报告：3D Toric Code

按是否考虑 measurement noise 重排原始实验记录。

## Without Measurement Noise

## 2026-04-21 23:42 3D Toric `q=0` 首轮实现与本地 Smoke

摘要：
- 做什么：把 3D toric `q=0` 接入统一生产扫描管线，并做本地 smoke。
- 结论：3D 路径已能稳定产出 merged `npz` 与图，但 smoke 不足以判读 threshold。
- 看图：[scan_result_multi_L_3d_toric_q0_smoke.png](data/3d_toric_code/without_measurement_noise/exp01_q0_pipeline_smoke/scan_result_multi_L_3d_toric_q0_smoke.png)

### 实现目标

- 新开 3D toric `q=0` threshold 工作树，先把 3D code 接入统一生产扫描管线。
- 保持本地只做轻量验证，重型实验后续转移到 `nd-1`。

### 代码变更

- `build_toric_code_examples.py`
  - 新增 `build_3d_toric_zero_syndrome_move_data(...)`
  - 新增 `build_toric_code_by_family(...)`
  - 新增 `build_zero_syndrome_move_data_by_family(...)`
  - 3D `q=0` 的非平凡 move 最终采用“固定坐标的整张平面上的平行边集合”，
    而不是轴向单 loop；前者才满足 `H_Z c = 0`
- `main.py`
  - `q0_num_start_chains` 改为按 `start_sector_generators.shape[0]`
    自动确定上限，不再写死 `4`
  - `q=0` 初态标签从 2 bit 泛化为任意 bit 数
  - 单点并行扫描改为通过 `code_family` 选择 2D / 3D 构码与几何 move
- `production_chunked_scan.py`
  - 新增 `--code-family {2d_toric,3d_toric}`
  - manifest / merged `npz` 写入 `code_family`
  - 生产 chunk / merge / preflight 全部支持 3D toric
- `plot_scan_results.py`、`analyze_threshold_crossing.py`
  - 图标题和阈值分析 summary 现在写入 `code_family`
- `exact_enumeration.py`
  - 保留原 2D exact regression
  - 新增 3D `q=0` move 结构测试
  - 新增 3D 八个合法 start sector 的可区分性检查
- 新增脚本：
  - `scripts/run_local_3d_q0_smoke.sh`
  - `scripts/launch_nd1_3d_q0_threshold_scout.sh`
  - `scripts/launch_nd1_3d_q0_threshold_deep.sh`

### 正确性验证

- `conda run -n 12 python src/exact_enumeration.py`
- 结果：
  - 旧的 `Test 1/2/3/4` 2D exact-vs-MCMC regression 继续全部通过
  - 新增 `Test 0b` 3D sector distinguishability 通过
  - 说明：
    - 这轮 3D 改动没有破坏现有 2D 主路径
    - 3D `q=0` 的局部 move、非平凡 sector generator 和 logical mask 彼此一致

### 本地 Smoke

- 运行命令：
  - `scripts/run_local_3d_q0_smoke.sh`
- 输出目录：
  - `data/3d_toric_code/without_measurement_noise/exp01_q0_pipeline_smoke/`
- 运行参数：
  - `code_family = 3d_toric`
  - `L = [2, 3]`
  - `p = [0.0200, 0.0500, 0.0800]`
  - `q = 0.0`
  - `num_disorder_samples_total = 16`
  - `num_burn_in_sweeps = 200`
  - `num_sweeps_between_measurements = 4`
  - `num_measurements_per_disorder = 80`
  - `q0_num_start_chains = 8`

### Smoke 结果

```text
L=2: q_top = [0.9844, 0.9929, 0.8179]
L=3: q_top = [1.0000, 1.0000, 0.9473]

L=2: acceptance = [1.15e-3, 6.33e-4, 3.15e-2]
L=3: acceptance = [2.71e-6, 7.07e-3, 3.64e-3]

L=2: mean_m_u_spread = [0.1250, 0.0203, 0.3750]
L=3: mean_m_u_spread = [0.0000, 0.0000, 0.0766]
```

### 当前判断

- 3D `q=0` 生产路径已经能稳定跑通，并成功产出：
  - merged `npz`
  - 默认结果图
  - `code_family=3d_toric` 元数据
- 这轮 smoke 只覆盖 `L=2,3`，因此不用于判断 threshold，只用于确认：
  - 3D 几何 sampler 可运行
  - 多初态张量 shape 正确
  - 没有 `NaN` / shape mismatch / merge failure
- 从数值上看，`L=3` 在低 `p` 端 acceptance 仍偏低，说明重型 `nd-1` 扫描时
  仍需依赖更长 burn-in 和更大 disorder 数，不能把 smoke 预算当作物理结论

## 2026-04-22 00:06 ND-1 3D toric `q=0` scout 结果回收与分析

摘要：
- 做什么：3D `q=0` 首轮 scout，确定 threshold 窗口是否落在 `p≤0.12`。
- 结论：未见可信 crossing，左端命中只是饱和平台伪信号，窗口需要整体右移。
- 看图：[scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png](data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png)

### 运行与回收信息

- 远端运行目录：
  - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260421_235447/`
- 本地归档目录：
  - `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/`
- 主要文件：
  - 结果：
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/scan_result_multi_L_3d_toric_q0_threshold_scout.npz`
  - 默认图：
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/scan_result_multi_L_3d_toric_q0_threshold_scout.png`
  - 95% CI 图：
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/scan_result_multi_L_3d_toric_q0_threshold_scout_sem95.png`
  - gap 图：
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/scan_result_multi_L_3d_toric_q0_threshold_scout_gap_crossing.png`
  - 本地重跑分析图：
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/scan_result_multi_L_3d_toric_q0_threshold_scout_local_sem95.png`
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png`
  - manifest：
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/manifest.json`
  - summary：
    `data/3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/threshold_summary.json`

### 运行参数

- `code_family = 3d_toric`
- `L = [3, 4, 5]`
- `p = [0.0200, 0.0300, ..., 0.1200]`
- `q = 0.0`
- 每个 `(L, p)`：
  - `num_disorder_samples_total = 256`
  - `chunk_size = 16`
  - `num_chunks_per_point = 16`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 240`
  - `q0_num_start_chains = 8`
  - `workers = 48`

### 完成情况

- 远端 log 与 manifest 一致：

```text
completed_chunks = 528
failed_chunks = 0
pending_chunks = 0
final_outputs.status = completed
completed_at = 2026-04-21T15:55:38-04:00
```

### 关键结果

- `q_top(p)`：

```text
L=3: [1.0000, 1.0000, 0.9993, 0.9990, 0.9987, 0.9958, 0.9933, 0.9756, 0.9659, 0.9369, 0.9023]
L=4: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9996, 0.9999, 0.9998, 0.9976, 0.9923, 0.9830]
L=5: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9995]
```

- acceptance：

```text
L=3: 1.13e-4 -> 2.29e-2
L=4: 1.75e-4 -> 2.48e-2
L=5: 1.44e-4 -> 2.52e-2
```

- 多初态 spread：

```text
L=3: mean_m_u_spread 由 0 增长到 0.0696
L=4: mean_m_u_spread 由 0 增长到 0.0276
L=5: mean_m_u_spread 基本保持 0，到 p=0.12 也只有 0.0033
```

### 阈值判读

- `threshold_summary.json` 给出了：

```text
primary_crossing_window_hit = true
crossing_window = [0.02, 0.06]
recommended_server_window = [-0.02, 0.06]
```

- 但这个“命中”并不代表窗口内部存在可信 threshold crossing。
- 原因是：
  - `L=3-4` 的 pairwise gap 在 `p=0.02, 0.03` 恰好为 `0`
  - `L=4-5` 的 pairwise gap 在 `p=0.02 ~ 0.06` 也恰好为 `0`
  - 这些零 gap 来自左端所有尺寸都饱和到 `q_top = 1.0` 的平台，而不是
    一条曲线从上方穿过另一条曲线
- 更直接地看：

```text
delta_34: 从 p=0.04 开始一直为负，并且之后单调更负
delta_45: 从 p=0.07 开始一直为负，并且之后也保持为负
```

- 即在这轮 scout 的非饱和区内，始终是：

```text
L=3 < L=4 < L=5
```

- 这说明当前扫描窗口只看到了“大尺寸更稳定”的低误差相，而没有看到
  真实 crossing 进入窗口。

### 辅助判断

- 第一次“在 95% CI 外显著低于 1.0”的位置：

```text
L=3: p ≈ 0.07
L=4: p ≈ 0.11
L=5: 到 p = 0.12 仍未显著偏离 1.0
```

- 在 `p ∈ [0.07, 0.12]` 的窗口平均：

```text
L=3: mean_q_top = 0.9616, mean_accept = 0.01227, mean_spread = 0.02858
L=4: mean_q_top = 0.9954, mean_accept = 0.01300, mean_spread = 0.00766
L=5: mean_q_top = 0.9999, mean_accept = 0.01430, mean_spread = 0.00056
```

- 这进一步表明：
  - 不是因为大尺寸 mixing 更差才把 crossing 洗没
  - 恰恰相反，`L=5` 在当前窗口里几乎还停留在完美平台
  - 因而真正的 3D `q=0` threshold 若存在，至少没有落在 `p ≤ 0.12`
    这一段被 scout 覆盖的范围内

### 本轮结论

- 这轮 `nd-1` scout 成功完成了第一阶段目标中的“摸清窗口性质”：
  - 管线可靠
  - 统计量足够
  - 没有 chunk/merge/绘图失败
- 但从物理判读上：
  - 本轮**没有**看到可信的 3D `q=0` threshold crossing 证据
  - 因此**不应**按当前 `threshold_summary.json` 自动给出的
    `[-0.02, 0.06]` 窗口去开 deep run
  - 那个推荐窗口是由左端饱和平台上的零 gap 人工触发出来的，不具备物理意义

### 下一步建议

- 下一轮不做“deep around current crossing window”，而是把 scout 窗口整体右移。
- 更合理的下一轮范围应为：

```text
p ≈ 0.10 ~ 0.20
```

- 如果希望更稳妥，可以直接开：

```text
L = [3, 4, 5]
p = 0.10, 0.11, ..., 0.20
```

- 只有在右移后的窗口里真正看到 `delta_34` 和 `delta_45`
  从负变正或至少接近正负翻转，再值得进入 deep 阶段。

## 2026-04-22 15:10 3D toric `q=0` 右移搜索完成：在 `p≈0.22` 附近观察到 interior crossing

摘要：
- 做什么：3D `q=0` 连续做右移 Stage A / Stage B 搜索，定位真实 threshold 窗口。
- 结论：Stage A 仍未 crossing，Stage B 在 `p≈0.218~0.230` 首次出现可信 interior crossing。
- 看图：[scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png](data/3d_toric_code/without_measurement_noise/exp04_q0_crossing_window_scout/local_reanalysis/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png)

### 运行与归档

- `Stage A`：`nd-1` 右移 scout
  - 远端目录：
    `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_100557_rightshift_stageA/`
  - 本地归档：
    `data/3d_toric_code/without_measurement_noise/exp03_q0_right_shift_scout/`
- `Stage B`：`nd-3` extension scout
  - 远端目录：
    `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_101018_extension_stageB_nd3/`
  - 本地归档：
    `data/3d_toric_code/without_measurement_noise/exp04_q0_crossing_window_scout/`
- 两轮都另外在本地用最新的 `analyze_threshold_crossing.py` 重跑了一次分析，输出放在：
  - `.../local_reanalysis/scan_result_multi_L_3d_toric_q0_threshold_scout_local_sem95.png`
  - `.../local_reanalysis/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png`
  - `.../local_reanalysis/threshold_summary.json`

### `Stage A`：`p = 0.10 ~ 0.20` 仍未见 crossing

- 参数：
  - `L = [3, 4, 5]`
  - `p = 0.10, 0.11, ..., 0.20`
  - `num_disorder_samples_total = 256`
  - `chunk_size = 16`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 240`
  - `q0_num_start_chains = 8`
- 这轮 summary 给出：

```text
boundary_saturation_artifact = false
primary_crossing_window_hit  = false
secondary_proximity_hit      = false
recommended_server_window    = null
right_edge_gap_signs         = {'3-4': -1, '4-5': -1}
```

- 到窗口右端 `p = 0.20` 时：

```text
q_top(L=3,4,5) = [0.3385, 0.4419, 0.5715]
delta_34       = -0.1034
delta_45       = -0.1296
```

- 也就是说在整个 `0.10 ~ 0.20` 区间内始终保持：

```text
L=3 < L=4 < L=5
```

- 这说明真正的 crossing 还在更高 `p` 处，`Stage A` 只完成了“把窗口继续向右推”的任务。

### `Stage B`：`p = 0.16 ~ 0.28` 出现可信 interior crossing

- 参数：
  - `L = [3, 4, 5]`
  - `p = 0.16, 0.17, ..., 0.28`
  - `num_disorder_samples_total = 256`
  - `chunk_size = 16`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 240`
  - `q0_num_start_chains = 8`
  - 计算节点改到 `nd-3`，`workers = 96`
- 这轮 summary 给出：

```text
boundary_saturation_artifact = false
primary_crossing_window_hit  = true
secondary_proximity_hit      = false
interior_crossing_window     = [0.21825, 0.22969]
recommended_server_window    = [0.21825, 0.22969]
right_edge_gap_signs         = {'3-4': +1, '4-5': +1}
```

- pairwise gap 的首次 interior crossing 为：

```text
L3-L4: p ≈ 0.22969   (between 0.22 and 0.23)
L4-L5: p ≈ 0.21825   (between 0.21 and 0.22)
```

- `L4-L5` 在更右侧还出现了额外的小幅回穿：

```text
p ≈ 0.23705, 0.24196
```

- 但就“是否已经进入真实 threshold 窗口”这个问题而言，第一段共同 crossing 区已经足够明确。
- 到窗口右端 `p = 0.28` 时：

```text
q_top(L=3,4,5) = [0.04591, 0.02748, 0.01216]
delta_34       = +0.01843
delta_45       = +0.01532
```

- 因而 `Stage B` 的整体号数演化是：

```text
左端:  delta_34 < 0, delta_45 < 0
右端:  delta_34 > 0, delta_45 > 0
```

- 这正是之前想要看到的 interior crossing 行为。

### 采样质量与故障处理

- 平均 acceptance：

```text
Stage A: L=3/4/5 -> [0.0477, 0.0495, 0.0522]
Stage B: L=3/4/5 -> [0.1354, 0.1355, 0.1405]
```

- 平均多初态 `m_u` spread：

```text
Stage A: [0.1283, 0.1488, 0.2275]
Stage B: [0.2235, 0.2847, 0.5462]
```

- `Stage A` 在 `nd-1` 上的主采样和 merge 已完成，但远端后处理最初失败。
- 原因不是 `analyze_threshold_crossing.py` 本身，而是远端 `conda run` 读取
  `'$XDG_CONFIG_HOME/conda/.condarc'` 时触发了 `Stale file handle`。
- 修复方式：
  - 在 `nd-1` 上手动重跑后处理并成功产出 `threshold_summary.json`、`sem95`、`gap` 图
  - 同时把 launcher 统一改成 `CONDA_NO_PLUGINS=true`
  - 并把计算节点参数化，使同一套 3D `q=0` launcher 可直接切换 `nd-1/nd-3`

### 本轮结论

- 经过这两轮右移搜索，现在已经可以说：
  - 3D toric `q=0` 在当前 sampler 与 `L = [3,4,5]` 下，确实出现了可信的
    interior threshold crossing
  - crossing 的首个共同窗口大约落在

```text
p ≈ 0.218 ~ 0.230
```

- 这和之前 `p ≤ 0.12` 还处在“大尺寸几乎完美平台”里的图景是自洽的：
  - 旧窗口太靠左，所以只能看到 `L=5` 几乎不动
  - 真正 crossing 直到 `p ≈ 0.22` 左右才进入视野

- 因此，当前 3D `q=0` 第一阶段目标已经达到：
  - 不是边界饱和伪 crossing
  - 而是窗口内部的真实 crossing

### 下一步建议

- 既然现在已经锁定了第一段 interior crossing，下一步应转入 `deep`：

```text
center    ≈ 0.224
window    ≈ [0.209, 0.239]
step      = 0.005
L         = [3, 4, 5]
q         = 0
num_disorder_samples_total = 1024
```

- 如果 deep 结果仍保持两对 gap 在这个窗口中稳定、顺序合理地穿零，就可以把这套
  `q=0` 数据当作后续 `q>0` 搜索的 calibration 基线。

## With Measurement Noise

> 2026-04-22 审核说明：
> 本节 16:52 和 20:10 两轮 3D `q>0` 旧结果已判定失效。
> 根因不是 plotting，而是旧 `q>0` sampler 只做单比特更新、没有混入 zero-syndrome move，
> 导致低 `q` 时链冻结在局部 sector，`q_top` 被假性顶到接近 `1`。
> 对应本地目录
> `data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/`
> 与
> `（旧 Stage A broad scout 本地目录已清理，不作为结果保留）`
> 已删除，不再作为任何物理结论的依据。

## 2026-04-22 3D toric `q>0` 回归排查、sampler 修复与失效数据清理

摘要：
- 做什么：按 “3D exact regression → 单步 acceptance 对拍 → `q→0` continuity → 3D mask/section 自检” 的顺序排查 3D `q>0` 异常。
- 结论：定位到旧 `q>0` 路径的核心问题是采样核不完整，不是单步 acceptance 公式错误。修复后，3D `L=2` 的 9 个 `q>0` exact-vs-MCMC 点全部通过；旧预检与 Stage A 数据已删除。

### 本轮代码变更

- `src/exact_enumeration.py`
  - exact 枚举改为分块/流式实现，支持 3D `L=2` 的 `24` qubit 小系统
  - 新增 3D `q>0` exact-vs-MCMC 回归：
    - `q ∈ {1e-3, 1e-2, 4e-2}`
    - `p ∈ {0.05, 0.15, 0.25}`
  - 输出并校验：
    - `m_u`
    - `q_top`
    - `logZ`
    - posterior normalization `sum(exp(logw-logZ))`
  - 新增 `q→0` continuity 诊断
- `src/main.py`
  - 抽出单比特 `log_acceptance` 计算
  - 新增 `q>0` 单步 acceptance brute-force 对拍
  - `q>0` 采样路径改为 hybrid sweep：
    - 保留单比特更新
    - 额外混入 zero-syndrome move
  - 生产路径现在对 `q>0` 也会预构造 `zero_syndrome_move_data`
- `src/preprocessing.py`
  - 自检入口扩展到 3D `L=2` / `L=3`
  - 明确覆盖：
    - `H_Z r(σ) = σ`
    - 转置关系
    - logical mask 与 gauge-fixed representative 奇偶一致

### 关键回归结果

- `conda run -n 12 python src/exact_enumeration.py`
  - 旧 2D `Test 1/2/3/4` 全通过
  - `q>0` 单比特 brute-force local ratio test 全通过
  - 3D `L=2` 的 9 个 `q>0` exact-vs-MCMC 点全通过
  - 典型修复前后对比：

```text
旧 sampler, L=2, p=0.15, q=0.001:
  exact q_top = 0.266864
  mcmc  q_top = 1.000000

修复后:
  exact q_top = 0.266864
  mcmc  q_top = 0.265142
```

- `conda run -n 12 python src/mcmc.py`
  - 旧有 cache consistency / infinite-temperature / low-temperature / realistic smoke 全通过
- `conda run -n 12 python src/preprocessing.py`
  - 2D 与 3D `L=2,3` 的 linear section / logical mask 自检全通过

### 根因判断

- 单步 acceptance 公式本身没有错。
- 真正的问题是：
  - 旧 `q>0` 路径只做单比特 flip
  - 在低 `q` 区域，链会被 syndrome 罚项卡死在某个固定 `H_Z c` sector
  - 尤其 3D 小系统 exact regression 里，旧链会直接停在 `m_u = 1`
  - 这不是 posterior 本身，而是采样核缺少沿 `ker(H_Z)` 的零-syndrome 混合
- 修复方式是对 `q>0` 也混入 zero-syndrome move，使链能在固定 syndrome contour 内移动。

### `q→0` continuity 诊断

固定同一 disorder、`L=2`、`p=0.15`、同一 seed family 后：

```text
q=0      -> q_top = 0.381102
q=1e-6   -> q_top = 0.386342
q=1e-4   -> q_top = 0.386342
q=1e-3   -> q_top = 0.386342
```

- `q=1e-6` 与 `q=0` 已连续接近，不再出现旧路径那种“刚离开 0 就直接顶到 1”的假信号。

### 失效数据清理

- 已删除：
  - `data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/`
  - `（旧 Stage A broad scout 本地目录已清理，不作为结果保留）`
- 删除原因：
  - 它们全部来自修复前的旧 `q>0` sampler
  - 已被 exact regression 证明会在低 `q` 区域产生系统性假性高 `q_top`

### 后续建议

- 不要继续使用旧 Stage A 图做任何 threshold 判读。
- 下一步应基于修复后的 hybrid sampler 重新做：
  - 本地轻量预检
  - 远端 3D `q>0` Stage A broad scout
- 在新一轮远端扫描前，应把 `src/exact_enumeration.py` 作为 preflight 必跑项保留。

## 2026-04-22 16:52 3D toric `q>0` 管线接入与本地预检

摘要：
- 做什么：把 3D `q>0` threshold-search 编排层接入现有生产扫描管线，并完成本地轻量预检。
- 结论：3D measurement-noise 路径已能稳定跑通 `submit -> merge -> analyze`，输出目录、命名和 `threshold_summary.json` 全部落到 3D 专属路径。
- 看图：[scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_sem95.png](data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_sem95.png)

### 代码变更

- `src/measurement_noise_threshold_scout.py`
  - 新增 `--code-family {2d_toric,3d_toric}`
  - 默认 lattice sizes 改为按 code family 选择：
    - 2D：`[3,5,7]`
    - 3D：`[3,4,5]`
  - 默认输出根目录改为按 family 路由到
    `data/<family>_code/with_measurement_noise/...`
  - `q>0` 的 output stem 改为包含 `code_family`
  - 本地 scout 统一写出 `threshold_summary.json`
- `scripts/launch_3d_measurement_noise_threshold_search.sh`
  - 新增可复用的 3D measurement-noise 远端 launcher
  - 支持 `REMOTE_COMPUTE_HOST=nd-1/nd-2/nd-3`
  - 远端固定使用 `conda run -n 11 python`
  - 统一导出 `CONDA_NO_PLUGINS=true`
  - 通过 `Q_AND_P_WINDOWS` 参数化多 `q` 多窗口扫描
  - 单个 host 任务结束后自动汇总：
    - `threshold_summary.json`
    - `measurement_noise_threshold_search_summary.json`
    - overview 图

### 本地预检参数

- `code_family = 3d_toric`
- `L = [3,4,5]`
- `q = 0.0050`
- `p = [0.1600, 0.2200, 0.2800]`
- 每个 `(L,p)`：
  - `num_disorder_samples_total = 32`
  - `chunk_size = 16`
  - `num_chunks_per_point = 2`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 240`
  - `workers = 4`
  - `common_random_disorder_across_p = true`

### 预检输出

- 归档目录：
  - `data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/`
- 主要文件：
  - `q_0p0050/manifest.json`
  - `q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random.npz`
  - `q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random.png`
  - `q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_sem95.png`
  - `q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_gap_crossing.png`
  - `q_0p0050/threshold_summary.json`
  - `threshold_scout_index.json`

### 校验结果

- `manifest.json` 中：
  - `config.code_family = 3d_toric`
  - `final_outputs.status = completed`
  - `completed_chunks = 18`
  - `failed_chunks = 0`
- 预检期间运行了 `exact_enumeration.py`，其内置校验全部通过。
- `threshold_summary.json` 已正常生成；这轮轻量预检的物理判读是：

```text
boundary_saturation_artifact = true
primary_crossing_window_hit  = false
interior_crossing_window     = null
recommended_server_window    = null
right_edge_gap_signs         = {'3-4': 0, '4-5': 0}
```

- 这个结果只说明轻量预检窗口太窄且样本太少，不用于判断是否存在真实 threshold；
  本轮目的只是验证 3D `q>0` 管线、命名和分析接口已经全部接通。

### 运行时间观察

- 本地 `L=5` chunk 明显更重：

```text
L=3 chunk 约 9s
L=4 chunk 约 43s
L=5 chunk 约 154~164s
```

- 这和此前 `q=0` 的经验一致，说明真正的 Stage A/B/C 重型扫描应继续放在
  `nd-1/nd-2/nd-3` 上运行，不适合在本地做大样本搜索。

### 下一步

- 预检完成后，可以直接按既定 Stage A 在三个节点并行开 broad scout：
  - `nd-1`: `q = [0.0010, 0.0025]`
  - `nd-2`: `q = [0.0050, 0.0100]`
  - `nd-3`: `q = [0.0200, 0.0400]`
- Stage A 产物回收后，按每个 `q` 的 `threshold_summary.json`
  进入自适应 Stage B。

## 2026-04-22 20:10 3D toric `q>0` Stage A broad scout 回收、重绘与分析

摘要：
- 做什么：回收 `nd-1/nd-2/nd-3` 的 3D measurement-noise Stage A broad scout，全量下载数据与日志，在本地统一重绘 overview 并汇总判据。
- 结论：这轮 `q = 0.0010 ~ 0.0400` 没有任何一个点出现可信的三尺寸共同 interior crossing；`q=0.0200` 和 `q=0.0400` 只出现 secondary proximity，仍不足以宣称 finite threshold。
- 看图：旧 Stage A broad scout 本地目录已清理，不再保留图作为当前结果；本节仅保留历史结论。

### 运行与回收信息

- 远端 run root：
  - `nd-1`
    - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_measurement_noise_threshold_search_20260422_165518_stageA_nd1/`
  - `nd-2`
    - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_measurement_noise_threshold_search_20260422_165518_stageA_nd2/`
  - `nd-3`
    - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_measurement_noise_threshold_search_20260422_165518_stageA_nd3/`
- 本地统一归档目录：旧 Stage A broad scout 本地目录已清理，不作为当前结果保留。
- 原本地归档包含六个 `q_*` 子目录、overview 图、gap summary 图、summary JSON 和三台节点日志；当前不再把这些旧产物作为物理解读依据。

### 统一参数

- `code_family = 3d_toric`
- `L = [3,4,5]`
- `p = 0.04, 0.06, ..., 0.28`
- `q = [0.0010, 0.0025, 0.0050, 0.0100, 0.0200, 0.0400]`
- 每个 `(L,p,q)`：
  - `num_disorder_samples_total = 256`
  - `chunk_size = 16`
  - `num_chunks_per_point = 16`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 240`
  - `burn_in_scaling_reference_num_qubits = 18`
  - `common_random_disorder_across_p = true`
- workers：
  - `nd-1 = 48`
  - `nd-2 = 48`
  - `nd-3 = 96`

### 完成情况

- 三个 host 的主扫描都完整完成：
  - 每个 `q` 都是 `completed=624, failed=0, pending=0`
  - 六个 `q` 子目录都包含 merged `npz`、默认图、`sem95` 图、`gap_crossing` 图和 `threshold_summary.json`
- 远端 `nd-3` 的 host-level overview 正常生成。
- 远端 `nd-1` / `nd-2` 在 host-level overview 步骤失败，但不是采样失败。
  - 原因是旧版 `plot_threshold_search_overview.py` 假设每个 `q` 都有
    `recommended_server_window`
  - 对 `recommended_server_window = null` 的 `q` 会抛出 `TypeError`
  - 已在本地修复此脚本并成功重绘 Stage A 总览图

### 本地重绘与汇总产物

- 总览图与汇总表：旧 Stage A 本地重绘产物已清理，不作为当前结果保留；本节只记录当时得到的判据。

### 各 `q` 的核心判据

```text
q=0.0010
  boundary_saturation_artifact = true
  primary_crossing_window_hit  = false
  secondary_proximity_hit      = false
  recommended_server_window    = null
  right_edge_gap_signs         = {3-4: 0, 4-5: 0}

q=0.0025
  boundary_saturation_artifact = false
  primary_crossing_window_hit  = false
  secondary_proximity_hit      = false
  recommended_server_window    = null
  right_edge_gap_signs         = {3-4: +1, 4-5: -1}

q=0.0050
  boundary_saturation_artifact = true
  primary_crossing_window_hit  = false
  secondary_proximity_hit      = false
  recommended_server_window    = null
  right_edge_gap_signs         = {3-4: -1, 4-5: 0}

q=0.0100
  boundary_saturation_artifact = true
  primary_crossing_window_hit  = false
  secondary_proximity_hit      = false
  recommended_server_window    = null
  right_edge_gap_signs         = {3-4: +1, 4-5: -1}

q=0.0200
  boundary_saturation_artifact = false
  primary_crossing_window_hit  = false
  secondary_proximity_hit      = true
  recommended_server_window    ≈ [0.0400, 0.0693]
  right_edge_gap_signs         = {3-4: +1, 4-5: +1}

q=0.0400
  boundary_saturation_artifact = false
  primary_crossing_window_hit  = false
  secondary_proximity_hit      = true
  recommended_server_window    ≈ [0.0400, 0.1400]
  right_edge_gap_signs         = {3-4: +1, 4-5: +1}
```

### 端点行为

- 用 merged `npz` 直接检查窗口两端与中点，可见：

```text
q=0.0010
  p=0.04  -> [1.0000, 1.0000, 1.0000]
  p=0.16  -> [1.0000, 1.0000, 1.0000]
  p=0.28  -> [1.0000, 1.0000, 1.0000]

q=0.0200
  p=0.04  -> [0.9983, 0.9964, 0.9923]
  p=0.16  -> [0.9923, 0.9944, 0.9771]
  p=0.28  -> [0.9904, 0.9903, 0.9750]

q=0.0400
  p=0.04  -> [0.9901, 0.9782, 0.9743]
  p=0.16  -> [0.9560, 0.9301, 0.9021]
  p=0.28  -> [0.9059, 0.8971, 0.8355]
```

- 因而当前 Stage A 给出的图景是：
  - 极小 `q` 端：
    - `q=0.0010` 完全落在饱和平台上
    - `q=0.0025, 0.0050, 0.0100` 仍然强烈受平台/边界伪 crossing 影响
    - 这几组数据当前不足以定位真实 threshold 窗口
  - 中等 `q` 端：
    - `q=0.0200` 已开始离开完美平台
    - `L3-L4` 出现 interior sign flip，但 `L4-L5` 仍未共同过零
    - `q=0.0400` 没有 sign flip，只是两对 gap 在一段窗口内落入 pooled CI95

### 本轮判断

- 按预先设定的主判据，这轮 Stage A 的结论是：
  - 六个 `q` 都**没有**出现可信的三尺寸共同 interior crossing
  - 因而目前**不能**宣称 3D `q>0` 已观察到 finite threshold
- 其中最值得继续跟进的是：
  - `q=0.0200`
    - 已出现 secondary proximity
    - 且推荐窗口最窄，优先级最高
  - `q=0.0400`
    - 也有 secondary proximity
    - 但当前更像“接近”而非真正 crossing
- 对 `q≤0.0100`：
  - 当前 Stage A 窗口主要给出的是平台饱和或边界伪 crossing
  - 暂时不能据此判定“有 threshold”或“无 threshold”

### 下一步建议

- 若继续按原计划进入 Stage B，自适应细扫的优先顺序应改为：
  - 第一优先级：`q=0.0200`
    - 细扫 `p ≈ 0.0400 ~ 0.0700`
    - 步长 `0.005`
    - `num_disorder_samples_total = 512`
  - 第二优先级：`q=0.0400`
    - 细扫 `p ≈ 0.0400 ~ 0.1400`
    - 但这段窗口较宽，建议先压缩到
      `0.06, 0.08, ..., 0.14` 试跑一轮再决定是否 deep
- 对 `q = 0.0010, 0.0025, 0.0050, 0.0100`：
  - 先不要直接开 deep
  - 更合理的是先重做一轮更有判别力的 scout，再决定窗口该左移还是右移
  - 否则很容易被平台饱和与边界零 gap 误导

## 2026-04-22 21:16 修复后本地轻量预检复跑与 Stage A broad scout 重开

摘要：
- 做什么：在修复后的 hybrid `q>0` sampler 上重新跑本地 3D `q>0` 轻量预检，并据此重开远端 Stage A broad scout。
- 结论：本地 `q=0.0050` 轻量预检已经通过管线校验；但在 `p = 0.16, 0.22, 0.28` 上两条 pairwise gap 仍全为负，没有 interior sign flip，说明 crossing 仍在当前窗口右侧。这与 3D `q=0` 在 `p≈0.218~0.230` 已出现共同 interior crossing 的 calibration 明显不同。
- 当前状态：Stage A 已在 `nd-1/nd-2/nd-3` 重新挂入 `screen` 后台，等待回收合并。合并后第一优先级只看 pairwise gap 的符号翻转，不先看 `argmin gap`。

### 本地预检参数与产物

- 本地 run root：
  - `data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/`
- 参数：
  - `code_family = 3d_toric`
  - `L = [3,4,5]`
  - `q = 0.0050`
  - `p = [0.1600, 0.2200, 0.2800]`
  - `num_disorder_samples_total = 32`
  - `chunk_size = 16`
  - `num_chunks_per_point = 2`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 240`
  - `workers = 4`
  - `common_random_disorder_across_p = true`
- 完成情况：
  - `completed=18`
  - `failed=0`
  - `pending=0`

### 本地预检判读

- 看图：
  - [修复后本地预检 sem95 图](data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_sem95.png)
  - [修复后本地预检 gap 图](data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_gap_crossing.png)
- `threshold_summary.json` 读数：

```text
boundary_saturation_artifact = false
primary_crossing_window_hit  = false
secondary_proximity_hit      = true
recommended_server_window    = [0.28, 0.28]
right_edge_gap_signs         = {3-4: -1, 4-5: -1}
```

- pairwise gap 曲线符号：

```text
delta_34 sign pattern = [-1, -1, -1]
delta_45 sign pattern = [-1, -1, -1]

p = 0.16, 0.22, 0.28
delta_34 = [-0.1124, -0.1620, -0.0188]
delta_45 = [-0.0678, -0.1447, -0.0028]
```

- 判读：
  - 这轮预检不再出现修复前那种接近 `q_top=1` 的假饱和异常。
  - 但在当前窗口里，两条 pairwise gap 都没有翻正，因此 crossing 仍在窗口右侧。
  - 和 3D `q=0` 基线相比：
    - `q=0` 的可信 crossing 窗口仍在 `p≈0.218~0.230`
    - 且该基线窗口右端的 pairwise gap 已经翻成正号
    - 现在 `q=0.0050` 到 `p=0.28` 仍保持负号，说明其行为不能按 `q=0` 直接外推，必须继续看更右侧的 broad scout

### Remote Stage A 重开

- 远端 run id：
  - `nd-1`
    - `3d_toric_measurement_noise_threshold_search_20260422_211326_stageA_nd1`
    - `q = [0.0010, 0.0025]`
    - `workers = 48`
  - `nd-2`
    - `3d_toric_measurement_noise_threshold_search_20260422_211326_stageA_nd2`
    - `q = [0.0050, 0.0100]`
    - `workers = 48`
  - `nd-3`
    - `3d_toric_measurement_noise_threshold_search_20260422_211326_stageA_nd3`
    - `q = [0.0200, 0.0400]`
    - `workers = 96`
- 统一扫描参数：
  - `L = [3,4,5]`
  - `p = 0.0400, 0.0600, ..., 0.2800`
  - `num_disorder_samples_total = 256`
  - `chunk_size = 16`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 240`
  - `burn_in_scaling_reference_num_qubits = 18`
  - `common_random_disorder_across_p = true`
- 当前日志开头已确认：
  - `nd-1` 已启动 `q=0.0010`
  - `nd-2` 已启动 `q=0.0050`
  - `nd-3` 已启动 `q=0.0200`

### 合并后的读图规则

- 第一优先级：
  - 只看 pairwise gap 是否发生 interior sign flip
  - 特别是 `L=3-4` 与 `L=4-5` 是否在同一 `q` 上共同翻号
- 第二优先级：
  - 再看 `secondary_proximity_hit` 与推荐窗口
- 暂不采用：
  - 单独用 `argmin |gap|` 做任何 threshold 判读
- 所有新图都必须和 `q=0` 基线一起解释：
  - `q=0` calibration 仍以 `p≈0.218~0.230` 的共同 interior crossing 为准
  - 若 `q>0` 新图与这条 calibration 不一致，应先解释 pairwise gap 符号结构，再讨论任何“threshold 位置”

## 2026-04-22 21:4x 远端 Stage A 主动停止与产物清理

摘要：
- 做什么：在新一轮 3D `q>0` Stage A broad scout 启动后，主动停止 `nd-1/nd-2/nd-3` 的远端任务并清理中止产物。
- 原因：当前重新怀疑程序逻辑仍有错误，需要先回到本地继续排查，暂不继续消耗远端算力。
- 结论：本轮远端 Stage A 不再继续，也不保留任何 partial run 作为分析依据；仅保留修复后完整跑完的本地轻量预检 `exp05_q005_local_precheck_after_fix`。

### 主动停止的 run

- `nd-1`
  - `3d_toric_measurement_noise_threshold_search_20260422_211326_stageA_nd1`
- `nd-2`
  - `3d_toric_measurement_noise_threshold_search_20260422_211326_stageA_nd2`
- `nd-3`
  - `3d_toric_measurement_noise_threshold_search_20260422_211326_stageA_nd3`

### 清理情况

- 已停止：
  - 三台机器对应的 `screen`/采样进程
- 已删除：
  - `nd-1/nd-2/nd-3` 上这三轮 run 的远端 `runs/` 目录
  - 对应的远端 `repos/` 部署目录
  - 对应的远端日志文件
- 本地同步清理：
  - 删除第一次误参数启动后留下的半截预检目录
    - `data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/（旧 fixrerun 目录已清理，最终保留此目录）`
  - 删除用于发远端任务的临时 clone
    - `/tmp/projectD_stageA_launcher`

### 保留产物

- 继续保留并作为后续排错依据：
  - `data/3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix/`
- 保留原因：
  - 它是完整完成的本地轻量预检
  - `q=0.0050` 在 `p=0.16/0.22/0.28` 上给出了稳定的 pairwise gap 负号结构
  - 这份结果仍然对排查“为什么 `q>0` 行为和 `q=0` calibration 不一致”有直接价值

### 当前状态

- 暂停一切新的远端 broad scout / Stage B / deep 扫描。
- 后续优先级切换为：
  - 先做程序逻辑排查
  - 排查通过后，再决定是否重开 3D `q>0` Stage A

## 2026-04-23 10:4x `q>0` PT 生产基线冻结

摘要：
- 做什么：把已经接入的 multi-start + PT `q>0` 生产路径做一次完整本地基线冻结，覆盖 exact regression、混合诊断、production smoke，以及远端 launcher 参数透传。
- 结论：这版 `q>0` 生产管线已经满足“可以重开正式 broad scan”的最低条件；接下来不再使用旧 Stage A 记录做物理解读，而是只重开 `q=0.0050` 的可信 broad scan。
- 本节只记录代码冻结与本地验证，不把远端 broad scan 尚未完成的结果写成物理结论。

### 本地回归与诊断

- `conda run -n 12 python src/exact_enumeration.py`
  - 完整通过。
  - 关键锚点：
    - `3D q>0 exact-vs-MCMC regression` 通过
    - `3D q>0 exact-vs-PT regression` 通过
    - `3D q>0 production-path regression` 通过
  - 生产路径锚点误差：
    - `multistart |Δm_u|max = 0.0059`
    - `multistart |Δq_top| = 0.0007`
    - `multistart+pt |Δm_u|max = 0.0043`
    - `multistart+pt |Δq_top| = 0.0004`

- `conda run -n 12 python src/diagnose_3d_q_positive_mixing.py --suite c2`
  - 本地使用轻量 smoke 参数复核 `L=5, q=0.0050, p_cold=0.22`。
  - 关键诊断读数：
    - `num_chains_that_never_flipped_sector = 0`
    - `max_r_hat_across_disorders = 1.0056846205413785`
    - `mean_q_top_spread_per_disorder = 0.04775892857142857`
    - `winding_acceptance_rate_mean = 2.0942622950819675e-04`
  - 判读：
    - sector 互通已恢复，没有再出现“完全不翻 sector”的冻结。
    - 这组轻量参数下 `q_top spread` 仍偏大，因此它只能说明 PT 路径在动，不能替代正式生产预算。

### Production smoke

- 本地 `production_chunked_scan.py submit` 小预算 smoke 已完成：
  - `code_family = 3d_toric`
  - `L = [3,4,5]`
  - `p = [0.22, 0.26]`
  - `q = 0.0050`
  - `num_disorder_samples_total = 2`
  - `num_start_chains = 4`
  - `num_replicas_per_start = 2`
  - `pt_p_hot = 0.44`
  - `pt_num_temperatures = 5`
  - `num_measurements_per_disorder = 32`
- 产物：
  - merged NPZ 已生成
  - PNG 已生成
  - convergence sidecar JSON 已生成
  - merged NPZ 内已含 `converged_mask_matrix`
- convergence gate 结果：
  - `6 / 6` 点均未通过 gate
  - 失败主因是低预算下
    - `min_effective_sample_size <= 200`
    - `mean_q_top_spread >= 0.03`
- 判读：
  - 这正是 smoke 预算应有的行为，说明 gate 已真正接进 merge 产物，而不是只在单独脚本里做人读诊断。

### 远端 launcher 冻结

- 已更新 `scripts/launch_3d_measurement_noise_threshold_search.sh`，远端正式提交现在会显式透传：
  - `--num-start-chains`
  - `--num-replicas-per-start`
  - `--pt-p-hot`
  - `--pt-num-temperatures`
  - `--pt-swap-attempt-every-num-sweeps`
- 本地 `DRY_RUN=1` 已核对：
  - `NUM_START_CHAINS = 8`
  - `NUM_REPLICAS_PER_START = 2`
  - `PT_P_HOT = 0.44`
  - `PT_NUM_TEMPERATURES = 9`
  - `PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS = 1`
- 这一步修复后，远端 broad scan 不会再 silently 掉回旧单链模板。

### 下一步固定动作

- 第一轮正式 run 只做：
  - `q = 0.0050`
  - `L = [3,4,5]`
  - `p = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]`
- 正式预算固定为：
  - `num_disorder_samples_total = 4`
  - `chunk_size = 4`
  - `num_start_chains = 8`
  - `num_replicas_per_start = 2`
  - `num_burn_in_sweeps = 1200`
  - `num_sweeps_between_measurements = 6`
  - `num_measurements_per_disorder = 4096`
  - `pt_p_hot = 0.44`
  - `pt_num_temperatures = 9`
  - `common_random_disorder_across_p = true`
- 合并后判读顺序固定为：
  - 先看 `*_convergence.json` / `converged_mask_matrix`
  - 再看 pairwise gap 符号结构
  - 未通过 gate 的点只保留数值，不作为可信物理解读依据

## 2026-04-24 3D zero-disorder single-sample quick scan

摘要：
- 做什么：按“快速测试技巧”只取全零 disorder 构型，不做 disorder average，快速扫描两条 3D `q_top` 截面。
- 结论：两条截面都没有出现稳定三尺寸 crossing。`q_top` 在绝大多数点等于或非常接近 `1`，只能说明全零 disorder 分支在这组快速预算下保持强拓扑有序，不能替代 disorder-averaged 物理结论。
- 注意：本节所有结果都是 `all-zero disorder / num_disorder_samples=1 / no disorder averaging`，不用于估计正式阈值。

### 运行与产物

- 远端运行：
  - `nd-3`
  - `conda run -n 11`
  - `workers = 24`
  - `num_measurements_per_point = 128`
  - `num_burn_in_sweeps = 600`
  - `num_sweeps_between_measurements = 4`
  - `num_start_chains = 8`
  - `num_replicas_per_start = 1`
  - `pt_p_hot = 0.44`
  - `pt_num_temperatures = 7`
- 两条扫描：
  - fixed-q：`q = 0.0050`，`p = 0.10, 0.12, ..., 0.30`，`L = 3,4,5`
  - fixed-p：`p = 0.0050`，`q = 0.00, 0.02, ..., 0.20`，`L = 3,4,5`
- 运行时间：
  - fixed-q：`2026-04-23 17:08` 到 `18:21 CST`
  - fixed-p rerun：`2026-04-23 21:50` 到 `23:02 CST`
- 本地同步目录：
  - `data/3d_toric_code/with_measurement_noise/exp06_zero_disorder_quick_scan/`
- 关键产物：
  - `zero_disorder_combined_analysis.png`
  - `zero_disorder_combined_summary.json`
  - `fixed_q/fixed_q_q0p0050.npz`
  - `fixed_p/fixed_p_p0p0050.npz`

### 数值摘要

- 数据完整性：
  - `observed_syndrome_weight_matrix` 全为 `0`
  - `disorder_data_weight_matrix` 全为 `0`
  - 因此确实是全零 disorder 分支
- fixed-q (`q=0.0050`)：
  - `q_top_min = 0.98444693`
  - `max(1 - q_top) = 0.01555307`
  - 最大偏离出现在 `L=3, p=0.30`
  - `L=4` 与 `L=5` 在整条 `p=0.10~0.30` 上保持 `q_top=1`
  - `max |L3-L4 gap| = 0.01555307`
  - `max |L4-L5 gap| = 0`
  - 没有 `L3-L4` 或 `L4-L5` 的 sign-change crossing
- fixed-p (`p=0.0050`)：
  - `q_top_min = 0.99777004`
  - `max(1 - q_top) = 0.00222996`
  - 只有 `L=3, q=0.16` 与 `L=5, q=0.20` 出现单点轻微下降
  - `max |L3-L4 gap| = 0.00222996`
  - `max |L4-L5 gap| = 0.00222996`
  - 没有稳定 sign-change crossing
- 诊断：
  - fixed-q 最大 finite `R-hat = 1.00463498`
  - fixed-p 最大 finite `R-hat = 1.0`
  - 最小 effective sample size 为 `128`，等于本轮轻量测量数
  - fixed-q 在 `L=3, p=0.30` 的 `q_top_spread` 达到 `0.05231585`，提示该点的轻量预算仍有 multi-start spread

### 判读

- 这轮全零 disorder 结果没有支持“在扫描窗口内有清晰 finite-size crossing”。
- fixed-q 中较小尺寸 `L=3` 在高 `p` 端先下降，而 `L=4/5` 仍保持 `q_top=1`，更像有限采样/小尺寸先动，而不是可靠阈值信号。
- fixed-p 中 `q` 增大到 `0.20` 也没有造成系统性塌缩，说明在 `p=0.0050` 的全零 disorder 分支上，syndrome-noise 参数本身没有在这组预算内打破拓扑有序。
- 后续正式物理解读仍必须回到 disorder-averaged PT 生产扫描；本轮的价值是提供一个快速 sanity check：全零分支没有显示导致先前疑问的明显异常相变结构。

### 脚本修复记录

- 新增 `src/zero_disorder_quick_scan.py`：
  - 用全 `1` 的 precomputed uniform 数组强制生成全零 disorder
  - 每点调用现有 `run_disorder_average_simulation(...)`
  - 输出 NPZ/JSON/PNG
- 新增 `src/analyze_zero_disorder_quick_scan.py`：
  - 合并 fixed-q 与 fixed-p 结果
  - 生成 combined analysis plot 与 summary JSON
- fixed-p 首次运行暴露混合 `q=0/q>0` 诊断矩阵处理 bug：
  - 原脚本把整条 fixed-p scan 当成同一种诊断分支
  - 已修复为 `q=0` 点填 `q0_*` 诊断，`q>0` 点填 `q_top_spread/R-hat/ESS` 诊断，另一侧用 `NaN`
  - 本地 mixed-q smoke 通过后，远端 fixed-p rerun 成功

## 2026-04-24 3D `q=0.0050` disorder-averaged PT broad scan 回收

摘要：
- 做什么：回收并本地重绘 `nd-2` 上的正式 3D toric `q=0.0050` disorder-averaged broad scan。
- 结论：pairwise gaps 给出一个共同 interior crossing 指示，线性估计窗口为 `p≈0.2067~0.2086`，代表值 `p≈0.20765`。但本轮只有 `4` disorder，低 `p` 端 CI 很宽，且 `L=5,p=0.20` 未通过 convergence gate，因此还不能作为最终 threshold，只能作为 deep scan 的窗口定位。
- 下一步：围绕 `p=0.19~0.24` 做 `16` disorder deep run，步长建议 `0.005`。

### 运行与产物

- 远端 run：
  - `MASTER_RUN_ID = 3d_toric_measurement_noise_threshold_search_20260423_104417`
  - host：`nd-2`
  - remote root：`/home/DATA1/users/yuany/.single_shot/runs/3d_toric_measurement_noise_threshold_search_20260423_104417/q_0p0050`
- 本地同步目录：
  - `data/3d_toric_code/with_measurement_noise/exp07_q005_broad_scan/q_0p0050/`
- 关键产物：
  - `scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_search_common_random.npz`
  - `scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_search_common_random_sem95.png`
  - `scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_search_common_random_gap_crossing.png`
  - `scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_search_common_random_convergence.json`
  - `threshold_summary.json`

### 参数

- `L = [3,4,5]`
- `p = [0.20,0.22,0.24,0.26,0.28,0.30]`
- `q = 0.0050`
- `num_disorder_samples_total = 4`
- `chunk_size = 4`
- `num_start_chains = 8`
- `num_replicas_per_start = 2`
- `num_measurements_per_disorder = 4096`
- `num_sweeps_between_measurements = 6`
- `pt_p_hot = 0.44`
- `pt_num_temperatures = 9`
- `common_random_disorder_across_p = true`

### 数值摘要

`q_top` 矩阵：

```text
L=3: [0.55388, 0.40799, 0.16907, 0.08846, 0.04046, 0.01831]
L=4: [0.64430, 0.28805, 0.17657, 0.04186, 0.01811, 0.00564]
L=5: [0.65252, 0.27178, 0.16328, 0.02594, 0.00694, 0.00434]
```

Pairwise gaps：

```text
L3-L4: [-0.09043,  0.11994, -0.00751, 0.04660, 0.02235, 0.01267]
L4-L5: [-0.00821,  0.01627,  0.01330, 0.01593, 0.01117, 0.00131]
```

- `threshold_summary.json`：
  - `primary_crossing_window_hit = true`
  - common crossing window：`p_min = 0.2067082609839296`, `p_max = 0.20859685913546228`
  - representative：`p = 0.20765256005969596`
  - raw crossing window：`0.2067082609839296~0.2427754024873967`
- pairwise crossing estimates：
  - `L3-L4`: `0.20860`, `0.23882`, `0.24278`
  - `L4-L5`: `0.20671`
- convergence gate：
  - `17 / 18` 点通过
  - 唯一失败点：`L=5, p=0.20`
  - 失败原因：`mean_q_top_spread >= 0.030`
  - 该点 metrics：`mean_q_top_spread = 0.03281944`, `max_r_hat = 1.00099135`, `min_ESS = 715.079`, `mean_pt_min_swap_acceptance_rate = 0.29018`
- 诊断范围：
  - `max_r_hat <= 1.00099`
  - `min_ESS >= 665.56`
  - PT swap transport 正常，所有点 `mean_pt_min_swap_acceptance_rate > 0`

### 判读

- 这轮 broad scan 第一次给出 `q=0.0050` 下的可用 crossing 窗口：`p≈0.2077`。
- 低 `p` 端的 `q_top` 误差条很宽，尤其 `p=0.20/0.22`；因此 crossing 的存在更像“需要 deep 验证的候选”，不是最终阈值估计。
- `L3-L4` gap 在 `0.22~0.26` 间还有额外 sign flips，说明 `4` disorder 下曲线仍有统计波动；deep run 应用更多 disorder 收紧这个区域。
- 与 zero-disorder scan 对照：
  - zero-disorder 分支在相同或更宽窗口内几乎保持 `q_top≈1`
  - disorder-averaged 曲线显著下降并出现 crossing 候选
  - 这说明正式物理信号来自 disorder average，不应由全零 disorder 分支外推

### 下一步参数

推荐 deep run：

```text
L = 3,4,5
q = 0.0050
p = 0.190,0.195,0.200,0.205,0.210,0.215,0.220,0.225,0.230,0.235,0.240
num_disorder_samples_total = 16
chunk_size = 4
num_start_chains = 8
num_replicas_per_start = 2
num_measurements_per_disorder = 4096
num_sweeps_between_measurements = 6
pt_p_hot = 0.44
pt_num_temperatures = 9
```

若先压预算，可先跑 `p=0.200~0.230`、步长 `0.005`，保持其余参数不变。

## 2026-04-25 3D 实验目录统一重命名

摘要：
- 做什么：把 `data/3d_toric_code/` 下所有实验结果目录统一改为 `expNN_做什么` 命名，并补全目录级 `README.md`。
- 为什么做：此前目录名混合了 host、timestamp、stage 和临时修复名，难以判断先后顺序和每次重跑原因。
- 当前结论：后续查 3D 数据优先读 [3D 数据索引](data/3d_toric_code/README.md)，再进入具体 `expNN` 目录。

重命名后的主索引：

```text
without_measurement_noise/
  exp01_q0_pipeline_smoke
  exp02_q0_low_p_scout
  exp03_q0_right_shift_scout
  exp04_q0_crossing_window_scout
  exp09_q0_oneday_deep_fixed

with_measurement_noise/
  exp05_q005_local_precheck_after_fix
  exp06_zero_disorder_quick_scan
  exp07_q005_broad_scan
  exp08_q005_oneday_deep_scan
```

每个目录的 `README.md` 固定包含：
- 实验目的
- 为什么做
- 当前结论
- 主看图或 summary 入口

## 2026-04-25 3D `q=0` one-day deep 重跑完成

摘要：
- 做什么：停掉此前错误的 `q=0` run 后，在 `nd-1` 重新启动一天内快速 deep 的 `q=0` 基线实验。
- 结论：`384/384` chunks 全部完成；本地重跑 threshold 分析后得到共同 interior crossing window `p≈0.2146~0.2391`，代表点 `p≈0.2268`。
- 用途：作为当前 one-day deep 系列的 `q=0` calibration 基线，可与 `q=0.0050` / `q=0.0100` 结果对照。

### 运行与产物

- 远端 host：`nd-1`
- 远端 run root：
  - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_exp10_q0_oneday_deep_relaunch_20260425/q_0p0000`
- 本地目录：
  - `data/3d_toric_code/without_measurement_noise/exp10_q0_oneday_deep_relaunch/`
- 主产物：
  - `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch.npz`
  - `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch.png`
  - `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch_sem95.png`
  - `scan_result_multi_L_3d_toric_q0_oneday_deep_relaunch_gap_crossing.png`
  - `threshold_summary.json`

### 参数

```text
L = 3,4,5
p = 0.205,0.210,0.215,0.220,0.225,0.230,0.235,0.240
q = 0
num_disorder_samples_total = 256
chunk_size = 16
q0_num_start_chains = 8
num_start_chains = 8
num_replicas_per_start = 1
num_burn_in_sweeps = 1000
num_sweeps_between_measurements = 6
num_measurements_per_disorder = 240
common_random_disorder_across_p = true
```

注意：本轮 q=0 命令不包含任何 `--pt-*` 参数，避免触发 `q=0` 与 PT 的不兼容路径。

### 数值摘要

`q_top` 矩阵：

```text
L=3: [0.35583, 0.32534, 0.29850, 0.26304, 0.23165, 0.20454, 0.17466, 0.14830]
L=4: [0.40401, 0.34686, 0.29650, 0.25029, 0.21387, 0.17733, 0.14954, 0.12750]
L=5: [0.51275, 0.43556, 0.37493, 0.31023, 0.24699, 0.19645, 0.15577, 0.12615]
```

`threshold_summary.json`：

- `primary_crossing_window_hit = true`
- common crossing window：
  - `p_min = 0.21457655911521925`
  - `p_max = 0.23910998018700172`
  - representative `p = 0.22684326965111049`
- pairwise gap crossing:
  - `L3-L4` crossing near `p≈0.21458`
  - `L4-L5` crossing near `p≈0.23911`

Pairwise gaps：

```text
L3-L4: [-0.04818, -0.02152,  0.00199,  0.01275,  0.01778,  0.02721,  0.02513,  0.02081]
L4-L5: [-0.10874, -0.08871, -0.07843, -0.05994, -0.03311, -0.01911, -0.00623,  0.00135]
```

### 判读

- 这轮重跑恢复了合理的 `q=0` calibration：`L3-L4` 与 `L4-L5` 两条 gap 都在窗口内部出现 sign flip。
- representative `p≈0.2268` 与此前 `exp04_q0_crossing_window_scout` 的 `p≈0.218~0.230` calibration 一致，说明当前 one-day q=0 基线没有明显漂移。
- 窗口宽度仍较大，主要来自 `L4-L5` crossing 落在右端 `p≈0.239` 附近；若要精化 q=0 阈值，可在 `p=0.215~0.240` 内继续加密，但当前已经足够作为 q>0 对照。

## 2026-04-25 3D `q=0.0100` one-day deep 停止与 partial 分析

摘要：
- 做什么：停止 `nd-3` 上过慢的 `q=0.0100` one-day deep run，下载已有数据并做 partial L3-L4 分析。
- 结论：只有 `28/42` chunks 完成；`L=3,4` 完成，`L=5` 的 `14` 个正式 chunks 全部缺失。因此本轮不能给出三尺寸 threshold，只能作为失败诊断。
- 当前信号：L3-L4 gap 在 `p=0.17~0.23` 全部为负，没有 crossing；但 CI 很宽，且 `L=4` 高 `p` 端 convergence 多数失败。

### 运行与产物

- 远端 host：`nd-3`
- 停止前 screen：
  - `533858.ssprep_3d_toric_oneday_deep_q0p010_20260424`
- 远端 run root：
  - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_oneday_deep_q0p010_20260424/q_0p0100`
- 本地目录：
  - `data/3d_toric_code/with_measurement_noise/exp11_q001_oneday_deep_partial/`
- 主产物：
  - `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4.npz`
  - `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4.png`
  - `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4_sem95.png`
  - `scan_result_multi_L_3d_toric_q0p0100_partial_L3_L4_gap.png`
  - `partial_L3_L4_analysis_summary.json`

### 参数

原计划：

```text
L = 3,4,5
p = 0.170,0.180,0.190,0.200,0.210,0.220,0.230
q = 0.0100
num_disorder_samples_total = 6
chunk_size = 3
num_start_chains = 8
num_replicas_per_start = 2
pt_p_hot = 0.44
pt_num_temperatures = 9
pt_swap_attempt_every_num_sweeps = 1
num_burn_in_sweeps = 1000
num_sweeps_between_measurements = 6
num_measurements_per_disorder = 3072
common_random_disorder_across_p = true
```

实际完成：

```text
completed chunks = 28
pending chunks = 14
完成尺寸 = L=3,4
缺失尺寸 = L=5
```

### Partial L3-L4 数值摘要

`q_top`：

```text
L=3: [0.61762, 0.52301, 0.42590, 0.39006, 0.38174, 0.28764, 0.26255]
L=4: [0.81496, 0.75833, 0.76402, 0.69565, 0.58911, 0.34212, 0.26602]
```

L3-L4 gap：

```text
[-0.19734, -0.23532, -0.33812, -0.30559, -0.20737, -0.05449, -0.00347]
```

L3-L4 pooled CI95：

```text
[0.37237, 0.39610, 0.38140, 0.35676, 0.35435, 0.32553, 0.28483]
```

Convergence gate：

```text
L=3: [pass, fail, pass, pass, pass, pass, pass]
L=4: [pass, pass, pass, fail, fail, fail, fail]
```

诊断范围：

- `max_r_hat <= 1.00047`
- `min_ESS >= 1127`
- `mean_q_top_spread` 在 `L=4,p>=0.20` 达到 `0.0318~0.0418`，是主要失败原因之一

### 判读

- 本轮 `q=0.0100` 不能用于 threshold 宣称，因为三尺寸分析的关键 `L=5` 完全缺失。
- 已有 L3-L4 数据没有 sign flip，gap 到 `p=0.23` 仍为负；但 CI95 远大于 gap，说明 `6` disorder 的统计方差仍然太大。
- `L=4` 高 `p` 端 convergence gate 系统性失败，提示当前参数即使只看 `L=4` 也不足够稳。
- 下一轮若继续 `q=0.0100`，不建议照搬本轮 `L=5` 参数硬跑；更实际的方案是先缩小窗口并降低单 chunk 尾部风险，例如：
  - 先跑 `p=0.22,0.24,0.26` 的 `L=5` smoke，确认耗时和 convergence
  - 或把 `num_measurements_per_disorder` 降到 `1536`，先用更多 disorder 控制 disorder 方差
  - 若目标是一天内完成，优先保证 `L=3,4,5` 都有数据，而不是让 `L=5` 尾部吞掉整轮预算

## 2026-04-25 3D `q=0.0050/0.0100` Numba 后 overnight runs 与左侧综合分析

摘要：
- 做什么：回收 Numba fast path 后的三节点正式结果 `exp12~exp17`，并对 `q=0.0100` 左侧窗口生成综合分析 `exp18_q001_left_combined_summary`。
- 结论：`q=0.0100` 不应继续往 `p>0.24` 扫。右侧 coarse/fine 结果显示大 `p` 已经偏过 crossing 区域；左侧补样本池化后把重点窗口收缩到 `p≈0.22~0.235`。
- 当前不能宣称最终 threshold：池化 dense 数据中 `L3-L4` crossing 约 `p≈0.2233`，但 `L4-L5` 到 `p=0.230` 仍略负；还需要在 `0.225~0.235` 继续加 disorder 或做更窄窗口复本。

### 运行与产物

本地目录：

```text
data/3d_toric_code/with_measurement_noise/exp12_q005_fine_20260425_nd1
data/3d_toric_code/with_measurement_noise/exp13_q001_coarse_20260425_nd2
data/3d_toric_code/with_measurement_noise/exp14_q001_fine_20260425_nd3
data/3d_toric_code/with_measurement_noise/exp15_q001_left_denseA_20260425_nd1
data/3d_toric_code/with_measurement_noise/exp16_q001_left_denseB_20260425_nd2
data/3d_toric_code/with_measurement_noise/exp17_q001_left_fine_20260425_nd3
data/3d_toric_code/with_measurement_noise/exp18_q001_left_combined_summary
```

完成状态：

```text
exp12 q=0.0050 fine:        528/528 chunks, failed=0
exp13 q=0.0100 coarse:      528/528 chunks, failed=0
exp14 q=0.0100 fine-right:  480/480 chunks, failed=0
exp15 q=0.0100 denseA:      768/768 chunks, failed=0
exp16 q=0.0100 denseB:      768/768 chunks, failed=0
exp17 q=0.0100 fine-left:   792/792 chunks, failed=0
```

主看图：

- [q=0.0100 pooled dense q_top](data/3d_toric_code/with_measurement_noise/exp18_q001_left_combined_summary/q001_left_dense_pooled_sem95.png)
- [q=0.0100 pooled dense gap](data/3d_toric_code/with_measurement_noise/exp18_q001_left_combined_summary/q001_left_dense_pooled_gap_ci95.png)
- [q=0.0100 all-run gap comparison](data/3d_toric_code/with_measurement_noise/exp18_q001_left_combined_summary/q001_left_gap_comparison_all_runs.png)
- [q=0.0100 dense-vs-fine gap detail](data/3d_toric_code/with_measurement_noise/exp18_q001_left_combined_summary/q001_left_gap_detail_dense_vs_fine.png)
- [machine-readable summary](data/3d_toric_code/with_measurement_noise/exp18_q001_left_combined_summary/q001_left_combined_summary.json)

### 参数摘要

`exp12`:

```text
q = 0.0050
p = 0.200,0.205,...,0.250
num_disorder_samples_total = 32
chunk_size = 2
num_measurements_per_disorder = 2048
num_start_chains = 8
num_replicas_per_start = 1
pt_num_temperatures = 7
max_effective_num_burn_in_sweeps = 4000
```

`exp13/14`:

```text
q = 0.0100
exp13 p = 0.220,0.230,...,0.320, 16 disorder
exp14 p = 0.225,0.230,...,0.270, 16 disorder
num_measurements_per_disorder = 1536
pt_num_temperatures = 7
max_effective_num_burn_in_sweeps = 3000
```

`exp15/16/17`:

```text
q = 0.0100
exp15/16 p = 0.195,0.200,...,0.230, 64 disorder each
exp17 p = 0.205,0.2075,...,0.230, 48 disorder
num_measurements_per_disorder = 1536
pt_num_temperatures = 7
max_effective_num_burn_in_sweeps = 3000
```

### 数值摘要

`q=0.0050 exp12`:

```text
primary_crossing_window_hit = false
recommended window          = p≈0.2164~0.2200
right_edge_gap_signs        = {L3-L4: +, L4-L5: +}
```

`q=0.0100 exp13/14` 右侧检查：

```text
exp13 recommended window = p≈0.2400~0.2900
exp14 recommended window = p≈0.2250~0.2700
right_edge_gap_signs     = {L3-L4: +, L4-L5: +}
```

这两轮的关键作用不是给 threshold，而是说明 `p>0.24` 已经太靠右；继续往更大 `p` 扫会浪费时间。

`q=0.0100 exp15+exp16` 池化 dense 结果，`128` disorder：

```text
p:      [0.195, 0.200, 0.205, 0.210, 0.215, 0.220, 0.225, 0.230]
L3-L4: [-0.16096, -0.13740, -0.10564, -0.07481, -0.03588, -0.01341,  0.00666,  0.01536]
L4-L5: [-0.08052, -0.05088, -0.04156, -0.04969, -0.06047, -0.05669, -0.04128, -0.02143]
```

线性 crossing：

```text
L3-L4 crossing ≈ 0.22334
L4-L5 crossing: not observed by p=0.230
```

`q=0.0100 exp17` 左侧 fine grid：

```text
L3-L4 stays negative from p=0.205 to 0.230
L4-L5 only crosses near the right edge, around p≈0.2294
```

### 判读

- `q=0.0100` 的旧失败不是因为 threshold 必然在更右侧，而是因为 `0.20~0.23` 左侧窗口 disorder 方差太大。
- `exp13/14` 已足够说明 `p>0.24` 不值得优先继续扩展；那里两条 gap 多数已为正，物理上更像已过 crossing 区。
- `exp15+16` 池化后明显压低了 error bar，并把 `L3-L4` crossing 稳定到 `p≈0.223` 附近。
- 但 `L4-L5` 仍未在 `p<=0.230` 明确翻号，所以目前最合理的下一轮不是大范围扫描，而是在 `p=0.225~0.240` 做窄窗口高 disorder 复本。
- 当前结论应表述为：`q=0.0100` threshold 候选窗口在 `p≈0.22~0.235`，尚未完成三尺寸共同 crossing 的最终确认。

### 下一步建议

若继续 `q=0.0100`：

```text
L = 3,4,5
p = 0.2225,0.2250,0.2275,0.2300,0.2325,0.2350,0.2375,0.2400
num_disorder_samples_total = 96 或 128
chunk_size = 2
num_measurements_per_disorder = 1536
num_start_chains = 8
num_replicas_per_start = 1
pt_num_temperatures = 7
max_effective_num_burn_in_sweeps = 3000
```

若 convergence 诊断显示 `mean_q_top_spread` 仍系统性偏高，再考虑只在 `L=5` 或关键 `p` 点提高 `num_measurements_per_disorder` / `replicas`，不要先扩大 `p` 范围。
