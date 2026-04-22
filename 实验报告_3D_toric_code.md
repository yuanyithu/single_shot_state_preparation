# 实验报告：3D Toric Code

按是否考虑 measurement noise 重排原始实验记录。

## Without Measurement Noise

## 2026-04-21 23:42 3D Toric `q=0` 首轮实现与本地 Smoke

摘要：
- 做什么：把 3D toric `q=0` 接入统一生产扫描管线，并做本地 smoke。
- 结论：3D 路径已能稳定产出 merged `npz` 与图，但 smoke 不足以判读 threshold。
- 看图：[scan_result_multi_L_3d_toric_q0_smoke.png](data/3d_toric_code/without_measurement_noise/q0_smoke_local_20260421_234202/scan_result_multi_L_3d_toric_q0_smoke.png)

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
  - `data/3d_toric_code/without_measurement_noise/q0_smoke_local_20260421_234202/`
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
- 看图：[scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png](data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png)

### 运行与回收信息

- 远端运行目录：
  - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260421_235447/`
- 本地归档目录：
  - `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/`
- 主要文件：
  - 结果：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout.npz`
  - 默认图：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout.png`
  - 95% CI 图：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout_sem95.png`
  - gap 图：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout_gap_crossing.png`
  - 本地重跑分析图：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout_local_sem95.png`
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png`
  - manifest：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/manifest.json`
  - summary：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_nd1_20260421_235447/threshold_summary.json`

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
- 看图：[scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png](data/3d_toric_code/without_measurement_noise/q0_threshold_scout_stageB_nd3_20260422_101018/local_reanalysis/scan_result_multi_L_3d_toric_q0_threshold_scout_local_gap_crossing.png)

### 运行与归档

- `Stage A`：`nd-1` 右移 scout
  - 远端目录：
    `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_100557_rightshift_stageA/`
  - 本地归档：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_stageA_nd1_20260422_100557/`
- `Stage B`：`nd-3` extension scout
  - 远端目录：
    `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_101018_extension_stageB_nd3/`
  - 本地归档：
    `data/3d_toric_code/without_measurement_noise/q0_threshold_scout_stageB_nd3_20260422_101018/`
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
> `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_scout_local/precheck_20260422/`
> 与
> `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_search_stageA_20260422_165518/`
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
  - `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_scout_local/precheck_20260422/`
  - `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_search_stageA_20260422_165518/`
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
- 看图：[scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_sem95.png](data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_scout_local/precheck_20260422/q_0p0050/scan_result_multi_L_3d_toric_q0p0050_measurement_noise_threshold_scout_common_random_sem95.png)

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
  - `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_scout_local/precheck_20260422/`
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
- 看图：[measurement_noise_threshold_search_gap_summary.png](data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_search_stageA_20260422_165518/measurement_noise_threshold_search_gap_summary.png)

### 运行与回收信息

- 远端 run root：
  - `nd-1`
    - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_measurement_noise_threshold_search_20260422_165518_stageA_nd1/`
  - `nd-2`
    - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_measurement_noise_threshold_search_20260422_165518_stageA_nd2/`
  - `nd-3`
    - `/home/DATA1/users/yuany/.single_shot/runs/3d_toric_measurement_noise_threshold_search_20260422_165518_stageA_nd3/`
- 本地统一归档目录：
  - `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_search_stageA_20260422_165518/`
- 本地归档包含：
  - 六个 `q_*` 子目录
  - `measurement_noise_threshold_search_sem95_overview.png`
  - `measurement_noise_threshold_search_gap_summary.png`
  - `measurement_noise_threshold_search_summary.json`
  - `nd1_stageA.log`
  - `nd2_stageA.log`
  - `nd3_stageA.log`

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

- 总览图：
  - `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_search_stageA_20260422_165518/measurement_noise_threshold_search_sem95_overview.png`
  - `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_search_stageA_20260422_165518/measurement_noise_threshold_search_gap_summary.png`
- 汇总表：
  - `data/3d_toric_code/with_measurement_noise/measurement_noise_threshold_search_stageA_20260422_165518/measurement_noise_threshold_search_summary.json`

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
